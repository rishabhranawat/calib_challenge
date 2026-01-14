"""
Camera Calibration via Focus of Expansion (FOE) Estimation

This module solves the comma.ai calibration challenge by estimating camera pitch and yaw
angles from dashcam video using optical flow analysis.

Key Insight:
-----------
Camera calibration (pitch/yaw offset between camera and vehicle axes) is essentially
CONSTANT throughout a video - the camera mount doesn't move. Therefore, instead of
per-frame smoothing, we collect FOE estimates from all frames and take the median
to get a robust video-level calibration estimate.

Algorithm Overview:
------------------
1. For each frame pair, detect Shi-Tomasi corners and track with Lucas-Kanade optical flow
2. Validate tracks using forward-backward consistency check
3. Filter out low-magnitude flows (unreliable at low speeds)
4. Estimate Focus of Expansion (FOE) using RANSAC on the linear constraint system
5. Convert FOE to pitch/yaw angles using camera intrinsics
6. Take median of all frame estimates as the final calibration (robust to outliers)

The FOE is the point in the image where all optical flow vectors appear to radiate from
during forward motion. Its offset from the image center gives the camera calibration angles.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class FlowConfig:
    """Configuration parameters for the FOE-based calibration predictor."""

    # Camera intrinsics
    focal_length_px: float = 910.0  # Focal length in pixels (given by challenge)

    # Region of Interest - focus on road area, exclude sky and hood
    roi_top_frac: float = 0.40      # Exclude top 40% (sky, trees)
    roi_bottom_frac: float = 0.90   # Exclude bottom 10% (hood)
    roi_horizontal_margin_frac: float = 0.10  # Exclude 10% on each side

    # Frame preprocessing
    gaussian_kernel: int = 5  # Gaussian blur kernel size for noise reduction

    # Feature detection (Shi-Tomasi corners)
    feature_count: int = 3000       # Maximum corners to detect per frame
    feature_quality: float = 0.01   # Minimum quality level for corners
    feature_min_distance: int = 7   # Minimum distance between corners (pixels)

    # Lucas-Kanade optical flow tracking
    lk_win_size: int = 21    # Search window size
    lk_max_level: int = 3    # Pyramid levels for multi-scale tracking

    # Flow validation thresholds
    fb_check_thresh: float = 0.5   # Max forward-backward error (pixels)
    min_flow_magnitude: float = 7.0  # Min flow magnitude to include (filters low-speed frames)
    min_points_for_ransac: int = 30  # Minimum points needed for robust estimation

    # RANSAC parameters for FOE estimation
    ransac_iterations: int = 1000   # Number of RANSAC iterations
    ransac_residual_thresh: float = 1.0  # Inlier threshold (pixels)
    max_flow_weight: float = 25.0   # Cap on flow magnitude for weighted fitting

    # Random seed for reproducibility
    seed: int = 13


class FlowFOEPredictor:
    """
    Predicts camera calibration (pitch/yaw) from video using optical flow and FOE estimation.

    The key insight is that calibration is constant per video, so we estimate FOE for each
    frame and take the median across all frames for a robust final estimate.
    """

    def __init__(self, config: FlowConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(config.seed)

    def predict(self, video_path: Path) -> np.ndarray:
        """
        Process a video and return per-frame calibration predictions.

        Since calibration is constant, all frames get the same predicted value
        (the median of per-frame FOE estimates).

        Args:
            video_path: Path to the input video file

        Returns:
            Array of shape (num_frames, 2) with [pitch, yaw] per frame
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        prev_gray: Optional[np.ndarray] = None
        roi_mask: Optional[np.ndarray] = None
        cx = cy = None
        frame_count = 0
        raw_measurements: List[np.ndarray] = []

        # Process all frames and collect FOE-based angle estimates
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            gray = self._preprocess_frame(frame)

            if roi_mask is None:
                roi_mask = self._build_roi_mask(gray.shape)
                h, w = gray.shape
                cx, cy = w / 2.0, h / 2.0  # Principal point (image center)

            if prev_gray is None:
                prev_gray = gray
                continue

            # Estimate angles from optical flow between consecutive frames
            measurement = self._estimate_angles(prev_gray, gray, roi_mask, cx, cy)
            if measurement is not None:
                raw_measurements.append(measurement)
            prev_gray = gray

        cap.release()

        if not raw_measurements:
            return np.zeros((frame_count or 1, 2), dtype=np.float32)

        # Compute robust video-level calibration using median
        # Median is robust to outliers from turns, stops, and tracking failures
        all_measurements = np.vstack(raw_measurements)
        calibration = np.median(all_measurements, axis=0)

        # Return constant calibration for all frames
        return np.tile(calibration, (frame_count, 1)).astype(np.float32)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Convert to grayscale and apply Gaussian blur for noise reduction."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.config.gaussian_kernel > 0:
            k = self.config.gaussian_kernel
            if k % 2 == 0:
                k += 1  # Ensure odd kernel size
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        return gray

    def _build_roi_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """
        Build a mask for the region of interest (road area).

        Excludes sky (top), hood (bottom), and edges where distortion is higher.
        """
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        y0 = int(self.config.roi_top_frac * h)
        y1 = int(self.config.roi_bottom_frac * h)
        margin = int(self.config.roi_horizontal_margin_frac * w)
        mask[y0:y1, margin:w-margin] = 255
        return mask

    def _estimate_angles(
        self,
        prev_gray: np.ndarray,
        curr_gray: np.ndarray,
        roi_mask: np.ndarray,
        cx: float,
        cy: float,
    ) -> Optional[np.ndarray]:
        """
        Estimate pitch and yaw angles from optical flow between two frames.

        Pipeline:
        1. Detect Shi-Tomasi corners in previous frame
        2. Track corners to current frame using Lucas-Kanade
        3. Validate with forward-backward consistency check
        4. Filter by flow magnitude (removes low-speed noise)
        5. Estimate FOE using RANSAC
        6. Convert FOE offset to angles
        """
        # Step 1: Detect corners (Shi-Tomasi)
        points = cv2.goodFeaturesToTrack(
            prev_gray,
            mask=roi_mask,
            maxCorners=self.config.feature_count,
            qualityLevel=self.config.feature_quality,
            minDistance=self.config.feature_min_distance,
            blockSize=7,
        )
        if points is None or len(points) < self.config.min_points_for_ransac:
            return None

        # Step 2: Track corners forward (prev -> curr)
        next_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, points, None,
            winSize=(self.config.lk_win_size, self.config.lk_win_size),
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if next_pts is None:
            return None

        # Step 3: Track corners backward (curr -> prev) for validation
        back_pts, back_status, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, prev_gray, next_pts, None,
            winSize=(self.config.lk_win_size, self.config.lk_win_size),
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if back_pts is None:
            return None

        # Reshape for easier manipulation
        pts0 = points.reshape(-1, 2)
        pts1 = next_pts.reshape(-1, 2)
        pts_back = back_pts.reshape(-1, 2)
        status = status.reshape(-1)
        back_status = back_status.reshape(-1)

        # Forward-backward consistency check: reject tracks where
        # tracking forward then backward doesn't return to start
        fb_err = np.linalg.norm(pts0 - pts_back, axis=1)
        good = (status == 1) & (back_status == 1) & (fb_err < self.config.fb_check_thresh)
        pts0, pts1 = pts0[good], pts1[good]

        if len(pts0) < self.config.min_points_for_ransac:
            return None

        # Step 4: Filter by flow magnitude (removes noisy low-speed estimates)
        flows = pts1 - pts0
        mags = np.linalg.norm(flows, axis=1)
        valid = mags > self.config.min_flow_magnitude
        pts0, flows = pts0[valid], flows[valid]

        if len(pts0) < self.config.min_points_for_ransac:
            return None

        # Step 5: Estimate FOE using RANSAC
        foe = self._estimate_foe_ransac(pts0, flows)
        if foe is None:
            return None

        # Step 6: Convert FOE to pitch/yaw angles
        # FOE offset from principal point, normalized by focal length
        dx = (foe[0] - cx) / self.config.focal_length_px
        dy = (foe[1] - cy) / self.config.focal_length_px

        yaw = math.atan(dx)    # Horizontal angle (left/right)
        pitch = -math.atan(dy)  # Vertical angle (up/down), negated for coordinate convention

        return np.array([pitch, yaw], dtype=np.float32)

    def _estimate_foe_ransac(self, points: np.ndarray, flows: np.ndarray) -> Optional[np.ndarray]:
        """
        Estimate Focus of Expansion using RANSAC for robustness to outliers.

        The FOE constraint: for a point (x,y) with flow (u,v), the FOE (x0,y0) satisfies:
            (x - x0) * v = (y - y0) * u

        Rearranging to linear form: v*x0 - u*y0 = v*x - u*y
        This gives us a linear system A @ [x0, y0] = b

        RANSAC finds the FOE that maximizes inlier count, then refits on inliers
        with flow-magnitude weighting (larger flows are more reliable).
        """
        n = len(points)
        if n < 2:
            return None

        best_inliers: Optional[np.ndarray] = None
        best_model: Optional[np.ndarray] = None
        A_all, b_all = self._build_linear_system(points, flows)

        # RANSAC loop: sample minimal set, fit, count inliers
        for _ in range(self.config.ransac_iterations):
            idx = self._rng.choice(n, size=2, replace=False)
            model = self._solve_linear(A_all[idx], b_all[idx])
            if model is None:
                continue

            residuals = np.abs(A_all @ model - b_all)
            inliers = residuals < self.config.ransac_residual_thresh

            if best_inliers is None or np.count_nonzero(inliers) > np.count_nonzero(best_inliers):
                best_inliers = inliers
                best_model = model

        if best_inliers is None or best_model is None:
            return None

        inlier_idx = best_inliers.nonzero()[0]
        if len(inlier_idx) < 2:
            return None

        # Refit on inliers with flow-magnitude weighting
        # Larger flows give more reliable constraints
        weights = np.minimum(
            np.linalg.norm(flows[inlier_idx], axis=1),
            self.config.max_flow_weight,
        )
        foe = self._solve_linear(
            A_all[inlier_idx] * weights[:, None],
            b_all[inlier_idx] * weights,
        )
        return foe

    @staticmethod
    def _build_linear_system(points: np.ndarray, flows: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the linear system for FOE estimation.

        From constraint: v*x0 - u*y0 = v*x - u*y
        Matrix form: [v, -u] @ [x0, y0]^T = v*x - u*y
        """
        u, v = flows[:, 0], flows[:, 1]
        x, y = points[:, 0], points[:, 1]
        A = np.stack([v, -u], axis=1)
        b = v * x - u * y
        return A, b

    @staticmethod
    def _solve_linear(A: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
        """Solve linear system using least squares."""
        try:
            solution, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(solution)):
            return None
        return solution


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict camera calibration (pitch/yaw) using optical flow and FOE estimation."
    )
    parser.add_argument("--video", type=Path, help="Path to a single video file.")
    parser.add_argument("--output", type=Path, help="Path to save predictions for --video.")
    parser.add_argument(
        "--video-dir", type=Path,
        help="Directory containing numbered .hevc files (e.g., labeled/ or unlabeled/).",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Directory where numbered .txt predictions will be saved.",
    )
    parser.add_argument(
        "--ids", type=str, nargs="+",
        help="Video IDs to process when using --video-dir/--output-dir. Example: 0 1 2 3 4",
    )
    parser.add_argument(
        "--focal-length", type=float, default=910.0,
        help="Effective focal length in pixels.",
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for RANSAC.")
    return parser.parse_args()


def run() -> None:
    args = parse_args()
    config = FlowConfig(focal_length_px=args.focal_length, seed=args.seed)
    predictor = FlowFOEPredictor(config)

    if args.video:
        if args.output is None:
            raise SystemExit("--output is required when --video is provided")
        if args.output.parent:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        predictions = predictor.predict(args.video)
        np.savetxt(args.output, predictions, fmt="%.6f")
        return

    if args.video_dir and args.output_dir:
        ids = args.ids or sorted(p.stem for p in args.video_dir.glob("*.hevc"))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for vid_id in ids:
            video_path = args.video_dir / f"{vid_id}.hevc"
            output_path = args.output_dir / f"{vid_id}.txt"
            predictions = predictor.predict(video_path)
            np.savetxt(output_path, predictions, fmt="%.6f")
        return

    raise SystemExit("Provide either --video/--output or --video-dir/--output-dir (with optional --ids).")


if __name__ == "__main__":
    run()
