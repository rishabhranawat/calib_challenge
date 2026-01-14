# Camera Calibration Challenge Solution

## Final Score: 7.77% (Target: <25%)

This document describes the solution methodology for the comma.ai camera calibration challenge.

## Problem Overview

The challenge is to predict camera calibration angles (pitch and yaw) from dashcam video. These angles represent the offset between where the camera is pointing versus the vehicle's direction of travel. Accurate calibration is essential for autonomous driving systems.

- **Input**: 1-minute dashcam videos at 20 fps (1,200 frames each)
- **Output**: Per-frame pitch and yaw angles in radians
- **Evaluation**: MSE relative to a zero-baseline (lower is better)

## Key Insight

**Calibration is constant per video** - the camera mount doesn't move during driving. This is the crucial insight that improved our score from 449% to 7.77%.

Instead of trying to smooth per-frame estimates (which are noisy due to turns, stops, etc.), we collect FOE estimates from all frames and take the **median** as a robust video-level calibration estimate.

## Algorithm

### 1. Focus of Expansion (FOE) Theory

When a vehicle moves forward, the optical flow pattern forms a radial expansion from a single point called the **Focus of Expansion (FOE)**. If the camera was perfectly aligned with the vehicle axis, the FOE would be at the image center (principal point). Any offset of the FOE from the center indicates the camera calibration angles.

```
pitch = -atan((FOE_y - cy) / focal_length)
yaw   =  atan((FOE_x - cx) / focal_length)
```

### 2. Per-Frame Pipeline

For each consecutive frame pair:

1. **Feature Detection**: Detect up to 3000 Shi-Tomasi corners in the ROI (road area only, excluding sky and hood)

2. **Optical Flow Tracking**: Track features using Lucas-Kanade pyramidal optical flow

3. **Forward-Backward Validation**: Track matched points back to original frame. Reject tracks where round-trip error exceeds 0.5 pixels

4. **Flow Magnitude Filtering**: Discard flows with magnitude < 7.0 pixels (removes unreliable low-speed estimates)

5. **RANSAC FOE Estimation**:
   - FOE constraint: `(x - x0) * v = (y - y0) * u` for point (x,y) with flow (u,v)
   - Run 1000 RANSAC iterations to find FOE with most inliers
   - Refit on inliers with flow-magnitude weighting

6. **Angle Conversion**: Convert FOE offset to pitch/yaw using focal length

### 3. Video-Level Aggregation

Take the **median** of all per-frame estimates. The median is robust to outliers from:
- Vehicle turns (FOE moves during turns)
- Stops and slow speeds (filtered, but some slip through)
- Tracking failures
- Moving objects in the scene

## Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `focal_length_px` | 910 | Camera focal length (given) |
| `roi_top_frac` | 0.40 | Exclude top 40% (sky) |
| `roi_bottom_frac` | 0.90 | Exclude bottom 10% (hood) |
| `fb_check_thresh` | 0.5 | Forward-backward error threshold |
| `min_flow_magnitude` | 7.0 | Minimum flow to include |
| `ransac_iterations` | 1000 | RANSAC iterations for FOE |
| `ransac_residual_thresh` | 1.0 | Inlier threshold |

## Results

### Per-Video Breakdown

| Video | GT Pitch | Pred Pitch | GT Yaw | Pred Yaw | MSE |
|-------|----------|------------|--------|----------|-----|
| 0 | 1.94° | 1.81° | 1.84° | 2.03° | 0.000010 |
| 1 | 3.51° | 3.78° | 0.63° | 0.43° | 0.000020 |
| 2 | 1.64° | 1.56° | 3.24° | 2.73° | 0.000078 |
| 3 | 1.32° | 0.82° | 1.55° | 1.85° | 0.000467 |
| 4 | 0.72° | 0.67° | 3.01° | 3.09° | 0.000015 |

### Summary

- **Overall MSE**: 0.000118
- **Zero Baseline MSE**: 0.001518
- **Error Score**: 7.77%

## Evolution of the Solution

| Change | Score |
|--------|-------|
| Initial implementation (per-frame smoothing) | 449.52% |
| Fixed pitch sign convention | 319.52% |
| **Video-level median (key insight)** | 16.80% |
| Tighter ROI (40-90% vertical) | 16.38% |
| min_flow_magnitude = 3.0 | 9.86% |
| min_flow_magnitude = 6.0 | 8.43% |
| min_flow_magnitude = 7.0, fb_check = 0.5 | **7.77%** |

## What Didn't Work

- **Per-frame smoothing (EMA)**: Calibration is constant, smoothing just adds lag
- **Trimmed mean**: Slightly worse than median (18.88% vs 16.80%)
- **Too tight fb_check (0.3)**: Filters out too many valid points
- **Too high min_flow (10.0)**: Loses too many frames, score jumped to 18.26%

## Usage

```bash
# Generate predictions for labeled videos
python foe_predictor.py --video-dir labeled/ --output-dir outputs/ --ids 0 1 2 3 4

# Evaluate
python eval.py outputs/

# Generate for unlabeled (submission)
python foe_predictor.py --video-dir unlabeled/ --output-dir outputs/ --ids 5 6 7 8 9
```

## Dependencies

- Python 3.x
- OpenCV (`cv2`)
- NumPy

## Files

- `foe_predictor.py` - Main prediction code with detailed comments
- `eval.py` - Evaluation script
- `outputs/` - Generated predictions
- `labeled/` - Training videos with ground truth
- `unlabeled/` - Test videos for submission
