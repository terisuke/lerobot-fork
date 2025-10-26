# Phase 5: SO-101 Stereo Vision Integration - Implementation Guide

## 📋 Overview

This document describes the Phase 5 implementation for integrating stereo vision capabilities into the SO-101 robot follower. This is part of the larger Phase 5-7 roadmap for end-to-end stereo vision training and deployment.

**Implementation Status:** ✅ **Phase 5 Complete**

## 🎯 What Was Implemented

### 1. SO101FollowerConfig Update
**File:** `src/lerobot/robots/so101_follower/config_so101_follower.py`

**Changes:**
- Added `OpenCVCameraConfig` import
- Updated `cameras` field with default stereo camera configuration:
  - Left camera: `index_or_path=0` (1280x720 @ 30fps)
  - Right camera: `index_or_path=1` (1280x720 @ 30fps)
- Added `stereo_calibration_path` field for calibration file path

**Key Point:** Uses `index_or_path` field (NOT `camera_index`), as per the actual OpenCVCameraConfig implementation.

### 2. Stereo Integration Test Script
**File:** `examples/stereo/05_test_with_so101.py`

**Features:**
- Tests stereo camera integration with SO-101
- Side-by-side stereo view
- Optional depth map visualization (with calibration)
- Real-time depth statistics
- Frame saving capability

**Usage:**
```bash
# Basic test (side-by-side view)
python examples/stereo/05_test_with_so101.py

# With depth estimation
python examples/stereo/05_test_with_so101.py --show-depth

# Custom camera indices
python examples/stereo/05_test_with_so101.py --left-index 0 --right-index 1

# Custom calibration
python examples/stereo/05_test_with_so101.py \
    --calibration configs/camera/stereo_calibration.yaml \
    --show-depth
```

**Controls:**
- `q` - Quit
- `d` - Toggle depth view
- `s` - Save current frames

### 3. Dataset Quality Check Script
**File:** `examples/stereo/06_check_dataset_quality.py`

**Features:**
- Validates dataset integrity (episodes, frames, FPS)
- Verifies stereo camera keys (`observation.images.left`, `observation.images.right`)
- Checks image dimensions and quality
- Validates action and state data presence
- Episode-specific analysis

**Usage:**
```bash
# Check a HuggingFace Hub dataset
python examples/stereo/06_check_dataset_quality.py \
    --repo-id username/so101-stereo-pickup

# Check local dataset only
python examples/stereo/06_check_dataset_quality.py \
    --repo-id username/so101-stereo-pickup \
    --local-files-only

# Check specific episode
python examples/stereo/06_check_dataset_quality.py \
    --repo-id username/so101-stereo-pickup \
    --episode-index 0
```

### 4. Success Rate Calculator
**File:** `examples/stereo/07_calculate_success_rate.py`

**Features:**
- Calculates success rate from lerobot-eval info JSON
- Reads 'pc_success' percentage from eval_info.json output
- Generates summary statistics with rewards
- Saves results to JSON

**Usage:**
```bash
# Calculate success rate from eval_info.json
python examples/stereo/07_calculate_success_rate.py \
    --eval-info outputs/eval/so101_stereo_results/eval_info.json

# With custom output file
python examples/stereo/07_calculate_success_rate.py \
    --eval-info outputs/eval/so101_stereo_results/eval_info.json \
    --output results.json
```

**Exit Codes:**
- `0` - Success rate >= 60%
- `1` - Success rate < 60% (needs improvement)

## 📁 Project Structure

```
lerobot/
├── src/lerobot/
│   ├── robots/so101_follower/
│   │   └── config_so101_follower.py (✅ Updated)
│   ├── cameras/opencv/
│   │   ├── camera_opencv.py (✅ Used)
│   │   └── configuration_opencv.py (✅ Used)
│   └── utils/stereo/
│       ├── calibration.py (✅ Used from Phase 1-4)
│       └── depth_estimation.py (✅ Used from Phase 1-4)
└── examples/stereo/
    ├── 05_test_with_so101.py (✅ New)
    ├── 06_check_dataset_quality.py (✅ New)
    └── 07_calculate_success_rate.py (✅ New)
```

## ✅ Verification Checklist

### Phase 5 Completion Criteria

- [x] SO101FollowerConfig updated with stereo camera configuration
- [x] Uses `index_or_path` field (correct field name)
- [x] Integration test script created and syntax-checked
- [x] Dataset quality check script created and syntax-checked
- [x] Success rate calculator script created and syntax-checked
- [x] All scripts executable with proper permissions
- [x] Documentation completed

## 🚀 Next Steps: Phase 6 & 7

### Phase 6: Data Collection (4-5 days)

**Prerequisites:**
1. Stereo calibration completed:
   ```bash
   python examples/stereo/02_calibrate_stereo.py \
       --left-index 0 --right-index 1 \
       --output configs/camera/stereo_calibration.yaml
   ```

2. Hardware setup:
   - Dual webcams mounted and aligned
   - SO-101 robot follower and leader connected
   - Workspace configured with good lighting

**Data Collection Command:**
```bash
# Set HuggingFace username
export HF_USER=$(huggingface-cli whoami | head -n 1)

# Pilot collection (5 episodes)
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841 \
    --robot.cameras='{ \
        left: {type: opencv, index_or_path: 0, fps: 30, width: 1280, height: 720}, \
        right: {type: opencv, index_or_path: 1, fps: 30, width: 1280, height: 720} \
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --dataset.repo_id=${HF_USER}/so101-stereo-pilot \
    --dataset.tags="so101,stereo,pilot" \
    --dataset.num_episodes=5 \
    --dataset.episode_time_s=15 \
    --dataset.reset_time_s=10 \
    --dataset.fps=30 \
    --display_data=true

# Verify data quality
python examples/stereo/06_check_dataset_quality.py \
    --repo-id ${HF_USER}/so101-stereo-pilot

# Production collection (50 episodes)
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841 \
    --robot.cameras='{ \
        left: {type: opencv, index_or_path: 0, fps: 30, width: 1280, height: 720}, \
        right: {type: opencv, index_or_path: 1, fps: 30, width: 1280, height: 720} \
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/tty.usbmodem58760431551 \
    --dataset.repo_id=${HF_USER}/so101-stereo-pickup \
    --dataset.tags="so101,stereo,pick-place" \
    --dataset.single_task="Pick up red cube and place in box" \
    --dataset.num_episodes=50 \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=15 \
    --dataset.fps=30 \
    --dataset.push_to_hub=true \
    --display_data=true
```

### Phase 7: Training & Evaluation (1-2 weeks)

**Training Command:**
```bash
# ACT Policy Training
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=${HF_USER}/so101-stereo-pickup \
    --training.offline_steps=80000 \
    --training.batch_size=8 \
    --training.eval_freq=1000 \
    --training.save_freq=5000 \
    --training.save_checkpoint=true \
    --training.num_workers=4 \
    --policy.device=mps \
    --output_dir=outputs/train/so101_stereo_act \
    --wandb.enable=true \
    --wandb.project=lerobot-so101-stereo \
    --wandb.notes="SO-101 with stereo cameras"
```

**Evaluation Command:**
```bash
# Evaluate trained policy
lerobot-eval \
    --env.type=so101_real \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.usbmodem585A0076841 \
    --robot.cameras='{ \
        left: {type: opencv, index_or_path: 0, fps: 30, width: 1280, height: 720}, \
        right: {type: opencv, index_or_path: 1, fps: 30, width: 1280, height: 720} \
    }' \
    --policy.path=outputs/train/so101_stereo_act/checkpoints/last/pretrained_model \
    --eval.n_episodes=10 \
    --output_dir=outputs/eval/so101_stereo_results \
    --policy.device=mps

# Calculate success rate from eval_info.json
python examples/stereo/07_calculate_success_rate.py \
    --eval-info outputs/eval/so101_stereo_results/eval_info.json
```

## 🔧 Troubleshooting

### Camera Connection Issues

**Problem:** Cameras not detected
```bash
# Find available cameras
lerobot-find-cameras opencv

# Test individual cameras
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read()[0])"
python -c "import cv2; cap = cv2.VideoCapture(1); print(cap.read()[0])"
```

**Solution:** Ensure both cameras are connected and have different indices.

### Calibration Issues

**Problem:** Calibration file not found
```bash
# Check if calibration exists
test -f configs/camera/stereo_calibration.yaml && \
  echo "✅ Calibration exists" || \
  echo "❌ Run calibration first"

# Run calibration
python examples/stereo/02_calibrate_stereo.py \
    --left-index 0 --right-index 1 \
    --output configs/camera/stereo_calibration.yaml
```

### Port Configuration

**Problem:** Cannot connect to robot motors
```bash
# Find motor ports
ls /dev/tty.usb*

# Update port in command
# Replace: robot.port: "/dev/tty.usbmodem585A0076841"
# With your actual port from ls output
```

## 📊 Expected Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 5: Integration | 2-3 days | ✅ **Complete** |
| Phase 6: Data Collection | 4-5 days | 🔜 Next |
| Phase 7: Training & Eval | 10-14 days | ⏸️ Pending |

## 🎓 Key Learnings

### Important Configuration Details

1. **Field Name:** Use `index_or_path`, NOT `camera_index`
   ```python
   # ✅ Correct
   OpenCVCameraConfig(index_or_path=0)

   # ❌ Wrong
   OpenCVCameraConfig(camera_index=0)
   ```

2. **Official Entry Points:** Always use official lerobot-* commands
   ```bash
   # ✅ Correct
   lerobot-train
   lerobot-eval
   lerobot-record
   lerobot-dataset-viz

   # ❌ Wrong
   lerobot-visualize-dataset  # Not registered
   python -m lerobot.scripts.*  # Deprecated
   ```

3. **Camera Configuration:** Use dot notation for CLI arguments
   ```bash
   # ✅ Correct
   --robot.cameras='{ \
       left: {type: opencv, index_or_path: 0}, \
       right: {type: opencv, index_or_path: 1} \
   }'

   # ❌ Wrong
   --robot-overrides='{ robot.cameras: {...} }'  # Not supported
   ```

## 📚 References

- [SO-101 Setup Guide](https://huggingface.co/docs/lerobot/so101)
- [LeRobot Getting Started](https://huggingface.co/docs/lerobot/getting_started_real_world_robot)
- [Main Implementation Guide](../../CLAUDE.md)
- [Stereo Utilities Documentation](README.md)

## 🤝 Support

For issues or questions:
- Check [LeRobot Discord](https://discord.gg/lerobot)
- Review [GitHub Issues](https://github.com/huggingface/lerobot/issues)
- Consult project documentation in `CLAUDE.md`

---

**Phase 5 Implementation Completed:** October 26, 2025

**Ready for Phase 6:** Data collection can now begin!
