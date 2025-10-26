# Stereo Vision for LeRobot

Dual webcam stereo vision system for SO-101 robot with depth perception capabilities.

---

## 📋 Overview

This directory contains the complete stereo vision implementation for LeRobot's SO-101 robot arm:

- **Phase 1-4**: Stereo calibration and depth estimation utilities
- **Phase 5**: SO-101 integration, dataset validation, and evaluation tools

**Status:** ✅ Production-ready for data collection and training

---

## 🚀 Quick Start

### 1. Stereo Calibration

Calibrate your dual webcam setup before use:

```bash
python examples/stereo/02_calibrate_stereo.py \
    --left-index 0 --right-index 1 \
    --output configs/camera/stereo_calibration.yaml
```

**Controls:**
- `SPACE` - Capture calibration image
- `q` - Quit and save calibration

### 2. Test Stereo Vision

Verify your stereo setup:

```bash
# Test depth estimation
python examples/stereo/03_test_depth.py \
    --left-index 0 --right-index 1 \
    --calibration configs/camera/stereo_calibration.yaml

# Test SO-101 integration
python examples/stereo/05_test_with_so101.py --show-depth
```

### 3. Data Collection

Record demonstrations with stereo cameras:

```bash
export HF_USER=$(huggingface-cli whoami | head -n 1)

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
    --dataset.num_episodes=50
```

### 4. Dataset Validation

Check dataset quality after recording:

```bash
python examples/stereo/06_check_dataset_quality.py \
    --repo-id ${HF_USER}/so101-stereo-pickup
```

### 5. Training & Evaluation

Train a policy with stereo vision:

```bash
# Train ACT policy
lerobot-train \
    --policy.type=act \
    --dataset.repo_id=${HF_USER}/so101-stereo-pickup \
    --policy.device=mps \
    --output_dir=outputs/train/so101_stereo_act

# Evaluate trained model
lerobot-eval \
    --env.type=so101_real \
    --robot.type=so101_follower \
    --robot.cameras='{ \
        left: {type: opencv, index_or_path: 0, fps: 30, width: 1280, height: 720}, \
        right: {type: opencv, index_or_path: 1, fps: 30, width: 1280, height: 720} \
    }' \
    --policy.path=outputs/train/so101_stereo_act/checkpoints/last/pretrained_model \
    --eval.n_episodes=10 \
    --output_dir=outputs/eval/so101_stereo_results

# Calculate success rate
python examples/stereo/07_calculate_success_rate.py \
    --eval-info outputs/eval/so101_stereo_results/eval_info.json
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `02_calibrate_stereo.py` | Stereo camera calibration tool |
| `03_test_depth.py` | Depth estimation testing |
| `05_test_with_so101.py` | SO-101 stereo integration test |
| `06_check_dataset_quality.py` | Dataset validation tool |
| `07_calculate_success_rate.py` | Evaluation success rate calculator |
| `PHASE5_IMPLEMENTATION.md` | Complete Phase 5 implementation guide |

---

## 🔧 Hardware Requirements

- **Cameras**: 2x USB webcams (1280x720 recommended)
- **Robot**: SO-101 follower + leader arms
- **USB Hub**: Powered USB 3.0 hub recommended
- **Calibration**: A4 checkerboard pattern (9x6 corners)

---

## 📚 Camera Configuration

The stereo system uses OpenCV cameras with the following default configuration:

```python
left_camera:
  type: opencv
  index_or_path: 0
  fps: 30
  width: 1280
  height: 720

right_camera:
  type: opencv
  index_or_path: 1
  fps: 30
  width: 1280
  height: 720
```

To find available cameras:

```bash
lerobot-find-cameras opencv
```

---

## 🐛 Troubleshooting

### Camera Not Found

```bash
# List available cameras
lerobot-find-cameras opencv

# Test individual cameras
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.read()[0])"
python -c "import cv2; cap = cv2.VideoCapture(1); print(cap.read()[0])"
```

### Calibration Issues

- Ensure good lighting and clear checkerboard visibility
- Capture at least 20 images from various angles and distances
- Move checkerboard slowly for best results

### Port Configuration

```bash
# Find robot motor ports
ls /dev/tty.usb*

# Update --robot.port and --teleop.port accordingly
```

---

## 📖 Documentation

For detailed implementation information, see:
- **[PHASE5_IMPLEMENTATION.md](PHASE5_IMPLEMENTATION.md)** - Complete Phase 5 guide
- **[Main Documentation](../../CLAUDE.md)** - Project overview
- **[LeRobot Docs](https://huggingface.co/docs/lerobot)** - Official documentation

---

## 🤝 Support

- **Discord**: [LeRobot Community](https://discord.gg/lerobot)
- **Issues**: [GitHub Issues](https://github.com/huggingface/lerobot/issues)
- **Documentation**: [HuggingFace Docs](https://huggingface.co/docs/lerobot)

---

**Implementation Date:** October 2025
**Status:** ✅ Production-ready
