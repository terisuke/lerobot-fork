# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LeRobot is a state-of-the-art machine learning framework for real-world robotics in PyTorch. This repository contains models, datasets, and tools for robotics applications, with a focus on imitation learning and reinforcement learning.

**This fork** extends the base LeRobot framework with Intel RealSense D435 depth camera integration and object detection capabilities for the SO-101 robot arm.

## Development Commands

### Environment Setup

```bash
# Create and activate conda environment (Python 3.10+)
conda create -y -n lerobot python=3.10
conda activate lerobot

# Install ffmpeg (required)
conda install ffmpeg -c conda-forge

# Install from source (editable mode)
pip install -e .

# Install with specific extras (simulation environments)
pip install -e ".[aloha,pusht]"

# Install with hardware support
pip install -e ".[feetech,intelrealsense]"

# Install development dependencies
pip install -e ".[dev,test]"
```

### Testing

```bash
# Run all tests
pytest tests -vv

# Run specific test module
pytest tests/policies/ -vv

# Run with coverage
pytest tests -vv --cov=lerobot --cov-report=html

# Run end-to-end tests (via Makefile)
make test-end-to-end DEVICE=cpu
make test-act-ete-train DEVICE=cpu
make test-diffusion-ete-eval DEVICE=cpu
```

### Code Quality

```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Auto-format with ruff
ruff format .

# Lint and auto-fix issues
ruff check . --fix

# Type checking (gradually enabled per module)
mypy src/lerobot/configs
mypy src/lerobot/cameras

# Security scanning
bandit -c pyproject.toml -r src/
```

### Training & Evaluation

```bash
# Train a policy from config
lerobot-train --config_path=lerobot/diffusion_pusht

# Train with custom parameters
lerobot-train \
  --policy.type=act \
  --env.type=aloha \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --batch_size=32 \
  --steps=100000 \
  --wandb.enable=true

# Evaluate a trained policy
lerobot-eval \
  --policy.path=outputs/train/.../checkpoints/.../pretrained_model \
  --env.type=aloha \
  --eval.n_episodes=50

# Visualize a dataset
lerobot-dataset-viz \
  --repo-id lerobot/pusht \
  --episode-index 0
```

### Robot Operations

```bash
# Find connected cameras
lerobot-find-cameras

# Find motor ports
lerobot-find-port

# Calibrate motors
lerobot-calibrate

# Setup motors
lerobot-setup-motors

# Record demonstrations
lerobot-record

# Teleoperate robot
lerobot-teleoperate
```

## Architecture Overview

### Core Components

**Policies** (`src/lerobot/policies/`)
- Policy implementations organized by type (ACT, Diffusion, TDMPC, SmolVLA, etc.)
- `vision_policy.py` - Custom vision-based policy with object detection integration
- Each policy has its own subdirectory with model architecture and configuration

**Datasets** (`src/lerobot/datasets/`)
- `LeRobotDataset` - Core dataset class using HuggingFace datasets backend (Arrow/Parquet)
- Supports temporal frame queries via `delta_timestamps`
- Video compression using MP4 to save space
- `realsense_dataset.py` - Custom dataset recorder for RealSense depth camera data

**Robots** (`src/lerobot/robots/`)
- Robot control abstractions for various hardware platforms
- `so100_follower/`, `so101_follower/` - SO-100/SO-101 robot arm implementations
- `hope_jr/` - HopeJR humanoid robot arm
- `lekiwi/` - Mobile robot platform

**Object Detection** (`src/lerobot/object_detection/`) - **CUSTOM EXTENSION**
- `detector.py` - YOLOv8-based object detection with RealSense integration
- `depth_tracker.py` - Depth-aware object tracking
- `tracker.py` - Generic object tracking utilities
- `utils.py` - Detection utilities and helpers

**Cameras** (`src/lerobot/cameras/`)
- Camera interface abstractions
- OpenCV, RealSense, and other camera backends

**Motors** (`src/lerobot/motors/`)
- Motor control for Dynamixel, Feetech servos

**Environments** (`src/lerobot/envs/`)
- Simulation environment wrappers (gym-aloha, gym-pusht, etc.)

**Configuration** (`src/lerobot/configs/`)
- Draccus-based configuration system
- Type-safe config dataclasses in `types.py`
- Training/eval configs in `train.py`, `eval.py`

### Data Flow Architecture

1. **Data Collection**: Robot teleoperation → Camera capture → Dataset recording
2. **Training**: Dataset loading → Policy training → Checkpoint saving
3. **Inference**: Camera stream → Object detection → Policy inference → Robot control

### LeRobotDataset Format

The dataset format uses HuggingFace datasets (Arrow/Parquet) with this structure:

```python
dataset.hf_dataset:
  ├ observation.images.cam_high (VideoFrame): {path: str, timestamp: float32}
  ├ observation.state (list[float32]): robot joint positions
  ├ action (list[float32]): target joint positions
  ├ episode_index (int64): episode identifier
  ├ frame_index (int64): frame index within episode
  ├ timestamp (float32): episode timestamp
  ├ next.done (bool): episode termination flag
  └ index (int64): global frame index

dataset.meta:
  ├ info: {codebase_version, fps, features, total_episodes, total_frames, robot_type}
  ├ episodes: DataFrame with episode metadata
  ├ stats: {max, mean, min, std} per feature
  └ tasks: DataFrame of task information
```

## RealSense Camera Integration

### Current Status

⚠️ **IMPORTANT**: RealSense D435 integration is experimental on macOS (Apple Silicon):
- Connection is unstable on macOS with frequent "failed to set power state" errors
- Currently operates in **mock mode** by default (actual camera disabled)
- Linux/Windows recommended for production use
- Alternative: Dual webcam stereo vision implementation (planned)

### Hardware Requirements

- Intel RealSense D435 depth camera
- Powered USB 3.0 hub (recommended)
- USB-C to USB-C cable for better power delivery
- Requires `sudo` privileges on macOS

### Testing RealSense System

```bash
# Test integrated object picking system (mock mode)
sudo python examples/object_picking/test_integrated_system.py --test all

# Test individual components
sudo python examples/object_picking/test_integrated_system.py --test camera
sudo python examples/object_picking/test_integrated_system.py --test detection
sudo python examples/object_picking/test_integrated_system.py --test tracking
```

### Switching from Mock Mode to Real Camera

To enable actual RealSense camera (when hardware is stable):

1. In `src/lerobot/object_detection/detector.py`:
   - Change `self.camera_available = False` to `True`
   - Uncomment `self._setup_realsense()`

2. In `src/lerobot/datasets/realsense_dataset.py`:
   - Change `self.camera_available = False` to `True`
   - Uncomment `self._setup_camera()`

## Important Development Notes

### Code Style

- Python 3.10+ required
- PyTorch 2.2+ required
- Use Ruff for formatting and linting (configured in `pyproject.toml`)
- Type hints gradually being added (mypy enabled for specific modules)
- Google-style docstrings
- Pre-commit hooks enforce style automatically

### Testing Guidelines

- Tests use pytest
- Mock hardware in tests (see `tests/mocks/`)
- CI runs fast tests on every PR, full tests nightly
- End-to-end tests verify policy training/evaluation pipelines
- Hardware-dependent tests should be skipped in CI

### Configuration System

Uses Draccus for type-safe configs:
- Override via CLI: `--policy.type=act --batch_size=32`
- Load from file: `--config_path=path/to/config.json`
- Configs are dataclasses in `src/lerobot/configs/`

### Adding New Policies

1. Create subdirectory in `src/lerobot/policies/your_policy/`
2. Implement policy class inheriting from appropriate base
3. Add config dataclass to `src/lerobot/configs/policies.py`
4. Register in `src/lerobot/policies/factory.py`
5. Add tests in `tests/policies/`

### Dataset Management

Datasets hosted on HuggingFace Hub under `lerobot/` organization:
- Use `LeRobotDataset("lerobot/dataset_name")` to auto-download
- Local datasets: specify `root` parameter for custom path
- Create datasets: Use `lerobot-record` or custom recorder classes
- Upload: `huggingface-cli upload lerobot/dataset_name path/to/data`

### Weights & Biases Integration

```bash
# Login first
wandb login

# Enable in training config
lerobot-train --wandb.enable=true --wandb.project=my_project
```

### Common Pitfalls

1. **ffmpeg**: Must install `ffmpeg 7.X` with `libsvtav1` encoder support
2. **CUDA**: Check PyTorch CUDA version matches your system
3. **Permissions**: RealSense camera requires `sudo` on macOS
4. **Episode data**: Frame indexing is 0-based per episode
5. **Delta timestamps**: Specify temporal context for observations correctly

## File Locations

- Training outputs: `outputs/train/YYYY-MM-DD/HH-MM-SS_policy_name/`
- Checkpoints: `outputs/.../checkpoints/STEP/pretrained_model/`
- Logs: Check WandB if enabled, or local logs in output directory
- Datasets: `~/.cache/huggingface/lerobot/` (default) or custom `root`

## References

- Main docs: https://huggingface.co/docs/lerobot
- Hub: https://huggingface.co/lerobot
- Discord: https://discord.gg/s3KuuzsPFb
- Issues: https://github.com/huggingface/lerobot/issues

## Fork-Specific Documentation

- `IMPLEMENTATION_HANDOVER.md` - Japanese documentation on object picking implementation
- `ARCHIVE_README.md` - Technical details on RealSense camera issues and status
- `docs/WEB_CAMERA_STEREO_IMPLEMENTATION.md` - Alternative webcam stereo vision plan
- `examples/object_picking/` - Custom object detection and picking examples
