# Vertex AI Training Troubleshooting Guide

This document records the issues encountered and solutions applied during the setup and execution of Vertex AI training jobs for LeRobot models.

## Table of Contents

- [Configuration Issues](#configuration-issues)
- [Docker Container Issues](#docker-container-issues)
- [Training Script Issues](#training-script-issues)
- [Success Configuration](#success-configuration)

---

## Configuration Issues

### Issue 1: Nested Training Configuration Structure

**Error:**

```
TypeError: TrainPipelineConfig.__init__() got an unexpected keyword argument 'lr_scheduler'
```

**Root Cause:**
The initial YAML configuration used a nested structure with a `training:` top-level key, and included `lr_scheduler` field which should be `scheduler`.

**Solution:**
`TrainPipelineConfig` expects a flat structure with specific field names. The correct structure is:

- `dataset: DatasetConfig` (not `dataset_repo_id`)
- `scheduler: LRSchedulerConfig` (not `lr_scheduler`)
- All training parameters at the top level (not nested under `training:`)

**Example:**

```yaml
# ❌ Wrong
training:
  dataset_repo_id: "..."
  lr_scheduler:
    name: "cosine"

# ✅ Correct
dataset:
  repo_id: "..."
scheduler:
  type: "cosine"
```

---

### Issue 2: Environment Configuration Type

**Error:**

```
draccus.utils.DecodingError: `env`: Expected a dict with a 'type' key for <class 'lerobot.envs.configs.EnvConfig'>
```

**Root Cause:**
`EnvConfig` is a `draccus.ChoiceRegistry` that requires a `type` field to determine which environment subclass to instantiate.

**Solution:**
Either:

1. Set `env: null` to disable gym-based evaluation during training (recommended for cloud training)
2. Provide a valid environment config with a `type` field:

```yaml
# ✅ Option 1: Disable evaluation
env: null

# ✅ Option 2: Specify environment type
env:
  type: pusht  # or aloha, etc.
  task: "PushT-v0"
  fps: 10
```

---

### Issue 3: WandB Configuration Fields

**Error:**

```
draccus.utils.DecodingError: 'wandb': The fields 'tags' are not valid for WandBConfig
```

**Root Cause:**
`WandBConfig` only supports specific fields: `enable`, `disable_artifact`, `project`, `entity`, `notes`, `run_id`, and `mode`. The `tags` field is not supported.

**Solution:**
Remove unsupported fields from the wandb configuration:

```yaml
# ❌ Wrong
wandb:
  enable: false
  tags:
    - tag1
    - tag2

# ✅ Correct
wandb:
  enable: false
  project: "my-project"
  notes: "Training notes"
```

**Valid WandBConfig fields:**

- `enable: bool`
- `disable_artifact: bool`
- `project: str`
- `entity: str | None`
- `notes: str | None`
- `run_id: str | None`
- `mode: str | None` (values: 'online', 'offline', 'disabled')

---

### Issue 4: Policy Hub Push Configuration

**Error:**

```
ValueError: 'policy.repo_id' argument missing. Please specify it to push the model to the hub.
```

**Root Cause:**
By default, policies may attempt to push to the Hugging Face Hub after training. Without `policy.repo_id` configured, this fails.

**Solution:**
Explicitly disable hub pushing if not needed:

```yaml
policy:
  type: act
  # ... other config ...
  push_to_hub: false
```

---

## Docker Container Issues

### Issue 1: Platform Mismatch

**Error:**

```
WARNING: The requested image's platform (linux/arm64) does not match the detected host platform (linux/amd64)
```

**Root Cause:**
Building Docker images on Apple Silicon (M1/M2/M3) Macs produces ARM64 images by default, but Vertex AI requires AMD64 images.

**Solution:**
Use `docker buildx` with explicit platform specification:

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.vertex \
  -t gcr.io/PROJECT_ID/lerobot-trainer:TAG \
  --push \
  .
```

---

### Issue 2: Optional Dependencies

**Error:**

```
ERROR: Could not find a version that satisfies the requirement decord
ERROR: Could not find a version that satisfies the requirement egl-probe
```

**Root Cause:**
Some LeRobot dependencies (decord, egl-probe) are optional and may not be available for all platforms.

**Solution:**
Install these packages with `|| echo "skipped"` to make them optional:

```dockerfile
RUN pip install decord || echo "decord skipped (optional)"
RUN pip install egl-probe || echo "egl-probe skipped (optional)"
```

---

### Issue 3: Missing draccus Dependency

**Error:**

```
ModuleNotFoundError: No module named 'draccus'
```

**Root Cause:**
Installing with `pip install .` doesn't include all development dependencies. The `draccus` package is required for configuration parsing but may not be in the base requirements.

**Solution:**
Use editable install which ensures all dependencies are resolved:

```dockerfile
RUN pip install -e .
```

---

## Training Script Issues

### Issue 1: Configuration Loading with draccus

**Root Cause:**
Manual YAML parsing and dictionary manipulation doesn't properly handle the complex dataclass structure of `TrainPipelineConfig`.

**Solution:**
Use `draccus.parse()` with command-line style overrides:

```python
import draccus
import sys

# Override sys.argv to simulate command-line parsing
old_argv = sys.argv.copy()
sys.argv = [
    "train_on_vertex.py",
    f"--config_path={config_path}",
    f"--dataset.repo_id={dataset_path}",
    f"--output_dir={output_dir}",
]

try:
    cfg = draccus.parse(TrainPipelineConfig)
finally:
    sys.argv = old_argv
```

This approach:

- Properly handles nested dataclass structures
- Validates all fields according to the dataclass definitions
- Supports type conversions and default values
- Works with `draccus.ChoiceRegistry` for polymorphic configs

---

## Success Configuration

### Working YAML Configuration

Here's the final working configuration structure for `train_so101_vertex.yaml`:

```yaml
# Dataset configuration
dataset:
  repo_id: "Terisuke/my_so101_dataset_v2"

# Output directory (will be overridden by command line)
output_dir: "outputs/so101_training"

# Training hyperparameters
batch_size: 8
steps: 10000
save_checkpoint: true
save_freq: 500
log_freq: 10
eval_freq: 500
num_workers: 4
resume: false

# Policy configuration - ACT (Action Chunking Transformer)
policy:
  type: act
  n_obs_steps: 1
  chunk_size: 100
  n_action_steps: 100
  vision_backbone: resnet18
  pretrained_backbone_weights: ResNet18_Weights.IMAGENET1K_V1
  dim_model: 512
  n_heads: 8
  dim_feedforward: 3200
  n_encoder_layers: 4
  n_decoder_layers: 1
  use_vae: true
  latent_dim: 32
  n_vae_encoder_layers: 4
  dropout: 0.1
  kl_weight: 10.0
  optimizer_lr: 1e-4
  optimizer_weight_decay: 1e-4
  optimizer_lr_backbone: 1e-5
  push_to_hub: false

# Environment configuration - set to null to disable gym evaluation
env: null

# WandB logging (optional)
wandb:
  enable: false
  project: "lerobot-so101-vertex"
  entity: null
  notes: "Training SO101 robot to pat head using ACT policy on Vertex AI"

# Use policy training preset (includes optimizer and scheduler)
use_policy_training_preset: true

# Random seed
seed: 1000
```

### Key Points

1. **Flat structure**: No nested `training:` key
2. **Dataset**: Use `dataset.repo_id`, not `dataset_repo_id`
3. **Policy**: Include `type: act` and `push_to_hub: false`
4. **Environment**: Set to `null` for cloud training (no gym evaluation)
5. **WandB**: Only use supported fields (`enable`, `project`, `entity`, `notes`)
6. **Scheduler**: Use `use_policy_training_preset: true` to let policy define optimizer/scheduler

---

## Docker Container Build

### Working Dockerfile

`docker/Dockerfile.vertex`:

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip install google-cloud-storage google-cloud-aiplatform

WORKDIR /app

COPY . /app/

RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

RUN pip install -e .

RUN pip install flash-attn==2.8.3 || echo "flash-attn skipped (optional)"

RUN mkdir -p /app/data /app/outputs

ENTRYPOINT ["python", "/app/scripts/train_on_vertex.py"]
```

### Build and Push Command

```bash
docker buildx build \
  --platform linux/amd64 \
  -f docker/Dockerfile.vertex \
  -t gcr.io/lerobot-480101/lerobot-trainer:v5 \
  --push \
  .
```

---

## Job Submission

### Working Command

```bash
./scripts/vertex_ai_train.sh \
  --job-name so101-v8 \
  --dataset-path gs://lerobot-datasets-480101/my_so101_dataset_v2 \
  --output-path gs://lerobot-models-480101/outputs/so101-20251203 \
  --config gs://lerobot-datasets-480101/configs/train_so101_vertex.yaml \
  --machine-type n1-standard-8 \
  --accelerator-type NVIDIA_TESLA_T4 \
  --accelerator-count 1 \
  --region asia-northeast1 \
  --container-image gcr.io/lerobot-480101/lerobot-trainer:v5
```

---

## Monitoring

### Check Job Status

```bash
gcloud ai custom-jobs describe \
  projects/595166083070/locations/asia-northeast1/customJobs/JOB_ID \
  --region=asia-northeast1
```

### Stream Logs

```bash
gcloud ai custom-jobs stream-logs \
  projects/595166083070/locations/asia-northeast1/customJobs/JOB_ID
```

### Or use the monitoring script

```bash
./MONITOR_TRAINING.sh
```

---

## Training Metrics (Success Example)

From job `so101-v8` (ID: 8168809131617026048):

**Configuration:**

- Policy: ACT (51.6M parameters)
- Dataset: 16,988 frames, 50 episodes
- Batch size: 8
- Steps: 10,000
- GPU: NVIDIA Tesla T4

**Training Progress:**

- Initial loss: ~26.7 (step 10)
- Loss after 100 steps: ~3.0
- Loss after 200 steps: ~1.8
- Loss after 400 steps: ~1.5

**Memory Usage:**

- ~3068.9 MB GPU memory

**Timeline:**

- Job provisioning: 2 minutes
- Dataset download: 40 seconds (12 files, ~768MB)
- Model initialization: 30 seconds
- Training: ~2-3 hours for 10,000 steps

---

## Common Checklist

Before submitting a Vertex AI training job:

- [ ] Configuration file uses flat structure (no nested `training:` key)
- [ ] `dataset.repo_id` is specified (not `dataset_repo_id`)
- [ ] `policy.type` is specified
- [ ] `policy.push_to_hub` is set to `false` (unless pushing to hub)
- [ ] `env` is set to `null` (for cloud training without gym evaluation)
- [ ] `wandb` config only includes supported fields
- [ ] `use_policy_training_preset: true` is set (or custom optimizer/scheduler defined)
- [ ] Docker image is built for `linux/amd64` platform
- [ ] Docker image is pushed to GCR
- [ ] Dataset is uploaded to GCS
- [ ] Config file is uploaded to GCS
- [ ] Service account has required permissions
- [ ] GPU quota is available in the selected region

---

## Version History

### v1-v4: Initial Attempts

- Platform mismatch issues
- Config structure problems
- Dependency resolution issues

### v5: Docker Fix

- Fixed platform to AMD64
- Resolved draccus dependency
- Fixed config loading with draccus.parse()

### v6: Environment Config Fix

- Set `env: null` to disable gym evaluation

### v7: WandB Config Fix

- Removed unsupported `tags` field

### v8: Success ✅

- Added `policy.push_to_hub: false`
- Training started successfully with stable loss convergence

---

## References

- [Vertex AI Documentation](https://console.cloud.google.com/vertex-ai)
- [LeRobot Training Configuration](../src/lerobot/configs/train.py)
- [Vertex AI Quickstart](./VERTEX_AI_QUICKSTART.md)
- [Vertex AI Training Guide](./VERTEX_AI_TRAINING.md)
