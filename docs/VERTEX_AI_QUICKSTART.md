# Vertex AI Training - Quick Start Guide

This is a step-by-step guide to get you started with training LeRobot models on Google Cloud Vertex AI.

## Prerequisites

✅ The following has already been configured:
- Google Cloud project: `lerobot-480101`
- Service account with proper permissions
- Cloud Storage buckets for datasets and models
- Vertex AI API enabled

## Step 1: Prepare Your Dataset

First, ensure your dataset is in the LeRobot format (see main README for dataset creation).

Then upload it to Cloud Storage:

```bash
# Navigate to your lerobot directory
cd /Users/teradakousuke/developer/lerobot

# Upload your local dataset
gsutil -m cp -r ./data/your-dataset gs://lerobot-datasets-480101/

# Verify the upload
gsutil ls gs://lerobot-datasets-480101/your-dataset/
```

## Step 2: Configure Your Training

Copy and customize the example config:

```bash
# Copy the example config
cp configs/vertex_ai_example.yaml configs/my_training.yaml

# Edit the config to match your dataset and requirements
# Update dataset paths, hyperparameters, policy settings, etc.
```

Key settings to adjust in `my_training.yaml`:
- `policy.name`: Choose your policy (e.g., "act", "diffusion")
- `policy.input_shapes`: Match your observation space
- `policy.output_shapes`: Match your action space
- `training.batch_size`: Adjust based on your GPU
- `training.num_epochs`: Set training duration
- `env.name`: Your robot/environment name

## Step 3: Build the Training Container

Build and push the Docker container:

```bash
# Authenticate with Google Container Registry
gcloud auth configure-docker

# Build the container (this may take 10-15 minutes)
docker build -f docker/Dockerfile.vertex -t gcr.io/lerobot-480101/lerobot-trainer:latest .

# Push to Container Registry
docker push gcr.io/lerobot-480101/lerobot-trainer:latest
```

**Note:** You only need to rebuild the container when you update the code or dependencies.

## Step 4: Submit Training Job

Submit your training job to Vertex AI:

```bash
./scripts/vertex_ai_train.sh \
  --job-name="lerobot-training-$(date +%Y%m%d-%H%M%S)" \
  --dataset-path="gs://lerobot-datasets-480101/your-dataset" \
  --output-path="gs://lerobot-models-480101/outputs/$(date +%Y%m%d-%H%M%S)" \
  --config="configs/my_training.yaml" \
  --machine-type="n1-standard-8" \
  --accelerator-type="NVIDIA_TESLA_T4" \
  --accelerator-count=1
```

**Machine Type Recommendations:**
- **Quick test (CPU)**: `n1-standard-4` (no accelerator)
- **Small model**: `n1-standard-8` + `NVIDIA_TESLA_T4` x1
- **Large model**: `n1-standard-16` + `NVIDIA_TESLA_V100` x1
- **Production**: `n1-standard-16` + `NVIDIA_TESLA_A100` x1-4

## Step 5: Monitor Training

Monitor your training job:

```bash
# List all jobs
gcloud ai custom-jobs list --region=us-central1

# Get job details (replace JOB_ID with your job ID from the list command)
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Stream training logs in real-time
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

You can also monitor jobs in the Google Cloud Console:
https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=lerobot-480101

## Step 6: Download Trained Model

After training completes, download your trained model:

```bash
# List available outputs
gsutil ls gs://lerobot-models-480101/outputs/

# Download the entire output directory
gsutil -m cp -r gs://lerobot-models-480101/outputs/YOUR_JOB_TIMESTAMP ./outputs/

# Or download just the final checkpoint
gsutil -m cp -r gs://lerobot-models-480101/outputs/YOUR_JOB_TIMESTAMP/checkpoints/final ./outputs/
```

## Step 7: Evaluate Your Model

### Option 1: Evaluate on Real Robot

Use the trained policy to control your robot:

```bash
# Download the checkpoint if not already downloaded
gsutil -m cp -r gs://lerobot-models-480101/outputs/YOUR_JOB_TIMESTAMP/checkpoints/final ./outputs/checkpoint-final/

# Evaluate on real robot
export KMP_DUPLICATE_LIB_OK=TRUE
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.wchusbserial5AB90691861 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 960, "fps": 25, "fourcc": "MJPG"}, "side": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
    --display_data=true \
    --dataset.repo_id=YOUR_USERNAME/eval_my_so101_dataset \
    --dataset.num_episodes=10 \
    --dataset.single_task="Your task description" \
    --policy.path=./outputs/checkpoint-final/pretrained_model
```

### Option 2: Upload to Hugging Face Hub and Use

Upload the trained model to Hugging Face Hub for easier access:

```bash
# Upload checkpoint to Hugging Face Hub
huggingface-cli upload YOUR_USERNAME/my_act_policy \
    ./outputs/YOUR_JOB_TIMESTAMP/checkpoints/final/pretrained_model

# Then use it directly from the hub
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/tty.wchusbserial5AB90691861 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras='{"front": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 960, "fps": 25, "fourcc": "MJPG"}, "side": {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30}}' \
    --display_data=true \
    --dataset.repo_id=YOUR_USERNAME/eval_my_so101_dataset \
    --dataset.num_episodes=10 \
    --dataset.single_task="Your task description" \
    --policy.path=YOUR_USERNAME/my_act_policy
```

### Option 3: Use Standard Evaluation Script

For simulation environments, use the standard evaluation script:

```bash
lerobot-eval \
  --policy.path=./outputs/YOUR_JOB_TIMESTAMP/checkpoints/final/pretrained_model \
  --env.type=pusht \
  --eval.batch_size=10 \
  --eval.n_episodes=10 \
  --policy.use_amp=false \
  --policy.device=cuda
```

## Cost Optimization Tips

### 1. Use Preemptible Instances (up to 80% savings)
```bash
./scripts/vertex_ai_train.sh \
  --preemptible \
  ... # other arguments
```

Note: Preemptible instances can be shut down by Google Cloud at any time. Make sure to enable checkpointing!

### 2. Start with CPU for Testing
Test your config locally or on CPU first:
```bash
./scripts/vertex_ai_train.sh \
  --machine-type="n1-standard-4" \
  # No accelerator flags
  ...
```

### 3. Use Smaller GPUs for Development
T4 GPUs are much cheaper than V100/A100:
- Development: T4 ($0.35/hour)
- Production: V100 ($2.48/hour) or A100 ($3.67/hour)

### 4. Monitor and Cancel Failed Jobs Quickly
```bash
# Cancel a running job if it's not working
gcloud ai custom-jobs cancel JOB_ID --region=us-central1
```

## Common Issues

### Container Build Fails
```bash
# Check Docker is running
docker ps

# Make sure you're authenticated
gcloud auth configure-docker

# Try building without cache
docker build --no-cache -f docker/Dockerfile.vertex -t gcr.io/lerobot-480101/lerobot-trainer:latest .
```

### Job Fails Immediately
```bash
# Check the logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1

# Common issues:
# - Config path is wrong
# - Dataset path doesn't exist
# - Container image not found
# - Insufficient permissions
```

### Out of Memory Errors
Reduce batch size in your config:
```yaml
training:
  batch_size: 16  # Try 8, 16, or 32 depending on your model and GPU
```

### Dataset Not Found
Make sure your dataset is uploaded:
```bash
gsutil ls gs://lerobot-datasets-480101/your-dataset/
```

## Next Steps

- Read the [full Vertex AI Training Guide](VERTEX_AI_TRAINING.md) for advanced features
- Try distributed training with multiple GPUs
- Set up hyperparameter tuning
- Enable WandB logging for better experiment tracking
- Explore different policies (ACT, Diffusion, VQ-BeT, etc.)

## Support

For issues:
1. Check the [troubleshooting section](VERTEX_AI_TRAINING.md#troubleshooting) in the full guide
2. Review Cloud Logging for detailed errors
3. Ask in the LeRobot Discord or GitHub discussions

## Pricing Estimate

Approximate costs for training (us-central1 region):

| Configuration | Cost per Hour | Typical Training Time | Total Cost |
|--------------|---------------|----------------------|------------|
| n1-standard-4 (CPU only) | $0.19 | 20-40 hours | $4-8 |
| n1-standard-8 + T4 | $0.54 | 4-8 hours | $2-4 |
| n1-standard-16 + V100 | $3.23 | 2-4 hours | $6-13 |
| n1-standard-16 + A100 | $4.42 | 1-2 hours | $4-9 |

*Storage costs: ~$0.02/GB/month for datasets and models*

Use the [GCP Pricing Calculator](https://cloud.google.com/products/calculator) for detailed estimates.
