# Vertex AI Training Setup Guide

This guide explains how to train LeRobot models on Google Cloud Vertex AI using your local datasets.

## Prerequisites

- Google Cloud Platform account with billing enabled
- Project ID: `lerobot-480101`
- `gcloud` CLI installed and authenticated
- Local datasets in the LeRobot format

## Architecture

The training setup consists of:

- **Cloud Storage Buckets**: Store datasets and trained models
- **Vertex AI Custom Jobs**: Execute training on cloud infrastructure
- **Service Account**: Manages permissions and authentication
- **Container Registry**: Hosts custom training images

## Initial Setup (Already Completed)

The following infrastructure has been configured:

### 1. Authentication & Project

```bash
gcloud config set project lerobot-480101
gcloud auth application-default set-quota-project lerobot-480101
```

### 2. APIs Enabled

- Vertex AI API (`aiplatform.googleapis.com`)
- Cloud Storage API (`storage.googleapis.com`)

### 3. Service Account

- **Email**: `vertex-ai-training@lerobot-480101.iam.gserviceaccount.com`
- **Key Location**: `~/vertex-ai-key.json`
- **Roles**:
  - `roles/aiplatform.user` - Execute Vertex AI jobs
  - `roles/storage.objectAdmin` - Read/write datasets and models
  - `roles/logging.logWriter` - Write training logs

### 4. Cloud Storage Buckets

- `gs://lerobot-datasets-480101` - Training datasets
- `gs://lerobot-models-480101` - Trained models and checkpoints

### 5. Default Region

- Region: `us-central1`

## Training Workflow

### Step 1: Upload Dataset to Cloud Storage

Upload your local dataset to the Cloud Storage bucket:

```bash
# Upload a single dataset
gsutil -m cp -r ./data/local_dataset gs://lerobot-datasets-480101/

# Or sync an entire directory
gsutil -m rsync -r ./data gs://lerobot-datasets-480101/datasets/
```

Verify the upload:

```bash
gsutil ls gs://lerobot-datasets-480101/
```

### Step 2: Build Training Container

Build a Docker container with your training code:

```bash
# Build the container
cd /Users/teradakousuke/developer/lerobot
docker build -f docker/Dockerfile.vertex -t gcr.io/lerobot-480101/lerobot-trainer:latest .

# Push to Google Container Registry
docker push gcr.io/lerobot-480101/lerobot-trainer:latest
```

> **Note**: See `docker/Dockerfile.vertex` for the container specification.

### Step 3: Submit Training Job

Use the provided script to submit a training job:

```bash
./scripts/vertex_ai_train.sh \
  --job-name="lerobot-training-$(date +%Y%m%d-%H%M%S)" \
  --dataset-path="gs://lerobot-datasets-480101/your-dataset" \
  --output-path="gs://lerobot-models-480101/outputs" \
  --config="configs/your_config.yaml" \
  --machine-type="n1-standard-8" \
  --accelerator-type="NVIDIA_TESLA_T4" \
  --accelerator-count=1
```

Available machine types:

- CPU: `n1-standard-4`, `n1-standard-8`, `n1-highmem-8`
- GPU: `n1-standard-8` + `NVIDIA_TESLA_T4`, `NVIDIA_TESLA_V100`, `NVIDIA_TESLA_A100`

### Step 4: Monitor Training

Monitor your training job:

```bash
# List all jobs
gcloud ai custom-jobs list --region=us-central1

# Get job details
gcloud ai custom-jobs describe JOB_ID --region=us-central1

# Stream logs
gcloud ai custom-jobs stream-logs JOB_ID --region=us-central1
```

### Step 5: Download Trained Models

After training completes, download the models:

```bash
# Download to local outputs directory
gsutil -m cp -r gs://lerobot-models-480101/outputs/your-job ./outputs/

# Or download specific checkpoints
gsutil cp gs://lerobot-models-480101/outputs/your-job/checkpoint-1000.pth ./outputs/
```

## Configuration Files

### Training Config Example

Create a training configuration at `configs/vertex_ai_config.yaml`:

```yaml
training:
  dataset_repo_id: "gs://lerobot-datasets-480101/your-dataset"
  output_dir: "gs://lerobot-models-480101/outputs"

  num_epochs: 100
  batch_size: 32
  learning_rate: 1e-4

  save_checkpoint_steps: 1000
  eval_steps: 500

policy:
  name: "act"
  # Policy-specific configuration

env:
  name: "so101"
  # Environment-specific configuration
```

## Cost Optimization

### Preemptible VMs

Use preemptible instances to reduce costs by up to 80%:

```bash
./scripts/vertex_ai_train.sh \
  --job-name="my-job" \
  --preemptible \
  ...
```

### Instance Selection

- **Development/Testing**: `n1-standard-4` (CPU only)
- **Small models**: `n1-standard-8` + 1x T4 GPU
- **Large models**: `n1-standard-16` + 1x V100 or A100 GPU
- **Distributed training**: Multiple workers with A100 GPUs

### Storage

- Use `STANDARD` storage class for active datasets
- Move old datasets to `NEARLINE` or `COLDLINE` for archival

## Troubleshooting

### Authentication Issues

```bash
# Re-authenticate
gcloud auth application-default login
gcloud auth application-default set-quota-project lerobot-480101
```

### Permission Errors

```bash
# Verify service account permissions
gcloud projects get-iam-policy lerobot-480101 \
  --flatten="bindings[].members" \
  --filter="bindings.members:vertex-ai-training@lerobot-480101.iam.gserviceaccount.com"
```

### Container Build Failures

```bash
# Test locally first
docker build -f docker/Dockerfile.vertex -t test-image .
docker run --rm test-image python -c "import lerobot; print(lerobot.__version__)"
```

### Job Failures

```bash
# Check detailed logs
gcloud logging read "resource.type=ml_job AND resource.labels.job_id=YOUR_JOB_ID" \
  --limit=100 \
  --format=json
```

## Advanced Usage

### Distributed Training

For large-scale training, use multiple workers:

```bash
./scripts/vertex_ai_train.sh \
  --job-name="distributed-job" \
  --worker-count=4 \
  --machine-type="n1-standard-16" \
  --accelerator-type="NVIDIA_TESLA_A100" \
  --accelerator-count=2 \
  ...
```

### Hyperparameter Tuning

Use Vertex AI Hyperparameter Tuning:

```bash
gcloud ai hp-tuning-jobs create \
  --region=us-central1 \
  --display-name="lerobot-hptuning" \
  --config=configs/hptuning_config.yaml
```

### Custom Training Loop

For full control, create a Python training script:

```python
# scripts/train_on_vertex.py
from google.cloud import aiplatform
from lerobot import train

aiplatform.init(
    project='lerobot-480101',
    location='us-central1',
)

# Your training logic here
train.run(config_path='gs://lerobot-datasets-480101/config.yaml')
```

## Security Best Practices

1. **Rotate Service Account Keys**: Regenerate keys periodically
2. **Least Privilege**: Only grant necessary permissions
3. **VPC Service Controls**: Restrict API access (for production)
4. **Audit Logging**: Enable Cloud Audit Logs for compliance
5. **Secret Management**: Use Secret Manager for sensitive data

## Resources

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Custom Training Guide](https://cloud.google.com/vertex-ai/docs/training/custom-training)
- [Pricing Calculator](https://cloud.google.com/products/calculator)
- [LeRobot Documentation](https://huggingface.co/docs/lerobot)

## Support

For issues specific to this setup:

1. Check the troubleshooting section above
2. Review Cloud Logging for detailed error messages
3. Verify IAM permissions and service account configuration
4. Consult the LeRobot community on Discord or GitHub
