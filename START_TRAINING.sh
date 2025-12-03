#!/bin/bash
# Step-by-step guide to start Vertex AI training for SO101 dataset
# Execute these commands one by one after starting Docker Desktop

set -e

echo "=================================================="
echo "Vertex AI Training Setup for SO101 Dataset"
echo "=================================================="
echo ""
echo "Prerequisites:"
echo "- Docker Desktop must be running"
echo "- Dataset already uploaded to gs://lerobot-datasets-480101/my_so101_dataset_v2"
echo ""
echo "Press ENTER to continue..."
read

# Step 1: Check Docker is running
echo ""
echo "Step 1: Checking Docker..."
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop and try again."
    exit 1
fi
echo "✓ Docker is running"

# Step 2: Build Docker container
echo ""
echo "Step 2: Building Docker container (10-15 minutes)..."
echo "Building: gcr.io/lerobot-480101/lerobot-trainer:latest"
docker build -f docker/Dockerfile.vertex -t gcr.io/lerobot-480101/lerobot-trainer:latest .
echo "✓ Docker container built successfully"

# Step 3: Push to Google Container Registry
echo ""
echo "Step 3: Pushing container to GCR (5-10 minutes)..."
docker push gcr.io/lerobot-480101/lerobot-trainer:latest
echo "✓ Container pushed successfully"

# Step 4: Upload config to Cloud Storage
echo ""
echo "Step 4: Uploading training config..."
gsutil cp configs/train_so101_vertex.yaml gs://lerobot-datasets-480101/configs/
echo "✓ Config uploaded"

# Step 5: Submit training job
JOB_NAME="so101-training-$(date +%Y%m%d-%H%M%S)"
OUTPUT_PATH="gs://lerobot-models-480101/outputs/so101-$(date +%Y%m%d)"

echo ""
echo "Step 5: Submitting training job..."
echo "Job Name: $JOB_NAME"
echo "Output: $OUTPUT_PATH"
echo ""

./scripts/vertex_ai_train.sh \
  --job-name="$JOB_NAME" \
  --dataset-path="gs://lerobot-datasets-480101/my_so101_dataset_v2" \
  --output-path="$OUTPUT_PATH" \
  --config="gs://lerobot-datasets-480101/configs/train_so101_vertex.yaml" \
  --machine-type="n1-standard-8" \
  --accelerator-type="NVIDIA_TESLA_T4" \
  --accelerator-count=1

echo ""
echo "=================================================="
echo "Training job submitted successfully!"
echo "=================================================="
echo ""
echo "Monitor your job:"
echo "  gcloud ai custom-jobs list --region=us-central1"
echo ""
echo "View logs:"
echo "  gcloud ai custom-jobs stream-logs $JOB_NAME --region=us-central1"
echo ""
echo "Web Console:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=lerobot-480101"
echo ""
