#!/bin/bash
# Submit a training job to Google Cloud Vertex AI
#
# Usage:
#   ./scripts/vertex_ai_train.sh \
#     --job-name="my-training-job" \
#     --dataset-path="gs://lerobot-datasets-480101/dataset" \
#     --output-path="gs://lerobot-models-480101/outputs" \
#     --config="configs/train_config.yaml" \
#     --machine-type="n1-standard-8" \
#     --accelerator-type="NVIDIA_TESLA_T4" \
#     --accelerator-count=1 \
#     --region="asia-northeast1"

set -e

# Default values
PROJECT_ID="lerobot-480101"
REGION="us-central1"
SERVICE_ACCOUNT="vertex-ai-training@lerobot-480101.iam.gserviceaccount.com"
CONTAINER_IMAGE="gcr.io/lerobot-480101/lerobot-trainer:latest"
MACHINE_TYPE="n1-standard-8"
ACCELERATOR_TYPE=""
ACCELERATOR_COUNT=0
WORKER_COUNT=1
PREEMPTIBLE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --job-name)
      JOB_NAME="$2"
      shift 2
      ;;
    --dataset-path)
      DATASET_PATH="$2"
      shift 2
      ;;
    --output-path)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --machine-type)
      MACHINE_TYPE="$2"
      shift 2
      ;;
    --accelerator-type)
      ACCELERATOR_TYPE="$2"
      shift 2
      ;;
    --accelerator-count)
      ACCELERATOR_COUNT="$2"
      shift 2
      ;;
    --worker-count)
      WORKER_COUNT="$2"
      shift 2
      ;;
    --region)
      REGION="$2"
      shift 2
      ;;
    --preemptible)
      PREEMPTIBLE="--enable-web-access"
      shift
      ;;
    --container-image)
      CONTAINER_IMAGE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$JOB_NAME" ]; then
  echo "Error: --job-name is required"
  exit 1
fi

if [ -z "$DATASET_PATH" ]; then
  echo "Error: --dataset-path is required"
  exit 1
fi

if [ -z "$OUTPUT_PATH" ]; then
  echo "Error: --output-path is required"
  exit 1
fi

if [ -z "$CONFIG_PATH" ]; then
  echo "Error: --config is required"
  exit 1
fi

# Build worker pool spec
WORKER_POOL_SPEC="machine-type=$MACHINE_TYPE,replica-count=$WORKER_COUNT,container-image-uri=$CONTAINER_IMAGE"

# Add accelerator if specified
if [ -n "$ACCELERATOR_TYPE" ] && [ "$ACCELERATOR_COUNT" -gt 0 ]; then
  WORKER_POOL_SPEC="$WORKER_POOL_SPEC,accelerator-type=$ACCELERATOR_TYPE,accelerator-count=$ACCELERATOR_COUNT"
fi

# Upload config to Cloud Storage if it's a local file
if [ -f "$CONFIG_PATH" ]; then
  echo "Uploading config file to Cloud Storage..."
  CONFIG_GCS_PATH="gs://lerobot-datasets-480101/configs/$(basename $CONFIG_PATH)"
  gsutil cp "$CONFIG_PATH" "$CONFIG_GCS_PATH"
  CONFIG_PATH="$CONFIG_GCS_PATH"
fi

# Build command arguments
COMMAND_ARGS="--dataset_path=$DATASET_PATH"
COMMAND_ARGS="$COMMAND_ARGS --output_dir=$OUTPUT_PATH"
COMMAND_ARGS="$COMMAND_ARGS --config=$CONFIG_PATH"

echo "=========================================="
echo "Submitting Vertex AI Training Job"
echo "=========================================="
echo "Job Name:        $JOB_NAME"
echo "Project:         $PROJECT_ID"
echo "Region:          $REGION"
echo "Machine Type:    $MACHINE_TYPE"
echo "Accelerator:     $ACCELERATOR_TYPE ($ACCELERATOR_COUNT)"
echo "Workers:         $WORKER_COUNT"
echo "Container:       $CONTAINER_IMAGE"
echo "Dataset:         $DATASET_PATH"
echo "Output:          $OUTPUT_PATH"
echo "Config:          $CONFIG_PATH"
echo "Service Account: $SERVICE_ACCOUNT"
echo "=========================================="

# Submit the job
gcloud ai custom-jobs create \
  --region="$REGION" \
  --display-name="$JOB_NAME" \
  --worker-pool-spec="$WORKER_POOL_SPEC" \
  --service-account="$SERVICE_ACCOUNT" \
  --args="$COMMAND_ARGS" \
  $PREEMPTIBLE

echo ""
echo "Job submitted successfully!"
echo ""
echo "Monitor your job with:"
echo "  gcloud ai custom-jobs list --region=$REGION"
echo ""
echo "View logs with:"
echo "  gcloud ai custom-jobs stream-logs JOB_ID --region=$REGION"
echo ""
echo "Cancel job with:"
echo "  gcloud ai custom-jobs cancel JOB_ID --region=$REGION"
