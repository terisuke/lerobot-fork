#!/bin/bash
# Monitor Vertex AI Training Job
# Job ID: 4646572010548232192
# Job Name: so101-v9-20k-fixed

JOB_ID="4646572010548232192"
REGION="asia-northeast1"

echo "=========================================="
echo "Vertex AI Training Job Monitor"
echo "=========================================="
echo "Job ID: $JOB_ID"
echo "Region: $REGION"
echo ""

# Check job status
echo "1. Checking job status..."
gcloud ai custom-jobs describe projects/595166083070/locations/$REGION/customJobs/$JOB_ID \
  --region=$REGION \
  --format="value(state,startTime,endTime)"
echo ""

# Get recent logs
echo "2. Recent logs (last 20 entries):"
gcloud logging read "resource.type=ml_job AND resource.labels.job_id=$JOB_ID" \
  --limit=20 \
  --format=json \
  --project=lerobot-480101 2>/dev/null | \
  python3 -c "import json,sys; logs=json.load(sys.stdin); [print(f\"{l.get('timestamp','?')} {l.get('severity','?')}: {l.get('jsonPayload',{}).get('message',l.get('textPayload',''))}\") for l in reversed(logs)]"
echo ""

# Stream live logs (optional)
echo "=========================================="
echo "To stream live logs, run:"
echo "  gcloud ai custom-jobs stream-logs projects/595166083070/locations/$REGION/customJobs/$JOB_ID"
echo ""
echo "To view in Google Cloud Console:"
echo "  https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=lerobot-480101"
echo ""
echo "To cancel the job:"
echo "  gcloud ai custom-jobs cancel projects/595166083070/locations/$REGION/customJobs/$JOB_ID --region=$REGION"
echo ""
echo "To download outputs after training:"
echo "  gsutil -m cp -r gs://lerobot-models-480101/outputs/so101-20251203/ ./outputs/"
echo "=========================================="
