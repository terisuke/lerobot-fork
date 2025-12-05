# Cloud Storage Output Upload Issue and Fix

**Date:** 2025-12-03
**Issue:** Training outputs were not being uploaded to Cloud Storage
**Status:** ✅ Fixed in v6-fixed-upload container

---

## Problem Description

After v8 training job completed successfully, no outputs were found in Cloud Storage:

```bash
$ gsutil ls gs://lerobot-models-480101/outputs/
CommandException: One or more URLs matched no objects.
```

Despite the job completing without errors, the trained model checkpoints were not uploaded to GCS.

---

## Root Cause

The issue was in `scripts/train_on_vertex.py` Step 5 (Upload outputs to GCS).

### The Problem

1. **TrainPipelineConfig auto-renames output directory:**

   In `src/lerobot/configs/train.py` lines 115-125:

   ```python
   if not self.resume and isinstance(self.output_dir, Path) and self.output_dir.is_dir():
       # If output directory exists and not resuming, automatically append timestamp
       now = dt.datetime.now()
       timestamp_suffix = f"_{now:%Y%m%d_%H%M%S}"
       original_output_dir = self.output_dir
       self.output_dir = Path(str(self.output_dir) + timestamp_suffix)
       logger.warning(
           f"Output directory {original_output_dir} already exists. "
           f"Automatically renaming to {self.output_dir} to avoid overwriting. "
           f"To resume training, use --resume=true"
       )
   ```

2. **Upload script used wrong directory:**

   In `scripts/train_on_vertex.py` (original code):

   ```python
   # Step 5: Upload outputs to GCS
   if args.output_dir.startswith("gs://"):
       logging.info("Step 5: Uploading outputs to Cloud Storage")
       upload_to_gcs(args.local_output_dir, args.output_dir)  # ❌ Wrong path!
   ```

3. **What happened:**
   - Config specified: `output_dir: "/app/outputs"`
   - Training created: `/app/outputs_20251203_123727/` (with timestamp)
   - Upload tried: `/app/outputs/` (original path - empty!)
   - Result: No files to upload

---

## Solution

Modified `scripts/train_on_vertex.py` to use the actual output directory from the config:

```python
# Step 5: Upload outputs to GCS
# Note: cfg.output_dir may have been modified by TrainPipelineConfig.validate()
# (e.g., timestamped to avoid overwriting existing directory)
actual_output_dir = str(cfg.output_dir)
logging.info(f"Actual output directory: {actual_output_dir}")

if args.output_dir.startswith("gs://"):
    logging.info("Step 5: Uploading outputs to Cloud Storage")
    if os.path.exists(actual_output_dir):
        upload_to_gcs(actual_output_dir, args.output_dir)
    else:
        logging.warning(f"Output directory {actual_output_dir} does not exist. Nothing to upload.")
else:
    logging.info(f"Outputs saved locally at {actual_output_dir}")
```

### Key Changes

1. ✅ Use `cfg.output_dir` instead of `args.local_output_dir`
2. ✅ Log the actual output directory for debugging
3. ✅ Check if directory exists before uploading
4. ✅ Provide helpful warning if directory is missing

---

## Impact on Previous Jobs

### v8 Job (ID: 8168809131617026048)

- **Status:** Training completed successfully
- **Final Loss:** 0.195
- **Problem:** Outputs never uploaded to GCS
- **Model Location:** Lost (container was terminated)
- **Action Required:** Re-run training to get model outputs

### v9-20k Job (ID: 1527547793617453056)

- **Status:** Cancelled before completion
- **Reason:** Had same upload bug
- **Action Taken:** Cancelled and rebuilt with fix

### v9-20k-fixed Job (ID: 4646572010548232192)

- **Status:** ✅ Running with fix
- **Container:** gcr.io/lerobot-480101/lerobot-trainer:v6-fixed-upload
- **Expected Result:** Outputs will be uploaded correctly

---

## Verification

To verify the fix works, check for outputs after training completes:

```bash
# Check if outputs exist
gsutil ls -lh gs://lerobot-models-480101/outputs/so101-20251203-20k/

# Should see checkpoint directories and model files:
# - checkpoint-1000/
# - checkpoint-2000/
# - ...
# - checkpoint-20000/
# - pretrained_model/
# - train_config.json
```

---

## Lessons Learned

1. **Config validation can modify paths** - Always use the validated config values, not the original arguments.

2. **Directory timestamping is automatic** - TrainPipelineConfig adds timestamps to avoid overwriting existing outputs. This is a safety feature but needs to be accounted for.

3. **Add existence checks** - Always verify directories exist before attempting operations on them.

4. **Log actual paths** - Include debug logging to show which directories are actually being used.

5. **Test upload logic early** - This issue could have been caught with a quick check of GCS after the first successful training run.

---

## Prevention for Future

### Code Review Checklist

- [ ] Verify output paths match between training and upload steps
- [ ] Check for automatic path modifications in config validation
- [ ] Add existence checks before file operations
- [ ] Include debug logging for critical paths
- [ ] Test full pipeline including upload in development

### Monitoring

After each training job:

1. Check job completion status
2. Verify outputs in Cloud Storage
3. Confirm checkpoint files are present
4. Test model can be downloaded

---

## Related Files

- **Fixed Script:** `scripts/train_on_vertex.py`
- **Config Validation:** `src/lerobot/configs/train.py` (lines 115-129)
- **Docker Image:** `gcr.io/lerobot-480101/lerobot-trainer:v6-fixed-upload`
- **Issue Discovery:** TRAINING_V8_RESULTS.md

---

## Cost Impact

**v8 Training (Lost):**

- Duration: ~4h56m on Tesla T4
- Cost: ~$1.50 USD
- Result: Model lost, must re-train

**v9-20k-fixed (In Progress):**

- Duration: Expected ~8-10 hours
- Cost: ~$2.50-3.00 USD
- Result: Will include proper output upload

**Total Additional Cost:** ~$4.00-4.50 USD to recover from issue

---

## Status Update

**Current Active Job:**

- **Job ID:** 4646572010548232192
- **Job Name:** so101-v9-20k-fixed
- **Container:** v6-fixed-upload (includes fix)
- **Config:** 20,000 steps training
- **Monitor:** `./MONITOR_TRAINING.sh`
- **Web Console:** https://console.cloud.google.com/vertex-ai/training/custom-jobs/asia-northeast1/4646572010548232192?project=lerobot-480101

---

**Created:** 2025-12-03
**Fixed In:** gcr.io/lerobot-480101/lerobot-trainer:v6-fixed-upload
**Status:** ✅ Resolved
