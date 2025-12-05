#!/usr/bin/env python3
"""
Training script for Google Cloud Vertex AI.

This script wraps the standard LeRobot training pipeline to:
1. Download datasets from Cloud Storage
2. Execute training using lerobot_train.py
3. Upload trained models back to Cloud Storage

Usage:
    python scripts/train_on_vertex.py \
        --dataset_path gs://lerobot-datasets-480101/your-dataset \
        --output_dir gs://lerobot-models-480101/outputs \
        --config configs/your_config.yaml
"""

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

from google.cloud import storage

# Add lerobot to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lerobot.scripts.lerobot_train import train
from lerobot.configs.train import TrainPipelineConfig


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args():
    """Parse command line arguments."""
    arg_parser = argparse.ArgumentParser(description="Train LeRobot on Vertex AI")
    arg_parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="GCS path to dataset (e.g., gs://bucket/dataset)",
    )
    arg_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="GCS path for outputs (e.g., gs://bucket/outputs)",
    )
    arg_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path or GCS path to training config YAML",
    )
    arg_parser.add_argument(
        "--local_dataset_dir",
        type=str,
        default="/app/data",
        help="Local directory to download datasets",
    )
    arg_parser.add_argument(
        "--local_output_dir",
        type=str,
        default="/app/outputs",
        help="Local directory for training outputs",
    )
    return arg_parser.parse_args()


def download_from_gcs(gcs_path: str, local_path: str):
    """
    Download files from Google Cloud Storage.

    Args:
        gcs_path: GCS path (gs://bucket/path)
        local_path: Local destination path
    """
    logging.info(f"Downloading from {gcs_path} to {local_path}")

    # Parse GCS path
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    path_parts = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_prefix = path_parts[1] if len(path_parts) > 1 else ""

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Create local directory
    os.makedirs(local_path, exist_ok=True)

    # Download all blobs with the prefix
    blobs = bucket.list_blobs(prefix=blob_prefix)
    download_count = 0

    for blob in blobs:
        # Skip directory markers
        if blob.name.endswith("/"):
            continue

        # Calculate relative path
        relative_path = blob.name[len(blob_prefix) :].lstrip("/")
        if not relative_path:
            relative_path = os.path.basename(blob.name)

        local_file_path = os.path.join(local_path, relative_path)

        # Create parent directories
        os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

        # Download blob
        logging.info(f"  Downloading {blob.name} -> {local_file_path}")
        blob.download_to_filename(local_file_path)
        download_count += 1

    logging.info(f"Downloaded {download_count} files from {gcs_path}")


def upload_to_gcs(local_path: str, gcs_path: str):
    """
    Upload files to Google Cloud Storage.

    Args:
        local_path: Local source path
        gcs_path: GCS destination path (gs://bucket/path)
    """
    logging.info(f"Uploading from {local_path} to {gcs_path}")

    # Parse GCS path
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}")

    path_parts = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_prefix = path_parts[1] if len(path_parts) > 1 else ""

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)

    # Upload all files recursively
    upload_count = 0

    for root, _dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)

            # Calculate relative path
            relative_path = os.path.relpath(local_file_path, local_path)
            blob_name = os.path.join(blob_prefix, relative_path).replace("\\", "/")

            # Upload blob
            logging.info(f"  Uploading {local_file_path} -> gs://{bucket_name}/{blob_name}")
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file_path)
            upload_count += 1

    logging.info(f"Uploaded {upload_count} files to {gcs_path}")


def main():
    """Main training function."""
    setup_logging()
    args = parse_args()

    logging.info("=" * 60)
    logging.info("Starting Vertex AI Training Job")
    logging.info("=" * 60)
    logging.info(f"Dataset path: {args.dataset_path}")
    logging.info(f"Output dir: {args.output_dir}")
    logging.info(f"Config: {args.config}")

    # Step 1: Download dataset from GCS
    if args.dataset_path.startswith("gs://"):
        logging.info("Step 1: Downloading dataset from Cloud Storage")
        download_from_gcs(args.dataset_path, args.local_dataset_dir)
        dataset_path = args.local_dataset_dir
    else:
        dataset_path = args.dataset_path
        logging.info(f"Using local dataset at {dataset_path}")

    # Step 2: Download config if it's on GCS
    if args.config.startswith("gs://"):
        logging.info("Step 2: Downloading config from Cloud Storage")
        # Create temp directory for config
        temp_dir = tempfile.mkdtemp()
        # Download the config file
        blob_client = storage.Client()
        
        # Parse GCS path
        path_parts = args.config[5:].split("/", 1)
        bucket_name = path_parts[0]
        blob_name = path_parts[1]
        
        bucket = blob_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        
        # Download to temp file
        config_path = os.path.join(temp_dir, "config.yaml")
        blob.download_to_filename(config_path)
        logging.info(f"Downloaded config to {config_path}")
    else:
        config_path = args.config
        logging.info(f"Using local config at {config_path}")

    # Step 3: Load training configuration
    logging.info("Step 3: Loading training configuration")

    # Load config from YAML file using draccus to properly handle dataclass structure
    import draccus
    from pathlib import Path
    
    logging.info(f"Loading config from: {config_path}")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Override arguments for dataset and output paths
    import sys
    old_argv = sys.argv.copy()
    sys.argv = [
        "train_on_vertex.py",
        f"--config_path={config_path}",
        f"--dataset.repo_id={dataset_path}",
        f"--output_dir={args.local_output_dir}",
    ]
    
    try:
        # Use draccus to parse the config with overrides
        cfg = draccus.parse(TrainPipelineConfig)
        logging.info(f"Config loaded successfully")
        logging.info(f"Dataset: {cfg.dataset.repo_id}")
        logging.info(f"Output dir: {cfg.output_dir}")
        logging.info(f"Policy type: {cfg.policy.type if cfg.policy else 'None'}")
    finally:
        sys.argv = old_argv

    # Step 4: Run training
    logging.info("Step 4: Starting training")
    logging.info("=" * 60)

    try:
        train(cfg)
        logging.info("Training completed successfully")
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise

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

    logging.info("=" * 60)
    logging.info("Vertex AI Training Job Completed Successfully")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
