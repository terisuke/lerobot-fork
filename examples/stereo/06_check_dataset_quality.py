#!/usr/bin/env python
"""
Dataset Quality Check Tool

This script checks the quality of a LeRobot dataset, specifically verifying:
- Dataset integrity (episodes, frames, FPS)
- Camera keys (verifying stereo cameras: left and right)
- Image dimensions and quality
- Action and state data presence

Usage:
    # Check a HuggingFace Hub dataset
    python examples/stereo/06_check_dataset_quality.py --repo-id username/so101-stereo-pickup

    # Check a local dataset
    python examples/stereo/06_check_dataset_quality.py --repo-id username/so101-stereo-pickup --local-files-only

    # Check specific episode
    python examples/stereo/06_check_dataset_quality.py --repo-id username/so101-stereo-pickup --episode-index 0
"""

import argparse
from pathlib import Path

import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def check_dataset_quality(
    repo_id: str,
    local_files_only: bool = False,
    episode_index: int | None = None,
) -> bool:
    """
    Check dataset quality and integrity.

    Args:
        repo_id: HuggingFace Hub repository ID or local path
        local_files_only: Use only local files (no download)
        episode_index: Specific episode to check (None = check all)

    Returns:
        True if all checks pass, False otherwise
    """
    print("=" * 70)
    print("📂 Dataset Quality Check")
    print("=" * 70)
    print(f"Repository ID:      {repo_id}")
    print(f"Local files only:   {local_files_only}")
    print(f"Episode to check:   {episode_index if episode_index is not None else 'All'}")
    print("=" * 70)

    # Load dataset
    print("\n📥 Loading dataset...")
    try:
        dataset = LeRobotDataset(
            repo_id=repo_id,
            local_files_only=local_files_only
        )
        print(f"✅ Dataset loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return False

    # Basic dataset info
    print(f"\n📊 Dataset Information:")
    print(f"  Total frames:       {len(dataset)}")
    print(f"  Total episodes:     {dataset.num_episodes}")
    print(f"  FPS:                {dataset.fps}")
    print(f"  Robot type:         {dataset.robot_type if hasattr(dataset, 'robot_type') else 'N/A'}")

    # Check for required metadata
    if hasattr(dataset, 'meta'):
        print(f"\n📋 Metadata:")
        if 'info' in dataset.meta:
            info = dataset.meta['info']
            print(f"  Codebase version:   {info.get('codebase_version', 'N/A')}")
            print(f"  Total episodes:     {info.get('total_episodes', 'N/A')}")
            print(f"  Total frames:       {info.get('total_frames', 'N/A')}")

    # Check camera keys
    print(f"\n📷 Camera Configuration:")
    sample = dataset[0]
    sample_keys = list(sample.keys())

    # Find all image-related keys
    camera_keys = [k for k in sample_keys if 'image' in k.lower()]
    print(f"  Camera keys found:  {len(camera_keys)}")
    for key in camera_keys:
        print(f"    - {key}")

    # Verify stereo cameras (left and right)
    required_keys = ["observation.images.left", "observation.images.right"]
    all_required_present = True

    print(f"\n✅ Required Stereo Camera Keys:")
    for key in required_keys:
        if key in camera_keys:
            print(f"  ✅ {key}")
        else:
            print(f"  ❌ {key} MISSING!")
            all_required_present = False

    if not all_required_present:
        print("\n⚠️  WARNING: Missing required stereo camera keys!")
        print("   Dataset may not be suitable for stereo vision tasks.")

    # Check image dimensions
    if camera_keys:
        print(f"\n📐 Image Dimensions:")
        for key in camera_keys:
            sample_img = sample[key]
            if isinstance(sample_img, np.ndarray):
                print(f"  {key}:")
                print(f"    Shape:            {sample_img.shape}")
                print(f"    Dtype:            {sample_img.dtype}")

                # Check for expected dimensions (720p stereo)
                expected_height, expected_width = 720, 1280
                if sample_img.shape[0] == expected_height and sample_img.shape[1] == expected_width:
                    print(f"    Resolution:       ✅ Matches expected {expected_width}x{expected_height}")
                else:
                    print(f"    Resolution:       ⚠️  Does not match expected {expected_width}x{expected_height}")

                # Check color channels
                if len(sample_img.shape) == 3 and sample_img.shape[2] == 3:
                    print(f"    Color channels:   ✅ RGB (3 channels)")
                else:
                    print(f"    Color channels:   ⚠️  Unexpected format")

    # Check action and state
    print(f"\n🎮 Control Data:")
    if "action" in sample:
        action = sample["action"]
        print(f"  Action:")
        print(f"    Shape:            {action.shape}")
        print(f"    Dtype:            {action.dtype}")
        if isinstance(action, np.ndarray):
            print(f"    Range:            [{action.min():.3f}, {action.max():.3f}]")
    else:
        print(f"  ❌ Action data MISSING!")
        all_required_present = False

    if "observation.state" in sample:
        state = sample["observation.state"]
        print(f"  Observation State:")
        print(f"    Shape:            {state.shape}")
        print(f"    Dtype:            {state.dtype}")
        if isinstance(state, np.ndarray):
            print(f"    Range:            [{state.min():.3f}, {state.max():.3f}]")
    else:
        print(f"  ⚠️  Observation state not found")

    # Episode-specific checks
    if episode_index is not None:
        print(f"\n📹 Episode {episode_index} Details:")
        try:
            episode_data = dataset.hf_dataset.filter(
                lambda x: x['episode_index'] == episode_index
            )
            episode_length = len(episode_data)
            print(f"  Episode length:     {episode_length} frames")
            print(f"  Duration:           {episode_length / dataset.fps:.2f} seconds")

            # Check for episode completion
            if 'next.done' in episode_data.column_names:
                done_flags = episode_data['next.done']
                if any(done_flags):
                    print(f"  Completion:         ✅ Episode has 'done' flag")
                else:
                    print(f"  Completion:         ⚠️  Episode has no 'done' flag")
        except Exception as e:
            print(f"  ❌ Failed to analyze episode {episode_index}: {e}")

    # Image quality checks (sample-based)
    print(f"\n🖼️  Image Quality Checks:")
    if camera_keys:
        for key in camera_keys[:2]:  # Check first two cameras only
            img = sample[key]
            if isinstance(img, np.ndarray):
                # Check brightness
                mean_brightness = img.mean()
                print(f"  {key}:")
                print(f"    Mean brightness:  {mean_brightness:.1f}/255")

                if mean_brightness < 50:
                    print(f"    ⚠️  Image may be too dark")
                elif mean_brightness > 200:
                    print(f"    ⚠️  Image may be too bright")
                else:
                    print(f"    ✅ Brightness looks good")

                # Check for blank images (all pixels same value)
                if img.std() < 5:
                    print(f"    ⚠️  Image may be blank or uniform")
                else:
                    print(f"    ✅ Image has variation")

    # Final summary
    print("\n" + "=" * 70)
    print("📋 Quality Check Summary")
    print("=" * 70)

    if all_required_present:
        print("✅ All required data present")
        print("✅ Dataset appears to be valid for training")
        result = True
    else:
        print("❌ Some required data is missing")
        print("⚠️  Dataset may not be suitable for training")
        result = False

    print("=" * 70)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Check LeRobot dataset quality"
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace Hub repository ID (e.g., username/dataset-name)"
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Use only local files (do not download from Hub)"
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        default=None,
        help="Check specific episode (optional)"
    )
    args = parser.parse_args()

    success = check_dataset_quality(
        repo_id=args.repo_id,
        local_files_only=args.local_files_only,
        episode_index=args.episode_index,
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
