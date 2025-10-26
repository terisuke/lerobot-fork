#!/usr/bin/env python3

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stereo depth estimation test script.

Uses stereo calibration to compute and visualize depth maps in real-time.

Usage:
    python examples/stereo/03_test_depth.py \\
        --calibration configs/camera/stereo_calibration.yaml \\
        --method sgbm

Requirements:
    - Completed stereo calibration (run 02_calibrate_stereo.py first)
    - Two physical webcams at same indices used during calibration
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.utils.stereo import StereoDepthEstimator, load_stereo_calibration


def parse_args():
    parser = argparse.ArgumentParser(description="Test stereo depth estimation")
    parser.add_argument(
        "--calibration",
        type=str,
        required=True,
        help="Path to stereo calibration file (.yaml or .json)",
    )
    parser.add_argument(
        "--left-camera",
        type=int,
        default=0,
        help="Left camera index (default: 0)",
    )
    parser.add_argument(
        "--right-camera",
        type=int,
        default=1,
        help="Right camera index (default: 1)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["bm", "sgbm"],
        default="sgbm",
        help="Stereo matching method: bm (fast) or sgbm (accurate) (default: sgbm)",
    )
    parser.add_argument(
        "--num-disparities",
        type=int,
        default=80,
        help="Maximum disparity (must be divisible by 16) (default: 80)",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=11,
        help="Matched block size (must be odd) (default: 11)",
    )
    parser.add_argument(
        "--save-output",
        action="store_true",
        help="Save output images and depth maps",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/stereo_depth",
        help="Output directory for saved images (default: outputs/stereo_depth)",
    )
    return parser.parse_args()


def colorize_depth(depth_map: np.ndarray, min_depth: float = 0.3, max_depth: float = 3.0) -> np.ndarray:
    """
    Colorize depth map for visualization.

    Args:
        depth_map: Depth map in meters
        min_depth: Minimum depth for color mapping (meters)
        max_depth: Maximum depth for color mapping (meters)

    Returns:
        BGR color image for visualization
    """
    # Clip and normalize depth
    depth_clipped = np.clip(depth_map, min_depth, max_depth)
    depth_normalized = (depth_clipped - min_depth) / (max_depth - min_depth)
    depth_normalized = (depth_normalized * 255).astype(np.uint8)

    # Apply colormap (TURBO is good for depth visualization)
    depth_colorized = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_TURBO)

    # Mask invalid depths (inf or negative)
    invalid_mask = np.isinf(depth_map) | (depth_map <= 0)
    depth_colorized[invalid_mask] = [0, 0, 0]  # Black for invalid depths

    return depth_colorized


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Stereo Depth Estimation Test")
    print("=" * 70)

    # Load calibration
    print(f"\nLoading calibration from: {args.calibration}")
    try:
        calibration_data = load_stereo_calibration(args.calibration)
        print("✓ Calibration loaded successfully")
        print(f"  RMS Error:    {calibration_data.get('rms_error', 'N/A')}")
        print(f"  Image size:   {calibration_data['image_size']}")
        print(f"  Pattern size: {calibration_data.get('pattern_size', 'N/A')}")
    except Exception as e:
        print(f"❌ Error loading calibration: {e}")
        return

    # Initialize depth estimator
    print(f"\nInitializing depth estimator...")
    print(f"  Method:          {args.method.upper()}")
    print(f"  Num disparities: {args.num_disparities}")
    print(f"  Block size:      {args.block_size}")

    try:
        estimator = StereoDepthEstimator(
            calibration_data=calibration_data,
            method=args.method,
            num_disparities=args.num_disparities,
            block_size=args.block_size,
        )
        print("✓ Depth estimator initialized")
    except Exception as e:
        print(f"❌ Error initializing depth estimator: {e}")
        return

    # Get camera configuration from calibration
    width, height = calibration_data["image_size"]

    # Initialize cameras
    print("\n" + "-" * 70)
    print("Initializing cameras...")

    camera_configs = {
        "left": OpenCVCameraConfig(
            index_or_path=args.left_camera,
            fps=30,
            width=width,
            height=height,
            color_mode=ColorMode.BGR,
        ),
        "right": OpenCVCameraConfig(
            index_or_path=args.right_camera,
            fps=30,
            width=width,
            height=height,
            color_mode=ColorMode.BGR,
        ),
    }

    cameras = {}
    for name, config in camera_configs.items():
        print(f"  Connecting {name} camera (index {config.index_or_path})...")
        cameras[name] = OpenCVCamera(config)
        cameras[name].connect(warmup=False)

    print("✓ Cameras initialized")

    # Create output directory if saving
    if args.save_output:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nOutput will be saved to: {output_dir}")

    print("\n" + "-" * 70)
    print("Controls:")
    print("  ESC   - Exit")
    print("  S     - Save current frame and depth map")
    print("  SPACE - Toggle depth visualization mode")
    print("-" * 70)

    frame_count = 0
    save_count = 0
    show_colorized = True
    fps_history = []

    try:
        while True:
            start_time = time.time()

            # Read frames
            frames = {}
            retry_count = 0
            max_retries = 3

            while retry_count < max_retries:
                try:
                    for name, camera in cameras.items():
                        frames[name] = camera.read()
                    break
                except RuntimeError as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        if frame_count == 0:
                            print(f"⚠️  Initial read failed (retry {retry_count}/{max_retries})")
                        time.sleep(0.1)
                    else:
                        raise

            left_frame = frames["left"]
            right_frame = frames["right"]

            # Compute depth
            depth_map = estimator.compute_depth(left_frame, right_frame)

            # Get depth at center point
            center_y, center_x = height // 2, width // 2
            center_depth = depth_map[center_y, center_x]
            center_depth_str = (
                f"{center_depth:.2f}m" if not np.isinf(center_depth) else "inf"
            )

            # Compute statistics
            valid_depths = depth_map[~np.isinf(depth_map) & (depth_map > 0)]
            if len(valid_depths) > 0:
                min_depth = np.min(valid_depths)
                max_depth = np.max(valid_depths)
                mean_depth = np.mean(valid_depths)
                stats_str = f"Min: {min_depth:.2f}m, Max: {max_depth:.2f}m, Mean: {mean_depth:.2f}m"
            else:
                stats_str = "No valid depth data"

            # Visualize depth
            if show_colorized:
                depth_vis = colorize_depth(depth_map)
            else:
                # Show raw disparity
                disparity = estimator.compute_disparity(left_frame, right_frame)
                disparity_normalized = cv2.normalize(
                    disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                )
                depth_vis = cv2.applyColorMap(disparity_normalized, cv2.COLORMAP_JET)

            # Draw center crosshair
            cv2.drawMarker(
                depth_vis,
                (center_x, center_y),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                20,
                2,
            )

            # Add text overlay
            end_time = time.time()
            fps = 1.0 / (end_time - start_time)
            fps_history.append(fps)
            if len(fps_history) > 30:
                fps_history.pop(0)
            avg_fps = np.mean(fps_history)

            mode_str = "Colorized Depth" if show_colorized else "Raw Disparity"
            cv2.putText(
                depth_vis,
                f"FPS: {avg_fps:.1f} | Mode: {mode_str}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                depth_vis,
                f"Center: {center_depth_str}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                depth_vis,
                stats_str,
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

            # Show frames
            cv2.imshow("Left Camera", left_frame)
            cv2.imshow("Right Camera", right_frame)
            cv2.imshow("Depth Map", depth_vis)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("\nExiting...")
                break
            elif key == ord("s"):  # Save
                if args.save_output:
                    output_dir = Path(args.output_dir)
                    cv2.imwrite(str(output_dir / f"left_{save_count:03d}.jpg"), left_frame)
                    cv2.imwrite(str(output_dir / f"right_{save_count:03d}.jpg"), right_frame)
                    cv2.imwrite(str(output_dir / f"depth_{save_count:03d}.jpg"), depth_vis)
                    np.save(str(output_dir / f"depth_{save_count:03d}.npy"), depth_map)
                    print(f"✓ Saved frame {save_count} to {output_dir}")
                    save_count += 1
                else:
                    print("⚠️  Saving disabled. Use --save-output to enable.")
            elif key == ord(" "):  # Toggle mode
                show_colorized = not show_colorized
                print(f"Switched to {'colorized depth' if show_colorized else 'raw disparity'} mode")

            frame_count += 1

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        # Cleanup
        for camera in cameras.values():
            camera.disconnect()
        cv2.destroyAllWindows()

        print(f"\nProcessed {frame_count} frames")
        if save_count > 0:
            print(f"Saved {save_count} frames to {args.output_dir}")


if __name__ == "__main__":
    main()
