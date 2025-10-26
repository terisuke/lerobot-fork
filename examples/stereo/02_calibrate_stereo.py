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
Stereo camera calibration script.

Captures images of a checkerboard pattern from two cameras and computes
stereo calibration parameters.

Usage:
    python examples/stereo/02_calibrate_stereo.py \\
        --left-camera 0 \\
        --right-camera 1 \\
        --num-images 20 \\
        --output configs/camera/stereo_calibration.yaml

Requirements:
    - Checkerboard pattern (default: 9x6 inner corners, 25mm squares)
    - Two physical webcams
    - Good lighting
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.configs import ColorMode
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.utils.stereo import calibrate_stereo_cameras, save_stereo_calibration


def parse_args():
    parser = argparse.ArgumentParser(description="Stereo camera calibration")
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
        "--width",
        type=int,
        default=1280,
        help="Camera width (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Camera height (default: 720)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Camera FPS (default: 30)",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=20,
        help="Number of calibration images to capture (default: 20)",
    )
    parser.add_argument(
        "--pattern-width",
        type=int,
        default=9,
        help="Checkerboard pattern width in inner corners (default: 9)",
    )
    parser.add_argument(
        "--pattern-height",
        type=int,
        default=6,
        help="Checkerboard pattern height in inner corners (default: 6)",
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=0.025,
        help="Checkerboard square size in meters (default: 0.025 = 25mm)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="configs/camera/stereo_calibration.yaml",
        help="Output calibration file path (default: configs/camera/stereo_calibration.yaml)",
    )
    parser.add_argument(
        "--preview-dir",
        type=str,
        default="outputs/stereo_calibration",
        help="Directory to save preview images (default: outputs/stereo_calibration)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("\n" + "=" * 70)
    print("Stereo Camera Calibration")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Left camera:  #{args.left_camera}")
    print(f"  Right camera: #{args.right_camera}")
    print(f"  Resolution:   {args.width}x{args.height} @ {args.fps}fps")
    print(f"  Pattern:      {args.pattern_width}x{args.pattern_height} inner corners")
    print(f"  Square size:  {args.square_size * 1000:.1f}mm")
    print(f"  Target:       {args.num_images} image pairs")
    print(f"  Output:       {args.output}")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    preview_dir = Path(args.preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # Initialize cameras
    print("\n" + "-" * 70)
    print("Initializing cameras...")

    camera_configs = {
        "left": OpenCVCameraConfig(
            index_or_path=args.left_camera,
            fps=args.fps,
            width=args.width,
            height=args.height,
            color_mode=ColorMode.BGR,
        ),
        "right": OpenCVCameraConfig(
            index_or_path=args.right_camera,
            fps=args.fps,
            width=args.width,
            height=args.height,
            color_mode=ColorMode.BGR,
        ),
    }

    cameras = {}
    for name, config in camera_configs.items():
        print(f"  Connecting {name} camera (index {config.index_or_path})...")
        cameras[name] = OpenCVCamera(config)
        cameras[name].connect(warmup=False)

    print("✓ Cameras initialized")

    # Prepare for image capture
    left_images = []
    right_images = []
    pattern_size = (args.pattern_width, args.pattern_height)
    capture_count = 0

    print("\n" + "-" * 70)
    print("Calibration Instructions:")
    print("-" * 70)
    print("1. Hold the checkerboard pattern in view of both cameras")
    print("2. Press SPACE to capture an image when the pattern is detected")
    print("3. Move the pattern to different positions and angles")
    print("4. Capture at least", args.num_images, "good images")
    print("5. Press ESC to finish and compute calibration")
    print("\nTip: Capture images from various:")
    print("  - Distances (near and far)")
    print("  - Angles (tilted left/right, up/down)")
    print("  - Positions (center, edges, corners)")
    print("-" * 70)

    try:
        while capture_count < args.num_images:
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
                        print(f"⚠️  Read failed (retry {retry_count}/{max_retries}): {e}")
                        time.sleep(0.1)
                    else:
                        raise

            left_frame = frames["left"]
            right_frame = frames["right"]

            # Convert to grayscale for detection
            left_gray = cv2.cvtColor(left_frame, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_frame, cv2.COLOR_BGR2GRAY)

            # Detect checkerboard
            ret_left, corners_left = cv2.findChessboardCorners(
                left_gray, pattern_size, None
            )
            ret_right, corners_right = cv2.findChessboardCorners(
                right_gray, pattern_size, None
            )

            # Draw corners for visualization
            display_left = left_frame.copy()
            display_right = right_frame.copy()

            if ret_left:
                cv2.drawChessboardCorners(display_left, pattern_size, corners_left, ret_left)
            if ret_right:
                cv2.drawChessboardCorners(display_right, pattern_size, corners_right, ret_right)

            # Status overlay
            both_detected = ret_left and ret_right
            status_color = (0, 255, 0) if both_detected else (0, 0, 255)
            status_text = f"Captured: {capture_count}/{args.num_images}"

            if both_detected:
                status_text += " - Pattern OK - Press SPACE"
            else:
                status_text += " - No pattern detected"

            cv2.putText(
                display_left,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.putText(
                display_right,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )

            # Show frames
            cv2.imshow("Left Camera", display_left)
            cv2.imshow("Right Camera", display_right)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("\nCalibration capture interrupted by user")
                break
            elif key == ord(" ") and both_detected:  # SPACE
                # Save images
                left_images.append(left_gray)
                right_images.append(right_gray)

                # Save preview images
                cv2.imwrite(
                    str(preview_dir / f"left_{capture_count:02d}.jpg"), display_left
                )
                cv2.imwrite(
                    str(preview_dir / f"right_{capture_count:02d}.jpg"), display_right
                )

                capture_count += 1
                print(f"✓ Captured image pair {capture_count}/{args.num_images}")

                # Brief pause to avoid duplicate captures
                time.sleep(0.3)

    finally:
        # Cleanup
        for camera in cameras.values():
            camera.disconnect()
        cv2.destroyAllWindows()

    # Check if we have enough images
    if len(left_images) < 10:
        print(
            f"\n❌ Error: Only {len(left_images)} image pairs captured. "
            "Need at least 10 for calibration."
        )
        return

    print("\n" + "=" * 70)
    print(f"Computing calibration with {len(left_images)} image pairs...")
    print("=" * 70)

    # Perform calibration
    try:
        calibration_data = calibrate_stereo_cameras(
            left_images=left_images,
            right_images=right_images,
            pattern_size=pattern_size,
            square_size=args.square_size,
        )

        # Save calibration
        save_stereo_calibration(calibration_data, args.output)

        print("\n" + "=" * 70)
        print("✓ Calibration Complete!")
        print("=" * 70)
        print(f"\nCalibration Results:")
        print(f"  RMS Error:     {calibration_data['rms_error']:.4f} pixels")
        print(f"  Image size:    {calibration_data['image_size']}")
        print(f"  Pattern size:  {calibration_data['pattern_size']}")
        print(f"  Square size:   {calibration_data['square_size'] * 1000:.1f}mm")
        print(f"\nCalibration saved to: {args.output}")
        print(f"Preview images saved to: {preview_dir}/")
        print("\nNext step:")
        print(f"  python examples/stereo/03_test_depth.py --calibration {args.output}")

    except Exception as e:
        print(f"\n❌ Error during calibration: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
