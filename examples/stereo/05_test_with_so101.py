#!/usr/bin/env python
"""
SO-101 Stereo System Integration Test

This script tests the stereo camera integration with the SO-101 robot follower.
It uses the existing OpenCVCamera implementation and stereo utilities from Phase 1-4.

Usage:
    # Basic test (side-by-side view)
    python examples/stereo/05_test_with_so101.py

    # With depth estimation
    python examples/stereo/05_test_with_so101.py --show-depth

    # Custom camera indices
    python examples/stereo/05_test_with_so101.py --left-index 0 --right-index 1

    # Custom calibration file
    python examples/stereo/05_test_with_so101.py --calibration configs/camera/my_calib.yaml
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.utils.stereo.calibration import load_stereo_calibration
from lerobot.utils.stereo.depth_estimation import StereoDepthEstimator


def main():
    parser = argparse.ArgumentParser(
        description="Test SO-101 with stereo cameras"
    )
    parser.add_argument(
        "--left-index", type=int, default=0,
        help="Left camera index (default: 0)"
    )
    parser.add_argument(
        "--right-index", type=int, default=1,
        help="Right camera index (default: 1)"
    )
    parser.add_argument(
        "--calibration", type=str,
        default="configs/camera/stereo_calibration.yaml",
        help="Stereo calibration file path"
    )
    parser.add_argument(
        "--show-depth", action="store_true",
        help="Display depth map (requires calibration file)"
    )
    parser.add_argument(
        "--fps", type=int, default=30,
        help="Camera FPS (default: 30)"
    )
    parser.add_argument(
        "--width", type=int, default=1280,
        help="Camera width (default: 1280)"
    )
    parser.add_argument(
        "--height", type=int, default=720,
        help="Camera height (default: 720)"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("🎥 SO-101 Stereo Integration Test")
    print("=" * 70)
    print(f"Left camera index:  {args.left_index}")
    print(f"Right camera index: {args.right_index}")
    print(f"Resolution:         {args.width}x{args.height} @ {args.fps}fps")
    print(f"Calibration file:   {args.calibration}")
    print(f"Depth estimation:   {'Enabled' if args.show_depth else 'Disabled'}")
    print("=" * 70)

    # Initialize cameras with correct field name (index_or_path)
    left_config = OpenCVCameraConfig(
        index_or_path=args.left_index,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )
    right_config = OpenCVCameraConfig(
        index_or_path=args.right_index,
        fps=args.fps,
        width=args.width,
        height=args.height,
    )

    print("\n🔌 Initializing cameras...")
    left_camera = OpenCVCamera(left_config)
    right_camera = OpenCVCamera(right_config)

    print("🔌 Connecting cameras...")
    try:
        left_camera.connect(warmup=True)
        right_camera.connect(warmup=True)
        print("✅ Cameras connected successfully")
    except Exception as e:
        print(f"❌ Failed to connect cameras: {e}")
        return 1

    # Setup depth estimator if requested
    depth_estimator = None
    if args.show_depth:
        calibration_path = Path(args.calibration)
        if calibration_path.exists():
            print(f"\n📐 Loading calibration: {args.calibration}")
            try:
                calib_data = load_stereo_calibration(args.calibration)
                depth_estimator = StereoDepthEstimator(calib_data, method="sgbm")
                print("✅ Depth estimation enabled (SGBM method)")
            except Exception as e:
                print(f"⚠️  Failed to load calibration: {e}")
                print("   Continuing without depth estimation...")
                args.show_depth = False
        else:
            print(f"⚠️  Calibration file not found: {args.calibration}")
            print("   Run calibration first:")
            print(f"   python examples/stereo/02_calibrate_stereo.py")
            print("   Continuing without depth estimation...")
            args.show_depth = False

    print("\n🎮 Controls:")
    print("  q - Quit")
    print("  d - Toggle depth view")
    print("  s - Save current frames")
    print("\n▶️  Starting camera stream...\n")

    show_depth = args.show_depth
    frame_count = 0
    saved_count = 0

    try:
        while True:
            # Capture stereo frames
            left_frame = left_camera.read()
            right_frame = right_camera.read()
            frame_count += 1

            if show_depth and depth_estimator:
                # Compute depth using Phase 1-4 utilities
                depth_map = depth_estimator.compute_depth(
                    left_frame, right_frame
                )

                # Visualize depth (normalize and apply colormap)
                depth_vis = cv2.normalize(
                    depth_map, None, 0, 255, cv2.NORM_MINMAX
                )
                depth_vis = depth_vis.astype(np.uint8)
                depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

                # 2x2 grid layout: [left, right] / [depth, depth]
                top_row = np.hstack([left_frame, right_frame])
                bottom_row = np.hstack([depth_vis, depth_vis])
                display = np.vstack([top_row, bottom_row])

                # Add depth statistics
                valid_depths = depth_map[np.isfinite(depth_map)]
                if len(valid_depths) > 0:
                    min_depth = valid_depths.min()
                    max_depth = valid_depths.max()
                    mean_depth = valid_depths.mean()
                    depth_info = f"Depth: min={min_depth:.2f}m, max={max_depth:.2f}m, mean={mean_depth:.2f}m"
                else:
                    depth_info = "Depth: No valid data"

                cv2.putText(
                    display, depth_info, (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )
            else:
                # Side-by-side stereo view
                display = np.hstack([left_frame, right_frame])

            # Add status overlay
            status = f"Depth: {'ON' if show_depth else 'OFF'} | Frame: {frame_count} | Press 'q' to quit"
            cv2.putText(
                display, status, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

            cv2.imshow("SO-101 Stereo System", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n⏹️  Quitting...")
                break
            elif key == ord('d'):
                if depth_estimator:
                    show_depth = not show_depth
                    mode = "ON" if show_depth else "OFF"
                    print(f"🔄 Depth view: {mode}")
                else:
                    print("⚠️  Depth estimation not available (no calibration)")
            elif key == ord('s'):
                # Save current frames
                output_dir = Path("outputs/stereo_test")
                output_dir.mkdir(parents=True, exist_ok=True)

                cv2.imwrite(str(output_dir / f"left_{saved_count:04d}.png"), left_frame)
                cv2.imwrite(str(output_dir / f"right_{saved_count:04d}.png"), right_frame)

                if show_depth and depth_estimator:
                    cv2.imwrite(str(output_dir / f"depth_{saved_count:04d}.png"), depth_vis)

                saved_count += 1
                print(f"💾 Saved frame {saved_count} to {output_dir}/")

    except KeyboardInterrupt:
        print("\n⏹️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during streaming: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n🔌 Disconnecting cameras...")
        left_camera.disconnect()
        right_camera.disconnect()
        cv2.destroyAllWindows()
        print(f"✅ Test completed ({frame_count} frames captured)")

    return 0


if __name__ == "__main__":
    exit(main())
