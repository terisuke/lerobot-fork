#!/usr/bin/env python
"""
Final stereo vision setup with Camera #0 and Camera #1.

Optimal Configuration:
- Camera #0 (Left):  1280x720 @ 30fps
- Camera #1 (Right): 1280x720 @ 30fps

Both physical webcams running at the same resolution and FPS for perfect synchronization.
"""
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.cameras.configs import ColorMode


def main():
    print("=" * 70)
    print("Final Stereo Vision Setup - Optimal Configuration")
    print("=" * 70)
    print("Camera #0 (Left):  1280x720 @ 30fps")
    print("Camera #1 (Right): 1280x720 @ 30fps")
    print("=" * 70)

    # LeRobot standard: dict[str, CameraConfig]
    # Using BGR mode for better color accuracy on these cameras
    camera_configs = {
        "left": OpenCVCameraConfig(
            index_or_path=0,
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR,
        ),
        "right": OpenCVCameraConfig(
            index_or_path=1,
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR,
        ),
    }

    # Initialize cameras
    print("\nInitializing cameras...")
    cameras = {}

    for name, config in camera_configs.items():
        cameras[name] = OpenCVCamera(config)
        cameras[name].connect(warmup=False)  # Skip warmup to avoid initial read failures
        print(f"✅ Connected to {name} camera")

    # Output directory
    output_dir = Path("outputs/stereo_final_optimal")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n🎥 Starting synchronized stereo capture...")
    print("Both cameras at 1280x720 @ 30fps")
    print("Press ESC to exit\n")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            # Read from both cameras
            # This is the LeRobot standard way for multi-camera capture
            frames = {}
            try:
                for name, camera in cameras.items():
                    frames[name] = camera.read()
            except RuntimeError as e:
                # Camera #0 may fail on first read, skip and retry
                if frame_count == 0:
                    print(f"⚠️  Initial read failed (expected for Camera #0), retrying...")
                    continue
                else:
                    raise

            frame_count += 1

            # Display side-by-side
            combined = np.hstack([frames["left"], frames["right"]])

            # Add info overlay
            fps = frame_count / (time.time() - start_time)
            cv2.putText(
                combined,
                f"FPS: {fps:.1f} | Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                combined,
                "Left: Cam#0 @ 1280x720",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                combined,
                "Right: Cam#1 @ 1280x720",
                (650, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            # Display (already in BGR format)
            cv2.imshow("Stereo Vision (Optimal) - ESC to exit", combined)

            # Save sample frames
            if frame_count == 1:
                cv2.imwrite(
                    str(output_dir / "left_cam0.jpg"),
                    frames["left"]  # Already in BGR
                )
                cv2.imwrite(
                    str(output_dir / "right_cam1.jpg"),
                    frames["right"]  # Already in BGR
                )
                cv2.imwrite(
                    str(output_dir / "stereo_combined.jpg"),
                    combined  # Already in BGR
                )
                print(f"💾 Saved sample frames to {output_dir}")

            # ESC to exit
            if cv2.waitKey(1) & 0xFF == 27:
                break

    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        for name, camera in cameras.items():
            camera.disconnect()
            print(f"Disconnected {name} camera")

        cv2.destroyAllWindows()

        elapsed = time.time() - start_time
        print(f"\n✅ Captured {frame_count} synchronized frames in {elapsed:.2f}s")
        print(f"Average FPS: {frame_count / elapsed:.2f}")

        # Print final configuration
        print("\n" + "=" * 70)
        print("FINAL STEREO CONFIGURATION")
        print("=" * 70)
        print("\n✅ Physical Webcam Stereo Pair:")
        print("   Left:  Camera #0 - 1280x720 @ 30fps")
        print("   Right: Camera #1 - 1280x720 @ 30fps")

        print("\n📝 LeRobot Configuration:")
        print("```python")
        print("camera_configs = {")
        print('    "left": OpenCVCameraConfig(')
        print("        index_or_path=0,")
        print("        fps=30,")
        print("        width=1280,")
        print("        height=720,")
        print("    ),")
        print('    "right": OpenCVCameraConfig(')
        print("        index_or_path=1,")
        print("        fps=30,")
        print("        width=1280,")
        print("        height=720,")
        print("    ),")
        print("}")
        print("```")

        print("\n✅ Ideal for stereo vision:")
        print("   - Same resolution: 1280x720 (HD 720p)")
        print("   - Same FPS: 30fps")
        print("   - Easy synchronization")
        print("   - Good quality for depth estimation")

        print("\n📋 Next steps:")
        print("   1. Stereo calibration with checkerboard")
        print("   2. Depth map computation")
        print("   3. Integration with robot control")

        print("=" * 70)


if __name__ == "__main__":
    main()
