#!/usr/bin/env python3
"""
Single Webcam Test Script for LeRobot

This script tests object detection using a single USB webcam.
Useful for verifying camera setup before stereo vision implementation.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not found. Install with: pip install ultralytics")

def test_single_webcam():
    """Test single webcam capture and object detection."""
    
    print("=" * 60)
    print("Single Webcam Test")
    print("=" * 60)
    
    # Initialize webcam
    print("\n1. Initializing webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("💡 Check USB connection and try a different camera index")
        return False
    
    # Get camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✅ Camera opened: {width}x{height} @ {fps:.1f} FPS")
    
    # Initialize YOLO model
    model = None
    if YOLO_AVAILABLE:
        print("\n2. Loading YOLOv8 model...")
        try:
            model = YOLO("yolov8n.pt")
            print("✅ YOLOv8 model loaded")
        except Exception as e:
            print(f"⚠️  Could not load YOLO model: {e}")
            model = None
    else:
        print("\n⚠️  Skipping object detection (YOLO not available)")
    
    # Main loop
    print("\n3. Starting capture (press 'q' to quit, 's' to save)...")
    print("   Camera position test:")
    print("   - Place an object in view")
    print("   - Check if detection works")
    print("   - Consider optimal camera placement\n")
    
    frame_count = 0
    saved_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to read frame")
                break
            
            # Run object detection
            if model is not None:
                results = model.predict(frame, conf=0.5, verbose=False)
                
                # Draw detections
                annotated_frame = results[0].plot()
                
                # Print detection info
                if len(results[0].boxes) > 0:
                    print(f"\rFrame {frame_count}: {len(results[0].boxes)} objects detected" + " " * 10, end="")
            else:
                annotated_frame = frame
            
            # Add info text
            cv2.putText(annotated_frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, "Press 'q' to quit, 's' to save", (10, height - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display
            cv2.imshow('Webcam Test', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"webcam_frame_{saved_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"\n💾 Saved: {filename}")
                saved_count += 1
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"\rCapturing at ~{frame_count/30:.1f} FPS" + " " * 20, end="")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("\n✅ Test completed")
        print(f"   Total frames: {frame_count}")
        print(f"   Saved images: {saved_count}")
    
    return True

def test_camera_positioning():
    """Guide for testing different camera positions."""
    
    print("\n" + "=" * 60)
    print("Camera Positioning Guide")
    print("=" * 60)
    print("""
For robotic manipulation, camera placement is crucial:

RECOMMENDED POSITIONS:

1. OVERHEAD (Top-down view)
   - Mount camera above workspace
   - Best for: Pick-and-place tasks
   - Advantage: No parallax, clear depth perception
   - Setup: Tripod or ceiling mount

2. SIDE MOUNTED (45-degree angle)
   - Camera at ~45 degrees to workspace
   - Best for: General manipulation
   - Advantage: Good depth perception, minimal occlusion
   - Setup: Desk mount or tripod

3. ARM MOUNTED (End-effector camera)
   - Camera attached to robot arm
   - Best for: Precise manipulation
   - Advantage: Always sees target object
   - Setup: Custom bracket attachment

TESTING PROCEDURE:
1. Place test object (e.g., coffee cup) in workspace
2. Run this script with camera at position 1
3. Check object detection quality
4. Repeat for positions 2-3
5. Choose position with best visibility

CURRENT SETUP: Camera ID 0 (default USB webcam)
To use different camera, change the index in the code.
    """)

if __name__ == "__main__":
    print("\n🤖 LeRobot Webcam Test\n")
    
    # Show positioning guide
    test_camera_positioning()
    
    # Run test
    input("\nPress Enter to start camera test...")
    success = test_single_webcam()
    
    if success:
        print("\n✅ Camera test successful!")
        print("💡 Next steps:")
        print("   1. Test with different camera positions")
        print("   2. Verify object detection accuracy")
        print("   3. Prepare for stereo camera setup")
    else:
        print("\n❌ Camera test failed")
        print("💡 Troubleshooting:")
        print("   1. Check USB connection")
        print("   2. Try different camera index (0, 1, 2...)")
        print("   3. Check camera permissions on macOS")
