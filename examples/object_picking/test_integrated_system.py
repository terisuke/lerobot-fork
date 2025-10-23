#!/usr/bin/env python3
"""
Test Integrated Object Picking System

This script tests the integrated object picking system with
RealSense camera, object detection, tracking, and robot control.
"""

import sys
import os
import time
import numpy as np
import cv2
from pathlib import Path
from typing import Dict

# Add lerobot to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from lerobot.object_detection import ObjectDetector, DepthObjectTracker
from lerobot.control.realtime_controller import RealTimeController, ControlCommand, ControlState
from lerobot.datasets.realsense_dataset import RealSenseDatasetRecorder


class IntegratedSystemTester:
    """
    Tester for the integrated object picking system.
    
    This class tests all components working together:
    - RealSense camera
    - Object detection and tracking
    - Robot control
    - Dataset recording
    """
    
    def __init__(self, robot_port: str = "/dev/ttyUSB0"):
        """Initialize the integrated system tester."""
        self.robot_port = robot_port
        
        # Initialize components
        print("🧪 Initializing Integrated System Tester...")
        
        # Initialize detector and tracker
        self.detector = ObjectDetector(confidence_threshold=0.6)
        self.tracker = DepthObjectTracker()
        
        # Initialize controller
        self.controller = RealTimeController(robot_port=robot_port)
        
        # Initialize dataset recorder
        self.dataset_recorder = RealSenseDatasetRecorder("./test_dataset")
        
        # Test results
        self.test_results = {}
        
    def test_camera_system(self) -> bool:
        """Test RealSense camera system."""
        print("📹 Testing camera system...")
        
        try:
            # Get camera frames
            rgb_image, depth_image, intrinsics = self.detector.get_camera_frames()
            
            # Check image properties
            if rgb_image is None or depth_image is None:
                print("❌ Camera test failed: No images received")
                return False
            
            print(f"✅ Camera test passed: RGB {rgb_image.shape}, Depth {depth_image.shape}")
            return True
            
        except Exception as e:
            print(f"❌ Camera test failed: {e}")
            return False
    
    def test_object_detection(self) -> bool:
        """Test object detection system."""
        print("🔍 Testing object detection...")
        
        try:
            # Get camera frames
            rgb_image, depth_image, intrinsics = self.detector.get_camera_frames()
            
            # Detect objects
            detections = self.detector.detect_objects(rgb_image, depth_image, intrinsics)
            
            print(f"✅ Object detection test passed: {len(detections)} objects detected")
            
            # Display results
            display_image = rgb_image.copy()
            for i, detection in enumerate(detections):
                x, y, w, h = detection.bbox
                cv2.rectangle(display_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(display_image, f"{detection.class_name}: {detection.confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Object Detection Test", display_image)
            cv2.waitKey(1000)
            cv2.destroyAllWindows()
            
            return True
            
        except Exception as e:
            print(f"❌ Object detection test failed: {e}")
            return False
    
    def test_object_tracking(self) -> bool:
        """Test object tracking system."""
        print("🎯 Testing object tracking...")
        
        try:
            # Test tracking for multiple frames
            for frame_num in range(30):  # 30 frames
                # Get camera frames
                rgb_image, depth_image, intrinsics = self.detector.get_camera_frames()
                
                # Detect objects
                detections = self.detector.detect_objects(rgb_image, depth_image, intrinsics)
                
                # Track objects
                tracked_objects = self.tracker.update(detections)
                
                if frame_num % 10 == 0:
                    print(f"  Frame {frame_num}: {len(tracked_objects)} objects tracked")
                
                # Small delay
                time.sleep(0.033)  # ~30 FPS
            
            print(f"✅ Object tracking test passed: {len(tracked_objects)} objects tracked")
            return True
            
        except Exception as e:
            print(f"❌ Object tracking test failed: {e}")
            return False
    
    def test_robot_control(self) -> bool:
        """Test robot control system."""
        print("🤖 Testing robot control...")
        
        try:
            # Test controller initialization
            if not self.controller.robot_available:
                print("⚠️  Robot not available, testing simulation mode")
                return True
            
            # Test basic robot movements
            print("  Testing robot movements...")
            
            # Home position
            self.controller._home_robot()
            time.sleep(1.0)
            
            # Test position movements
            test_positions = [
                (0.1, 0.0, 0.2),
                (0.0, 0.1, 0.2),
                (0.0, 0.0, 0.3)
            ]
            
            for pos in test_positions:
                print(f"    Moving to {pos}")
                self.controller._move_to_position(pos)
                time.sleep(1.0)
            
            # Return home
            self.controller._home_robot()
            
            print("✅ Robot control test passed")
            return True
            
        except Exception as e:
            print(f"❌ Robot control test failed: {e}")
            return False
    
    def test_dataset_recording(self) -> bool:
        """Test dataset recording system."""
        print("💾 Testing dataset recording...")
        
        try:
            # Start recording
            episode_id = self.dataset_recorder.start_episode(
                metadata={'test': 'integrated_system', 'duration': 5.0}
            )
            print(f"  Started recording episode: {episode_id}")
            
            # Record for 5 seconds
            start_time = time.time()
            frame_count = 0
            
            while time.time() - start_time < 5.0:
                # Get camera frames
                rgb_image, depth_image, intrinsics = self.detector.get_camera_frames()
                
                # Detect and track objects
                detections = self.detector.detect_objects(rgb_image, depth_image, intrinsics)
                tracked_objects = self.tracker.update(detections)
                
                # Record frame
                frame = self.dataset_recorder.record_frame()
                frame_count += 1
                
                time.sleep(0.033)  # ~30 FPS
            
            # Stop recording
            episode_path = self.dataset_recorder.stop_episode()
            
            print(f"✅ Dataset recording test passed: {frame_count} frames recorded")
            print(f"  Episode saved to: {episode_path}")
            return True
            
        except Exception as e:
            print(f"❌ Dataset recording test failed: {e}")
            return False
    
    def test_integrated_system(self) -> bool:
        """Test the complete integrated system."""
        print("🔗 Testing integrated system...")
        
        try:
            # Start controller
            self.controller.start()
            time.sleep(1.0)
            
            # Send start detection command
            command = ControlCommand(action="start_detection")
            self.controller.send_command(command)
            
            # Monitor system for 10 seconds
            start_time = time.time()
            while time.time() - start_time < 10.0:
                status = self.controller.get_status()
                print(f"  State: {status.state.value}, Target: {status.current_target is not None}")
                
                # Display camera feed
                try:
                    rgb_image, depth_image, intrinsics = self.detector.get_camera_frames()
                    detections = self.detector.detect_objects(rgb_image, depth_image, intrinsics)
                    tracked_objects = self.tracker.update(detections)
                    
                    # Display results
                    display_image = rgb_image.copy()
                    for obj in tracked_objects:
                        x, y, w, h = obj.current_detection.bbox
                        color = (0, 255, 0) if obj == status.current_target else (255, 0, 0)
                        cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)
                        cv2.putText(display_image, f"ID:{obj.object_id}", 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.imshow("Integrated System Test", display_image)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    
                except Exception as e:
                    print(f"    Camera error: {e}")
                
                time.sleep(0.1)
            
            # Stop controller
            self.controller.stop()
            cv2.destroyAllWindows()
            
            print("✅ Integrated system test passed")
            return True
            
        except Exception as e:
            print(f"❌ Integrated system test failed: {e}")
            return False
    
    def run_all_tests(self) -> Dict[str, bool]:
        """Run all tests and return results."""
        print("🧪 Running Integrated System Tests")
        print("=" * 50)
        
        tests = [
            ("Camera System", self.test_camera_system),
            ("Object Detection", self.test_object_detection),
            ("Object Tracking", self.test_object_tracking),
            ("Robot Control", self.test_robot_control),
            ("Dataset Recording", self.test_dataset_recording),
            ("Integrated System", self.test_integrated_system)
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            print(f"\n🔍 Running {test_name} test...")
            try:
                result = test_func()
                results[test_name] = result
                if result:
                    print(f"✅ {test_name} test PASSED")
                else:
                    print(f"❌ {test_name} test FAILED")
            except Exception as e:
                print(f"❌ {test_name} test ERROR: {e}")
                results[test_name] = False
        
        # Print summary
        print("\n" + "=" * 50)
        print("📊 TEST SUMMARY")
        print("=" * 50)
        
        passed = sum(1 for result in results.values() if result)
        total = len(results)
        
        for test_name, result in results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            print(f"{test_name:20} {status}")
        
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All tests passed! System is ready for use.")
        else:
            print("⚠️  Some tests failed. Please check the issues above.")
        
        return results
    
    def cleanup(self):
        """Cleanup resources."""
        try:
            self.detector.close()
            self.controller.stop()
            cv2.destroyAllWindows()
            print("🧹 Cleanup completed")
        except Exception as e:
            print(f"⚠️  Cleanup error: {e}")


def main():
    """Main function to run the integrated system tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Integrated Object Picking System")
    parser.add_argument("--robot-port", default="/dev/ttyUSB0", 
                       help="Serial port for SO-101 robot")
    parser.add_argument("--test", choices=["camera", "detection", "tracking", "robot", "dataset", "integrated", "all"],
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    # Create tester
    tester = IntegratedSystemTester(robot_port=args.robot_port)
    
    try:
        if args.test == "all":
            # Run all tests
            results = tester.run_all_tests()
        else:
            # Run specific test
            test_name = args.test
            if test_name == "camera":
                test_name = "camera_system"
            elif test_name == "detection":
                test_name = "object_detection"
            elif test_name == "tracking":
                test_name = "object_tracking"
            elif test_name == "robot":
                test_name = "robot_control"
            elif test_name == "dataset":
                test_name = "dataset_recording"
            elif test_name == "integrated":
                test_name = "integrated_system"
            
            test_func = getattr(tester, f"test_{test_name}")
            result = test_func()
            print(f"\n{'✅ PASSED' if result else '❌ FAILED'}: {args.test} test")
    
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    finally:
        tester.cleanup()


if __name__ == "__main__":
    main()
