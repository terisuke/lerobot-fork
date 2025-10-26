#!/usr/bin/env python3
"""
Automatic Object Picking System

This script implements an automatic object picking system that detects
thrown objects using RealSense camera and controls SO-101 robot arm
to pick them up.
"""

import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Add lerobot to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from lerobot.datasets.realsense_dataset import RealSenseDatasetRecorder
from lerobot.object_detection import DepthObjectTracker, ObjectDetector
from lerobot.robots.so100 import SO100Robot
from lerobot.teleoperators.so100_teleop import SO100Teleoperator


class AutoPickSystem:
    """
    Automatic object picking system using RealSense camera and SO-101 robot.

    This system detects thrown objects, tracks their trajectory,
    and automatically controls the robot to pick them up.
    """

    def __init__(
        self,
        robot_port: str = "/dev/ttyUSB0",
        dataset_dir: str = "./auto_pick_dataset",
        confidence_threshold: float = 0.7,
        velocity_threshold: float = 0.05,
    ):
        """
        Initialize the automatic picking system.

        Args:
            robot_port: Serial port for SO-101 robot
            dataset_dir: Directory to save dataset
            confidence_threshold: Minimum confidence for object detection
            velocity_threshold: Minimum velocity to consider object as moving
        """
        self.confidence_threshold = confidence_threshold
        self.velocity_threshold = velocity_threshold

        # Initialize components
        print("🤖 Initializing Auto Pick System...")

        # Initialize RealSense camera and detection
        self.detector = ObjectDetector(confidence_threshold=confidence_threshold)
        self.tracker = DepthObjectTracker()

        # Initialize dataset recorder
        self.dataset_recorder = RealSenseDatasetRecorder(dataset_dir)

        # Initialize robot
        try:
            self.robot = SO100Robot(port=robot_port)
            self.teleop = SO100Teleoperator(self.robot)
            print("✅ SO-101 robot initialized")
        except Exception as e:
            print(f"⚠️  Robot initialization failed: {e}")
            self.robot = None
            self.teleop = None

        # System state
        self.is_running = False
        self.current_target = None
        self.pick_attempts = 0
        self.max_pick_attempts = 3

    def start_system(self):
        """Start the automatic picking system."""
        if self.is_running:
            print("⚠️  System is already running")
            return

        self.is_running = True
        print("🚀 Auto Pick System started")
        print("📹 Monitoring for thrown objects...")

        try:
            while self.is_running:
                # Get camera frames
                rgb_image, depth_image, intrinsics = self.detector.get_camera_frames()

                # Detect objects
                detections = self.detector.detect_objects(
                    rgb_image, depth_image, intrinsics
                )

                # Track objects
                tracked_objects = self.tracker.update(detections)

                # Find moving objects (potentially thrown)
                moving_objects = self.tracker.get_moving_objects(
                    self.velocity_threshold
                )

                # Process moving objects
                if moving_objects:
                    self._process_moving_objects(moving_objects, rgb_image, depth_image)

                # Display results
                self._display_results(rgb_image, tracked_objects, moving_objects)

                # Small delay
                time.sleep(0.033)  # ~30 FPS

        except KeyboardInterrupt:
            print("⏹️  System stopped by user")
        finally:
            self.stop_system()

    def _process_moving_objects(self, moving_objects, rgb_image, depth_image):
        """Process moving objects and attempt to pick them up."""
        for obj in moving_objects:
            # Check if object is suitable for picking
            if self._is_suitable_for_picking(obj):
                print(f"🎯 Target object detected: {obj.current_detection.class_name}")
                print(f"   Position: {obj.current_detection.world_position}")
                print(f"   Velocity: {obj.velocity}")

                # Set as current target
                self.current_target = obj

                # Attempt to pick up the object
                if self.robot is not None:
                    self._attempt_pickup(obj)
                else:
                    print("⚠️  Robot not available, simulating pickup")
                    self._simulate_pickup(obj)

    def _is_suitable_for_picking(self, obj) -> bool:
        """Check if object is suitable for picking."""
        # Check depth range
        if not (0.2 <= obj.current_detection.depth <= 1.5):
            return False

        # Check confidence
        if obj.depth_confidence < 0.8:
            return False

        # Check size
        x, y, w, h = obj.current_detection.bbox
        if w < 30 or h < 30 or w > 200 or h > 200:
            return False

        # Check if object is moving (thrown)
        velocity_magnitude = np.sqrt(
            obj.velocity[0] ** 2 + obj.velocity[1] ** 2 + obj.velocity[2] ** 2
        )
        if velocity_magnitude < self.velocity_threshold:
            return False

        return True

    def _attempt_pickup(self, obj):
        """Attempt to pick up the target object."""
        if self.pick_attempts >= self.max_pick_attempts:
            print("❌ Max pickup attempts reached, giving up")
            return

        self.pick_attempts += 1
        print(f"🤖 Attempting pickup #{self.pick_attempts}")

        try:
            # Get object position
            world_pos = obj.current_detection.world_position
            x, y, z = world_pos

            # Convert to robot coordinates (adjust based on camera-robot calibration)
            robot_x = x + 0.1  # Adjust based on camera position
            robot_y = y + 0.05
            robot_z = z + 0.02

            # Move to object position
            print(
                f"📍 Moving to position: ({robot_x:.3f}, {robot_y:.3f}, {robot_z:.3f})"
            )

            # Move robot to object (simplified - would need proper inverse kinematics)
            self.teleop.move_to_position(robot_x, robot_y, robot_z)

            # Close gripper
            print("🤏 Closing gripper")
            self.teleop.close_gripper()

            # Lift object
            print("⬆️  Lifting object")
            self.teleop.move_to_position(robot_x, robot_y, robot_z + 0.1)

            # Move to drop position
            print("📦 Moving to drop position")
            self.teleop.move_to_position(0.2, 0.0, 0.3)  # Drop position

            # Open gripper
            print("🤏 Opening gripper")
            self.teleop.open_gripper()

            # Return to home position
            print("🏠 Returning to home position")
            self.teleop.move_to_position(0.0, 0.0, 0.2)

            print("✅ Pickup completed successfully")
            self.current_target = None
            self.pick_attempts = 0

        except Exception as e:
            print(f"❌ Pickup failed: {e}")
            self.pick_attempts += 1

    def _simulate_pickup(self, obj):
        """Simulate pickup without actual robot control."""
        print("🎭 Simulating pickup sequence...")

        # Simulate robot movements
        time.sleep(1.0)  # Move to object
        print("✅ Object picked up (simulated)")
        time.sleep(0.5)  # Lift
        print("✅ Object lifted (simulated)")
        time.sleep(1.0)  # Move to drop
        print("✅ Object dropped (simulated)")

        self.current_target = None
        self.pick_attempts = 0

    def _display_results(self, rgb_image, tracked_objects, moving_objects):
        """Display detection and tracking results."""
        display_image = rgb_image.copy()

        # Draw all tracked objects
        for obj in tracked_objects:
            x, y, w, h = obj.current_detection.bbox
            color = (0, 255, 0) if obj in moving_objects else (255, 0, 0)
            cv2.rectangle(display_image, (x, y), (x + w, y + h), color, 2)

            # Add label
            label = f"ID:{obj.object_id} {obj.current_detection.class_name}"
            cv2.putText(
                display_image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Add velocity info for moving objects
            if obj in moving_objects:
                velocity_magnitude = np.sqrt(
                    obj.velocity[0] ** 2 + obj.velocity[1] ** 2 + obj.velocity[2] ** 2
                )
                vel_text = f"Speed: {velocity_magnitude:.3f} m/s"
                cv2.putText(
                    display_image,
                    vel_text,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        # Add system status
        status_text = (
            f"Tracking: {len(tracked_objects)} | Moving: {len(moving_objects)}"
        )
        cv2.putText(
            display_image,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Show image
        cv2.imshow("Auto Pick System", display_image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            self.stop_system()
        elif key == ord("r"):
            self._start_recording()
        elif key == ord("s"):
            self._stop_recording()

    def _start_recording(self):
        """Start recording dataset."""
        if not hasattr(self, "recording_episode"):
            episode_id = self.dataset_recorder.start_episode(
                metadata={"system": "auto_pick", "mode": "detection"}
            )
            self.recording_episode = episode_id
            print(f"📹 Started recording episode: {episode_id}")

    def _stop_recording(self):
        """Stop recording dataset."""
        if hasattr(self, "recording_episode"):
            episode_path = self.dataset_recorder.stop_episode()
            delattr(self, "recording_episode")
            print(f"💾 Stopped recording: {episode_path}")

    def stop_system(self):
        """Stop the automatic picking system."""
        if not self.is_running:
            return

        self.is_running = False

        # Stop recording if active
        if hasattr(self, "recording_episode"):
            self._stop_recording()

        # Close camera
        self.detector.close()

        # Close robot
        if self.robot is not None:
            self.robot.close()

        # Close display
        cv2.destroyAllWindows()

        print("🛑 Auto Pick System stopped")

    def run_interactive_mode(self):
        """Run the system in interactive mode with manual controls."""
        print("🎮 Interactive Mode")
        print("Controls:")
        print("  'q' - Quit")
        print("  'r' - Start recording")
        print("  's' - Stop recording")
        print("  'h' - Home robot")
        print("  'p' - Pickup current target")

        self.start_system()


def main():
    """Main function to run the automatic picking system."""
    import argparse

    parser = argparse.ArgumentParser(description="Automatic Object Picking System")
    parser.add_argument(
        "--robot-port", default="/dev/ttyUSB0", help="Serial port for SO-101 robot"
    )
    parser.add_argument(
        "--dataset-dir", default="./auto_pick_dataset", help="Directory to save dataset"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.7,
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=0.05,
        help="Velocity threshold for moving objects",
    )

    args = parser.parse_args()

    # Create system
    system = AutoPickSystem(
        robot_port=args.robot_port,
        dataset_dir=args.dataset_dir,
        confidence_threshold=args.confidence,
        velocity_threshold=args.velocity,
    )

    try:
        # Run interactive mode
        system.run_interactive_mode()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    finally:
        system.stop_system()


if __name__ == "__main__":
    main()
