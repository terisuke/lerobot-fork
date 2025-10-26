#!/usr/bin/env python3
"""
Record Cup Manipulation Demonstrations for SO101

This script records demonstrations of:
1. Detecting cup with webcam (upside-down mount supported)
2. Picking up cup via teleoperation
3. Moving cup to position near bottle
4. Placing cup

Recorded data will be used to train ACT model for autonomous execution.
"""

import sys
import cv2
import numpy as np
from pathlib import Path
import time
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  YOLOv8 not found. Install with: pip install ultralytics")

# TODO: Import SO101 robot control
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.robots.so100_follower.config_so100_follower import SO100FollowerConfig
from lerobot.robots.so100_follower.so100_follower import SO100Follower
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardTeleop, KeyboardTeleopConfig


class CupDemonstrationRecorder:
    """Record demonstrations for cup manipulation task."""
    
    def __init__(self, output_dir="data/cup_demos", camera_flip=True, enable_robot=True):
        """
        Initialize recorder.
        
        Args:
            output_dir: Directory to save demonstrations
            camera_flip: Whether to flip camera image (for upside-down mount)
            enable_robot: Whether to connect to actual robot
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.camera_flip = camera_flip
        self.enable_robot = enable_robot
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        print("✅ Camera initialized")
        if self.camera_flip:
            print("   📹 Camera mounted upside-down (will flip image)")
        
        # Initialize YOLO model
        if YOLO_AVAILABLE:
            self.model = YOLO("yolov8n.pt")
            print("✅ YOLOv8 model loaded")
        else:
            self.model = None
            print("⚠️  YOLO not available - running without detection")
        
        # Initialize robot and teleoperation
        self.robot = None
        self.teleop = None
        if self.enable_robot:
            try:
                # TODO: Configure robot port
                robot_config = SO100FollowerConfig(
                    port="/dev/ttyUSB0",  # Adjust to your port
                    id="my_awesome_follower_arm",
                    use_degrees=True
                )
                self.robot = SO100Follower(robot_config)
                
                # Initialize keyboard teleoperation
                teleop_config = KeyboardTeleopConfig(id="keyboard_teleop")
                self.teleop = KeyboardTeleop(teleop_config)
                
                self.robot.connect()
                self.teleop.connect()
                
                print("✅ Robot connected")
                print("✅ Keyboard teleoperation enabled")
            except Exception as e:
                print(f"⚠️  Robot connection failed: {e}")
                print("   Running in simulation mode")
                self.robot = None
                self.teleop = None
        else:
            print("ℹ️  Running in simulation mode (no robot)")
        
        # Recording state
        self.recording = False
        self.demo_data = []
        self.demo_count = 0
        self.frame_count = 0
        
        # Current action (for teleoperation)
        self.current_action = {
            'delta_x': 0.0,
            'delta_y': 0.0,
            'delta_z': 0.0,
            'gripper': 1.0  # 0=close, 1=stay, 2=open
        }
        
        print("\n📝 Initialization complete")
        print("=" * 60)
        print("Controls:")
        print("  Arrow Keys - Move arm (x, y)")
        print("  Shift/Ctrl - Move arm (z)")
        print("  Space - Gripper open/close")
        print("  'r' - Start/Stop recording")
        print("  's' - Save current demo")
        print("  'q' - Quit")
        print("=" * 60)
    
    def flip_image_if_needed(self, frame):
        """Flip image if camera is mounted upside-down."""
        if self.camera_flip:
            # Flip both horizontally and vertically
            return cv2.flip(frame, -1)
        return frame
    
    def detect_cup(self, frame):
        """Detect cup in frame using YOLO."""
        if self.model is None:
            return None
        
        results = self.model.predict(frame, conf=0.5, verbose=False)
        
        for box in results[0].boxes:
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            class_name = results[0].names[class_id]
            
            if class_name in ['cup', 'mug', 'vase', 'bowl']:
                bbox = box.xyxy[0].cpu().numpy()
                return {
                    'class': class_name,
                    'confidence': confidence,
                    'bbox': bbox.tolist(),
                    'position_2d': [float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2)]
                }
        
        return None
    
    def get_robot_state(self):
        """Get current robot state."""
        if self.robot and self.robot.is_connected:
            try:
                obs = self.robot.get_observation()
                return {
                    'joint_positions': obs.get('state', [0.0] * 6).tolist() if hasattr(obs.get('state', []), 'tolist') else obs.get('state', [0.0] * 6),
                    'gripper_position': obs.get('gripper', 0.0),
                    'connected': True
                }
            except Exception as e:
                print(f"⚠️  Failed to get robot state: {e}")
                return self._default_robot_state()
        return self._default_robot_state()
    
    def _default_robot_state(self):
        """Return default robot state for simulation."""
        return {
            'joint_positions': [0.0] * 6,
            'gripper_position': 0.5,
            'connected': False
        }
    
    def update_teleoperation(self):
        """Update action from teleoperation input."""
        if self.teleop and self.teleop.is_connected:
            try:
                action = self.teleop.get_action()
                # Update current action based on teleop input
                # This is a simplified version - would need proper key mapping
                self.current_action = {
                    'delta_x': 0.0,
                    'delta_y': 0.0,
                    'delta_z': 0.0,
                    'gripper': 1.0
                }
            except Exception as e:
                # No teleop input yet
                pass
    
    def visualize(self, frame, cup_detection, recording_status):
        """Draw visualization on frame."""
        annotated = frame.copy()
        
        # Recording status
        status_color = (0, 0, 255) if recording_status else (0, 255, 0)
        status_text = "RECORDING" if recording_status else "STANDBY"
        cv2.putText(annotated, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        
        # Demo count
        cv2.putText(annotated, f"Demos: {self.demo_count}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Robot connection status
        robot_status = "Connected" if (self.robot and self.robot.is_connected) else "Simulation"
        robot_color = (0, 255, 0) if robot_status == "Connected" else (100, 100, 100)
        cv2.putText(annotated, f"Robot: {robot_status}", (10, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, robot_color, 2)
        
        # Camera flip status
        if self.camera_flip:
            cv2.putText(annotated, "Flip: ON", (10, 130),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Cup detection
        if cup_detection:
            x1, y1, x2, y2 = cup_detection['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            label = f"{cup_detection['class']} {cup_detection['confidence']:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Info text at bottom
        cv2.putText(annotated, "r: Record, s: Save, q: Quit | Arrow: Move, Space: Gripper",
                   (10, annotated.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return annotated
    
    def record_frame(self, frame, cup_detection, robot_state):
        """Record a single frame of demonstration data."""
        timestamp = time.time()
        
        frame_data = {
            'timestamp': timestamp,
            'frame_number': self.frame_count,
            'cup_detection': cup_detection,
            'robot_state': robot_state,
            'action': self.current_action.copy()
        }
        
        self.demo_data.append(frame_data)
    
    def save_demo(self):
        """Save current demonstration to file."""
        if len(self.demo_data) == 0:
            print("⚠️  No data to save")
            return
        
        # Create demo directory
        demo_dir = self.output_dir / f"demo_{self.demo_count:03d}"
        demo_dir.mkdir(exist_ok=True)
        
        # Save frames as images
        frames_dir = demo_dir / "frames"
        frames_dir.mkdir(exist_ok=True)
        
        metadata = {
            'num_frames': len(self.demo_data),
            'start_time': self.demo_data[0]['timestamp'],
            'end_time': self.demo_data[-1]['timestamp'],
            'task': 'cup_to_bottle',
            'camera_flipped': self.camera_flip,
            'robot_connected': self.demo_data[0]['robot_state']['connected']
        }
        
        # Save metadata
        with open(demo_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save frame data
        frame_data_list = []
        for i, frame_data in enumerate(self.demo_data):
            # Create a simple timestamp-based filename
            frame_filename = f"frame_{i:05d}.txt"
            frame_path = frames_dir / frame_filename
            
            # Save frame info (in real implementation, would save actual image)
            with open(frame_path, 'w') as f:
                json.dump({
                    'timestamp': frame_data['timestamp'],
                    'frame_number': frame_data['frame_number'],
                    'cup_detection': frame_data['cup_detection'],
                    'robot_state': frame_data['robot_state'],
                    'action': frame_data['action']
                }, f, indent=2)
            
            frame_data_list.append({
                'frame_number': i,
                'frame_file': f"frames/{frame_filename}",
                'timestamp': frame_data['timestamp']
            })
        
        # Save index
        with open(demo_dir / "frame_index.json", 'w') as f:
            json.dump(frame_data_list, f, indent=2)
        
        print(f"✅ Saved demo {self.demo_count:03d} with {len(self.demo_data)} frames")
        print(f"   Location: {demo_dir}")
        
        self.demo_count += 1
        self.demo_data = []
        self.frame_count = 0
    
    def run(self):
        """Main recording loop."""
        print("\n🎥 Starting recording loop...\n")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Flip image if camera is upside-down
                frame = self.flip_image_if_needed(frame)
                
                # Detect cup
                cup_detection = None
                if self.model is not None:
                    cup_detection = self.detect_cup(frame)
                
                # Update teleoperation
                self.update_teleoperation()
                
                # Get robot state
                robot_state = self.get_robot_state()
                
                # Visualize
                annotated = self.visualize(frame, cup_detection, self.recording)
                cv2.imshow('Cup Demonstration Recorder', annotated)
                
                # Record if recording
                if self.recording:
                    self.record_frame(frame, cup_detection, robot_state)
                    self.frame_count += 1
                    
                    # Status update
                    if self.frame_count % 30 == 0:
                        print(f"\r📹 Recording... {self.frame_count} frames recorded", end="")
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('r'):
                    # Toggle recording
                    self.recording = not self.recording
                    if self.recording:
                        print("\n🔴 Started recording")
                        self.frame_count = 0
                    else:
                        print(f"\n⏹️  Stopped recording ({len(self.demo_data)} frames)")
                
                elif key == ord('q'):
                    break
                
                elif key == ord('s'):
                    # Save demo
                    if len(self.demo_data) > 0:
                        self.save_demo()
                    else:
                        print("⚠️  No data recorded yet")
                
                # TODO: Add keyboard controls for teleoperation
                # elif key == ord('w'):  # Move forward
                #     self.current_action['delta_y'] = -0.01
                # elif key == ord('s'):  # Move backward
                #     self.current_action['delta_y'] = 0.01
                # ... etc
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted by user")
        
        finally:
            # Cleanup
            self.cap.release()
            cv2.destroyAllWindows()
            
            if self.teleop:
                self.teleop.disconnect()
            if self.robot:
                self.robot.disconnect()
            
            # Save any remaining data
            if len(self.demo_data) > 0:
                print(f"\n💾 Saving remaining {len(self.demo_data)} frames...")
                self.save_demo()
            
            print(f"\n✅ Recording complete")
            print(f"   Total demos: {self.demo_count}")
            print(f"   Output directory: {self.output_dir}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Cup Manipulation Demonstration Recorder")
    print("=" * 60)
    print("Task: Pick cup → Place near bottle")
    print("")
    
    # Configuration
    output_dir = "data/cup_demos_webcam"
    camera_flip = True  # Set to True if camera is mounted upside-down
    enable_robot = True  # Set to False for simulation mode
    
    print(f"Configuration:")
    print(f"  Output directory: {output_dir}")
    print(f"  Camera flip: {camera_flip}")
    print(f"  Robot enabled: {enable_robot}")
    print("")
    
    try:
        recorder = CupDemonstrationRecorder(
            output_dir=output_dir,
            camera_flip=camera_flip,
            enable_robot=enable_robot
        )
        recorder.run()
        
        print("\n📊 Summary:")
        print(f"   Demos recorded: {recorder.demo_count}")
        
        print("\n💡 Next steps:")
        print("   1. Review recorded demonstrations")
        print("   2. Convert to LeRobot dataset format")
        print("   3. Train ACT model")
        print("   4. Evaluate autonomous execution")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
