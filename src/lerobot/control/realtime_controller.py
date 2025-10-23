"""
Real-time Control System for Object Picking

This module implements a real-time control system that coordinates
vision-based object detection, tracking, and robot control for
automatic object picking tasks.
"""

import time
import threading
import queue
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import cv2

from ..object_detection import ObjectDetector, DepthObjectTracker
from ..object_detection.detector import DetectedObject
from ..object_detection.depth_tracker import DepthTrackedObject
from ..policies.vision_policy import VisionPolicy, VisionPolicyInput, VisionPolicyOutput
from ..robots.so100_follower.so100_follower import SO100Follower
from ..robots.so100_follower.config_so100_follower import SO100FollowerConfig
from ..teleoperators.so100_leader.so100_leader import SO100Leader


class ControlState(Enum):
    """Control system states."""
    IDLE = "idle"
    DETECTING = "detecting"
    TRACKING = "tracking"
    APPROACHING = "approaching"
    PICKING = "picking"
    PLACING = "placing"
    RETURNING = "returning"
    ERROR = "error"


@dataclass
class ControlCommand:
    """Command for the control system."""
    action: str
    target_position: Optional[Tuple[float, float, float]] = None
    target_object_id: Optional[int] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class ControlStatus:
    """Status of the control system."""
    state: ControlState
    current_target: Optional[DepthTrackedObject]
    robot_position: Optional[Tuple[float, float, float]]
    confidence: float
    error_message: Optional[str] = None


class RealTimeController:
    """
    Real-time control system for object picking.
    
    This controller coordinates vision-based object detection,
    tracking, and robot control for automatic object picking.
    """
    
    def __init__(self, 
                 robot_port: str = "/dev/ttyUSB0",
                 control_frequency: float = 30.0,
                 detection_confidence: float = 0.7,
                 tracking_confidence: float = 0.8):
        """
        Initialize the real-time controller.
        
        Args:
            robot_port: Serial port for SO-101 robot
            control_frequency: Control loop frequency in Hz
            detection_confidence: Minimum confidence for object detection
            tracking_confidence: Minimum confidence for object tracking
        """
        self.control_frequency = control_frequency
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        
        # Initialize components
        self.detector = ObjectDetector(confidence_threshold=detection_confidence)
        self.tracker = DepthObjectTracker()
        self.policy = None  # Will be loaded when needed
        
        # Initialize robot
        try:
            config = SO100FollowerConfig(port=robot_port)
            self.robot = SO100Follower(config)
            self.teleop = SO100Leader(self.robot)
            self.robot_available = True
            print("✅ SO-101 robot initialized")
        except Exception as e:
            print(f"⚠️  Robot initialization failed: {e}")
            self.robot = None
            self.teleop = None
            self.robot_available = False
        
        # Control state
        self.state = ControlState.IDLE
        self.current_target = None
        self.robot_position = (0.0, 0.0, 0.2)  # Home position
        self.confidence = 0.0
        self.error_message = None
        
        # Control loop
        self.is_running = False
        self.control_thread = None
        self.command_queue = queue.Queue()
        self.status_callbacks: List[Callable[[ControlStatus], None]] = []
        
        # Safety parameters
        self.max_velocity = 0.1  # m/s
        self.max_acceleration = 0.05  # m/s²
        self.safety_limits = {
            'x': (-0.5, 0.5),
            'y': (-0.5, 0.5),
            'z': (0.0, 0.5)
        }
        
    def start(self):
        """Start the real-time control system."""
        if self.is_running:
            print("⚠️  Controller is already running")
            return
        
        self.is_running = True
        self.control_thread = threading.Thread(target=self._control_loop, daemon=True)
        self.control_thread.start()
        
        print("🚀 Real-time controller started")
    
    def stop(self):
        """Stop the real-time control system."""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.control_thread:
            self.control_thread.join(timeout=1.0)
        
        # Close robot
        if self.robot_available:
            self.robot.close()
        
        # Close camera
        self.detector.close()
        
        print("🛑 Real-time controller stopped")
    
    def _control_loop(self):
        """Main control loop."""
        last_time = time.time()
        
        while self.is_running:
            current_time = time.time()
            dt = current_time - last_time
            
            try:
                # Process commands
                self._process_commands()
                
                # Update state based on current state
                self._update_state()
                
                # Execute control actions
                self._execute_control()
                
                # Update status
                self._update_status()
                
                # Maintain control frequency
                sleep_time = 1.0 / self.control_frequency - dt
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
                last_time = current_time
                
            except Exception as e:
                print(f"❌ Control loop error: {e}")
                self.state = ControlState.ERROR
                self.error_message = str(e)
    
    def _process_commands(self):
        """Process queued commands."""
        try:
            while not self.command_queue.empty():
                command = self.command_queue.get_nowait()
                self._execute_command(command)
        except queue.Empty:
            pass
    
    def _execute_command(self, command: ControlCommand):
        """Execute a control command."""
        if command.action == "start_detection":
            self.state = ControlState.DETECTING
        elif command.action == "pick_object":
            if command.target_object_id is not None:
                self._set_target_object(command.target_object_id)
                self.state = ControlState.APPROACHING
        elif command.action == "stop":
            self.state = ControlState.IDLE
        elif command.action == "home":
            self._home_robot()
        elif command.action == "emergency_stop":
            self._emergency_stop()
    
    def _update_state(self):
        """Update control state based on current situation."""
        if self.state == ControlState.IDLE:
            return
        
        # Get current camera data
        try:
            rgb_image, depth_image, intrinsics = self.detector.get_camera_frames()
            detections = self.detector.detect_objects(rgb_image, depth_image, intrinsics)
            tracked_objects = self.tracker.update(detections)
        except Exception as e:
            print(f"⚠️  Camera error: {e}")
            return
        
        # Update state based on current state
        if self.state == ControlState.DETECTING:
            self._update_detecting_state(tracked_objects)
        elif self.state == ControlState.TRACKING:
            self._update_tracking_state(tracked_objects)
        elif self.state == ControlState.APPROACHING:
            self._update_approaching_state(tracked_objects)
        elif self.state == ControlState.PICKING:
            self._update_picking_state(tracked_objects)
        elif self.state == ControlState.PLACING:
            self._update_placing_state()
        elif self.state == ControlState.RETURNING:
            self._update_returning_state()
    
    def _update_detecting_state(self, tracked_objects: List[DepthTrackedObject]):
        """Update detecting state."""
        # Look for suitable objects
        suitable_objects = self._find_suitable_objects(tracked_objects)
        
        if suitable_objects:
            # Select best object
            best_object = self._select_best_object(suitable_objects)
            self.current_target = best_object
            self.state = ControlState.TRACKING
            print(f"🎯 Target object selected: {best_object.object_id}")
    
    def _update_tracking_state(self, tracked_objects: List[DepthTrackedObject]):
        """Update tracking state."""
        if self.current_target is None:
            self.state = ControlState.DETECTING
            return
        
        # Find current target in tracked objects
        current_target = None
        for obj in tracked_objects:
            if obj.object_id == self.current_target.object_id:
                current_target = obj
                break
        
        if current_target is None:
            print("❌ Target object lost")
            self.current_target = None
            self.state = ControlState.DETECTING
            return
        
        # Update target
        self.current_target = current_target
        
        # Check if object is stable enough for picking
        if (current_target.depth_confidence >= self.tracking_confidence and
            current_target.age >= 10):  # Tracked for at least 10 frames
            self.state = ControlState.APPROACHING
            print("✅ Target object stable, approaching...")
    
    def _update_approaching_state(self, tracked_objects: List[DepthTrackedObject]):
        """Update approaching state."""
        if self.current_target is None:
            self.state = ControlState.DETECTING
            return
        
        # Find current target
        current_target = None
        for obj in tracked_objects:
            if obj.object_id == self.current_target.object_id:
                current_target = obj
                break
        
        if current_target is None:
            self.state = ControlState.DETECTING
            return
        
        # Move robot towards target
        target_pos = current_target.current_detection.world_position
        self._move_towards_target(target_pos)
        
        # Check if close enough to pick
        distance = np.sqrt(
            (target_pos[0] - self.robot_position[0])**2 +
            (target_pos[1] - self.robot_position[1])**2 +
            (target_pos[2] - self.robot_position[2])**2
        )
        
        if distance < 0.05:  # 5cm threshold
            self.state = ControlState.PICKING
            print("🤏 Close enough to pick, starting pickup...")
    
    def _update_picking_state(self, tracked_objects: List[DepthTrackedObject]):
        """Update picking state."""
        if not self.robot_available:
            print("⚠️  Robot not available, simulating pickup")
            time.sleep(2.0)  # Simulate pickup time
            self.state = ControlState.PLACING
            return
        
        try:
            # Close gripper
            self.teleop.close_gripper()
            time.sleep(1.0)
            
            # Lift object
            lift_pos = (
                self.robot_position[0],
                self.robot_position[1],
                self.robot_position[2] + 0.1
            )
            self._move_to_position(lift_pos)
            
            self.state = ControlState.PLACING
            print("✅ Object picked up successfully")
            
        except Exception as e:
            print(f"❌ Pickup failed: {e}")
            self.state = ControlState.ERROR
            self.error_message = str(e)
    
    def _update_placing_state(self):
        """Update placing state."""
        if not self.robot_available:
            print("⚠️  Robot not available, simulating placement")
            time.sleep(1.0)
            self.state = ControlState.RETURNING
            return
        
        try:
            # Move to drop position
            drop_pos = (0.2, 0.0, 0.3)  # Drop position
            self._move_to_position(drop_pos)
            
            # Open gripper
            self.teleop.open_gripper()
            time.sleep(1.0)
            
            self.state = ControlState.RETURNING
            print("📦 Object placed successfully")
            
        except Exception as e:
            print(f"❌ Placement failed: {e}")
            self.state = ControlState.ERROR
            self.error_message = str(e)
    
    def _update_returning_state(self):
        """Update returning state."""
        if not self.robot_available:
            print("⚠️  Robot not available, simulating return")
            time.sleep(1.0)
            self.state = ControlState.IDLE
            return
        
        try:
            # Return to home position
            home_pos = (0.0, 0.0, 0.2)
            self._move_to_position(home_pos)
            
            self.state = ControlState.IDLE
            self.current_target = None
            print("🏠 Returned to home position")
            
        except Exception as e:
            print(f"❌ Return failed: {e}")
            self.state = ControlState.ERROR
            self.error_message = str(e)
    
    def _find_suitable_objects(self, tracked_objects: List[DepthTrackedObject]) -> List[DepthTrackedObject]:
        """Find objects suitable for picking."""
        suitable = []
        
        for obj in tracked_objects:
            # Check depth range
            if not (0.2 <= obj.current_detection.depth <= 1.5):
                continue
            
            # Check confidence
            if obj.depth_confidence < self.tracking_confidence:
                continue
            
            # Check size
            x, y, w, h = obj.current_detection.bbox
            if w < 30 or h < 30 or w > 200 or h > 200:
                continue
            
            # Check if object is stable (not moving too much)
            velocity_magnitude = np.sqrt(
                obj.velocity[0]**2 + obj.velocity[1]**2 + obj.velocity[2]**2
            )
            if velocity_magnitude > 0.1:  # Moving too fast
                continue
            
            suitable.append(obj)
        
        return suitable
    
    def _select_best_object(self, objects: List[DepthTrackedObject]) -> DepthTrackedObject:
        """Select the best object for picking."""
        best_object = None
        best_score = -1
        
        for obj in objects:
            score = 0
            
            # Depth score (prefer objects in good range)
            depth = obj.current_detection.depth
            if 0.3 <= depth <= 1.0:
                score += 2.0
            elif 0.2 <= depth <= 1.5:
                score += 1.0
            
            # Confidence score
            score += obj.current_detection.confidence * 2
            score += obj.depth_confidence * 2
            
            # Size score
            x, y, w, h = obj.current_detection.bbox
            size = w * h
            if 1000 <= size <= 10000:
                score += 1.0
            
            # Stability score
            if obj.age >= 20:  # Tracked for a while
                score += 1.0
            
            if score > best_score:
                best_score = score
                best_object = obj
        
        return best_object
    
    def _move_towards_target(self, target_pos: Tuple[float, float, float]):
        """Move robot towards target position."""
        if not self.robot_available:
            print(f"🎭 Simulating move to: {target_pos}")
            self.robot_position = target_pos
            return
        
        try:
            # Calculate safe movement
            safe_pos = self._calculate_safe_position(target_pos)
            
            # Move robot
            self._move_to_position(safe_pos)
            
        except Exception as e:
            print(f"❌ Movement failed: {e}")
            self.state = ControlState.ERROR
            self.error_message = str(e)
    
    def _move_to_position(self, position: Tuple[float, float, float]):
        """Move robot to specific position."""
        if not self.robot_available:
            print(f"🎭 Simulating move to: {position}")
            self.robot_position = position
            return
        
        try:
            x, y, z = position
            self.teleop.move_to_position(x, y, z)
            self.robot_position = position
            
        except Exception as e:
            print(f"❌ Position move failed: {e}")
            raise
    
    def _calculate_safe_position(self, target_pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Calculate safe position within limits."""
        x, y, z = target_pos
        
        # Apply safety limits
        x = max(self.safety_limits['x'][0], min(self.safety_limits['x'][1], x))
        y = max(self.safety_limits['y'][0], min(self.safety_limits['y'][1], y))
        z = max(self.safety_limits['z'][0], min(self.safety_limits['z'][1], z))
        
        return (x, y, z)
    
    def _set_target_object(self, object_id: int):
        """Set target object by ID."""
        # This would be implemented to find object by ID
        pass
    
    def _home_robot(self):
        """Home the robot."""
        if self.robot_available:
            try:
                self._move_to_position((0.0, 0.0, 0.2))
                print("🏠 Robot homed")
            except Exception as e:
                print(f"❌ Homing failed: {e}")
        else:
            print("🎭 Simulating robot homing")
    
    def _emergency_stop(self):
        """Emergency stop the robot."""
        if self.robot_available:
            try:
                self.robot.emergency_stop()
                print("🛑 Emergency stop activated")
            except Exception as e:
                print(f"❌ Emergency stop failed: {e}")
        else:
            print("🎭 Simulating emergency stop")
        
        self.state = ControlState.IDLE
    
    def _execute_control(self):
        """Execute control actions based on current state."""
        # This would implement the actual control logic
        pass
    
    def _update_status(self):
        """Update system status."""
        status = ControlStatus(
            state=self.state,
            current_target=self.current_target,
            robot_position=self.robot_position,
            confidence=self.confidence,
            error_message=self.error_message
        )
        
        # Notify callbacks
        for callback in self.status_callbacks:
            try:
                callback(status)
            except Exception as e:
                print(f"⚠️  Status callback error: {e}")
    
    def send_command(self, command: ControlCommand):
        """Send a command to the controller."""
        self.command_queue.put(command)
    
    def add_status_callback(self, callback: Callable[[ControlStatus], None]):
        """Add a status callback."""
        self.status_callbacks.append(callback)
    
    def get_status(self) -> ControlStatus:
        """Get current system status."""
        return ControlStatus(
            state=self.state,
            current_target=self.current_target,
            robot_position=self.robot_position,
            confidence=self.confidence,
            error_message=self.error_message
        )
