#!/usr/bin/env python3
"""
Cup Manipulation Demo - Single Webcam

This demo shows how to detect a cup, estimate its position, and plan
manipulation actions using a single webcam with depth estimation.

Current Status: Demo/Prototype
Next Step: Add stereo vision for accurate depth
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

class CupManipulationDemo:
    """Demo for cup detection and manipulation planning."""
    
    def __init__(self):
        """Initialize the demo system."""
        print("=" * 60)
        print("Cup Manipulation Demo")
        print("=" * 60)
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        print("✅ Camera initialized")
        
        # YOLO model
        if YOLO_AVAILABLE:
            self.model = YOLO("yolov8n.pt")
            print("✅ YOLOv8 model loaded")
        else:
            self.model = None
            print("⚠️  YOLO not available")
        
        # Object classes we care about
        self.target_classes = {
            'cup': 41,
            'bottle': 39,
            'vase': 75,
            'bowl': 46
        }
        
        # Workspace setup (estimated from camera view)
        self.workspace_size = [0.4, 0.3, 0.05]  # meters: width, depth, height
        
        print("\n📐 Workspace assumed:")
        print(f"   Width: {self.workspace_size[0]}m")
        print(f"   Depth: {self.workspace_size[1]}m")
        print(f"   Height: {self.workspace_size[2]}m")
    
    def estimate_3d_position(self, bbox, frame_shape):
        """
        Estimate 3D position from bounding box.
        
        This is a SIMPLIFIED estimation:
        - Assumes objects are on workspace surface
        - Uses bounding box center
        - Estimates depth from apparent size
        
        TODO: Replace with actual stereo depth estimation
        """
        height, width = frame_shape[:2]
        
        # 2D pixel coordinates
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Estimate depth from apparent size
        object_width = x2 - x1
        object_height = y2 - y1
        apparent_size = max(object_width, object_height)
        
        # Rough depth estimation (requires calibration)
        # TODO: Replace with stereo depth
        estimated_depth = 0.3 + (1000 / apparent_size) * 0.1  # meters
        
        # Convert to 3D workspace coordinates
        # Assume camera is centered above workspace
        workspace_width = self.workspace_size[0]
        workspace_depth = self.workspace_size[1]
        
        # Normalize coordinates
        norm_x = (center_x / width - 0.5) * 2  # -1 to 1
        norm_y = (center_y / height - 0.5) * 2  # -1 to 1
        
        # Convert to meters
        x = norm_x * workspace_width / 2
        y = -norm_y * workspace_depth / 2  # Flip Y axis
        z = 0.05  # Assume on workspace surface
        
        return (x, y, z, estimated_depth)
    
    def detect_objects(self, frame):
        """Detect objects in frame using YOLO."""
        if self.model is None:
            return []
        
        results = self.model.predict(frame, conf=0.5, verbose=False)
        
        detections = []
        for box in results[0].boxes:
            class_id = int(box.cls[0].cpu().numpy())
            confidence = float(box.conf[0].cpu().numpy())
            bbox = box.xyxy[0].cpu().numpy()
            
            # Check if it's a target object
            class_name = results[0].names[class_id]
            if class_name in ['cup', 'bottle', 'vase', 'bowl', 'mug']:
                # Get 3D position estimate
                x, y, z, depth = self.estimate_3d_position(bbox, frame.shape)
                
                detections.append({
                    'class': class_name,
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': bbox,
                    'position_3d': (x, y, z),
                    'estimated_depth': depth
                })
        
        return detections
    
    def visualize_detections(self, frame, detections):
        """Draw detections and information on frame."""
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox'].astype(int)
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Label
            label = f"{det['class']} {det['confidence']:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            
            # 3D position info
            x, y, z = det['position_3d']
            depth = det['estimated_depth']
            info = f"X:{x:.3f} Y:{y:.3f} Z:{z:.3f} D:{depth:.3f}m"
            cv2.putText(annotated, info, (x1, y2 + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        return annotated
    
    def plan_grasp_action(self, detection):
        """Plan a grasp action for detected cup."""
        if detection['class'] not in ['cup', 'mug', 'bowl', 'vase']:
            return None
        
        x, y, z = detection['position_3d']
        
        # Grasp planning (simplified)
        grasp_action = {
            'type': 'pick',
            'target_position': (x, y, z + 0.05),  # Above object
            'grasp_position': (x, y, z),
            'object_class': detection['class'],
            'confidence': detection['confidence']
        }
        
        return grasp_action
    
    def plan_place_action(self, grasp_action, target_position):
        """Plan a place action after grasping."""
        place_action = {
            'type': 'place',
            'target_position': target_position,
            'from_position': grasp_action['grasp_position']
        }
        
        return place_action
    
    def run_demo(self):
        """Run the manipulation demo."""
        print("\n🎯 Demo: Cup Manipulation")
        print("   Goal: Detect cup → Pick → Place near bottle\n")
        print("Controls:")
        print("  'q' - Quit")
        print("  'p' - Plan grasp action")
        print("  'r' - Reset")
        print("\nStarting capture...\n")
        
        frame_count = 0
        last_detection = None
        grasp_planned = False
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = self.detect_objects(frame)
            
            # Find cups
            cups = [d for d in detections if d['class'] in ['cup', 'mug', 'vase', 'bowl']]
            bottles = [d for d in detections if d['class'] == 'bottle']
            
            # Visualize
            annotated = self.visualize_detections(frame, detections)
            
            # Status text
            status_y = 30
            cv2.putText(annotated, f"Cups: {len(cups)}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            status_y += 30
            cv2.putText(annotated, f"Bottles: {len(bottles)}", (10, status_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Plan manipulation
            if len(cups) > 0:
                cup = cups[0]  # Pick first cup
                last_detection = cup
                
                # Show grasp planning info
                if grasp_planned:
                    cv2.putText(annotated, "Grasp planned!", (10, 100),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    grasp_action = self.plan_grasp_action(cup)
                    if grasp_action:
                        x, y, z = grasp_action['target_position']
                        cv2.putText(annotated, f"Grasp at: ({x:.2f}, {y:.2f}, {z:.2f})", 
                                   (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            
            # Display
            cv2.imshow('Cup Manipulation Demo', annotated)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p') and last_detection:
                # Plan and show grasp action
                grasp_action = self.plan_grasp_action(last_detection)
                if grasp_action:
                    print(f"\n🎯 Planned Grasp Action:")
                    print(f"   Object: {grasp_action['object_class']}")
                    print(f"   Grasp at: {grasp_action['grasp_position']}")
                    print(f"   Approach from: {grasp_action['target_position']}")
                    grasp_planned = True
            elif key == ord('r'):
                grasp_planned = False
                print("🔄 Reset")
            
            frame_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()
        
        print("\n✅ Demo completed")
        print("\n📝 Next Steps:")
        print("1. Add stereo vision for accurate depth")
        print("2. Implement actual robot control")
        print("3. Add object tracking for robust detection")
        print("4. Calibrate workspace coordinates")

if __name__ == "__main__":
    try:
        demo = CupManipulationDemo()
        demo.run_demo()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
