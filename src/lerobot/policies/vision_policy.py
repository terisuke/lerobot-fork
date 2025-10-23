"""
Vision-based Policy for Object Picking

This module implements a vision-based policy that learns to pick up
objects using RGB and depth information from RealSense cameras.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import cv2

from ..object_detection import ObjectDetector, DepthObjectTracker
from ..object_detection.detector import DetectedObject
from ..object_detection.depth_tracker import DepthTrackedObject


@dataclass
class VisionPolicyInput:
    """Input data for vision-based policy."""
    rgb_image: np.ndarray
    depth_image: np.ndarray
    detected_objects: List[DetectedObject]
    tracked_objects: List[DepthTrackedObject]
    robot_state: Optional[Dict[str, Any]] = None


@dataclass
class VisionPolicyOutput:
    """Output from vision-based policy."""
    action: np.ndarray  # Robot action (position, gripper)
    confidence: float  # Policy confidence
    target_object_id: Optional[int] = None
    predicted_trajectory: Optional[np.ndarray] = None


class VisionEncoder(nn.Module):
    """Encoder for RGB and depth images."""
    
    def __init__(self, input_channels: int = 4,  # RGB + Depth
                 hidden_dim: int = 256,
                 output_dim: int = 512):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # RGB encoder
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Depth encoder
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(256 * 8 * 8 * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, rgb_image: torch.Tensor, depth_image: torch.Tensor) -> torch.Tensor:
        """Forward pass through vision encoder."""
        # Encode RGB
        rgb_features = self.rgb_encoder(rgb_image)
        rgb_features = rgb_features.view(rgb_features.size(0), -1)
        
        # Encode depth
        depth_features = self.depth_encoder(depth_image)
        depth_features = depth_features.view(depth_features.size(0), -1)
        
        # Fuse features
        combined_features = torch.cat([rgb_features, depth_features], dim=1)
        fused_features = self.fusion(combined_features)
        
        return fused_features


class ObjectEncoder(nn.Module):
    """Encoder for object detection and tracking information."""
    
    def __init__(self, object_dim: int = 10,  # bbox + center + depth + confidence + class
                 hidden_dim: int = 128,
                 output_dim: int = 256):
        super().__init__()
        
        self.object_dim = object_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(object_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        """Encode object information."""
        return self.encoder(objects)


class VisionPolicy(nn.Module):
    """
    Vision-based policy for object picking.
    
    This policy learns to pick up objects using RGB and depth information
    from RealSense cameras and object detection/tracking data.
    """
    
    def __init__(self, 
                 action_dim: int = 7,  # 3D position + 3D orientation + gripper
                 hidden_dim: int = 512,
                 num_objects: int = 10):
        super().__init__()
        
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_objects = num_objects
        
        # Vision encoder
        self.vision_encoder = VisionEncoder()
        
        # Object encoder
        self.object_encoder = ObjectEncoder()
        
        # Robot state encoder
        self.robot_encoder = nn.Sequential(
            nn.Linear(6, 64),  # 6DOF robot state
            nn.ReLU(),
            nn.Linear(64, 128)
        )
        
        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(512 + 256 + 128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(512 + 256 + 128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1)
        )
        
        # Object attention
        self.object_attention = nn.MultiheadAttention(
            embed_dim=256, num_heads=8, batch_first=True
        )
    
    def forward(self, 
                rgb_image: torch.Tensor,
                depth_image: torch.Tensor,
                objects: torch.Tensor,
                robot_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the policy network.
        
        Args:
            rgb_image: RGB image tensor (B, 3, H, W)
            depth_image: Depth image tensor (B, 1, H, W)
            objects: Object features tensor (B, N, 10)
            robot_state: Robot state tensor (B, 6)
            
        Returns:
            Tuple of (action, value)
        """
        batch_size = rgb_image.size(0)
        
        # Encode vision
        vision_features = self.vision_encoder(rgb_image, depth_image)
        
        # Encode objects
        object_features = self.object_encoder(objects.view(-1, objects.size(-1)))
        object_features = object_features.view(batch_size, -1, 256)
        
        # Apply attention to objects
        attended_objects, _ = self.object_attention(
            object_features, object_features, object_features
        )
        attended_objects = attended_objects.mean(dim=1)  # Global average
        
        # Encode robot state
        robot_features = self.robot_encoder(robot_state)
        
        # Combine features
        combined_features = torch.cat([
            vision_features, attended_objects, robot_features
        ], dim=1)
        
        # Get action and value
        action = self.policy_net(combined_features)
        value = self.value_net(combined_features)
        
        return action, value
    
    def predict(self, 
                rgb_image: np.ndarray,
                depth_image: np.ndarray,
                detected_objects: List[DetectedObject],
                tracked_objects: List[DepthTrackedObject],
                robot_state: Optional[Dict[str, Any]] = None) -> VisionPolicyOutput:
        """
        Predict action for given input.
        
        Args:
            rgb_image: RGB image
            depth_image: Depth image
            detected_objects: List of detected objects
            tracked_objects: List of tracked objects
            robot_state: Optional robot state
            
        Returns:
            Policy output
        """
        self.eval()
        
        with torch.no_grad():
            # Preprocess inputs
            rgb_tensor = self._preprocess_rgb(rgb_image)
            depth_tensor = self._preprocess_depth(depth_image)
            objects_tensor = self._preprocess_objects(detected_objects, tracked_objects)
            robot_tensor = self._preprocess_robot_state(robot_state)
            
            # Forward pass
            action, value = self.forward(
                rgb_tensor, depth_tensor, objects_tensor, robot_tensor
            )
            
            # Postprocess outputs
            action_np = action.cpu().numpy()[0]
            confidence = value.cpu().numpy()[0][0]
            
            # Find target object
            target_object_id = self._find_target_object(tracked_objects)
            
            # Predict trajectory
            predicted_trajectory = self._predict_trajectory(
                tracked_objects, target_object_id
            )
            
            return VisionPolicyOutput(
                action=action_np,
                confidence=confidence,
                target_object_id=target_object_id,
                predicted_trajectory=predicted_trajectory
            )
    
    def _preprocess_rgb(self, rgb_image: np.ndarray) -> torch.Tensor:
        """Preprocess RGB image."""
        # Resize to 224x224
        rgb_resized = cv2.resize(rgb_image, (224, 224))
        # Normalize to [0, 1]
        rgb_normalized = rgb_resized.astype(np.float32) / 255.0
        # Convert to tensor and add batch dimension
        rgb_tensor = torch.from_numpy(rgb_normalized).permute(2, 0, 1).unsqueeze(0)
        return rgb_tensor
    
    def _preprocess_depth(self, depth_image: np.ndarray) -> torch.Tensor:
        """Preprocess depth image."""
        # Resize to 224x224
        depth_resized = cv2.resize(depth_image, (224, 224))
        # Normalize to [0, 1]
        depth_normalized = depth_resized.astype(np.float32) / 1000.0  # Convert mm to m
        # Convert to tensor and add batch dimension
        depth_tensor = torch.from_numpy(depth_normalized).unsqueeze(0).unsqueeze(0)
        return depth_tensor
    
    def _preprocess_objects(self, 
                          detected_objects: List[DetectedObject],
                          tracked_objects: List[DepthTrackedObject]) -> torch.Tensor:
        """Preprocess object information."""
        # Create object features
        object_features = []
        
        for obj in tracked_objects:
            # Extract features
            x, y, w, h = obj.current_detection.bbox
            center_x, center_y = obj.current_detection.center
            depth = obj.current_detection.depth
            confidence = obj.current_detection.confidence
            class_id = obj.current_detection.class_id
            
            # Normalize features
            features = np.array([
                x / 640.0,  # Normalized x
                y / 480.0,  # Normalized y
                w / 640.0,  # Normalized width
                h / 480.0,  # Normalized height
                center_x / 640.0,  # Normalized center x
                center_y / 480.0,  # Normalized center y
                depth / 2.0,  # Normalized depth
                confidence,  # Confidence
                class_id / 80.0,  # Normalized class ID
                obj.depth_confidence  # Depth confidence
            ], dtype=np.float32)
            
            object_features.append(features)
        
        # Pad or truncate to fixed size
        while len(object_features) < self.num_objects:
            object_features.append(np.zeros(10, dtype=np.float32))
        
        object_features = object_features[:self.num_objects]
        
        # Convert to tensor
        objects_tensor = torch.from_numpy(np.array(object_features)).unsqueeze(0)
        return objects_tensor
    
    def _preprocess_robot_state(self, robot_state: Optional[Dict[str, Any]]) -> torch.Tensor:
        """Preprocess robot state."""
        if robot_state is None:
            # Default robot state
            state = np.array([0.0, 0.0, 0.2, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            # Extract robot state
            state = np.array([
                robot_state.get('x', 0.0),
                robot_state.get('y', 0.0),
                robot_state.get('z', 0.2),
                robot_state.get('rx', 0.0),
                robot_state.get('ry', 0.0),
                robot_state.get('rz', 0.0)
            ], dtype=np.float32)
        
        return torch.from_numpy(state).unsqueeze(0)
    
    def _find_target_object(self, tracked_objects: List[DepthTrackedObject]) -> Optional[int]:
        """Find the best target object for picking."""
        if not tracked_objects:
            return None
        
        # Score objects based on various criteria
        best_object = None
        best_score = -1
        
        for obj in tracked_objects:
            score = 0
            
            # Depth score (prefer objects in good range)
            depth = obj.current_detection.depth
            if 0.3 <= depth <= 1.0:
                score += 1.0
            elif 0.2 <= depth <= 1.5:
                score += 0.5
            
            # Confidence score
            score += obj.current_detection.confidence
            
            # Depth confidence score
            score += obj.depth_confidence
            
            # Size score (prefer medium-sized objects)
            x, y, w, h = obj.current_detection.bbox
            size = w * h
            if 1000 <= size <= 10000:  # Reasonable size range
                score += 0.5
            
            if score > best_score:
                best_score = score
                best_object = obj
        
        return best_object.object_id if best_object else None
    
    def _predict_trajectory(self, 
                          tracked_objects: List[DepthTrackedObject],
                          target_object_id: Optional[int]) -> Optional[np.ndarray]:
        """Predict trajectory for target object."""
        if target_object_id is None:
            return None
        
        # Find target object
        target_object = None
        for obj in tracked_objects:
            if obj.object_id == target_object_id:
                target_object = obj
                break
        
        if target_object is None:
            return None
        
        # Predict future positions
        trajectory = []
        current_pos = target_object.current_detection.world_position
        
        for i in range(10):  # Predict 10 steps ahead
            # Simple linear prediction
            predicted_pos = (
                current_pos[0] + target_object.velocity[0] * i * 0.033,  # 30 FPS
                current_pos[1] + target_object.velocity[1] * i * 0.033,
                current_pos[2] + target_object.velocity[2] * i * 0.033
            )
            trajectory.append(predicted_pos)
        
        return np.array(trajectory)
    
    def save_model(self, path: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'num_objects': self.num_objects
        }, path)
    
    @classmethod
    def load_model(cls, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location='cpu')
        
        model = cls(
            action_dim=checkpoint['action_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            num_objects=checkpoint['num_objects']
        )
        
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
