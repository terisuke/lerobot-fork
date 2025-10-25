"""
Depth-based Object Tracking for Robotic Manipulation

This module implements advanced object tracking using depth information
from RealSense cameras for precise robotic manipulation tasks.
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from .detector import DetectedObject
from .tracker import TrackedObject
import pyrealsense2 as rs


@dataclass
class DepthTrackedObject:
    """Represents a depth-tracked object with enhanced 3D information."""
    object_id: int
    current_detection: DetectedObject
    position_history: List[Tuple[float, float, float]]  # (x, y, z) world positions
    velocity: Tuple[float, float, float]  # (vx, vy, vz) in m/s
    age: int  # number of frames tracked
    lost_frames: int  # number of frames since last detection
    depth_confidence: float  # confidence based on depth consistency
    size_history: List[Tuple[int, int]]  # (width, height) history
    depth_range: Tuple[float, float]  # (min_depth, max_depth) for this object


class DepthObjectTracker:
    """
    Advanced object tracker using depth information for robotic manipulation.
    
    This tracker provides enhanced 3D tracking capabilities using depth data
    from RealSense cameras for precise robotic manipulation tasks.
    """
    
    def __init__(self, 
                 max_distance: float = 0.1,  # 10cm
                 max_lost_frames: int = 10,
                 velocity_smoothing: float = 0.7,
                 depth_consistency_threshold: float = 0.05,  # 5cm
                 size_change_threshold: float = 0.2):  # 20% size change
        """
        Initialize the depth-based object tracker.
        
        Args:
            max_distance: Maximum distance for object association (meters)
            max_lost_frames: Maximum frames to keep lost objects
            velocity_smoothing: Smoothing factor for velocity calculation
            depth_consistency_threshold: Maximum depth change for consistency
            size_change_threshold: Maximum size change ratio
        """
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.velocity_smoothing = velocity_smoothing
        self.depth_consistency_threshold = depth_consistency_threshold
        self.size_change_threshold = size_change_threshold
        
        self.tracked_objects: Dict[int, DepthTrackedObject] = {}
        self.next_object_id = 0
        
    def update(self, detections: List[DetectedObject]) -> List[DepthTrackedObject]:
        """
        Update tracker with new detections using depth information.
        
        Args:
            detections: List of detected objects from current frame
            
        Returns:
            List of currently tracked objects
        """
        # Update existing tracks with depth validation
        self._update_existing_tracks_with_depth(detections)
        
        # Create new tracks for unmatched detections
        self._create_new_tracks_with_depth(detections)
        
        # Remove lost tracks
        self._remove_lost_tracks()
        
        # Calculate enhanced velocities and depth confidence
        self._calculate_enhanced_metrics()
        
        return list(self.tracked_objects.values())
    
    def _update_existing_tracks_with_depth(self, detections: List[DetectedObject]):
        """Update existing tracks with depth validation."""
        # Create enhanced distance matrix with depth consistency
        distances = self._calculate_enhanced_distance_matrix(detections)
        
        # Find best matches considering depth consistency
        matched_pairs = self._find_best_matches_with_depth(distances, detections)
        
        # Update matched tracks
        for track_id, detection_idx in matched_pairs:
            if track_id in self.tracked_objects:
                track = self.tracked_objects[track_id]
                detection = detections[detection_idx]
                
                # Validate depth consistency
                if self._validate_depth_consistency(track, detection):
                    track.current_detection = detection
                    track.position_history.append(detection.world_position)
                    track.age += 1
                    track.lost_frames = 0
                    
                    # Update size history
                    x, y, w, h = detection.bbox
                    track.size_history.append((w, h))
                    
                    # Update depth range
                    self._update_depth_range(track, detection.depth)
                    
                    # Limit history length
                    if len(track.position_history) > 50:
                        track.position_history.pop(0)
                    if len(track.size_history) > 50:
                        track.size_history.pop(0)
                else:
                    # Depth inconsistency - mark as lost
                    track.lost_frames += 1
    
    def _create_new_tracks_with_depth(self, detections: List[DetectedObject]):
        """Create new tracks for unmatched detections with depth validation."""
        # Find unmatched detections
        matched_detection_indices = set()
        for track in self.tracked_objects.values():
            if hasattr(track, '_matched_detection_idx'):
                matched_detection_indices.add(track._matched_detection_idx)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                # Validate depth for new track
                if self._validate_depth_for_new_track(detection):
                    new_track = DepthTrackedObject(
                        object_id=self.next_object_id,
                        current_detection=detection,
                        position_history=[detection.world_position],
                        velocity=(0.0, 0.0, 0.0),
                        age=1,
                        lost_frames=0,
                        depth_confidence=1.0,
                        size_history=[(detection.bbox[2], detection.bbox[3])],
                        depth_range=(detection.depth, detection.depth)
                    )
                    self.tracked_objects[self.next_object_id] = new_track
                    self.next_object_id += 1
    
    def _remove_lost_tracks(self):
        """Remove tracks that have been lost for too long."""
        tracks_to_remove = []
        for track_id, track in self.tracked_objects.items():
            track.lost_frames += 1
            if track.lost_frames > self.max_lost_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracked_objects[track_id]
    
    def _calculate_enhanced_metrics(self):
        """Calculate enhanced velocities and depth confidence."""
        for track in self.tracked_objects.values():
            if len(track.position_history) >= 2:
                # Calculate velocity from position history
                recent_positions = track.position_history[-5:]  # Last 5 positions
                if len(recent_positions) >= 2:
                    # Calculate average velocity
                    velocities = []
                    for i in range(1, len(recent_positions)):
                        pos1 = recent_positions[i-1]
                        pos2 = recent_positions[i]
                        vel = (pos2[0] - pos1[0], pos2[1] - pos1[1], pos2[2] - pos1[2])
                        velocities.append(vel)
                    
                    if velocities:
                        # Average velocity
                        avg_vel = np.mean(velocities, axis=0)
                        # Smooth with previous velocity
                        track.velocity = (
                            self.velocity_smoothing * track.velocity[0] + 
                            (1 - self.velocity_smoothing) * avg_vel[0],
                            self.velocity_smoothing * track.velocity[1] + 
                            (1 - self.velocity_smoothing) * avg_vel[1],
                            self.velocity_smoothing * track.velocity[2] + 
                            (1 - self.velocity_smoothing) * avg_vel[2]
                        )
                
                # Calculate depth confidence
                track.depth_confidence = self._calculate_depth_confidence(track)
    
    def _calculate_enhanced_distance_matrix(self, detections: List[DetectedObject]) -> np.ndarray:
        """Calculate enhanced distance matrix with depth consistency."""
        if not self.tracked_objects or not detections:
            return np.array([])
        
        distances = np.zeros((len(self.tracked_objects), len(detections)))
        
        for i, track in enumerate(self.tracked_objects.values()):
            for j, detection in enumerate(detections):
                # Calculate 3D distance
                track_pos = track.current_detection.world_position
                det_pos = detection.world_position
                
                distance = np.sqrt(
                    (track_pos[0] - det_pos[0])**2 +
                    (track_pos[1] - det_pos[1])**2 +
                    (track_pos[2] - det_pos[2])**2
                )
                
                # Add depth consistency penalty
                depth_penalty = abs(track.current_detection.depth - detection.depth)
                if depth_penalty > self.depth_consistency_threshold:
                    distance += depth_penalty * 2  # Penalty for depth inconsistency
                
                distances[i, j] = distance
        
        return distances
    
    def _find_best_matches_with_depth(self, distances: np.ndarray, 
                                    detections: List[DetectedObject]) -> List[Tuple[int, int]]:
        """Find best matches considering depth consistency."""
        if distances.size == 0:
            return []
        
        matched_pairs = []
        used_detections = set()
        
        # Sort by distance
        flat_indices = np.argsort(distances.flatten())
        
        for flat_idx in flat_indices:
            track_idx, det_idx = np.unravel_index(flat_idx, distances.shape)
            
            if (track_idx not in [pair[0] for pair in matched_pairs] and
                det_idx not in used_detections and
                distances[track_idx, det_idx] < self.max_distance):
                
                matched_pairs.append((track_idx, det_idx))
                used_detections.add(det_idx)
        
        return matched_pairs
    
    def _validate_depth_consistency(self, track: DepthTrackedObject, 
                                  detection: DetectedObject) -> bool:
        """Validate depth consistency for track update."""
        depth_diff = abs(track.current_detection.depth - detection.depth)
        return depth_diff <= self.depth_consistency_threshold
    
    def _validate_depth_for_new_track(self, detection: DetectedObject) -> bool:
        """Validate depth for new track creation."""
        # Check if depth is reasonable
        return 0.1 <= detection.depth <= 2.0  # 10cm to 2m
    
    def _update_depth_range(self, track: DepthTrackedObject, new_depth: float):
        """Update depth range for track."""
        min_depth, max_depth = track.depth_range
        track.depth_range = (min(min_depth, new_depth), max(max_depth, new_depth))
    
    def _calculate_depth_confidence(self, track: DepthTrackedObject) -> float:
        """Calculate depth confidence based on consistency."""
        if len(track.position_history) < 2:
            return 1.0
        
        # Calculate depth variance
        depths = [pos[2] for pos in track.position_history[-10:]]  # Last 10 positions
        depth_variance = np.var(depths)
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = max(0.0, 1.0 - depth_variance / (self.depth_consistency_threshold ** 2))
        return confidence
    
    def get_tracked_objects(self) -> List[DepthTrackedObject]:
        """Get all currently tracked objects."""
        return list(self.tracked_objects.values())
    
    def get_object_by_id(self, object_id: int) -> Optional[DepthTrackedObject]:
        """Get tracked object by ID."""
        return self.tracked_objects.get(object_id)
    
    def get_objects_in_depth_range(self, min_depth: float, max_depth: float) -> List[DepthTrackedObject]:
        """Get objects within specific depth range."""
        filtered_objects = []
        for track in self.tracked_objects.values():
            current_depth = track.current_detection.depth
            if min_depth <= current_depth <= max_depth:
                filtered_objects.append(track)
        return filtered_objects
    
    def get_objects_with_high_confidence(self, min_confidence: float = 0.8) -> List[DepthTrackedObject]:
        """Get objects with high depth confidence."""
        filtered_objects = []
        for track in self.tracked_objects.values():
            if track.depth_confidence >= min_confidence:
                filtered_objects.append(track)
        return filtered_objects
    
    def predict_object_position(self, object_id: int, frames_ahead: int = 1) -> Optional[Tuple[float, float, float]]:
        """Predict future position of tracked object."""
        track = self.get_object_by_id(object_id)
        if not track or len(track.position_history) < 2:
            return None
        
        # Enhanced prediction using depth information
        current_pos = track.current_detection.world_position
        velocity = track.velocity
        
        # Apply depth-based velocity scaling
        depth_factor = min(1.0, track.current_detection.depth / 1.0)  # Scale by depth
        scaled_velocity = (
            velocity[0] * depth_factor,
            velocity[1] * depth_factor,
            velocity[2] * depth_factor
        )
        
        predicted_pos = (
            current_pos[0] + scaled_velocity[0] * frames_ahead,
            current_pos[1] + scaled_velocity[1] * frames_ahead,
            current_pos[2] + scaled_velocity[2] * frames_ahead
        )
        
        return predicted_pos
    
    def is_object_moving(self, object_id: int, velocity_threshold: float = 0.01) -> bool:
        """Check if object is moving significantly."""
        track = self.get_object_by_id(object_id)
        if not track:
            return False
        
        velocity_magnitude = np.sqrt(
            track.velocity[0]**2 + track.velocity[1]**2 + track.velocity[2]**2
        )
        
        return velocity_magnitude > velocity_threshold
    
    def get_moving_objects(self, velocity_threshold: float = 0.01) -> List[DepthTrackedObject]:
        """Get all objects that are currently moving."""
        moving_objects = []
        for track in self.tracked_objects.values():
            if self.is_object_moving(track.object_id, velocity_threshold):
                moving_objects.append(track)
        
        return moving_objects
    
    def get_stable_objects(self, min_age: int = 5, min_confidence: float = 0.7) -> List[DepthTrackedObject]:
        """Get objects that are stable and well-tracked."""
        stable_objects = []
        for track in self.tracked_objects.values():
            if (track.age >= min_age and 
                track.depth_confidence >= min_confidence and
                track.lost_frames == 0):
                stable_objects.append(track)
        
        return stable_objects

