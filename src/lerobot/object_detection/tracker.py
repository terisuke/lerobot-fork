"""
Object Tracking for Robotic Manipulation

This module implements object tracking using depth information
to maintain object identity across frames.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .detector import DetectedObject
import cv2


@dataclass
class TrackedObject:
    """Represents a tracked object with its history."""
    object_id: int
    current_detection: DetectedObject
    position_history: List[Tuple[float, float, float]]  # (x, y, z) world positions
    velocity: Tuple[float, float, float]  # (vx, vy, vz) in m/s
    age: int  # number of frames tracked
    lost_frames: int  # number of frames since last detection


class ObjectTracker:
    """
    Object tracker that maintains object identity across frames.
    
    This tracker uses depth information and position history to track
    objects for robotic manipulation tasks.
    """
    
    def __init__(self, 
                 max_distance: float = 0.1,  # 10cm
                 max_lost_frames: int = 10,
                 velocity_smoothing: float = 0.7):
        """
        Initialize the object tracker.
        
        Args:
            max_distance: Maximum distance for object association (meters)
            max_lost_frames: Maximum frames to keep lost objects
            velocity_smoothing: Smoothing factor for velocity calculation
        """
        self.max_distance = max_distance
        self.max_lost_frames = max_lost_frames
        self.velocity_smoothing = velocity_smoothing
        
        self.tracked_objects: Dict[int, TrackedObject] = {}
        self.next_object_id = 0
        
    def update(self, detections: List[DetectedObject]) -> List[TrackedObject]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detected objects from current frame
            
        Returns:
            List of currently tracked objects
        """
        # Update existing tracks
        self._update_existing_tracks(detections)
        
        # Create new tracks for unmatched detections
        self._create_new_tracks(detections)
        
        # Remove lost tracks
        self._remove_lost_tracks()
        
        # Calculate velocities
        self._calculate_velocities()
        
        return list(self.tracked_objects.values())
    
    def _update_existing_tracks(self, detections: List[DetectedObject]):
        """Update existing tracks with new detections."""
        # Create distance matrix
        distances = self._calculate_distance_matrix(detections)
        
        # Find best matches
        matched_pairs = self._find_best_matches(distances)
        
        # Update matched tracks
        for track_id, detection_idx in matched_pairs:
            if track_id in self.tracked_objects:
                track = self.tracked_objects[track_id]
                track.current_detection = detections[detection_idx]
                track.position_history.append(detections[detection_idx].world_position)
                track.age += 1
                track.lost_frames = 0
                
                # Limit history length
                if len(track.position_history) > 50:
                    track.position_history.pop(0)
    
    def _create_new_tracks(self, detections: List[DetectedObject]):
        """Create new tracks for unmatched detections."""
        # Find unmatched detections
        matched_detection_indices = set()
        for track in self.tracked_objects.values():
            if hasattr(track, '_matched_detection_idx'):
                matched_detection_indices.add(track._matched_detection_idx)
        
        # Create new tracks for unmatched detections
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                new_track = TrackedObject(
                    object_id=self.next_object_id,
                    current_detection=detection,
                    position_history=[detection.world_position],
                    velocity=(0.0, 0.0, 0.0),
                    age=1,
                    lost_frames=0
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
    
    def _calculate_velocities(self):
        """Calculate velocities for all tracks."""
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
    
    def _calculate_distance_matrix(self, detections: List[DetectedObject]) -> np.ndarray:
        """Calculate distance matrix between tracks and detections."""
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
                
                distances[i, j] = distance
        
        return distances
    
    def _find_best_matches(self, distances: np.ndarray) -> List[Tuple[int, int]]:
        """Find best matches between tracks and detections."""
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
    
    def get_tracked_objects(self) -> List[TrackedObject]:
        """Get all currently tracked objects."""
        return list(self.tracked_objects.values())
    
    def get_object_by_id(self, object_id: int) -> Optional[TrackedObject]:
        """Get tracked object by ID."""
        return self.tracked_objects.get(object_id)
    
    def predict_object_position(self, object_id: int, frames_ahead: int = 1) -> Optional[Tuple[float, float, float]]:
        """
        Predict future position of tracked object.
        
        Args:
            object_id: ID of the object to predict
            frames_ahead: Number of frames to predict ahead
            
        Returns:
            Predicted position (x, y, z) or None if object not found
        """
        track = self.get_object_by_id(object_id)
        if not track or len(track.position_history) < 2:
            return None
        
        # Simple linear prediction
        current_pos = track.current_detection.world_position
        velocity = track.velocity
        
        predicted_pos = (
            current_pos[0] + velocity[0] * frames_ahead,
            current_pos[1] + velocity[1] * frames_ahead,
            current_pos[2] + velocity[2] * frames_ahead
        )
        
        return predicted_pos
    
    def is_object_moving(self, object_id: int, velocity_threshold: float = 0.01) -> bool:
        """
        Check if object is moving significantly.
        
        Args:
            object_id: ID of the object to check
            velocity_threshold: Minimum velocity to consider moving (m/s)
            
        Returns:
            True if object is moving, False otherwise
        """
        track = self.get_object_by_id(object_id)
        if not track:
            return False
        
        velocity_magnitude = np.sqrt(
            track.velocity[0]**2 + track.velocity[1]**2 + track.velocity[2]**2
        )
        
        return velocity_magnitude > velocity_threshold
    
    def get_moving_objects(self, velocity_threshold: float = 0.01) -> List[TrackedObject]:
        """
        Get all objects that are currently moving.
        
        Args:
            velocity_threshold: Minimum velocity to consider moving (m/s)
            
        Returns:
            List of moving tracked objects
        """
        moving_objects = []
        for track in self.tracked_objects.values():
            if self.is_object_moving(track.object_id, velocity_threshold):
                moving_objects.append(track)
        
        return moving_objects
