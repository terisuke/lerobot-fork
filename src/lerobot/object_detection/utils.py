"""
Utility functions for object detection and tracking.

This module provides utility functions for processing depth and color data
from RealSense cameras for robotic manipulation tasks.
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


class DepthProcessor:
    """Utility class for processing depth data from RealSense cameras."""

    @staticmethod
    def get_depth_at_point(depth_image: np.ndarray, x: int, y: int) -> float:
        """
        Get depth value at specific point in depth image.

        Args:
            depth_image: Depth image from RealSense camera
            x, y: Pixel coordinates

        Returns:
            Depth value in meters, or infinity if out of bounds
        """
        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
            depth_value = depth_image[y, x]
            return depth_value / 1000.0  # Convert mm to meters
        return float("inf")

    @staticmethod
    def get_depth_in_region(
        depth_image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> float:
        """
        Get average depth value in a rectangular region.

        Args:
            depth_image: Depth image from RealSense camera
            x1, y1, x2, y2: Bounding box coordinates

        Returns:
            Average depth value in meters
        """
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, depth_image.shape[1]))
        y1 = max(0, min(y1, depth_image.shape[0]))
        x2 = max(0, min(x2, depth_image.shape[1]))
        y2 = max(0, min(y2, depth_image.shape[0]))

        if x1 >= x2 or y1 >= y2:
            return float("inf")

        # Extract region and calculate average depth
        region = depth_image[y1:y2, x1:x2]
        valid_depths = region[region > 0]  # Filter out invalid depths

        if len(valid_depths) == 0:
            return float("inf")

        return np.mean(valid_depths) / 1000.0  # Convert mm to meters

    @staticmethod
    def create_depth_mask(
        depth_image: np.ndarray, min_depth: float = 0.1, max_depth: float = 2.0
    ) -> np.ndarray:
        """
        Create a mask for valid depth values within specified range.

        Args:
            depth_image: Depth image from RealSense camera
            min_depth: Minimum depth in meters
            max_depth: Maximum depth in meters

        Returns:
            Binary mask where True indicates valid depth
        """
        depth_meters = depth_image / 1000.0  # Convert mm to meters
        mask = (depth_meters >= min_depth) & (depth_meters <= max_depth)
        return mask.astype(np.uint8) * 255

    @staticmethod
    def depth_to_world_coords(
        x: float, y: float, depth: float, intrinsics: rs.intrinsics
    ) -> Tuple[float, float, float]:
        """
        Convert 2D image coordinates to 3D world coordinates.

        Args:
            x, y: Image coordinates
            depth: Depth value in meters
            intrinsics: Camera intrinsics

        Returns:
            World coordinates (x, y, z) in meters
        """
        world_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        return (world_point[0], world_point[1], world_point[2])

    @staticmethod
    def world_to_depth_coords(
        world_x: float, world_y: float, world_z: float, intrinsics: rs.intrinsics
    ) -> Tuple[float, float, float]:
        """
        Convert 3D world coordinates to 2D image coordinates and depth.

        Args:
            world_x, world_y, world_z: World coordinates in meters
            intrinsics: Camera intrinsics

        Returns:
            Image coordinates (x, y) and depth in meters
        """
        pixel = rs.rs2_project_point_to_pixel(intrinsics, [world_x, world_y, world_z])
        return (pixel[0], pixel[1], world_z)


class ColorProcessor:
    """Utility class for processing color data from RealSense cameras."""

    @staticmethod
    def apply_color_filter(
        image: np.ndarray,
        lower_bound: Tuple[int, int, int],
        upper_bound: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Apply color filter to image.

        Args:
            image: Input image
            lower_bound: Lower bound for color filter (B, G, R)
            upper_bound: Upper bound for color filter (B, G, R)

        Returns:
            Filtered binary mask
        """
        mask = cv2.inRange(image, lower_bound, upper_bound)
        return mask

    @staticmethod
    def find_color_objects(
        image: np.ndarray,
        color_range: Tuple[Tuple[int, int, int], Tuple[int, int, int]],
        min_area: int = 100,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Find objects of specific color in image.

        Args:
            image: Input image
            color_range: Color range as ((lower_B, lower_G, lower_R), (upper_B, upper_G, upper_R))
            min_area: Minimum area for object detection

        Returns:
            List of bounding boxes (x, y, width, height)
        """
        lower_bound, upper_bound = color_range
        mask = ColorProcessor.apply_color_filter(image, lower_bound, upper_bound)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(contour)
                objects.append((x, y, w, h))

        return objects

    @staticmethod
    def enhance_image(
        image: np.ndarray, brightness: float = 1.0, contrast: float = 1.0
    ) -> np.ndarray:
        """
        Enhance image brightness and contrast.

        Args:
            image: Input image
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)

        Returns:
            Enhanced image
        """
        enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return enhanced

    @staticmethod
    def detect_edges(
        image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150
    ) -> np.ndarray:
        """
        Detect edges in image using Canny edge detection.

        Args:
            image: Input image
            low_threshold: Lower threshold for edge detection
            high_threshold: Upper threshold for edge detection

        Returns:
            Edge image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_threshold, high_threshold)
        return edges


class ObjectFilter:
    """Utility class for filtering detected objects."""

    @staticmethod
    def filter_by_depth(
        objects: List, min_depth: float = 0.1, max_depth: float = 2.0
    ) -> List:
        """
        Filter objects by depth range.

        Args:
            objects: List of detected objects with depth attribute
            min_depth: Minimum depth in meters
            max_depth: Maximum depth in meters

        Returns:
            Filtered list of objects
        """
        filtered = []
        for obj in objects:
            if hasattr(obj, "depth") and min_depth <= obj.depth <= max_depth:
                filtered.append(obj)
        return filtered

    @staticmethod
    def filter_by_size(
        objects: List, min_width: int = 20, min_height: int = 20
    ) -> List:
        """
        Filter objects by size.

        Args:
            objects: List of detected objects with bbox attribute
            min_width: Minimum width in pixels
            min_height: Minimum height in pixels

        Returns:
            Filtered list of objects
        """
        filtered = []
        for obj in objects:
            if hasattr(obj, "bbox"):
                x, y, w, h = obj.bbox
                if w >= min_width and h >= min_height:
                    filtered.append(obj)
        return filtered

    @staticmethod
    def filter_by_confidence(objects: List, min_confidence: float = 0.5) -> List:
        """
        Filter objects by confidence.

        Args:
            objects: List of detected objects with confidence attribute
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of objects
        """
        filtered = []
        for obj in objects:
            if hasattr(obj, "confidence") and obj.confidence >= min_confidence:
                filtered.append(obj)
        return filtered

    @staticmethod
    def filter_by_class(objects: List, target_classes: List[str]) -> List:
        """
        Filter objects by class name.

        Args:
            objects: List of detected objects with class_name attribute
            target_classes: List of target class names

        Returns:
            Filtered list of objects
        """
        filtered = []
        for obj in objects:
            if hasattr(obj, "class_name") and obj.class_name in target_classes:
                filtered.append(obj)
        return filtered


class VisualizationUtils:
    """Utility class for visualizing detection and tracking results."""

    @staticmethod
    def draw_detection(
        image: np.ndarray,
        detection,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw detection bounding box on image.

        Args:
            image: Input image
            detection: Detection object with bbox attribute
            color: BGR color for bounding box
            thickness: Line thickness

        Returns:
            Image with drawn bounding box
        """
        if not hasattr(detection, "bbox"):
            return image

        x, y, w, h = detection.bbox
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)

        # Add label if available
        if hasattr(detection, "class_name") and hasattr(detection, "confidence"):
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            cv2.putText(
                image,
                label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

        return image

    @staticmethod
    def draw_tracking_info(
        image: np.ndarray,
        tracked_object,
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draw tracking information on image.

        Args:
            image: Input image
            tracked_object: TrackedObject with current_detection attribute
            color: BGR color for drawing
            thickness: Line thickness

        Returns:
            Image with drawn tracking info
        """
        if not hasattr(tracked_object, "current_detection"):
            return image

        detection = tracked_object.current_detection
        image = VisualizationUtils.draw_detection(image, detection, color, thickness)

        # Add tracking ID
        if hasattr(tracked_object, "object_id"):
            x, y, w, h = detection.bbox
            cv2.putText(
                image,
                f"ID: {tracked_object.object_id}",
                (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

        # Add velocity info
        if hasattr(tracked_object, "velocity"):
            vx, vy, vz = tracked_object.velocity
            speed = np.sqrt(vx**2 + vy**2 + vz**2)
            cv2.putText(
                image,
                f"Speed: {speed:.3f} m/s",
                (x, y + h + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                thickness,
            )

        return image
