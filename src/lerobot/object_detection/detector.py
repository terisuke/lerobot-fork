"""
Object Detection using RealSense Camera

This module implements object detection using RGB and depth information
from Intel RealSense cameras for robotic manipulation tasks.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pyrealsense2 as rs


@dataclass
class DetectedObject:
    """Represents a detected object with its properties."""

    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    center: Tuple[float, float]  # (x, y) in image coordinates
    depth: float  # depth in meters
    confidence: float  # detection confidence
    class_id: int  # object class ID
    class_name: str  # object class name
    world_position: Tuple[float, float, float]  # (x, y, z) in world coordinates


class ObjectDetector:
    """
    Object detector using RealSense camera with depth information.

    This detector combines RGB and depth data to detect and localize objects
    in 3D space for robotic manipulation tasks.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        depth_threshold: float = 2.0,
        min_object_size: int = 20,
    ):
        """
        Initialize the object detector.

        Args:
            confidence_threshold: Minimum confidence for detections
            depth_threshold: Maximum depth for object detection (meters)
            min_object_size: Minimum object size in pixels
        """
        self.confidence_threshold = confidence_threshold
        self.depth_threshold = depth_threshold
        self.min_object_size = min_object_size

        # Initialize YOLO model for object detection
        self._init_yolo_model()

        # RealSense pipeline - disabled for macOS compatibility
        self.pipeline = None
        self.config = None
        self.camera_available = False
        print("📷 RealSense camera disabled for macOS compatibility")

    def _init_yolo_model(self):
        """Initialize YOLO model for object detection."""
        try:
            # Try to load YOLOv8 model
            from ultralytics import YOLO

            self.model = YOLO("yolov8n.pt")  # nano version for speed
            print("✅ YOLOv8 model loaded successfully")
        except ImportError:
            print("⚠️  ultralytics not found, using OpenCV DNN")
            self._init_opencv_dnn()

    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN for object detection."""
        # Load YOLO weights and config
        net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
        self.model = net
        print("✅ OpenCV DNN model loaded")

    def _setup_realsense(self):
        """Setup RealSense camera configuration."""
        try:
            # Initialize RealSense pipeline and config
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            # Configure depth and color streams
            self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            self.pipeline.start(self.config)
            print("✅ RealSense camera started")
            self.camera_available = True
        except Exception as e:
            print(f"⚠️  RealSense camera setup failed: {e}")
            print("📷 Falling back to mock camera mode")
            self.camera_available = False
            self.pipeline = None
            self.config = None

    def detect_objects(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        intrinsics: rs.intrinsics,
    ) -> List[DetectedObject]:
        """
        Detect objects in the given image with depth information.

        Args:
            color_image: RGB image from camera
            depth_image: Depth image from camera
            intrinsics: Camera intrinsics for 3D projection

        Returns:
            List of detected objects
        """
        detected_objects = []

        # Run object detection
        if hasattr(self.model, "predict"):
            # YOLOv8 model
            results = self.model.predict(color_image, conf=self.confidence_threshold)
            detections = results[0].boxes

            for i in range(len(detections)):
                # Get bounding box
                bbox = detections.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = bbox.astype(int)

                # Get confidence and class
                confidence = detections.conf[i].cpu().numpy()
                class_id = int(detections.cls[i].cpu().numpy())

                # Filter by confidence and size
                if confidence < self.confidence_threshold:
                    continue

                width = x2 - x1
                height = y2 - y1
                if width < self.min_object_size or height < self.min_object_size:
                    continue

                # Calculate center
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # Get depth at center
                depth_value = self._get_depth_at_point(
                    depth_image, int(center_x), int(center_y)
                )

                if depth_value > self.depth_threshold:
                    continue

                # Convert to world coordinates
                world_pos = self._depth_to_world_coords(
                    center_x, center_y, depth_value, intrinsics
                )

                # Create detected object
                obj = DetectedObject(
                    bbox=(x1, y1, width, height),
                    center=(center_x, center_y),
                    depth=depth_value,
                    confidence=confidence,
                    class_id=class_id,
                    class_name=self._get_class_name(class_id),
                    world_position=world_pos,
                )

                detected_objects.append(obj)

        return detected_objects

    def _get_depth_at_point(self, depth_image: np.ndarray, x: int, y: int) -> float:
        """Get depth value at specific point."""
        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
            depth_value = depth_image[y, x]
            return depth_value / 1000.0  # Convert mm to meters
        return float("inf")

    def _depth_to_world_coords(
        self, x: float, y: float, depth: float, intrinsics: rs.intrinsics
    ) -> Tuple[float, float, float]:
        """Convert 2D image coordinates to 3D world coordinates."""
        # Use RealSense API to convert to world coordinates
        world_point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
        return (world_point[0], world_point[1], world_point[2])

    def _get_class_name(self, class_id: int) -> str:
        """Get class name from class ID."""
        class_names = [
            "person",
            "bicycle",
            "car",
            "motorcycle",
            "airplane",
            "bus",
            "train",
            "truck",
            "boat",
            "traffic light",
            "fire hydrant",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        return (
            class_names[class_id]
            if class_id < len(class_names)
            else f"class_{class_id}"
        )

    def get_camera_frames(self) -> Tuple[np.ndarray, np.ndarray, rs.intrinsics]:
        """
        Get current frames from RealSense camera.

        Returns:
            Tuple of (color_image, depth_image, intrinsics)
        """
        if not self.camera_available:
            # Return mock frames when camera is not available
            color_image = np.zeros((480, 640, 3), dtype=np.uint8)
            depth_image = np.zeros((480, 640), dtype=np.uint16)
            intrinsics = None
            return color_image, depth_image, intrinsics

        try:
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Convert to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Get camera intrinsics
            intrinsics = (
                color_frame.get_profile().as_video_stream_profile().get_intrinsics()
            )

            return color_image, depth_image, intrinsics
        except Exception as e:
            print(f"⚠️  Failed to get camera frames: {e}")
            # Return mock frames on error
            color_image = np.zeros((480, 640, 3), dtype=np.uint8)
            depth_image = np.zeros((480, 640), dtype=np.uint16)
            intrinsics = None
            return color_image, depth_image, intrinsics

    def detect_objects_realtime(self) -> List[DetectedObject]:
        """
        Detect objects in real-time from camera feed.

        Returns:
            List of detected objects
        """
        color_image, depth_image, intrinsics = self.get_camera_frames()
        return self.detect_objects(color_image, depth_image, intrinsics)

    def close(self):
        """Close the camera pipeline."""
        if hasattr(self, "camera_available") and self.camera_available:
            try:
                self.pipeline.stop()
                print("✅ RealSense camera stopped")
            except Exception as e:
                print(f"⚠️  Error stopping camera: {e}")
        else:
            print("📷 Camera was not running")
