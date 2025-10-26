"""
Object Detection Module for LeRobot

This module provides object detection capabilities using RealSense cameras
and depth information for robotic manipulation tasks.
"""

from .depth_tracker import DepthObjectTracker, DepthTrackedObject
from .detector import ObjectDetector
from .tracker import ObjectTracker
from .utils import ColorProcessor, DepthProcessor

__all__ = [
    "ObjectDetector",
    "ObjectTracker",
    "DepthObjectTracker",
    "DepthTrackedObject",
    "DepthProcessor",
    "ColorProcessor",
]
