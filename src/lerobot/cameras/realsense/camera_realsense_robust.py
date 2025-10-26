#!/usr/bin/env python3
"""
Robust RealSense Camera Implementation
Handles power state errors and hardware absence gracefully
"""

import os
import logging
from typing import Optional, Tuple, Any
import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    rs = None

logger = logging.getLogger(__name__)

class RealSenseCameraRobust:
    """
    Robust RealSense camera implementation that handles power state errors
    and hardware absence gracefully.
    """
    
    def __init__(self, serial_number: str, width: int = 640, height: int = 480, fps: int = 30):
        self.serial_number = serial_number
        self.width = width
        self.height = height
        self.fps = fps
        self.pipeline = None
        self.is_connected = False
        self.power_error_occurred = False
        
    def connect(self) -> bool:
        """
        Connect to RealSense camera with robust error handling.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        if not REALSENSE_AVAILABLE:
            logger.warning("pyrealsense2 not available. RealSense camera disabled.")
            return False
            
        # Skip hardware tests in CI environment
        if os.getenv("CI"):
            logger.info("Skipping RealSense camera connection in CI environment")
            return False
            
        try:
            # Initialize pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(self.serial_number)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            
            # Start pipeline
            self.pipeline.start(config)
            self.is_connected = True
            logger.info(f"RealSense camera {self.serial_number} connected successfully")
            return True
            
        except RuntimeError as e:
            error_msg = str(e)
            if "failed to set power state" in error_msg:
                self.power_error_occurred = True
                logger.error("❌ RealSense USB power error: check power supply or use a powered hub.")
                logger.error("💡 Solution: Use a powered USB hub to connect the RealSense camera.")
                return False
            else:
                logger.error(f"RealSense camera connection failed: {e}")
                return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to RealSense camera: {e}")
            return False
    
    def read_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read frames from RealSense camera with error handling.
        
        Returns:
            Tuple[Optional[np.ndarray], Optional[np.ndarray]]: (color_frame, depth_frame)
        """
        if not self.is_connected or self.pipeline is None:
            return None, None
            
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            
            color_array = np.asarray(color_frame.get_data()) if color_frame else None
            depth_array = np.asarray(depth_frame.get_data()) if depth_frame else None
            
            return color_array, depth_array
            
        except RuntimeError as e:
            if "failed to set power state" in str(e):
                self.power_error_occurred = True
                logger.error("❌ RealSense USB power error during frame capture")
                return None, None
            else:
                logger.error(f"RealSense frame capture failed: {e}")
                return None, None
        except Exception as e:
            logger.error(f"Unexpected error reading frames: {e}")
            return None, None
    
    def disconnect(self):
        """Disconnect from RealSense camera."""
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
            self.pipeline = None
        self.is_connected = False
    
    def get_power_error_status(self) -> bool:
        """Check if power error has occurred."""
        return self.power_error_occurred
    
    def get_connection_status(self) -> bool:
        """Check if camera is connected."""
        return self.is_connected

def test_realsense_robust():
    """Test function for robust RealSense camera."""
    print("=== Robust RealSense Camera Test ===")
    
    # Check if running in CI
    if os.getenv("CI"):
        print("⚠️  Running in CI environment - skipping hardware tests")
        return True
    
    # Check if RealSense is available
    if not REALSENSE_AVAILABLE:
        print("⚠️  pyrealsense2 not available - skipping hardware tests")
        return True
    
    # Test camera connection
    camera = RealSenseCameraRobust("test_serial")
    if camera.connect():
        print("✅ RealSense camera connected successfully")
        color_frame, depth_frame = camera.read_frames()
        if color_frame is not None:
            print(f"✅ Color frame captured: {color_frame.shape}")
        if depth_frame is not None:
            print(f"✅ Depth frame captured: {depth_frame.shape}")
        camera.disconnect()
        return True
    else:
        if camera.get_power_error_status():
            print("❌ Power error detected - use powered USB hub")
        else:
            print("❌ Camera connection failed")
        return False

if __name__ == "__main__":
    test_realsense_robust()
