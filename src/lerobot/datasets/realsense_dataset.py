"""
RealSense Dataset for Depth-Enhanced Robotic Learning

This module provides dataset recording capabilities using RealSense cameras
with depth information for robotic manipulation tasks.
"""

import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from ..cameras.realsense import RealSenseCamera, RealSenseCameraConfig
    from ..cameras.configs import ColorMode
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    RealSenseCamera = None
    RealSenseCameraConfig = None
    ColorMode = None

from ..object_detection import DepthObjectTracker, ObjectDetector
from ..object_detection.depth_tracker import DepthTrackedObject
from ..object_detection.detector import DetectedObject


@dataclass
class RealSenseFrame:
    """Represents a single frame with RGB, depth, and object information."""

    timestamp: float
    rgb_image: np.ndarray
    depth_image: np.ndarray
    intrinsics: Dict[str, float]
    detected_objects: List[Dict[str, Any]]
    tracked_objects: List[Dict[str, Any]]
    robot_state: Optional[Dict[str, Any]] = None


@dataclass
class RealSenseEpisode:
    """Represents a complete episode with multiple frames."""

    episode_id: str
    start_time: float
    end_time: float
    frames: List[RealSenseFrame]
    metadata: Dict[str, Any]


class RealSenseDatasetRecorder:
    """
    Dataset recorder for RealSense camera with depth information.

    This recorder captures RGB, depth, and object detection data
    for robotic manipulation learning tasks.
    """

    def __init__(
        self,
        output_dir: str,
        camera_resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        depth_range: Tuple[float, float] = (0.1, 2.0),
    ):
        """
        Initialize the RealSense dataset recorder.

        Args:
            output_dir: Directory to save dataset
            camera_resolution: Camera resolution (width, height)
            fps: Frames per second
            depth_range: Valid depth range in meters
        """
        self.output_dir = Path(output_dir)
        self.camera_resolution = camera_resolution
        self.fps = fps
        self.depth_range = depth_range

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize RealSense camera using LeRobot's official class
        self.camera_available = False
        self.camera = None
        self._setup_camera()  # Try to initialize camera
        if not self.camera_available:
            print("📷 RealSense camera setup failed in dataset recorder, using mock mode")

        # Initialize object detection and tracking
        self.detector = ObjectDetector()
        self.tracker = DepthObjectTracker()

        # Recording state
        self.is_recording = False
        self.current_episode = None
        self.episode_frames = []

    def _setup_camera(self):
        """Setup RealSense camera configuration."""
        if not REALSENSE_AVAILABLE:
            print("⚠️  RealSense not available")
            return
            
        try:
            # Find available RealSense cameras
            try:
                cameras = RealSenseCamera.find_cameras()
            except RuntimeError as e:
                if "failed to set power state" in str(e):
                    print(f"⚠️  RealSense camera power error: {e}")
                    print("💡 Try using a powered USB hub or Type-C to Type-C cable")
                    cameras = []
                else:
                    raise
                    
            if cameras and len(cameras) > 0:
                serial_number = cameras[0]['id']
                print(f"📷 Found RealSense camera for dataset: {serial_number}")
                
                # Create camera config - use depth only to reduce power
                config = RealSenseCameraConfig(
                    serial_number_or_name=serial_number,
                    width=self.camera_resolution[0],
                    height=self.camera_resolution[1],
                    fps=min(self.fps, 15),  # Cap at 15fps
                    use_depth=True,
                    color_mode=ColorMode.RGB,
                    warmup_s=1.0
                )
                
                # Initialize camera
                self.camera = RealSenseCamera(config)
                
                # Try to connect
                try:
                    self.camera.connect(warmup=False)
                    self.camera_available = True
                    print("✅ RealSense camera started for dataset recording (depth-only)")
                except Exception as e:
                    print(f"⚠️  RealSense camera connection failed: {e}")
                    self.camera_available = False
                    if self.camera:
                        try:
                            self.camera.disconnect()
                        except:
                            pass
                        self.camera = None
            else:
                print("⚠️  No RealSense cameras found for dataset recording")
        except Exception as e:
            print(f"⚠️  RealSense camera setup failed: {e}")
            print("📷 Falling back to mock camera mode for dataset recording")
            self.camera_available = False

    def start_episode(
        self,
        episode_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start recording a new episode.

        Args:
            episode_id: Optional episode ID, will generate if not provided
            metadata: Optional metadata for the episode

        Returns:
            Episode ID
        """
        if self.is_recording:
            raise RuntimeError("Already recording an episode")

        if episode_id is None:
            episode_id = f"episode_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.current_episode = episode_id
        self.episode_frames = []
        self.is_recording = True

        # Store episode metadata
        self.episode_metadata = metadata or {}
        self.episode_metadata["episode_id"] = episode_id
        self.episode_metadata["start_time"] = time.time()

        print(f"🎬 Started recording episode: {episode_id}")
        return episode_id

    def stop_episode(self) -> str:
        """
        Stop recording the current episode and save it.

        Returns:
            Path to saved episode file
        """
        if not self.is_recording:
            raise RuntimeError("No episode is currently being recorded")

        # Finalize episode metadata
        self.episode_metadata["end_time"] = time.time()
        self.episode_metadata["duration"] = (
            self.episode_metadata["end_time"] - self.episode_metadata["start_time"]
        )
        self.episode_metadata["frame_count"] = len(self.episode_frames)

        # Create episode object
        episode = RealSenseEpisode(
            episode_id=self.current_episode,
            start_time=self.episode_metadata["start_time"],
            end_time=self.episode_metadata["end_time"],
            frames=self.episode_frames,
            metadata=self.episode_metadata,
        )

        # Save episode
        episode_path = self._save_episode(episode)

        # Reset recording state
        self.is_recording = False
        self.current_episode = None
        self.episode_frames = []

        print(f"💾 Saved episode: {episode_path}")
        return str(episode_path)

    def record_frame(
        self, robot_state: Optional[Dict[str, Any]] = None
    ) -> RealSenseFrame:
        """
        Record a single frame with object detection and tracking.

        Args:
            robot_state: Optional robot state information

        Returns:
            Recorded frame
        """
        if not self.is_recording:
            raise RuntimeError("No episode is currently being recorded")

        if not self.camera_available:
            # Return mock frame when camera is not available
            rgb_image = np.zeros(
                (self.camera_resolution[1], self.camera_resolution[0], 3),
                dtype=np.uint8,
            )
            depth_image = np.zeros(
                (self.camera_resolution[1], self.camera_resolution[0]), dtype=np.uint16
            )
            intrinsics_dict = {
                "fx": 525.0,
                "fy": 525.0,
                "cx": 320.0,
                "cy": 240.0,
                "width": self.camera_resolution[0],
                "height": self.camera_resolution[1],
            }
        else:
            # Get camera frames
            frames = self.pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Convert to numpy arrays
            rgb_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())

            # Get camera intrinsics
            intrinsics = (
                color_frame.get_profile().as_video_stream_profile().get_intrinsics()
            )
            intrinsics_dict = {
                "fx": intrinsics.fx,
                "fy": intrinsics.fy,
                "cx": intrinsics.cx,
                "cy": intrinsics.cy,
                "width": intrinsics.width,
                "height": intrinsics.height,
            }

        # Detect objects
        detected_objects = self.detector.detect_objects(rgb_image, depth_image, None)

        # Track objects
        tracked_objects = self.tracker.update(detected_objects)

        # Convert objects to serializable format
        detected_objects_dict = [self._object_to_dict(obj) for obj in detected_objects]
        tracked_objects_dict = [
            self._tracked_object_to_dict(obj) for obj in tracked_objects
        ]

        # Create frame
        frame = RealSenseFrame(
            timestamp=time.time(),
            rgb_image=rgb_image,
            depth_image=depth_image,
            intrinsics=intrinsics_dict,
            detected_objects=detected_objects_dict,
            tracked_objects=tracked_objects_dict,
            robot_state=robot_state,
        )

        # Add to episode
        self.episode_frames.append(frame)

        return frame

    def _object_to_dict(self, obj: DetectedObject) -> Dict[str, Any]:
        """Convert DetectedObject to dictionary."""
        return {
            "bbox": obj.bbox,
            "center": obj.center,
            "depth": obj.depth,
            "confidence": obj.confidence,
            "class_id": obj.class_id,
            "class_name": obj.class_name,
            "world_position": obj.world_position,
        }

    def _tracked_object_to_dict(self, obj: DepthTrackedObject) -> Dict[str, Any]:
        """Convert DepthTrackedObject to dictionary."""
        return {
            "object_id": obj.object_id,
            "current_detection": self._object_to_dict(obj.current_detection),
            "position_history": obj.position_history,
            "velocity": obj.velocity,
            "age": obj.age,
            "lost_frames": obj.lost_frames,
            "depth_confidence": obj.depth_confidence,
            "size_history": obj.size_history,
            "depth_range": obj.depth_range,
        }

    def _save_episode(self, episode: RealSenseEpisode) -> Path:
        """Save episode to HDF5 file."""
        episode_path = self.output_dir / f"{episode.episode_id}.h5"

        with h5py.File(episode_path, "w") as f:
            # Save metadata
            f.attrs["episode_id"] = episode.episode_id
            f.attrs["start_time"] = episode.start_time
            f.attrs["end_time"] = episode.end_time
            f.attrs["frame_count"] = len(episode.frames)
            f.attrs["metadata"] = json.dumps(episode.metadata)

            # Create groups for different data types
            rgb_group = f.create_group("rgb_images")
            depth_group = f.create_group("depth_images")
            objects_group = f.create_group("objects")
            intrinsics_group = f.create_group("intrinsics")
            robot_group = f.create_group("robot_states")

            # Save frames
            for i, frame in enumerate(episode.frames):
                # Save images
                rgb_group.create_dataset(
                    f"frame_{i:06d}", data=frame.rgb_image, compression="gzip"
                )
                depth_group.create_dataset(
                    f"frame_{i:06d}", data=frame.depth_image, compression="gzip"
                )

                # Save intrinsics
                intrinsics_group.create_dataset(
                    f"frame_{i:06d}", data=json.dumps(frame.intrinsics)
                )

                # Save objects
                objects_group.create_dataset(
                    f"detected_{i:06d}", data=json.dumps(frame.detected_objects)
                )
                objects_group.create_dataset(
                    f"tracked_{i:06d}", data=json.dumps(frame.tracked_objects)
                )

                # Save robot state if available
                if frame.robot_state is not None:
                    robot_group.create_dataset(
                        f"frame_{i:06d}", data=json.dumps(frame.robot_state)
                    )

        return episode_path

    def record_episode(
        self,
        duration: float,
        robot_state_callback: Optional[callable] = None,
        episode_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a complete episode for specified duration.

        Args:
            duration: Duration to record in seconds
            robot_state_callback: Optional callback to get robot state
            episode_id: Optional episode ID
            metadata: Optional episode metadata

        Returns:
            Path to saved episode file
        """
        # Start episode
        episode_id = self.start_episode(episode_id, metadata)

        # Record frames
        start_time = time.time()
        frame_count = 0

        try:
            while time.time() - start_time < duration:
                # Get robot state if callback provided
                robot_state = None
                if robot_state_callback:
                    robot_state = robot_state_callback()

                # Record frame
                frame = self.record_frame(robot_state)
                frame_count += 1

                # Print progress
                elapsed = time.time() - start_time
                if frame_count % 30 == 0:  # Every second at 30fps
                    print(
                        f"📹 Recording: {elapsed:.1f}s / {duration:.1f}s ({frame_count} frames)"
                    )

                # Small delay to maintain frame rate
                time.sleep(1.0 / self.fps)

        except KeyboardInterrupt:
            print("⏹️  Recording stopped by user")

        # Stop episode
        episode_path = self.stop_episode()

        print(
            f"✅ Episode recorded: {frame_count} frames in {time.time() - start_time:.1f}s"
        )
        return episode_path

    def get_camera_frames(self) -> Tuple[np.ndarray, np.ndarray, rs.intrinsics]:
        """Get current frames from camera."""
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        rgb_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        intrinsics = (
            color_frame.get_profile().as_video_stream_profile().get_intrinsics()
        )

        return rgb_image, depth_image, intrinsics

    def close(self):
        """Close the camera pipeline."""
        self.pipeline.stop()
        print("✅ RealSense camera stopped")


class RealSenseDatasetLoader:
    """Loader for RealSense dataset files."""

    def __init__(self, dataset_dir: str):
        self.dataset_dir = Path(dataset_dir)

    def load_episode(self, episode_id: str) -> RealSenseEpisode:
        """Load a specific episode."""
        episode_path = self.dataset_dir / f"{episode_id}.h5"

        if not episode_path.exists():
            raise FileNotFoundError(f"Episode {episode_id} not found")

        with h5py.File(episode_path, "r") as f:
            # Load metadata
            episode_id = f.attrs["episode_id"]
            start_time = f.attrs["start_time"]
            end_time = f.attrs["end_time"]
            frame_count = f.attrs["frame_count"]
            metadata = json.loads(f.attrs["metadata"])

            # Load frames
            frames = []
            for i in range(frame_count):
                # Load images
                rgb_image = f["rgb_images"][f"frame_{i:06d}"][:]
                depth_image = f["depth_images"][f"frame_{i:06d}"][:]

                # Load intrinsics
                intrinsics = json.loads(f["intrinsics"][f"frame_{i:06d}"][()])

                # Load objects
                detected_objects = json.loads(f["objects"][f"detected_{i:06d}"][()])
                tracked_objects = json.loads(f["objects"][f"tracked_{i:06d}"][()])

                # Load robot state if available
                robot_state = None
                if f"robot_states/frame_{i:06d}" in f:
                    robot_state = json.loads(f["robot_states"][f"frame_{i:06d}"][()])

                # Create frame
                frame = RealSenseFrame(
                    timestamp=start_time + i / 30.0,  # Approximate timestamp
                    rgb_image=rgb_image,
                    depth_image=depth_image,
                    intrinsics=intrinsics,
                    detected_objects=detected_objects,
                    tracked_objects=tracked_objects,
                    robot_state=robot_state,
                )

                frames.append(frame)

            # Create episode
            episode = RealSenseEpisode(
                episode_id=episode_id,
                start_time=start_time,
                end_time=end_time,
                frames=frames,
                metadata=metadata,
            )

            return episode

    def list_episodes(self) -> List[str]:
        """List all available episodes."""
        episodes = []
        for file_path in self.dataset_dir.glob("*.h5"):
            episodes.append(file_path.stem)
        return sorted(episodes)
