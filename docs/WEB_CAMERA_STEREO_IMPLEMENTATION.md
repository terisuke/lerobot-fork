# Web Camera Stereo Vision Implementation Plan

## 📋 Overview

This document outlines the implementation plan for using **dual webcam setup** as an alternative to Intel RealSense D435 depth camera for LeRobot SO101 robotic manipulation tasks.

**Primary Motivation**: RealSense cameras have significant stability issues on macOS (Apple Silicon), making webcam-based solutions more practical for development.

## 🎯 Objectives

1. Replace unstable RealSense D435 with 2x USB webcams
2. Implement stereo vision for depth estimation
3. Maintain compatibility with existing LeRobot framework
4. Provide reliable object detection and tracking
5. Support dataset recording with depth information

## 🏗️ System Architecture

### Hardware Setup
```
USB Webcam 1 (Left)  ─┐
                      ├─> USB Hub ─> Mac (USB-C)
USB Webcam 2 (Right) ─┘
```

**Requirements:**
- 2x USB 3.0 compatible webcams
- Recommended: Logitech C920, C930e, or similar with autofocus
- USB-C hub with multiple USB 3.0 ports (powered recommended)
- Camera separation: 6-10cm baseline for optimal depth accuracy

### Software Stack
- **LeRobot Cameras Module**: `OpenCVCamera` class
- **OpenCV**: StereoBM/StereoSGBM for disparity calculation
- **YOLOv8**: Object detection (existing)
- **Custom Depth Estimator**: Stereo vision to depth conversion

## 📐 Technical Approach

### 1. Camera Calibration (Required First Step)

**Purpose**: Determine intrinsic and extrinsic parameters

**Process**:
1. Capture 20-30 chessboard calibration images with both cameras
2. Use OpenCV `cv2.stereoCalibrate()`
3. Calculate camera matrices, distortion coefficients, rotation/translation
4. Generate stereo rectification maps with `cv2.stereoRectify()`

**Key Parameters**:
```python
calibration_data = {
    'camera_matrix_left': np.array(3x3),
    'camera_matrix_right': np.array(3x3),
    'distortion_coeffs_left': np.array(5,),
    'distortion_coeffs_right': np.array(5,),
    'R': np.array(3x3),  # Rotation matrix
    'T': np.array(3,),   # Translation vector
    'E': np.array(3x3),  # Essential matrix
    'F': np.array(3x3)   # Fundamental matrix
}
```

### 2. Depth Estimation Pipeline

**Stereo Vision Formula**:
```
depth = (baseline × focal_length) / disparity
```

**Process**:
1. Capture synchronized frames from both cameras
2. Rectify images using calibration data
3. Compute disparity map using StereoBM or StereoSGBM
4. Convert disparity to depth map
5. Apply filters (median, bilateral) to reduce noise

### 3. Integration with LeRobot

**Modules to Create/Modify**:
- `src/lerobot/cameras/stereo_camera.py` - New stereo camera class
- `src/lerobot/object_detection/stereo_detector.py` - Detector with depth from stereo
- `src/lerobot/datasets/stereo_dataset.py` - Dataset recorder for stereo cameras

## 🔧 Implementation Plan

### Phase 1: Basic Stereo Capture (Week 1)
**Goal**: Capture synchronized frames from 2 cameras

**Tasks**:
- [ ] Create `StereoCameraConfig` class extending `CameraConfig`
- [ ] Implement `StereoCamera` class using LeRobot `OpenCVCamera`
- [ ] Add dual camera initialization and synchronization
- [ ] Implement frame grabbing with timestamp matching
- [ ] Create test script for basic stereo capture

**Deliverable**: `examples/stereo/basic_capture.py`

### Phase 2: Camera Calibration (Week 1-2)
**Goal**: Calibrate stereo camera pair

**Tasks**:
- [ ] Create calibration capture tool
- [ ] Implement chessboard detection
- [ ] Add stereo calibration computation
- [ ] Generate and save calibration data
- [ ] Create calibration visualization tool

**Deliverable**: `examples/stereo/calibrate_stereo.py`

### Phase 3: Depth Estimation (Week 2)
**Goal**: Compute depth maps from stereo images

**Tasks**:
- [ ] Implement stereo rectification
- [ ] Add disparity calculation (StereoBM/StereoSGBM)
- [ ] Create depth map computation
- [ ] Add post-processing (median filter, bilateral filter)
- [ ] Implement depth visualization

**Deliverable**: `examples/stereo/depth_estimation.py`

### Phase 4: Object Detection Integration (Week 3)
**Goal**: Combine YOLOv8 detection with stereo depth

**Tasks**:
- [ ] Create `StereoObjectDetector` extending `ObjectDetector`
- [ ] Implement object localization in 3D space
- [ ] Add depth-based filtering
- [ ] Integrate with existing tracking system
- [ ] Create example: real-time detection + depth

**Deliverable**: `examples/stereo/stereo_detection.py`

### Phase 5: Dataset Recording (Week 3-4)
**Goal**: Record datasets with stereo depth information

**Tasks**:
- [ ] Create `StereoDatasetRecorder` class
- [ ] Implement synchronized RGB+Depth recording
- [ ] Add calibration metadata to datasets
- [ ] Support multiple episode recording
- [ ] Create data replay functionality

**Deliverable**: `examples/stereo/record_stereo_dataset.py`

### Phase 6: Robot Integration (Week 4)
**Goal**: Integrate with SO101 robot arm

**Tasks**:
- [ ] Test depth accuracy at robot workspace distances
- [ ] Implement 3D object localization
- [ ] Add grasping point calculation
- [ ] Test real-time manipulation control
- [ ] Create complete demo

**Deliverable**: `examples/stereo/stereo_manipulation.py`

## 📊 Expected Performance

### Accuracy Comparison

| Metric          | RealSense D435 | Stereo Webcams    |
|-----------------|----------------|-------------------|
| Depth Accuracy  | ±1-2mm @ 1m    | ±5-10mm @ 1m      |
| Depth Range     | 0.3-10m        | 0.5-5m            |
| Frame Rate      | 30 FPS         | 30 FPS            |
| Resolution      | 640×480        | 640×480           |
| macOS Stability | ❌ Poor         | ✅ Excellent       |
| Cost            | ~$180          | ~$120 (2 cameras) |

### Computational Requirements
- CPU: Moderate (disparity calculation is CPU-intensive)
- RAM: ~2GB for 640×480 stereo processing
- Storage: ~1MB per synchronized frame pair

## 🛠️ Required Code Changes

### 1. New Camera Configuration
```python
# src/lerobot/cameras/stereo_camera.py

@dataclass
class StereoCameraConfig(CameraConfig):
    """Configuration for dual webcam stereo vision"""
    left_camera_id: int = 0
    right_camera_id: int = 1
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    calibration_file: Optional[str] = None
    baseline_meters: float = 0.06  # 6cm default

class StereoCamera:
    """Stereo camera using 2 webcams with depth estimation"""
    
    def __init__(self, config: StereoCameraConfig):
        self.left_cam = OpenCVCamera(...)
        self.right_cam = OpenCVCamera(...)
        self.load_calibration()
        self.stereo_matcher = cv2.StereoBM_create(...)
    
    def read(self) -> Tuple[Image, Image, DepthMap]:
        """Return synchronized left, right, and depth"""
        left = self.left_cam.read()
        right = self.right_cam.read()
        depth = self.compute_depth(left, right)
        return left, right, depth
```

### 2. Modified Object Detector
```python
# src/lerobot/object_detection/stereo_detector.py

class StereoObjectDetector(ObjectDetector):
    """Object detector with stereo vision depth"""
    
    def __init__(self, stereo_camera: StereoCamera, ...):
        super().__init__(...)
        self.stereo_camera = stereo_camera
    
    def detect_with_depth(self):
        """Detect objects with accurate depth from stereo"""
        left, right, depth = self.stereo_camera.read()
        detections = self.model.predict(left)
        
        # Use actual depth map instead of single point
        for det in detections:
            det.depth = np.median(depth[det.y1:det.y2, det.x1:det.x2])
```

## 📚 Reference Resources

### Official Documentation
- **LeRobot Cameras**: https://huggingface.co/docs/lerobot/cameras
- **Hugging Face Stereo Vision**: https://huggingface.co/learn/computer-vision-course/en/unit8/3d_measurements_stereo_vision

### OpenCV Resources
- Stereo Calibration: https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html
- Disparity Computation: https://docs.opencv.org/4.x/d2/d6e/classcv_1_1StereoBM.html
- Stereo Matching: https://docs.opencv.org/4.x/2d/stereo.html

### Example Implementations
- LearnOpenCV Stereo Vision: https://learnopencv.com/introduction-to-epipolar-geometry-and-stereo-vision/
- PyImageSearch Stereo Matching: https://pyimagesearch.com/2016/01/11/opencv-panorama-stitching/

## ✅ Success Criteria

1. ✅ Stable operation on macOS (no "failed to set power state" errors)
2. ✅ Depth accuracy ±5mm at 1 meter (sufficient for grasping)
3. ✅ Real-time processing at 30 FPS (640×480)
4. ✅ Successful object detection and 3D localization
5. ✅ Dataset recording with depth information
6. ✅ Integration with SO101 robot arm control

## 🚀 Next Steps

1. **Hardware Acquisition**: Purchase 2x compatible USB webcams
2. **Environment Setup**: Install OpenCV with stereo support
3. **Start Phase 1**: Implement basic dual camera capture
4. **Calibration**: Perform stereo camera calibration
5. **Integration**: Connect with existing LeRobot framework

## 📝 Notes

- **Baseline Distance**: 6cm is typical, but can be adjusted based on working distance
- **Calibration Frequency**: Recalibrate if cameras move or setup changes
- **Performance**: Disparity calculation can be GPU-accelerated for higher FPS
- **Fallback**: Keep RealSense implementation for Linux/Windows where stable

## 📅 Timeline

| Phase                          | Duration      | Dependencies       |
|--------------------------------|---------------|--------------------|
| Phase 1: Basic Capture         | 3-4 days      | Hardware           |
| Phase 2: Calibration           | 2-3 days      | Phase 1            |
| Phase 3: Depth Estimation      | 4-5 days      | Phase 2            |
| Phase 4: Detection Integration | 3-4 days      | Phase 3            |
| Phase 5: Dataset Recording     | 2-3 days      | Phase 4            |
| Phase 6: Robot Integration     | 3-4 days      | Phase 4            |
| **Total**                      | **3-4 weeks** | Hardware available |

---

**Status**: Planning Phase  
**Last Updated**: 2024-12-20  
**Maintainer**: LeRobot Development Team
