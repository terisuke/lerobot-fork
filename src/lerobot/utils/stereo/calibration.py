# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Stereo camera calibration utilities.

Provides functions for calibrating stereo camera pairs using checkerboard patterns.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def calibrate_stereo_cameras(
    left_images: List[NDArray],
    right_images: List[NDArray],
    pattern_size: Tuple[int, int] = (9, 6),
    square_size: float = 0.025,
) -> Dict[str, Any]:
    """
    Calibrate stereo camera pair using checkerboard images.

    Args:
        left_images: List of grayscale images from left camera
        right_images: List of grayscale images from right camera
        pattern_size: Internal corners of checkerboard (width, height)
        square_size: Size of each square in meters

    Returns:
        Dictionary containing calibration parameters:
            - camera_matrix_left: Left camera matrix
            - camera_matrix_right: Right camera matrix
            - distortion_left: Left camera distortion coefficients
            - distortion_right: Right camera distortion coefficients
            - R: Rotation matrix between cameras
            - T: Translation vector between cameras
            - E: Essential matrix
            - F: Fundamental matrix
            - rms_error: RMS reprojection error
            - image_size: Image dimensions (width, height)

    Raises:
        ValueError: If insufficient images or checkerboard detection fails
    """
    if len(left_images) != len(right_images):
        raise ValueError("Number of left and right images must match")

    if len(left_images) < 10:
        raise ValueError(f"Need at least 10 image pairs, got {len(left_images)}")

    logger.info(f"Calibrating with {len(left_images)} image pairs")

    # Prepare object points (3D points in real world)
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world
    imgpoints_left = []  # 2D points in left image
    imgpoints_right = []  # 2D points in right image

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for left_gray, right_gray in zip(left_images, right_images):
        # Find checkerboard corners
        ret_left, corners_left = cv2.findChessboardCorners(left_gray, pattern_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(right_gray, pattern_size, None)

        if ret_left and ret_right:
            objpoints.append(objp)

            # Refine corner locations
            corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
            corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)

            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)

    if len(objpoints) < 10:
        raise ValueError(
            f"Only {len(objpoints)} valid image pairs found. Need at least 10."
        )

    logger.info(f"Using {len(objpoints)} valid image pairs for calibration")

    image_size = left_images[0].shape[::-1]  # (width, height)

    # Calibrate individual cameras
    logger.info("Calibrating left camera...")
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, image_size, None, None
    )

    logger.info("Calibrating right camera...")
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, image_size, None, None
    )

    # Stereo calibration
    logger.info("Performing stereo calibration...")
    flags = cv2.CALIB_FIX_INTRINSIC

    retval, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        mtx_left,
        dist_left,
        mtx_right,
        dist_right,
        image_size,
        flags=flags,
        criteria=criteria,
    )

    logger.info(f"Stereo calibration complete. RMS error: {retval:.4f}")

    # Return calibration data
    return {
        "camera_matrix_left": mtx_left,
        "camera_matrix_right": mtx_right,
        "distortion_left": dist_left,
        "distortion_right": dist_right,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "rms_error": float(retval),
        "image_size": image_size,
        "pattern_size": pattern_size,
        "square_size": square_size,
    }


def save_stereo_calibration(calibration_data: Dict[str, Any], output_path: str | Path) -> None:
    """
    Save stereo calibration data to file.

    Args:
        calibration_data: Calibration parameters from calibrate_stereo_cameras()
        output_path: Path to output file (.json or .yaml)

    Raises:
        ValueError: If output format is not supported
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy arrays and tuples to lists for JSON/YAML serialization
    data_to_save = {}
    for key, value in calibration_data.items():
        if isinstance(value, np.ndarray):
            data_to_save[key] = value.tolist()
        elif isinstance(value, tuple):
            data_to_save[key] = list(value)
        else:
            data_to_save[key] = value

    if output_path.suffix == ".json":
        with open(output_path, "w") as f:
            json.dump(data_to_save, f, indent=2)
        logger.info(f"Saved calibration to {output_path}")
    elif output_path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(output_path, "w") as f:
            yaml.dump(data_to_save, f, default_flow_style=False)
        logger.info(f"Saved calibration to {output_path}")
    else:
        raise ValueError(f"Unsupported file format: {output_path.suffix}")


def load_stereo_calibration(calibration_path: str | Path) -> Dict[str, Any]:
    """
    Load stereo calibration data from file.

    Args:
        calibration_path: Path to calibration file (.json or .yaml)

    Returns:
        Dictionary containing calibration parameters with numpy arrays

    Raises:
        FileNotFoundError: If calibration file does not exist
        ValueError: If file format is not supported
    """
    calibration_path = Path(calibration_path)

    if not calibration_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calibration_path}")

    if calibration_path.suffix == ".json":
        with open(calibration_path, "r") as f:
            data = json.load(f)
    elif calibration_path.suffix in [".yaml", ".yml"]:
        import yaml
        with open(calibration_path, "r") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported file format: {calibration_path.suffix}")

    # Convert lists back to numpy arrays and tuples where appropriate
    calibration_data = {}
    for key, value in data.items():
        if key in ["camera_matrix_left", "camera_matrix_right", "distortion_left",
                   "distortion_right", "R", "T", "E", "F"]:
            calibration_data[key] = np.array(value)
        elif key in ["image_size", "pattern_size"] and isinstance(value, list):
            calibration_data[key] = tuple(value)
        else:
            calibration_data[key] = value

    logger.info(f"Loaded calibration from {calibration_path}")
    return calibration_data
