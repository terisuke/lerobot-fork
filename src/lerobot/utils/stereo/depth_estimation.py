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
Stereo depth estimation utilities.

Provides functions for computing depth maps from calibrated stereo camera pairs.
"""

import logging
from typing import Any, Dict, Literal

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class StereoDepthEstimator:
    """
    Stereo depth estimator using block matching or semi-global block matching.

    This class manages stereo matching configuration and provides methods to compute
    depth maps from calibrated stereo image pairs.
    """

    def __init__(
        self,
        calibration_data: Dict[str, Any],
        method: Literal["bm", "sgbm"] = "sgbm",
        num_disparities: int = 16 * 5,
        block_size: int = 11,
    ):
        """
        Initialize stereo depth estimator.

        Args:
            calibration_data: Calibration parameters from calibrate_stereo_cameras()
            method: Stereo matching method ("bm" for StereoBM, "sgbm" for StereoSGBM)
            num_disparities: Maximum disparity (must be divisible by 16)
            block_size: Matched block size (must be odd, typically 5-21)

        Raises:
            ValueError: If calibration data is missing required fields
            ValueError: If num_disparities is not divisible by 16
            ValueError: If block_size is not odd
        """
        # Validate calibration data
        required_keys = [
            "camera_matrix_left",
            "camera_matrix_right",
            "distortion_left",
            "distortion_right",
            "R",
            "T",
        ]
        for key in required_keys:
            if key not in calibration_data:
                raise ValueError(f"Calibration data missing required key: {key}")

        self.calibration_data = calibration_data
        self.method = method

        # Validate parameters
        if num_disparities % 16 != 0:
            raise ValueError(f"num_disparities must be divisible by 16, got {num_disparities}")
        if block_size % 2 == 0:
            raise ValueError(f"block_size must be odd, got {block_size}")

        self.num_disparities = num_disparities
        self.block_size = block_size

        # Compute rectification transforms
        self._compute_rectification()

        # Create stereo matcher
        self._create_matcher()

        logger.info(
            f"Initialized StereoDepthEstimator with method={method}, "
            f"num_disparities={num_disparities}, block_size={block_size}"
        )

    def _compute_rectification(self) -> None:
        """Compute stereo rectification transforms."""
        image_size = tuple(self.calibration_data["image_size"])

        # Stereo rectification
        (
            self.R1,
            self.R2,
            self.P1,
            self.P2,
            self.Q,
            self.roi_left,
            self.roi_right,
        ) = cv2.stereoRectify(
            self.calibration_data["camera_matrix_left"],
            self.calibration_data["distortion_left"],
            self.calibration_data["camera_matrix_right"],
            self.calibration_data["distortion_right"],
            image_size,
            self.calibration_data["R"],
            self.calibration_data["T"],
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,
        )

        # Compute rectification maps for efficient remapping
        self.map1_left, self.map2_left = cv2.initUndistortRectifyMap(
            self.calibration_data["camera_matrix_left"],
            self.calibration_data["distortion_left"],
            self.R1,
            self.P1,
            image_size,
            cv2.CV_32FC1,
        )

        self.map1_right, self.map2_right = cv2.initUndistortRectifyMap(
            self.calibration_data["camera_matrix_right"],
            self.calibration_data["distortion_right"],
            self.R2,
            self.P2,
            image_size,
            cv2.CV_32FC1,
        )

    def _create_matcher(self) -> None:
        """Create stereo matcher based on selected method."""
        if self.method == "bm":
            # StereoBM - faster but less accurate
            self.matcher = cv2.StereoBM_create(
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
            )
        elif self.method == "sgbm":
            # StereoSGBM - slower but more accurate
            # P1 and P2 control smoothness (P2 > P1)
            P1 = 8 * 3 * self.block_size**2
            P2 = 32 * 3 * self.block_size**2

            self.matcher = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.num_disparities,
                blockSize=self.block_size,
                P1=P1,
                P2=P2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
            )
        else:
            raise ValueError(f"Unknown stereo matching method: {self.method}")

    def rectify_images(
        self, left_image: NDArray, right_image: NDArray
    ) -> tuple[NDArray, NDArray]:
        """
        Rectify stereo image pair.

        Args:
            left_image: Left camera image (grayscale or BGR)
            right_image: Right camera image (grayscale or BGR)

        Returns:
            Tuple of (rectified_left, rectified_right) images
        """
        left_rectified = cv2.remap(
            left_image, self.map1_left, self.map2_left, cv2.INTER_LINEAR
        )
        right_rectified = cv2.remap(
            right_image, self.map1_right, self.map2_right, cv2.INTER_LINEAR
        )

        return left_rectified, right_rectified

    def compute_disparity(
        self, left_image: NDArray, right_image: NDArray, rectify: bool = True
    ) -> NDArray:
        """
        Compute disparity map from stereo image pair.

        Args:
            left_image: Left camera image (BGR or grayscale)
            right_image: Right camera image (BGR or grayscale)
            rectify: Whether to rectify images before matching (default: True)

        Returns:
            Disparity map (float32, values in pixels)
        """
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image

        if len(right_image.shape) == 3:
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            right_gray = right_image

        # Rectify images
        if rectify:
            left_gray, right_gray = self.rectify_images(left_gray, right_gray)

        # Compute disparity
        disparity = self.matcher.compute(left_gray, right_gray)

        # Convert to float32 and scale
        disparity = disparity.astype(np.float32) / 16.0

        return disparity

    def compute_depth(
        self, left_image: NDArray, right_image: NDArray, rectify: bool = True
    ) -> NDArray:
        """
        Compute depth map from stereo image pair.

        Args:
            left_image: Left camera image (BGR or grayscale)
            right_image: Right camera image (BGR or grayscale)
            rectify: Whether to rectify images before matching (default: True)

        Returns:
            Depth map in meters (float32, invalid depths are np.inf)
        """
        # Compute disparity
        disparity = self.compute_disparity(left_image, right_image, rectify=rectify)

        # Convert disparity to depth using OpenCV's reprojection
        # The Q matrix from stereoRectify encodes the transformation:
        # [X Y Z W]^T = Q * [x y disparity 1]^T
        # where actual 3D coordinates are [X/W, Y/W, Z/W]
        points_3d = cv2.reprojectImageTo3D(disparity, self.Q)

        # Extract depth (Z coordinate)
        depth = points_3d[:, :, 2].astype(np.float32)

        # Mark invalid depths (negative or very large values) as inf
        invalid_mask = (disparity <= 0) | (depth <= 0) | (depth > 10000)
        depth[invalid_mask] = np.inf

        return depth


def compute_depth_map(
    left_image: NDArray,
    right_image: NDArray,
    calibration_data: Dict[str, Any],
    method: Literal["bm", "sgbm"] = "sgbm",
    num_disparities: int = 16 * 5,
    block_size: int = 11,
) -> NDArray:
    """
    Compute depth map from calibrated stereo image pair.

    This is a convenience function that creates a StereoDepthEstimator and computes
    the depth map in one call. For repeated depth computation, it's more efficient
    to create a StereoDepthEstimator instance and reuse it.

    Args:
        left_image: Left camera image (BGR or grayscale)
        right_image: Right camera image (BGR or grayscale)
        calibration_data: Calibration parameters from calibrate_stereo_cameras()
        method: Stereo matching method ("bm" or "sgbm")
        num_disparities: Maximum disparity (must be divisible by 16)
        block_size: Matched block size (must be odd)

    Returns:
        Depth map in meters (float32, invalid depths are np.inf)

    Example:
        >>> from lerobot.utils.stereo import compute_depth_map, load_stereo_calibration
        >>> calib = load_stereo_calibration("stereo_calib.yaml")
        >>> depth = compute_depth_map(left_frame, right_frame, calib)
        >>> print(f"Depth at center: {depth[360, 640]:.2f}m")
    """
    estimator = StereoDepthEstimator(
        calibration_data=calibration_data,
        method=method,
        num_disparities=num_disparities,
        block_size=block_size,
    )

    return estimator.compute_depth(left_image, right_image)
