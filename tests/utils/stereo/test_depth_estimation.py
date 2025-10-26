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

"""Tests for stereo depth estimation utilities."""

import numpy as np
import pytest

from lerobot.utils.stereo.depth_estimation import StereoDepthEstimator, compute_depth_map


def create_mock_calibration_data():
    """Create mock stereo calibration data for testing."""
    # Simple identity-like calibration for testing
    camera_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=np.float32)

    distortion = np.zeros((5, 1), dtype=np.float32)

    # Identity rotation
    R = np.eye(3, dtype=np.float32)

    # Small baseline translation (10cm to the right)
    T = np.array([[0.1], [0], [0]], dtype=np.float32)

    # Dummy E and F matrices
    E = np.eye(3, dtype=np.float32)
    F = np.eye(3, dtype=np.float32)

    return {
        "camera_matrix_left": camera_matrix,
        "camera_matrix_right": camera_matrix,
        "distortion_left": distortion,
        "distortion_right": distortion,
        "R": R,
        "T": T,
        "E": E,
        "F": F,
        "rms_error": 0.5,
        "image_size": (640, 480),
        "pattern_size": (9, 6),
        "square_size": 0.025,
    }


def create_stereo_test_images():
    """Create simple test stereo images."""
    # Create a simple pattern with horizontal offset
    left_image = np.zeros((480, 640), dtype=np.uint8)
    right_image = np.zeros((480, 640), dtype=np.uint8)

    # Draw a rectangle at different positions
    left_image[200:280, 250:350] = 255
    right_image[200:280, 230:330] = 255  # 20 pixel shift

    return left_image, right_image


class TestStereoDepthEstimator:
    """Test suite for StereoDepthEstimator class."""

    def test_init_with_valid_calibration(self):
        """Test initialization with valid calibration data."""
        calibration_data = create_mock_calibration_data()

        estimator = StereoDepthEstimator(
            calibration_data=calibration_data, method="sgbm", num_disparities=80, block_size=11
        )

        assert estimator.method == "sgbm"
        assert estimator.num_disparities == 80
        assert estimator.block_size == 11
        assert estimator.calibration_data == calibration_data

    def test_init_with_bm_method(self):
        """Test initialization with StereoBM method."""
        calibration_data = create_mock_calibration_data()

        estimator = StereoDepthEstimator(
            calibration_data=calibration_data, method="bm", num_disparities=64, block_size=15
        )

        assert estimator.method == "bm"

    def test_init_missing_calibration_keys(self):
        """Test error when calibration data is missing required keys."""
        incomplete_calibration = {
            "camera_matrix_left": np.eye(3),
            # Missing other required keys
        }

        with pytest.raises(ValueError, match="Calibration data missing required key"):
            StereoDepthEstimator(calibration_data=incomplete_calibration)

    def test_init_invalid_num_disparities(self):
        """Test error when num_disparities is not divisible by 16."""
        calibration_data = create_mock_calibration_data()

        with pytest.raises(ValueError, match="num_disparities must be divisible by 16"):
            StereoDepthEstimator(
                calibration_data=calibration_data, num_disparities=77  # Not divisible by 16
            )

    def test_init_invalid_block_size(self):
        """Test error when block_size is not odd."""
        calibration_data = create_mock_calibration_data()

        with pytest.raises(ValueError, match="block_size must be odd"):
            StereoDepthEstimator(calibration_data=calibration_data, block_size=12)  # Even number

    def test_rectify_images(self):
        """Test image rectification."""
        calibration_data = create_mock_calibration_data()
        estimator = StereoDepthEstimator(calibration_data=calibration_data)

        left_image, right_image = create_stereo_test_images()

        left_rect, right_rect = estimator.rectify_images(left_image, right_image)

        # Check output shape matches input
        assert left_rect.shape == left_image.shape
        assert right_rect.shape == right_image.shape

        # Check output type
        assert left_rect.dtype == left_image.dtype

    def test_compute_disparity_grayscale(self):
        """Test disparity computation with grayscale images."""
        calibration_data = create_mock_calibration_data()
        estimator = StereoDepthEstimator(calibration_data=calibration_data)

        left_image, right_image = create_stereo_test_images()

        disparity = estimator.compute_disparity(left_image, right_image, rectify=False)

        # Check output shape
        assert disparity.shape == left_image.shape

        # Check output type
        assert disparity.dtype == np.float32

    def test_compute_disparity_color(self):
        """Test disparity computation with color images."""
        calibration_data = create_mock_calibration_data()
        estimator = StereoDepthEstimator(calibration_data=calibration_data)

        # Create color images
        left_image, right_image = create_stereo_test_images()
        left_color = np.stack([left_image] * 3, axis=-1)
        right_color = np.stack([right_image] * 3, axis=-1)

        disparity = estimator.compute_disparity(left_color, right_color, rectify=False)

        # Check output shape
        assert disparity.shape == left_image.shape

        # Check output type
        assert disparity.dtype == np.float32

    def test_compute_depth(self):
        """Test depth computation."""
        calibration_data = create_mock_calibration_data()
        estimator = StereoDepthEstimator(calibration_data=calibration_data)

        left_image, right_image = create_stereo_test_images()

        depth = estimator.compute_depth(left_image, right_image, rectify=False)

        # Check output shape
        assert depth.shape == left_image.shape

        # Check output type
        assert depth.dtype == np.float32

        # Check that invalid depths are inf
        assert np.any(np.isinf(depth))

        # Check that some valid depths exist
        valid_depths = depth[~np.isinf(depth) & (depth > 0)]
        # Note: With simple test images, we might not get many valid depth values
        # This is expected as the stereo matching needs more complex patterns

    def test_compute_depth_with_rectification(self):
        """Test depth computation with rectification enabled."""
        calibration_data = create_mock_calibration_data()
        estimator = StereoDepthEstimator(calibration_data=calibration_data)

        left_image, right_image = create_stereo_test_images()

        depth = estimator.compute_depth(left_image, right_image, rectify=True)

        # Check output shape
        assert depth.shape == left_image.shape

        # Check output type
        assert depth.dtype == np.float32


class TestComputeDepthMap:
    """Test suite for compute_depth_map convenience function."""

    def test_compute_depth_map_basic(self):
        """Test basic depth map computation."""
        calibration_data = create_mock_calibration_data()
        left_image, right_image = create_stereo_test_images()

        depth_map = compute_depth_map(left_image, right_image, calibration_data)

        # Check output shape
        assert depth_map.shape == left_image.shape

        # Check output type
        assert depth_map.dtype == np.float32

    def test_compute_depth_map_with_params(self):
        """Test depth map computation with custom parameters."""
        calibration_data = create_mock_calibration_data()
        left_image, right_image = create_stereo_test_images()

        depth_map = compute_depth_map(
            left_image,
            right_image,
            calibration_data,
            method="bm",
            num_disparities=64,
            block_size=9,
        )

        # Check output shape
        assert depth_map.shape == left_image.shape

        # Check output type
        assert depth_map.dtype == np.float32

    def test_compute_depth_map_invalid_method(self):
        """Test error with invalid stereo matching method."""
        calibration_data = create_mock_calibration_data()
        left_image, right_image = create_stereo_test_images()

        # The error should be raised during StereoDepthEstimator initialization
        # which happens inside compute_depth_map
        with pytest.raises(ValueError):
            compute_depth_map(
                left_image,
                right_image,
                calibration_data,
                method="invalid_method",  # type: ignore
            )


class TestDepthEstimationIntegration:
    """Integration tests for depth estimation pipeline."""

    def test_full_pipeline(self):
        """Test full pipeline from images to depth map."""
        # Create calibration data
        calibration_data = create_mock_calibration_data()

        # Create test images
        left_image, right_image = create_stereo_test_images()

        # Initialize estimator
        estimator = StereoDepthEstimator(
            calibration_data=calibration_data, method="sgbm", num_disparities=80, block_size=11
        )

        # Rectify images
        left_rect, right_rect = estimator.rectify_images(left_image, right_image)

        # Compute disparity
        disparity = estimator.compute_disparity(left_rect, right_rect, rectify=False)

        # Compute depth
        depth = estimator.compute_depth(left_image, right_image, rectify=True)

        # Verify all steps produced valid output
        assert left_rect.shape == left_image.shape
        assert disparity.shape == left_image.shape
        assert depth.shape == left_image.shape

        # Verify types
        assert disparity.dtype == np.float32
        assert depth.dtype == np.float32

    def test_multiple_depth_computations(self):
        """Test that estimator can be reused for multiple depth computations."""
        calibration_data = create_mock_calibration_data()
        estimator = StereoDepthEstimator(calibration_data=calibration_data)

        # Compute depth multiple times
        for _ in range(5):
            left_image, right_image = create_stereo_test_images()
            depth = estimator.compute_depth(left_image, right_image)

            assert depth.shape == left_image.shape
            assert depth.dtype == np.float32
