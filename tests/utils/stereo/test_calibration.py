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

"""Tests for stereo calibration utilities."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from lerobot.utils.stereo.calibration import (
    calibrate_stereo_cameras,
    load_stereo_calibration,
    save_stereo_calibration,
)


def generate_checkerboard_image(
    image_size=(640, 480),
    pattern_size=(9, 6),
    square_size=40,
    rotation=0,
    translation=(0, 0),
):
    """
    Generate a synthetic checkerboard image for testing.

    Args:
        image_size: Output image size (width, height)
        pattern_size: Checkerboard pattern size (width, height)
        square_size: Size of each square in pixels
        rotation: Rotation angle in degrees
        translation: Translation offset (x, y) in pixels

    Returns:
        Grayscale checkerboard image
    """
    # Create a larger canvas for rotation
    canvas_size = max(image_size) * 3
    canvas = np.ones((canvas_size, canvas_size), dtype=np.uint8) * 200  # Light gray background

    # Calculate board dimensions - add border around checkerboard
    board_width = (pattern_size[0] + 1) * square_size
    board_height = (pattern_size[1] + 1) * square_size

    # Center the board on canvas
    start_x = (canvas_size - board_width) // 2
    start_y = (canvas_size - board_height) // 2

    # Draw white background for checkerboard
    cv2.rectangle(
        canvas,
        (start_x, start_y),
        (start_x + board_width, start_y + board_height),
        255,
        -1
    )

    # Draw black squares (checkerboard pattern)
    for i in range(pattern_size[1] + 1):
        for j in range(pattern_size[0] + 1):
            if (i + j) % 2 == 1:  # Black squares on odd positions
                x1 = start_x + j * square_size
                y1 = start_y + i * square_size
                x2 = x1 + square_size
                y2 = y1 + square_size
                cv2.rectangle(canvas, (x1, y1), (x2, y2), 0, -1)

    # Apply rotation around center
    if rotation != 0:
        center = (canvas_size // 2, canvas_size // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        canvas = cv2.warpAffine(canvas, rot_matrix, (canvas_size, canvas_size),
                                borderValue=200)

    # Crop and translate
    x_offset = (canvas_size - image_size[0]) // 2 + translation[0]
    y_offset = (canvas_size - image_size[1]) // 2 + translation[1]

    # Ensure we don't go out of bounds
    x_offset = max(0, min(x_offset, canvas_size - image_size[0]))
    y_offset = max(0, min(y_offset, canvas_size - image_size[1]))

    image = canvas[y_offset : y_offset + image_size[1], x_offset : x_offset + image_size[0]]

    return image


def generate_stereo_pair(
    image_size=(640, 480),
    pattern_size=(9, 6),
    num_images=15,
):
    """
    Generate synthetic stereo image pairs with checkerboards.

    Args:
        image_size: Image size (width, height)
        pattern_size: Checkerboard pattern size
        num_images: Number of image pairs to generate

    Returns:
        Tuple of (left_images, right_images)
    """
    left_images = []
    right_images = []

    # Generate images at different positions and rotations
    for i in range(num_images):
        # Vary rotation and translation with smaller ranges to keep board in frame
        rotation = (i * 6) % 30 - 15  # -15 to 15 degrees
        tx = ((i * 20) % 60) - 30  # -30 to 30 pixels
        ty = ((i * 15) % 40) - 20  # -20 to 20 pixels

        left_img = generate_checkerboard_image(
            image_size=image_size,
            pattern_size=pattern_size,
            square_size=35,  # Slightly smaller squares for better fit
            rotation=rotation,
            translation=(tx, ty),
        )

        # Right image with slight offset (simulating stereo baseline)
        right_img = generate_checkerboard_image(
            image_size=image_size,
            pattern_size=pattern_size,
            square_size=35,
            rotation=rotation,
            translation=(tx + 10, ty),  # Smaller offset for better overlap
        )

        left_images.append(left_img)
        right_images.append(right_img)

    return left_images, right_images


class TestStereoCalibration:
    """Test suite for stereo calibration functions."""

    def test_calibrate_stereo_cameras_success(self):
        """Test successful stereo calibration with synthetic data."""
        # Generate synthetic stereo pairs
        left_images, right_images = generate_stereo_pair(num_images=15)

        # Calibrate
        calibration_data = calibrate_stereo_cameras(
            left_images=left_images,
            right_images=right_images,
            pattern_size=(9, 6),
            square_size=0.025,
        )

        # Check that all required keys are present
        required_keys = [
            "camera_matrix_left",
            "camera_matrix_right",
            "distortion_left",
            "distortion_right",
            "R",
            "T",
            "E",
            "F",
            "rms_error",
            "image_size",
        ]
        for key in required_keys:
            assert key in calibration_data, f"Missing key: {key}"

        # Check data types
        assert isinstance(calibration_data["camera_matrix_left"], np.ndarray)
        assert isinstance(calibration_data["camera_matrix_right"], np.ndarray)
        assert isinstance(calibration_data["distortion_left"], np.ndarray)
        assert isinstance(calibration_data["distortion_right"], np.ndarray)
        assert isinstance(calibration_data["R"], np.ndarray)
        assert isinstance(calibration_data["T"], np.ndarray)
        assert isinstance(calibration_data["rms_error"], float)
        assert isinstance(calibration_data["image_size"], tuple)

        # Check matrix shapes
        assert calibration_data["camera_matrix_left"].shape == (3, 3)
        assert calibration_data["camera_matrix_right"].shape == (3, 3)
        assert calibration_data["R"].shape == (3, 3)
        assert calibration_data["T"].shape == (3, 1)

    def test_calibrate_mismatched_image_count(self):
        """Test error when left and right image counts don't match."""
        left_images, right_images = generate_stereo_pair(num_images=15)

        with pytest.raises(ValueError, match="Number of left and right images must match"):
            calibrate_stereo_cameras(
                left_images=left_images,
                right_images=right_images[:10],  # Mismatch
                pattern_size=(9, 6),
                square_size=0.025,
            )

    def test_calibrate_insufficient_images(self):
        """Test error when too few images are provided."""
        left_images, right_images = generate_stereo_pair(num_images=5)

        with pytest.raises(ValueError, match="Need at least 10 image pairs"):
            calibrate_stereo_cameras(
                left_images=left_images,
                right_images=right_images,
                pattern_size=(9, 6),
                square_size=0.025,
            )

    def test_save_and_load_json(self):
        """Test saving and loading calibration data to/from JSON."""
        # Generate calibration data
        left_images, right_images = generate_stereo_pair(num_images=15)
        calibration_data = calibrate_stereo_cameras(
            left_images=left_images,
            right_images=right_images,
            pattern_size=(9, 6),
            square_size=0.025,
        )

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "calibration.json"
            save_stereo_calibration(calibration_data, output_path)

            # Verify file exists
            assert output_path.exists()

            # Load and compare
            loaded_data = load_stereo_calibration(output_path)

            # Check all keys present
            assert set(loaded_data.keys()) == set(calibration_data.keys())

            # Check numpy arrays are equal
            for key in ["camera_matrix_left", "camera_matrix_right", "R", "T"]:
                np.testing.assert_array_almost_equal(
                    loaded_data[key], calibration_data[key], decimal=6
                )

            # Check scalar values
            assert loaded_data["rms_error"] == calibration_data["rms_error"]
            assert loaded_data["image_size"] == calibration_data["image_size"]

    def test_save_and_load_yaml(self):
        """Test saving and loading calibration data to/from YAML."""
        # Generate calibration data
        left_images, right_images = generate_stereo_pair(num_images=15)
        calibration_data = calibrate_stereo_cameras(
            left_images=left_images,
            right_images=right_images,
            pattern_size=(9, 6),
            square_size=0.025,
        )

        # Save to temporary file
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "calibration.yaml"
            save_stereo_calibration(calibration_data, output_path)

            # Verify file exists
            assert output_path.exists()

            # Load and compare
            loaded_data = load_stereo_calibration(output_path)

            # Check all keys present
            assert set(loaded_data.keys()) == set(calibration_data.keys())

            # Check numpy arrays are equal
            for key in ["camera_matrix_left", "camera_matrix_right", "R", "T"]:
                np.testing.assert_array_almost_equal(
                    loaded_data[key], calibration_data[key], decimal=6
                )

    def test_load_nonexistent_file(self):
        """Test error when loading from nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_stereo_calibration("nonexistent.json")

    def test_save_unsupported_format(self):
        """Test error when saving to unsupported file format."""
        left_images, right_images = generate_stereo_pair(num_images=15)
        calibration_data = calibrate_stereo_cameras(
            left_images=left_images,
            right_images=right_images,
            pattern_size=(9, 6),
            square_size=0.025,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "calibration.txt"
            with pytest.raises(ValueError, match="Unsupported file format"):
                save_stereo_calibration(calibration_data, output_path)

    def test_load_unsupported_format(self):
        """Test error when loading from unsupported file format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy file
            output_path = Path(tmpdir) / "calibration.txt"
            output_path.write_text("dummy")

            with pytest.raises(ValueError, match="Unsupported file format"):
                load_stereo_calibration(output_path)
