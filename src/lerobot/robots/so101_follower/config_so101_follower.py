#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("so101_follower")
@dataclass
class SO101FollowerConfig(RobotConfig):
    # Port to connect to the arm
    port: str

    disable_torque_on_disconnect: bool = True

    # `max_relative_target` limits the magnitude of the relative positional target vector for safety purposes.
    # Set this to a positive scalar to have the same value for all motors, or a dictionary that maps motor
    # names to the max_relative_target value for that motor.
    max_relative_target: float | dict[str, float] | None = None

    # cameras - Default stereo camera configuration for depth perception
    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            # Stereo camera pair for depth perception
            "left": OpenCVCameraConfig(
                index_or_path=0,
                fps=30,
                width=1280,
                height=720,
            ),
            "right": OpenCVCameraConfig(
                index_or_path=1,
                fps=30,
                width=1280,
                height=720,
            ),
        }
    )

    # Stereo calibration file path (optional)
    stereo_calibration_path: str = "configs/camera/stereo_calibration.yaml"

    # Set to `True` for backward compatibility with previous policies/dataset
    use_degrees: bool = False
