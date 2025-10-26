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
Stereo vision utilities for depth estimation using dual webcams.

This module provides post-processing utilities for stereo vision,
designed to work with LeRobot's standard multi-camera setup.
"""

from .calibration import calibrate_stereo_cameras, load_stereo_calibration, save_stereo_calibration
from .depth_estimation import StereoDepthEstimator, compute_depth_map

__all__ = [
    "calibrate_stereo_cameras",
    "load_stereo_calibration",
    "save_stereo_calibration",
    "StereoDepthEstimator",
    "compute_depth_map",
]
