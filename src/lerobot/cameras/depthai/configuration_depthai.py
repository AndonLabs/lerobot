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

"""Configuration for DepthAI (OAK) cameras."""

from dataclasses import dataclass
from enum import Enum

from lerobot.cameras import CameraConfig


class DepthAICameraSocket(str, Enum):
    """Camera socket on OAK device."""
    RGB = "CAM_A"      # Main RGB camera
    LEFT = "CAM_B"     # Left stereo camera
    RIGHT = "CAM_C"    # Right stereo camera


@CameraConfig.register_subclass("depthai")
@dataclass
class DepthAICameraConfig(CameraConfig):
    """
    Configuration for DepthAI (Luxonis OAK) cameras.

    Args:
        device_id: Device identifier. Can be:
            - MxId/Serial (e.g., "18443010215440F500") - stable across USB ports (recommended)
            - USB port path (e.g., "3.9") - changes if cable moves
            - None for first available device
        socket: Which camera socket to use (RGB, LEFT, RIGHT)
        fps: Frames per second
        width: Frame width
        height: Frame height
    """

    device_id: str | None = None
    socket: DepthAICameraSocket = DepthAICameraSocket.RGB
    fps: int = 30
    width: int = 1280
    height: int = 720

    # Manual camera controls for VLA consistency between training and inference.
    # Set these to fixed values so every session produces identical images.
    # Run `python -m lerobot.cameras.depthai.camera_depthai` to find good values.
    manual_focus: int | None = None          # 0-255 (0=infinity, 255=macro)
    manual_exposure: list[int] | None = None  # [exposure_us, iso] e.g. [10000, 400]
    manual_white_balance: int | None = None  # Color temperature in Kelvin, e.g. 4000
