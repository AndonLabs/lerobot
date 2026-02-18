#!/usr/bin/env python

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
Piper Leader Teleoperator for LeRobot.

This allows using an AgileX Piper arm as a leader/teacher arm for teleoperation.
The human moves this arm, and positions are read and sent to a follower robot.
"""

import logging
import time

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig

logger = logging.getLogger(__name__)

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None
    logger.warning("piper_sdk not installed. Install with: pip install piper_sdk")


class PiperLeader(Teleoperator):
    """
    AgileX Piper arm used as a leader/teacher for teleoperation.

    The arm should be in "teaching mode" (solid green LED) so it can be
    freely moved by hand while reading joint positions.
    """

    config_class = PiperLeaderConfig
    name = "piper_leader"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
        self.config = config
        self.piper = None
        self._is_connected = False

    @property
    def action_features(self) -> dict[str, type]:
        """Joint positions provided by this teleoperator."""
        return {
            "joint_0.pos": float,
            "joint_1.pos": float,
            "joint_2.pos": float,
            "joint_3.pos": float,
            "joint_4.pos": float,
            "joint_5.pos": float,
            "joint_6.pos": float,  # gripper
        }

    @property
    def feedback_features(self) -> dict[str, type]:
        """No feedback to the leader arm."""
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")

        if C_PiperInterface_V2 is None:
            raise ImportError(
                "piper_sdk is not installed. Install with: pip install piper_sdk"
            )

        logger.info(f"Connecting to Piper leader arm on {self.config.port}...")

        try:
            self.piper = C_PiperInterface_V2(self.config.port)
            self.piper.ConnectPort()
            time.sleep(0.1)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Piper on {self.config.port}: {e}\n"
                "Make sure CAN is activated: bash piper_sdk/can_activate.sh can0 1000000"
            )

        # Check if arm is in teaching mode (we want it freely movable)
        status = self.piper.GetArmStatus()
        if status.arm_status.ctrl_mode != 2:  # 2 = TEACHING_MODE
            logger.warning(
                "Piper leader arm is not in teaching mode. "
                "Press the button until LED is solid green for free movement."
            )

        self._is_connected = True

        # Flush stale CAN messages by reading a few times.
        # The first reads after ConnectPort() may return zeros which would
        # cause the follower arm to jump to a bad position.
        for _ in range(10):
            self.piper.GetArmJointMsgs()
            self.piper.GetArmGripperMsgs()
            time.sleep(0.02)

        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        # Piper doesn't need calibration - it has absolute encoders
        return True

    def calibrate(self) -> None:
        # No calibration needed for Piper
        pass

    def configure(self) -> None:
        # No special configuration needed for leader
        pass

    def get_action(self) -> dict[str, float]:
        """Read current joint positions from the leader arm."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        start = time.perf_counter()

        joint_status = self.piper.GetArmJointMsgs()
        gripper_status = self.piper.GetArmGripperMsgs()

        joint_state = joint_status.joint_state
        gripper_state = gripper_status.gripper_state

        # Return positions in millidegrees (raw SDK format)
        # The follower's SDK interface will handle conversion
        action = {
            "joint_0.pos": float(joint_state.joint_1),
            "joint_1.pos": float(joint_state.joint_2),
            "joint_2.pos": float(joint_state.joint_3),
            "joint_3.pos": float(joint_state.joint_4),
            "joint_4.pos": float(joint_state.joint_5),
            "joint_5.pos": float(joint_state.joint_6),
            "joint_6.pos": float(gripper_state.grippers_angle),
        }

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read action: {dt_ms:.1f}ms")
        return action

    def send_feedback(self, feedback: dict[str, float]) -> None:
        """No feedback to leader arm (it's freely moved by human)."""
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected")

        self._is_connected = False
        self.piper = None
        logger.info(f"{self} disconnected.")
