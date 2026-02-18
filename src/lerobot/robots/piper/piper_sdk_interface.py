# Piper SDK interface for LeRobot integration

import logging
import time
from typing import Any

try:
    from piper_sdk import C_PiperInterface_V2
except ImportError:
    C_PiperInterface_V2 = None

logger = logging.getLogger(__name__)


class PiperSDKInterface:
    def __init__(self, port: str = "can0"):
        if C_PiperInterface_V2 is None:
            raise ImportError("piper_sdk is not installed. Install with: pip install piper_sdk")
        self.port = port
        self.piper = None

    def connect(self):
        """Connect to the CAN bus and establish communication with the arm."""
        try:
            self.piper = C_PiperInterface_V2(self.port)
        except Exception as e:
            raise ConnectionError(
                f"Failed to connect to Piper on {self.port}: {e}\n"
                "Did you activate the CAN interface? bash piper_sdk/can_activate.sh"
            )
        self.piper.ConnectPort()
        # Flush stale CAN messages so the first get_status() returns real values
        for _ in range(10):
            self.piper.GetArmJointMsgs()
            self.piper.GetArmGripperMsgs()
            time.sleep(0.02)

    def enable(self):
        """Enable the arm for joint control. Call this right before the teleop loop."""
        if self.piper is None:
            raise RuntimeError("Not connected. Call connect() first.")

        # Enable the arm (matches working teleop.py sequence)
        logger.info("Enabling follower arm...")
        while not self.piper.EnablePiper():
            time.sleep(0.01)

        # Joint mode, 100% speed, high-follow mode (0xAD)
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
        time.sleep(0.1)
        logger.info("Follower arm enabled and ready.")

    def set_joint_positions(self, positions):
        # positions: list of 7 raw values from the leader arm
        # first 6 are joint positions in millidegrees, 7th is gripper angle
        # Re-assert motion control every cycle (required to keep arm responsive)
        self.piper.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
        self.piper.JointCtrl(
            int(positions[0]),
            int(positions[1]),
            int(positions[2]),
            int(positions[3]),
            int(positions[4]),
            int(positions[5]),
        )
        self.piper.GripperCtrl(int(positions[6]), 1000, 0x01, 0)

    def get_status(self) -> dict[str, Any]:
        joint_status = self.piper.GetArmJointMsgs()
        gripper = self.piper.GetArmGripperMsgs()

        joint_state = joint_status.joint_state
        return {
            "joint_0.pos": joint_state.joint_1,
            "joint_1.pos": joint_state.joint_2,
            "joint_2.pos": joint_state.joint_3,
            "joint_3.pos": joint_state.joint_4,
            "joint_4.pos": joint_state.joint_5,
            "joint_5.pos": joint_state.joint_6,
            "joint_6.pos": gripper.gripper_state.grippers_angle,
        }

    def disconnect(self):
        # Don't disable motors on disconnect - let the arm hold position.
        # The user can press the button to enter teaching mode when done.
        pass
