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
Provides the DepthAICamera class for capturing frames from Luxonis OAK cameras.
"""

import logging
import time
from threading import Event, Lock, Thread
from typing import Any

import numpy as np

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..camera import Camera
from .configuration_depthai import DepthAICameraConfig, DepthAICameraSocket

try:
    import depthai as dai
    DEPTHAI_AVAILABLE = True
except ImportError:
    DEPTHAI_AVAILABLE = False
    dai = None

logger = logging.getLogger(__name__)


class DepthAICamera(Camera):
    """
    Manages camera interactions using DepthAI SDK for Luxonis OAK cameras.

    This class provides a high-level interface to connect to, configure, and read
    frames from OAK cameras (OAK-D, OAK-D Lite, OAK-D Pro, etc.).

    Example:
        ```python
        from lerobot.cameras.depthai import DepthAICamera, DepthAICameraConfig

        config = DepthAICameraConfig(
            width=1280,
            height=720,
            fps=30,
        )
        camera = DepthAICamera(config)
        camera.connect()

        # Read frame
        frame = camera.read()
        print(frame.shape)

        camera.disconnect()
        ```
    """

    def __init__(self, config: DepthAICameraConfig):
        """Initialize the DepthAICamera instance."""
        super().__init__(config)

        if not DEPTHAI_AVAILABLE:
            raise ImportError("depthai is not installed. Install with: pip install depthai")

        self.config = config
        self.device_id = config.device_id
        self.socket = config.socket
        self.fps = config.fps

        self.device: "dai.Device | None" = None
        self.pipeline: "dai.Pipeline | None" = None
        self.queue: Any = None

        # Async reading
        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: np.ndarray | None = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        device_str = self.device_id or "auto"
        return f"{self.__class__.__name__}({device_str}, {self.socket.value})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera is currently connected."""
        return self.device is not None and self.pipeline is not None

    def connect(self, warmup: bool = True):
        """
        Connects to the OAK camera.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ConnectionError: If no OAK device is found.
        """
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} is already connected.")

        # Find device
        available_devices = dai.Device.getAllAvailableDevices()
        if not available_devices:
            raise ConnectionError("No OAK devices found. Check USB connection and udev rules.")

        if self.device_id:
            # Find specific device by MxId (serial) or USB path
            device_info = None
            for d in available_devices:
                # Check USB path first
                if d.name == self.device_id:
                    device_info = d
                    break
                # Check MxId (serial number) - need to connect briefly to get it
                try:
                    with dai.Device(d) as temp_device:
                        if temp_device.getDeviceId() == self.device_id:
                            device_info = d
                            break
                except Exception:
                    pass
            if device_info is None:
                available_info = []
                for d in available_devices:
                    try:
                        with dai.Device(d) as temp_device:
                            available_info.append(f"{d.name} (MxId: {temp_device.getDeviceId()})")
                    except Exception:
                        available_info.append(f"{d.name} (MxId: unknown)")
                raise ConnectionError(f"OAK device '{self.device_id}' not found. Available: {available_info}")
        else:
            device_info = available_devices[0]

        # Create device and pipeline
        self.device = dai.Device(device_info)

        # Map socket enum to depthai socket
        socket_map = {
            DepthAICameraSocket.RGB: dai.CameraBoardSocket.CAM_A,
            DepthAICameraSocket.LEFT: dai.CameraBoardSocket.CAM_B,
            DepthAICameraSocket.RIGHT: dai.CameraBoardSocket.CAM_C,
        }
        dai_socket = socket_map[self.socket]

        # Check if socket is available
        connected_cams = self.device.getConnectedCameras()
        if dai_socket not in connected_cams:
            self.device.close()
            self.device = None
            raise ConnectionError(f"Camera socket {self.socket.value} not available. Connected: {connected_cams}")

        # Create pipeline with v3 API - don't use context manager to keep it alive
        self.pipeline = dai.Pipeline(self.device)

        # Create camera node
        cam = self.pipeline.create(dai.node.Camera).build(dai_socket)

        # Apply manual controls before pipeline start
        cfg = self.config
        if cfg.manual_focus is not None:
            cam.initialControl.setManualFocus(cfg.manual_focus)
        if cfg.manual_exposure is not None:
            cam.initialControl.setManualExposure(cfg.manual_exposure[0], cfg.manual_exposure[1])
        if cfg.manual_white_balance is not None:
            cam.initialControl.setManualWhiteBalance(cfg.manual_white_balance)

        # Request output at specified resolution
        output = cam.requestOutput((self.width, self.height), type=dai.ImgFrame.Type.BGR888i)
        self.queue = output.createOutputQueue(maxSize=4, blocking=False)

        # Create control queue before starting pipeline (required by DepthAI v3)
        self._control_queue = cam.inputControl.createInputQueue()

        # Start pipeline
        self.pipeline.start()

        has_manual = cfg.manual_focus is not None or cfg.manual_exposure is not None or cfg.manual_white_balance is not None

        if warmup:
            # Let auto settings settle (longer warmup if we'll lock them)
            warmup_secs = 2.0 if not has_manual else 1.0
            start_time = time.time()
            while time.time() - start_time < warmup_secs:
                try:
                    self.read()
                except Exception:
                    pass
                time.sleep(0.1)

            # If no manual values set, lock auto values after warmup
            if not has_manual:
                ctrl = dai.CameraControl()
                ctrl.setAutoExposureLock(True)
                ctrl.setAutoWhiteBalanceLock(True)
                ctrl.setAutoFocusMode(dai.CameraControl.AutoFocusMode.OFF)
                self._control_queue.send(ctrl)
                logger.info(f"{self} locked auto-exposure, white balance, and focus.")
            else:
                locked = []
                if cfg.manual_focus is not None:
                    locked.append(f"focus={cfg.manual_focus}")
                if cfg.manual_exposure is not None:
                    locked.append(f"exposure={cfg.manual_exposure[0]}us/ISO{cfg.manual_exposure[1]}")
                if cfg.manual_white_balance is not None:
                    locked.append(f"wb={cfg.manual_white_balance}K")
                logger.info(f"{self} manual controls: {', '.join(locked)}")

        logger.info(f"{self} connected.")

    def read(self) -> np.ndarray:
        """
        Reads a single frame synchronously from the camera.

        Returns:
            np.ndarray: The captured frame (height, width, 3) in RGB format.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading fails.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        start_time = time.perf_counter()

        frame_data = self.queue.get()
        if frame_data is None:
            raise RuntimeError(f"{self} failed to get frame.")

        frame = frame_data.getCvFrame()

        # Convert BGR to RGB
        frame = frame[:, :, ::-1].copy()

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _read_loop(self):
        """Background thread loop for async reading."""
        while not self.stop_event.is_set():
            try:
                frame = self.read()
                with self.frame_lock:
                    self.latest_frame = frame
                self.new_frame_event.set()
            except DeviceNotConnectedError:
                break
            except Exception as e:
                logger.warning(f"Error reading frame in background thread for {self}: {e}")

    def _start_read_thread(self) -> None:
        """Starts the background read thread."""
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.1)
        if self.stop_event is not None:
            self.stop_event.set()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Stops the background read thread."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

    def read_latest(self, max_age_ms: int = 500) -> np.ndarray:
        """Return the most recent frame without waiting for a new one."""
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        with self.frame_lock:
            frame = self.latest_frame

        if frame is None:
            # No frame yet — fall back to blocking read for the first call
            return self.async_read(timeout_ms=max_age_ms)

        return frame

    def async_read(self, timeout_ms: float = 1000) -> np.ndarray:
        """
        Reads the latest available frame asynchronously.

        Args:
            timeout_ms: Maximum time to wait for a frame.

        Returns:
            np.ndarray: The latest captured frame.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame is available within timeout.
        """
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        if self.thread is None or not self.thread.is_alive():
            self._start_read_thread()

        # Retry up to 3 times on timeout (USB hiccups can cause brief frame drops)
        for attempt in range(3):
            if self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
                break
            logger.warning(f"Frame timeout from {self} (attempt {attempt + 1}/3)")
        else:
            raise TimeoutError(f"Timed out waiting for frame from {self} after 3 attempts ({timeout_ms} ms each).")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    def disconnect(self):
        """Disconnects from the camera and cleans up resources."""
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None

        if self.device is not None:
            try:
                self.device.close()
            except Exception:
                pass
            self.device = None

        self.queue = None

        logger.info(f"{self} disconnected.")

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available OAK cameras connected to the system.

        Returns:
            List of dictionaries with camera information.
        """
        if not DEPTHAI_AVAILABLE:
            return []

        found_cameras = []
        for device_info in dai.Device.getAllAvailableDevices():
            camera_info = {
                "name": f"OAK Camera @ {device_info.name}",
                "type": "DepthAI",
                "id": device_info.name,
                "state": device_info.state.name,
            }
            found_cameras.append(camera_info)

        return found_cameras


if __name__ == "__main__":
    """Calibration helper: connects to each camera, lets auto-adjust settle,
    then prints the values to use as manual_focus / manual_exposure / manual_white_balance."""
    import sys

    logging.basicConfig(level=logging.INFO)

    device_ids = sys.argv[1:] if len(sys.argv) > 1 else [None]
    for device_id in device_ids:
        config = DepthAICameraConfig(device_id=device_id, width=640, height=480, fps=30)
        cam = DepthAICamera(config)
        cam.connect(warmup=True)

        # Let auto settings settle for a few more seconds
        print(f"\n{cam}: waiting 3s for auto-adjust to settle...")
        start = time.time()
        while time.time() - start < 3.0:
            cam.read()
            time.sleep(0.05)

        # Read current camera metadata from a frame
        frame_data = cam.queue.get()
        exposure = frame_data.getExposureTime().total_seconds() * 1_000_000  # to microseconds
        iso = frame_data.getSensitivity()
        wb = frame_data.getColorTemperature()
        lens_pos = frame_data.getLensPosition()

        print(f"\n{'='*50}")
        print(f"  Camera: {cam}")
        print(f"  Device ID: {device_id}")
        print(f"{'='*50}")
        print(f"  Focus (lens position): {lens_pos}")
        print(f"  Exposure: {int(exposure)}us, ISO {iso}")
        print(f"  White Balance: {wb}K")
        print(f"\n  Config values to use:")
        print(f"    manual_focus: {lens_pos}")
        print(f"    manual_exposure: [{int(exposure)}, {iso}]")
        print(f"    manual_white_balance: {wb}")
        print(f"{'='*50}\n")

        cam.disconnect()
