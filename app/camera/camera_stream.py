import logging
import platform
import queue
import sys
import threading
import time
from types import TracebackType
from typing import Optional, Type

import cv2
import numpy as np

from app.camera.exceptions import CameraError
from app.camera.frame_data import FrameData
from app.config import CameraConfig

logger = logging.getLogger(__name__)


def _resolve_backend(config_backend: str) -> int:
    """Map a backend name to a cv2.CAP_* flag.

    When config.backend is "auto", the OS drives the choice:
      - Windows  -> cv2.CAP_DSHOW (DirectShow, low-latency default)
      - macOS    -> cv2.CAP_AVFOUNDATION (AVFoundation)
      - Linux    -> cv2.CAP_V4L2  (Video4Linux2)
    Any other value must be one of "dshow", "avfoundation", "v4l2" (validated
    at the pydantic level already via CameraConfig.backend: Literal[...]).
    """
    name_to_flag = {
        "dshow": cv2.CAP_DSHOW,
        "avfoundation": cv2.CAP_AVFOUNDATION,
        "v4l2": cv2.CAP_V4L2,
    }
    if config_backend != "auto":
        return name_to_flag[config_backend]
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    if system == "Darwin":
        return cv2.CAP_AVFOUNDATION
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


class CameraStream:
    """Thread-safe, low-latency webcam frame source.

    Architecture notes
    ------------------
    * Frames are captured on a dedicated daemon thread. This decouples the
      frame rate of the physical webcam from the consumer-side processing
      rate of Phases 2+, preventing the GUI/cursor loop from blocking the
      capture hardware.
    * A queue.Queue(maxsize=1) is used as a "latest frame wins" buffer.  When
      the consumer is slow, the oldest still-buffered frame is silently
      discarded before the new one is inserted, which guarantees end-to-end
      latency stays below 1 frame-duration instead of queuing up stale data.
    * Connection state (`_connected`) is written/read as a plain bool; because
      Python bool assignment is atomic under the GIL on CPython, no extra
      `threading.Lock` is required for this single flag.  Multi-field updates
      elsewhere should add a Lock if they ever become necessary.
    * The camera hardware is NOT touched in `__init__`.  Callers must invoke
      `start()` explicitly (or use the context-manager form) so the webcam
      LED turns on only when the user has requested capture — a privacy
      requirement from the spec.
    """

    def __init__(self, camera_index: int, config: CameraConfig) -> None:
        self.camera_index = camera_index
        self.config = config

        self._cap: Optional[cv2.VideoCapture] = None
        self._queue: "queue.Queue[FrameData]" = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._connected = False
        self._backend_flag: int = _resolve_backend(config.backend)

    # -------- lifecycle ----------

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            logger.warning("CameraStream.start() called while already running.")
            return

        self._cap = cv2.VideoCapture(self.camera_index, self._backend_flag)

        if not self._cap.isOpened():
            raise CameraError(f"No webcam detected at index {self.camera_index}")

        requests = [
            (cv2.CAP_PROP_FRAME_WIDTH, self.config.frame_width, "frame_width"),
            (cv2.CAP_PROP_FRAME_HEIGHT, self.config.frame_height, "frame_height"),
            (cv2.CAP_PROP_FPS, self.config.target_fps, "target_fps"),
            (cv2.CAP_PROP_BUFFERSIZE, 1, "buffer_size"),
        ]
        for prop_id, wanted, name in requests:
            self._cap.set(prop_id, wanted)
            actual = self._cap.get(prop_id)
            if isinstance(actual, (int, float)) and abs(float(actual) - float(wanted)) > 1e-6:
                logger.warning(
                    "CameraStream: requested %s=%s but driver reported %s "
                    "(camera may not support the exact value, continuing).",
                    name,
                    wanted,
                    actual,
                )

        self._stop_event.clear()
        self._connected = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name=f"CameraStream-{self.camera_index}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        if getattr(self, "_thread", None) is None:
            self._release_cap()
            return
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logger.warning(
                    "CameraStream: capture thread did not exit after 2s; "
                    "forcing VideoCapture release anyway."
                )
        self._thread = None
        self._release_cap()

    def _release_cap(self) -> None:
        cap = getattr(self, "_cap", None)
        if cap is not None:
            try:
                cap.release()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning("CameraStream: cap.release() raised: %s", exc)
        self._cap = None
        self._connected = False

    # -------- consumer API ----------

    def get_frame(self, timeout: float = 0.1) -> Optional[FrameData]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def is_connected(self) -> bool:
        return self._connected

    # -------- capture thread ----------

    def _capture_loop(self) -> None:
        consecutive_failures = 0
        frame_number = 0
        cap = self._cap
        if cap is None:  # pragma: no cover - impossible after start()
            return

        while not self._stop_event.is_set():
            try:
                ret, frame = cap.read()
            except Exception:
                ret, frame = False, None

            if not ret:
                consecutive_failures += 1
                if consecutive_failures > self.config.max_consecutive_failures:
                    if self._connected:
                        logger.error(
                            "CameraStream: %s consecutive read() failures; "
                            "marking disconnected and continuing to retry.",
                            consecutive_failures,
                        )
                        self._connected = False
                # Back off slightly so a fully dead camera doesn't spin a CPU core
                time.sleep(0.005)
                continue

            # success path
            consecutive_failures = 0
            if not self._connected:
                logger.info("CameraStream: read() recovered; connection restored.")
                self._connected = True

            frame_number += 1
            fd = FrameData(
                frame=frame if frame is not None else np.empty((0, 0, 3), dtype=np.uint8),
                timestamp=time.monotonic(),
                frame_number=frame_number,
            )

            if self._queue.full():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
            try:
                self._queue.put_nowait(fd)
            except queue.Full:  # pragma: no cover - racy but harmless
                pass

    # -------- context manager ----------

    def __enter__(self) -> "CameraStream":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        self.stop()
