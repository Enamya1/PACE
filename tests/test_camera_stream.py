import itertools
import time
from typing import Iterator, List, Tuple
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from app.camera.camera_stream import CameraStream
from app.camera.exceptions import CameraError
from app.config import CameraConfig


def _make_config(**overrides) -> CameraConfig:
    base = dict(
        camera_index=0,
        resolution_width=1280,
        resolution_height=720,
        fps=30,
        frame_width=640,
        frame_height=480,
        target_fps=30,
        max_consecutive_failures=3,
        backend="auto",
    )
    base.update(overrides)
    return CameraConfig(**base)


def _make_frame(w: int = 640, h: int = 480, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _infinite_read(seq: List[Tuple[bool, object]]) -> Iterator[Tuple[bool, object]]:
    """Yield each item of *seq* in order, then repeat the last item forever.

    MagicMock.side_effect will call next() on this generator on every
    `.read()` call from the capture thread.  The trailing infinite tail
    ensures the thread never hits StopIteration once the interesting
    portion of the sequence has been consumed by a fast background loop.
    """
    if not seq:
        while True:
            yield (False, None)
    last = seq[-1]
    for item in seq:
        yield item
    while True:
        yield last


# ---------------------------------------------------------------------------
# 1. start() / stop() lifecycle
# ---------------------------------------------------------------------------
@patch("app.camera.camera_stream.cv2.VideoCapture")
def test_camera_starts_and_stops_cleanly(mock_vc_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _make_frame(seed=1))
    mock_vc_cls.return_value = mock_cap

    cfg = _make_config()
    cam = CameraStream(cfg.camera_index, cfg)

    cam.start()

    mock_vc_cls.assert_called_once()
    args, kwargs = mock_vc_cls.call_args
    assert args[0] == cfg.camera_index
    mock_cap.isOpened.assert_called()
    assert mock_cap.set.call_count == 4  # w, h, fps, buffersize

    time.sleep(0.1)
    assert cam.is_connected() is True
    assert cam._thread is not None
    assert cam._thread.is_alive() is True

    cam.stop()

    mock_cap.release.assert_called_once()
    assert cam._thread is None


# ---------------------------------------------------------------------------
# 2. Latest-frame-wins buffer semantics
# ---------------------------------------------------------------------------
@patch("app.camera.camera_stream.cv2.VideoCapture")
def test_get_frame_returns_latest_frame(mock_vc_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    frames: List[Tuple[bool, np.ndarray]] = [
        (True, _make_frame(seed=100)),
        (True, _make_frame(seed=200)),
        (True, _make_frame(seed=300)),
        (True, _make_frame(seed=400)),
    ]
    mock_cap.read.side_effect = _infinite_read(frames)

    mock_vc_cls.return_value = mock_cap

    cfg = _make_config()
    cam = CameraStream(cfg.camera_index, cfg)
    try:
        cam.start()
        time.sleep(0.3)

        fd = cam.get_frame(timeout=0.5)
        assert fd is not None
        # Must be one of the later frames; at minimum should NOT be stale seed=100.
        # Since the queue has maxsize=1 and drops old frames before putting new,
        # after the thread has run for a bit we expect frame_number > 1.
        assert fd.frame_number >= 2, (
            "Expected drop-old-put-new buffer to advance past frame 1, "
            f"got frame_number={fd.frame_number}"
        )
    finally:
        cam.stop()


# ---------------------------------------------------------------------------
# 3. start() raises CameraError, not raw OpenCV, when camera missing
# ---------------------------------------------------------------------------
@patch("app.camera.camera_stream.cv2.VideoCapture")
def test_camera_error_on_failed_open(mock_vc_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False
    mock_vc_cls.return_value = mock_cap

    cfg = _make_config(camera_index=99)
    cam = CameraStream(cfg.camera_index, cfg)

    with pytest.raises(CameraError) as exc_info:
        cam.start()

    assert "index 99" in str(exc_info.value)
    mock_cap.release.assert_not_called()
    assert cam._thread is None


# ---------------------------------------------------------------------------
# 4. Disconnect detection after > max_consecutive_failures
# ---------------------------------------------------------------------------
@patch("app.camera.camera_stream.cv2.VideoCapture")
def test_disconnect_detection(mock_vc_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    # Some successes first, then *many* failures beyond threshold, then keep failing.
    threshold = 3
    successes = [(True, _make_frame(seed=i)) for i in range(5)]
    failures = [(False, None)] * (threshold + 10)
    mock_cap.read.side_effect = _infinite_read(successes + failures)

    mock_vc_cls.return_value = mock_cap
    cfg = _make_config(max_consecutive_failures=threshold)
    cam = CameraStream(cfg.camera_index, cfg)

    try:
        cam.start()
        time.sleep(0.4)

        # After the failure streak, camera must report disconnected.
        assert cam.is_connected() is False, (
            "Expected is_connected() to become False after exceeding the failure threshold."
        )
    finally:
        # Thread must still be alive (joinable) so stop() still works cleanly.
        thread = cam._thread
        assert thread is not None
        cam.stop()

    assert thread.is_alive() is False, "Thread was not joined cleanly after disconnect handling."


# ---------------------------------------------------------------------------
# 5. Transient failures (below threshold) do not disconnect; frames resume
# ---------------------------------------------------------------------------
@patch("app.camera.camera_stream.cv2.VideoCapture")
def test_reconnect_after_transient_failure(mock_vc_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True

    threshold = 5
    successes_1 = [(True, _make_frame(seed=i)) for i in range(3)]
    transient = [(False, None)] * 2  # well below threshold
    successes_2 = [(True, _make_frame(seed=100 + i)) for i in range(10)]
    mock_cap.read.side_effect = _infinite_read(successes_1 + transient + successes_2)

    mock_vc_cls.return_value = mock_cap
    cfg = _make_config(max_consecutive_failures=threshold)
    cam = CameraStream(cfg.camera_index, cfg)

    try:
        cam.start()
        time.sleep(0.35)

        # Connection must stay True because we never crossed the threshold.
        assert cam.is_connected() is True, (
            "Transient failures below threshold must not flip is_connected()."
        )

        # And we are still receiving *new* frames (successes_2 flow).
        fd = cam.get_frame(timeout=0.5)
        assert fd is not None
        assert fd.frame_number >= 4, (
            f"Expected frames to resume after transient, got frame_number={fd.frame_number}"
        )
    finally:
        cam.stop()


# ---------------------------------------------------------------------------
# 6. Context-manager form works correctly
# ---------------------------------------------------------------------------
@patch("app.camera.camera_stream.cv2.VideoCapture")
def test_context_manager(mock_vc_cls):
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, _make_frame(seed=1))
    mock_vc_cls.return_value = mock_cap

    cfg = _make_config()

    with CameraStream(cfg.camera_index, cfg) as cam:
        time.sleep(0.1)
        # Inside the with-block: VideoCapture was constructed + isOpened checked
        mock_vc_cls.assert_called_once()
        mock_cap.isOpened.assert_called()
        assert cam.is_connected() is True
        assert cam._thread is not None

    # Exit: stop() -> release called, thread cleaned up
    mock_cap.release.assert_called_once()
    assert cam._thread is None
