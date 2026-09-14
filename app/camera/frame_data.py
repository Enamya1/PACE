from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrameData:
    """A single captured webcam frame plus bookkeeping metadata.

    Attributes:
        frame: Raw BGR uint8 numpy array from cv2.VideoCapture.read().
        timestamp: Value of time.monotonic() at the moment read() returned.
        frame_number: Monotonically increasing counter of successful reads
            since start() was called (resets to 0 on each start()).
    """

    frame: np.ndarray
    timestamp: float
    frame_number: int
