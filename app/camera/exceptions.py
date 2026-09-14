class CameraError(Exception):
    """Raised when the camera cannot be opened, or a fatal webcam error occurs.

    Phase 14 (failure handling) will catch this exception type to display
    user-facing messages instead of leaking raw OpenCV/system exceptions.
    """

    pass
