MEDICAL_DISCLAIMER = (
    "PACE is a research prototype and is NOT a medical device. "
    "It is not intended to diagnose, treat, cure, or prevent any disease. "
    "Do not use this software for critical or life-dependent operations. "
    "Always maintain an alternative input method available."
)

UI_STATUS_MESSAGES = {
    "camera_init_ok": "Camera initialized successfully.",
    "camera_init_failed": "ERROR: Failed to initialize camera. Check index and permissions.",
    "camera_read_failed": "ERROR: Failed to read frame from camera.",
    "face_not_detected": "WARNING: No face detected in frame. Adjust lighting or position.",
    "eyes_not_detected": "WARNING: Eyes not detected. Ensure face is well-lit and centered.",
    "pupil_lost": "WARNING: Pupil tracking lost. Attempting re-acquisition.",
    "glint_lost": "WARNING: Glint(s) not detected. Calibration may drift.",
    "calibration_start": "Calibration starting. Follow on-screen prompts.",
    "calibration_complete": "Calibration complete. Ready for use.",
    "calibration_failed": "ERROR: Calibration failed. Please restart.",
    "blink_click_triggered": "Blink-click registered.",
    "blink_dwell_triggered": "Dwell-click registered.",
    "aws_upload_ok": "Logs uploaded to AWS successfully.",
    "aws_upload_failed": "ERROR: AWS upload failed. Check credentials and network.",
    "shutdown_ok": "PACE shut down cleanly. All resources released.",
}
