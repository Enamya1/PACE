# PACE
Pupil Assistive Calibrated Interface for Hands-Free Cursor Control

---

## 1. Purpose

TODO: fill in during Phase 1 (project overview, who it's for, what medical/accessibility problem it solves, the core hypothesis that pupil+glint OpenCV tracking beats MediaPipe-only for cursor accuracy).

## 2. Requirements

TODO: fill in during Phase 1 (system requirements: OS support, CPU/GPU, RAM, disk, USB webcam port, display resolution).

## 3. Python / OpenCV version

TODO: fill in during Phase 2. Target Python 3.11, OpenCV 5 (opencv-contrib-python for ximgproc / tracking extras). Confirm exact build hashes for reproducibility.

## 4. Installation

TODO: fill in during Phase 2. Exact venv + pip-install steps per OS (macOS / Linux / Windows), note about pyautogui permissions on macOS, pynput accessibility access.

### 4.1 macOS / Linux

TODO: Phase 2 — `python3.11 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && pre-commit install`

### 4.2 Windows

TODO: Phase 2 — `py -3.11 -m venv venv ; venv\Scripts\Activate.ps1 ; pip install -r requirements.txt ; pre-commit install`

## 5. Dependencies

TODO: fill in during Phase 2. Explain why each top-level dep exists:

| Package | Why it's here |
|---|---|
| opencv-python / opencv-contrib-python | Phase 3/4/5/12 — grayscale → blur → Otsu → morph → contour/ellipse/blob → solvePnP |
| numpy | All numerical work (frames, matrix ops, calibration solve) |
| mediapipe | Phase 2 initial face/eye ROI crop + EAR blink landmarks only |
| pyautogui | Cursor move + click dispatch (cross-platform) |
| pynput | Fallback mouse listener + dwell/click debugging |
| filterpy | Phase 8 — alpha-beta / Kalman smoothing for raw gaze jitter |
| PyQt6 | Phase 10 — calibration overlay + status GUI |
| pydantic | Config validation (no hardcoded tunables) |
| pyyaml | Configs/*.yaml loader |
| pandas / matplotlib | Phase 11 — metrics collection + benchmark plots |
| boto3 | Phase 12 — optional S3 log upload (disabled by default) |
| python-dotenv | AWS creds / local env |
| pytest / black / ruff / mypy | Dev tooling — tests, formatting, lint, type-check |
| pip-tools | Reproducible requirements pinning (requirements.in → requirements.txt) |

## 6. Webcam requirements

Phase 1 establishes the supported capture backends and how to select the right camera. (Minimum resolution, IR vs visible-light, glint/LED illuminator, eye-to-camera distance, and mounting advice will be added during Phase 3.)

### 6.1 Supported backends / platforms

PACE selects the OpenCV backend automatically based on the OS when `camera.backend: auto` (the default):

| OS        | backend flag used       | Value for `camera.backend:` in YAML |
|-----------|-------------------------|-------------------------------------|
| Windows   | `cv2.CAP_DSHOW`         | `dshow` (DirectShow — lowest latency)|
| macOS     | `cv2.CAP_AVFOUNDATION`  | `avfoundation`                       |
| Linux     | `cv2.CAP_V4L2`          | `v4l2`  (Video4Linux2)               |
| Any (fallback if unknown) | `cv2.CAP_ANY`    | — (used only on exotic platforms)    |

To override platform auto-detection, set `camera.backend` in `configs/default.yaml` (or a per-user profile under `configs/profiles/`) to one of the literal strings in the rightmost column above.

### 6.2 How to pick the right `camera_index`

- Default is `camera.camera_index: 0` — first device the OS enumerates.
- Machines with a built-in laptop cam + external USB cam typically expose indices `0` and `1`. If the wrong camera opens, increment `camera_index` to 1, 2, etc. and re-run `python -m app.main --camera-test` until you see frames flowing.
- On Windows with many capture devices (virtual cams from OBS / Teams / EpocCam), you can also list them with:
  ```powershell
  # PowerShell — enumerate DirectShow capture devices
  Get-CimInstance Win32_PnPEntity | Where-Object { $_.Name -match 'camera|webcam' } | Select Name, DeviceID
  ```
- On Linux, indices map to `/dev/video0`, `/dev/video1`, etc. so `ls -l /dev/video*` is the fastest lookup.
- On macOS, newer OS versions may ask for camera permissions the first time `--camera-test` runs; grant them in System Settings → Privacy & Security → Camera, then re-launch your terminal.

### 6.3 Manual smoke test

The 5-second live test (no OpenCV drawing — Phase 1 only counts frames):
```bash
python -m app.main --camera-test
```
Expected output: one line per second showing `connected=True` and avg FPS approaching `camera.target_fps` (usually 25–30 on a built-in webcam). If you see `[camera-test] FATAL: No webcam detected at index N`, raise `camera_index` in the YAML.

### 6.4 Disconnect behaviour (Phase 1 failure handling)

Unplugging the USB webcam mid-run will log an error after `camera.max_consecutive_failures` consecutive failed reads (default 15), flip `is_connected()` to `False`, and continue retrying silently — no traceback, no crash. Plug the camera back in and the next successful `read()` restores `is_connected()` to `True` automatically. Re-plugging on Windows sometimes hands you a new index; if frames don't resume, bump `camera_index`.

## 7. Configuration

TODO: fill in during Phase 2. Explain every key in `configs/default.yaml`, where per-user profiles go (`configs/profiles/` — gitignored), and how to pass `--config path/to/custom.yaml` to `app/main.py`.

## 8. Calibration

TODO: fill in during Phase 7. Explain the 9-point grid, polynomial fitting order, when to re-calibrate, expected calibration RMSE target (< 30 px on 1920×1080).

## 9. How to run

TODO: fill in during Phase 10 end-to-end. Main entry point flags: `--dry-run`, `--config`, `--no-gui`, `--skip-calibration`, `--record`.

## 10. How to test

TODO: fill in during Phase 12. `pytest tests/`, coverage target, how to run subsets (`pytest tests/test_foundation.py -v`), test data locations.

## 11. How to reproduce benchmarks

TODO: fill in during Phase 11. Standard benchmark dataset, `benchmarks/` runner scripts, where CSVs/PNGs land in `results/`, how to upload baseline plots for comparison.

## 12. AWS configuration

TODO: fill in during Phase 12. IAM policy, S3 bucket layout, KMS key guidance, `save_logs_to_aws: false` default, boto3 credential chain, NEVER commit AKIA/secret keys (enforced by `scripts/check_secrets.sh`).

## 13. Known limitations

TODO: fill in during Phase 12. Occlusions, glasses, lighting extremes, drift over time, multi-monitor edge cases, OS permission gotchas, latency budget breakdown.

## 14. Privacy considerations

TODO: fill in during Phase 12. No camera frames leave the machine unless `save_eye_frames: true` AND `save_logs_to_aws: true`. Per-user calibration profiles in `configs/profiles/` never go to git. AWS bucket ACLs, KMS, and retention policy requirements. Reference the medical disclaimer in `app/constants.py`.
