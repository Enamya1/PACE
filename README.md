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

TODO: fill in during Phase 3. Minimum resolution, frame rate, IR vs visible-light, glint/LED illuminator requirements, typical eye-to-camera distance, mounting suggestions. List tested webcams.

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
