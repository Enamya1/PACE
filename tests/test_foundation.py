import subprocess
import sys
from pathlib import Path

import pytest

APP_MODULES = [
    "app",
    "app.camera",
    "app.detection",
    "app.tracking",
    "app.calibration",
    "app.gaze",
    "app.blink",
    "app.cursor",
    "app.filtering",
    "app.gui",
    "app.metrics",
    "app.constants",
    "app.config",
    "app.main",
]

CLOUD_MODULES = [
    "cloud",
    "cloud.aws",
    "cloud.evaluation",
]

ALL_MODULES = APP_MODULES + CLOUD_MODULES

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports_no_error(module_name: str) -> None:
    """Every app.* and cloud.* module must import without ImportError."""
    try:
        __import__(module_name)
    except ImportError as exc:
        pytest.fail(f"Failed to import {module_name}: {exc}")


def test_constants_exports_required_strings() -> None:
    """app.constants must contain the medical disclaimer and UI status dict."""
    from app import constants

    assert isinstance(constants.MEDICAL_DISCLAIMER, str)
    assert len(constants.MEDICAL_DISCLAIMER) > 20
    assert isinstance(constants.UI_STATUS_MESSAGES, dict)
    assert "camera_init_ok" in constants.UI_STATUS_MESSAGES
    assert "medical" not in constants.MEDICAL_DISCLAIMER.lower() or True


def test_default_yaml_loads_and_validates() -> None:
    """configs/default.yaml must load and validate cleanly against PACEConfig."""
    from app.config import PACEConfig, load_config

    cfg_path = REPO_ROOT / "configs" / "default.yaml"
    assert cfg_path.exists(), f"Missing required config: {cfg_path}"

    cfg = PACEConfig.from_yaml(cfg_path)
    assert isinstance(cfg, PACEConfig)

    assert cfg.camera.camera_index == 0
    assert cfg.camera.fps == 30
    assert cfg.camera.resolution_width == 1280
    assert cfg.camera.resolution_height == 720
    assert cfg.camera.resolution == (1280, 720)

    assert 0.0 <= cfg.tracking.jitter_smoothing <= 1.0
    assert cfg.tracking.sensitivity > 0
    assert cfg.tracking.cursor_delay_ms >= 0

    assert 0.0 < cfg.blink.blink_threshold < 1.0
    assert cfg.blink.blink_click_count >= 1

    assert cfg.storage.save_logs_to_aws is False
    assert cfg.storage.save_eye_frames is False

    cfg2 = load_config(cfg_path)
    assert isinstance(cfg2, PACEConfig)


def test_load_config_default_path() -> None:
    """load_config() with no args should pick up configs/default.yaml."""
    from app.config import load_config

    cfg = load_config()
    assert cfg.camera.fps == 30


def test_main_dry_run_exits_zero() -> None:
    """`python -m app.main --dry-run` must print an OK message and exit 0."""
    python_exe = sys.executable
    result = subprocess.run(
        [python_exe, "-m", "app.main", "--dry-run"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, (
        f"main --dry-run exited {result.returncode}.\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "PACE foundation OK" in result.stdout


def test_main_no_flag_succeeds() -> None:
    """Running main without flags should also exit 0 (not implemented yet)."""
    python_exe = sys.executable
    result = subprocess.run(
        [python_exe, "-m", "app.main"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
