"""Phase 3 — Unit tests for PreprocessPipeline.

All tests run against the synthetic eye-ROI fixtures under
tests/fixtures/eye_rois/ — no live camera required, CI-friendly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from app.config import PreprocessConfig
from app.detection.data_types import DetectionResult, EyeROI, PreprocessResult
from app.detection.preprocess import PreprocessPipeline


FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "eye_rois"


# ------------------------------------------------------------------ #
#  Helpers
# ------------------------------------------------------------------ #
def _load_eye_roi(name: str, size: Tuple[int, int] = (120, 80)) -> EyeROI:
    img = cv2.imread(str(FIXTURE_DIR / name), cv2.IMREAD_COLOR)
    if img is None:
        pytest.skip(f"Fixture missing: {FIXTURE_DIR / name}")
    w, h = size
    if (img.shape[1], img.shape[0]) != (w, h):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    return EyeROI(
        image=img,
        original_bbox=(100, 100, 100 + w, 100 + h),
        scale_x=1.0,
        scale_y=1.0,
        iris_hint=None,
    )


def _fg(mask: np.ndarray) -> int:
    return int(np.count_nonzero(mask))


# ------------------------------------------------------------------ #
#  Test 1 — binary mask sanity (uint8, values ∈ {0, 255})
# ------------------------------------------------------------------ #
def test_produces_binary_mask() -> None:
    cfg = PreprocessConfig(threshold_mode="otsu")
    pipe = PreprocessPipeline(cfg)
    eye = _load_eye_roi("normal_lighting.jpg")
    pp = pipe.process(eye)

    assert isinstance(pp, PreprocessResult)
    assert pp.mask.dtype == np.uint8
    uniq = set(int(x) for x in np.unique(pp.mask))
    assert uniq.issubset({0, 255}), (
        f"PreprocessResult.mask must be strictly binary (0 or 255). Got values: {sorted(uniq)}"
    )
    # Pipeline also keeps the intermediate buffers:
    assert pp.gray.dtype == np.uint8
    assert pp.gray.ndim == 2
    assert pp.blurred.dtype == np.uint8
    assert pp.blurred.ndim == 2
    assert pp.gray.shape == pp.mask.shape


# ------------------------------------------------------------------ #
#  Test 2 — Otsu on normal lighting produces plausible pupil blob
# ------------------------------------------------------------------ #
def test_otsu_mode_normal_lighting() -> None:
    cfg = PreprocessConfig(threshold_mode="otsu")
    pipe = PreprocessPipeline(cfg)
    pp = pipe.process(_load_eye_roi("normal_lighting.jpg"))

    assert pp.lighting_ok is True
    assert pp.threshold_mode_used == "otsu"
    h, w = pp.mask.shape
    total = h * w
    fg = _fg(pp.mask)
    # Plausibility band: pupil blob on 120x80 ROI should occupy between
    # 0.5% and 25% of pixels (i.e. not 0 and not the whole mask).
    assert 0.005 < fg / total < 0.25, (
        f"Otsu mask foreground ratio out of band: {fg/total*100:.2f}% "
        f"(expected a non-trivial, non-universal pupil blob)."
    )
    # mean_brightness of the synthetic normal fixture ~ 112
    assert 80.0 < pp.mean_brightness < 160.0


# ------------------------------------------------------------------ #
#  Test 3 — low-light fixture fires lighting_ok=False (gate test)
# ------------------------------------------------------------------ #
def test_low_light_flags_lighting_not_ok() -> None:
    cfg_default = PreprocessConfig()  # min_valid_brightness=15
    pipe = PreprocessPipeline(cfg_default)
    pp = pipe.process(_load_eye_roi("low_light.jpg"))

    # Low-light fixture mean_gray ~ 12, strictly below 15 gate
    assert pp.lighting_ok is False
    assert pp.mean_brightness < 15.0
    # …but the pipeline still proceeds (no exception) and still returns
    # a real uint8 binary mask (Phase 4 must not crash on bad lighting):
    assert pp.mask.dtype == np.uint8
    uniq = set(int(x) for x in np.unique(pp.mask))
    assert uniq.issubset({0, 255})


# ------------------------------------------------------------------ #
#  Test 4 — adaptive thresholding improves low-light mask, *relatively*
# ------------------------------------------------------------------ #
def test_adaptive_mode_improves_low_light() -> None:
    eye = _load_eye_roi("low_light.jpg")
    # Baseline Otsu on the same fixture — tends to collapse to mostly
    # zeros because the histogram is skewed into the bottom bucket.
    pp_otsu = PreprocessPipeline(PreprocessConfig(threshold_mode="otsu")).process(eye)
    # Adaptive — local neighbourhood thresholding:
    pp_adapt = PreprocessPipeline(PreprocessConfig(threshold_mode="adaptive")).process(eye)

    normal_fg = _fg(
        PreprocessPipeline(PreprocessConfig(threshold_mode="otsu"))
        .process(_load_eye_roi("normal_lighting.jpg"))
        .mask
    )
    otsu_fg = _fg(pp_otsu.mask)
    adapt_fg = _fg(pp_adapt.mask)

    # Relative assertion: |adaptive_fg - normal_fg| must be strictly
    # smaller than |otsu_fg - normal_fg|.  In other words, adaptive
    # thresholding produces a foreground pixel count *closer* to the
    # "normal-lighting reference" than Otsu does for the same low-light
    # input.  (We don't assert absolute pixel counts, which would be
    # fragile across draw implementations / OpenCV versions.)
    assert abs(adapt_fg - normal_fg) < abs(otsu_fg - normal_fg), (
        "adaptive thresholding on low_light fixture should move the "
        f"foreground count toward the normal-light reference.  "
        f"normal={normal_fg}  otsu={otsu_fg}  adaptive={adapt_fg}."
    )
    assert pp_adapt.threshold_mode_used == "adaptive"


# ------------------------------------------------------------------ #
#  Test 5 — bad threshold_mode string fails *at construction* (lazy)
# ------------------------------------------------------------------ #
def test_invalid_threshold_mode_raises_at_construction() -> None:
    with pytest.raises(ValidationError):
        PreprocessConfig(threshold_mode="some-made-up-mode")


# ------------------------------------------------------------------ #
#  Test 6 — even blur_kernel size rejected by pydantic
# ------------------------------------------------------------------ #
def test_even_blur_kernel_rejected_by_config() -> None:
    with pytest.raises(ValidationError):
        PreprocessConfig(blur_kernel_size=4)
    # adaptive_block_size must also be odd:
    with pytest.raises(ValidationError):
        PreprocessConfig(adaptive_block_size=8)


# ------------------------------------------------------------------ #
#  Test 7 — process_both_eyes handles missing eyes gracefully
# ------------------------------------------------------------------ #
def test_process_both_eyes_handles_missing_eye() -> None:
    pipe = PreprocessPipeline(PreprocessConfig())

    left_eye: EyeROI = _load_eye_roi("normal_lighting.jpg")
    right_eye = None  # simulate Phase 2 detecting only one eye

    det = DetectionResult(
        found=False,  # found=False is fine; process_both_eyes only looks at ROIs
        confidence=0.0,
        face_bbox=None,
        left_eye=left_eye,
        right_eye=right_eye,
        backend_used="mediapipe",
        frame_timestamp=0.0,
        consecutive_misses=1,
    )

    l_pp, r_pp = pipe.process_both_eyes(det)
    assert isinstance(l_pp, PreprocessResult), (
        "Present left_eye must produce a non-None PreprocessResult."
    )
    assert r_pp is None, (
        "Missing right_eye must yield None (not an exception)."
    )

    # Swap it around — confirm symmetry (missing LEFT eye):
    det2 = DetectionResult(
        found=False,
        confidence=0.0,
        face_bbox=None,
        left_eye=None,
        right_eye=left_eye,
        backend_used="haar",
        frame_timestamp=1.0,
        consecutive_misses=2,
    )
    l_pp2, r_pp2 = pipe.process_both_eyes(det2)
    assert l_pp2 is None
    assert isinstance(r_pp2, PreprocessResult)


# ------------------------------------------------------------------ #
#  Bonus: sanity-check that glasses-glare fixture still yields a
#         non-degenerate binary mask with lighting_ok.
# ------------------------------------------------------------------ #
def test_glasses_glare_mask_remains_valid() -> None:
    pipe = PreprocessPipeline(PreprocessConfig())
    pp = pipe.process(_load_eye_roi("glasses_glare.jpg"))
    assert pp.lighting_ok is True
    uniq = set(int(x) for x in np.unique(pp.mask))
    assert uniq.issubset({0, 255})
    # glare fixture must not collapse into all-zero mask
    assert _fg(pp.mask) > 20


# ------------------------------------------------------------------ #
#  Bonus: check the 3 lighting preset YAMLs load cleanly through the
#  real PACEConfig.from_yaml loader (prevents silent YAML schema drift).
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "preset",
    ["normal_light.yaml", "low_light.yaml", "bright_light.yaml"],
)
def test_lighting_preset_yamls_load_via_pace_config(preset: str) -> None:
    from app.config import PACEConfig

    path = Path(__file__).resolve().parent.parent / "configs" / "presets" / preset
    if not path.exists():
        pytest.skip(f"preset yaml missing: {path}")
    cfg = PACEConfig.from_yaml(str(path))
    assert isinstance(cfg.preprocess, PreprocessConfig)
    # preset-specific assertions
    if preset == "low_light.yaml":
        assert cfg.preprocess.threshold_mode == "adaptive"
        assert cfg.preprocess.min_valid_brightness < 15.0
    if preset == "bright_light.yaml":
        assert cfg.preprocess.threshold_mode == "otsu"
        assert cfg.preprocess.max_valid_brightness >= 250.0
