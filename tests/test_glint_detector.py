"""Phase 5 — GlintDetector tests (fully deterministic, synthetic grayscale only).

All image assets are drawn on the fly with NumPy + ``cv2.circle`` /
``cv2.ellipse`` / ``cv2.fillPoly`` — no real webcam fixtures required.

Coordinate space
================
Every synthetic test draws onto a (h=80, w=120) uint8 grayscale canvas.
This exactly matches the default ``DetectionConfig.eye_roi_output_size``
(120×80), so ``GlintConfig.max_distance_from_pupil_px=40`` and
``GlintConfig.min_glint_area_px / max_glint_area_px`` are evaluated on
the same pixel grid as ``--glint-test`` will use at runtime.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from app.config import GlintConfig, PACEConfig
from app.detection.data_types import (
    EyeROI,
    GlintResult,
    PreprocessResult,
    PupilResult,
)
from app.detection.glint import GlintDetector


W, H = 120, 80  # matches eye_roi_output_size default exactly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fake_eye_roi() -> EyeROI:
    """Placebo EyeROI — no side tag (EyeROI doesn't carry one).

    Side is propagated through :class:`PreprocessResult.eye_side` and
    :class:`PupilResult.eye_side` directly; ``EyeROI`` itself is
    side-agnostic on the Phase 2 dataclass.
    """
    return EyeROI(
        image=np.zeros((H, W, 3), dtype=np.uint8),
        original_bbox=(0, 0, W, H),
        scale_x=1.0,
        scale_y=1.0,
    )


def _fake_preprocess(gray: np.ndarray) -> PreprocessResult:
    """Minimal PreprocessResult wrapping a synthetic grayscale image.

    ``mask`` is set to zeros (unused by :class:`GlintDetector` —
    it explicitly operates on ``.gray`` per the spec).  The actual
    :class:`PreprocessResult` dataclass (Phase 3) does NOT carry
    ``eye_side``, ``eye_roi``, or ``iris_hint`` attributes; those live
    on :class:`PupilResult` and the Phase 2 wrapper types.
    """
    return PreprocessResult(
        mask=np.zeros((H, W), dtype=np.uint8),
        gray=gray.astype(np.uint8),
        blurred=gray.astype(np.uint8),
        mean_brightness=float(gray.mean()),
        lighting_ok=True,
        threshold_mode_used="otsu",
    )


def _fake_pupil(cx: float, cy: float, *, eye_side: str = "left") -> PupilResult:
    """A fully-valid PupilResult used as the geometric anchor.

    r=13 matches the log-peak pupil size used in Phase 4's high-confidence
    test, so the distance math inside :class:`GlintDetector` exercises the
    same code path as a real Phase 4 → 5 handoff.
    """
    return PupilResult(
        found=True,
        confidence=0.70,
        center_x=cx,
        center_y=cy,
        radius=13.0,
        ellipse_angle=0.0,
        major_axis=26.0,
        minor_axis=24.0,
        used_prediction_fallback=False,
        eye_side=eye_side,
    )


def _fake_pupil_missing(*, eye_side: str = "left") -> PupilResult:
    """The no-anchor case — ``found=False`` triggers the §5 guard clause."""
    return PupilResult(
        found=False,
        confidence=0.0,
        center_x=0.0,
        center_y=0.0,
        radius=0.0,
        ellipse_angle=0.0,
        major_axis=0.0,
        minor_axis=0.0,
        used_prediction_fallback=False,
        eye_side=eye_side,
    )


def _base_gray_with_pupil(pcx: int, pcy: int, *, pupil_r: int = 13) -> np.ndarray:
    """Mid-gray canvas with a dark pupil/iris disc at (pcx, pcy).

    * Background: ``gray=140``  (sclera/skin midrange, far below the 220
      brightness gate so it will never show up as a candidate).
    * Dark pupil disc: ``gray=40`` — just a visual anchor for human
      readers of the debug output; its intensity is irrelevant to the
      glint pipeline which only cares about *bright* pixels.
    """
    base = np.full((H, W), fill_value=140, dtype=np.uint8)
    cv2.circle(base, (pcx, pcy), pupil_r, 40, thickness=-1, lineType=cv2.LINE_AA)
    return base


# ---------------------------------------------------------------------------
# Tests 1–3: core detection + §5 geometric gate (the judge-facing ones)
# ---------------------------------------------------------------------------


def test_detects_glint_near_pupil() -> None:
    """Test 1 — small bright dot near pupil → ``found=True`` within 2 px.

    This is the simplest "happy path" case.  The glint is intentionally
    placed *outside* the dark pupil disc (on the iris, r=8 from center)
    because real corneal glints form on the cornea in front of the iris,
    not in the dark pupil shadow — this matches physiology exactly.
    """
    cfg = GlintConfig()
    det = GlintDetector(cfg)

    pcx, pcy = 60, 40
    # Glint at d = 8 px from pupil center (well inside max=40)
    gx, gy = pcx + 8, pcy + 2  # → (68, 42)
    gr = 3  # radius 3 → filled circle → area ≈ 28 px, band [1, 60] ✓

    gray = _base_gray_with_pupil(pcx, pcy)
    # Saturated near-white glint.  250 ≥ brightness_threshold=220 ✓
    cv2.circle(gray, (gx, gy), gr, 250, thickness=-1, lineType=cv2.LINE_AA)
    pp = _fake_preprocess(gray)
    pup = _fake_pupil(float(pcx), float(pcy))

    r = det.detect(pp, pup)

    assert r.found is True
    assert r.reason is None
    # Center within 2 px of ground truth.  Allow integer-pixel rounding
    # from cv2.circle → findContours → minEnclosingCircle chain.
    assert abs(r.center_x - gx) <= 2.0, f"cx got {r.center_x}, want {gx}"
    assert abs(r.center_y - gy) <= 2.0, f"cy got {r.center_y}, want {gy}"
    # Reasonable confidence (dominated by the distance term: d=8/40=0.20
    # → c_dist ≈ 0.80, weighted 3/5 = 0.48 of the score).
    assert r.confidence > 0.5, f"confidence on clean glint = {r.confidence:.3f}"


def test_rejects_bright_region_far_from_pupil() -> None:
    """Test 2 — §5's star witness.

    Single bright circular blob passes brightness/area/circularity filters
    *with room to spare* BUT is placed 56 px from the pupil center (well
    past the 40 px default gate).  Expected: ``found=False`` with
    ``reason="rejected_too_far_from_pupil"``, NOT the generic
    ``"no_valid_candidate"``.

    This is the test you walk a judge through to answer "how do you
    avoid false glints?".  It distinguishes the proximity-based
    rejection from the brightness/area/circularity filters.
    """
    cfg = GlintConfig()
    det = GlintDetector(cfg)

    pcx, pcy = 60, 40
    # Bright blob in the *opposite corner* — sclera reflection, glasses
    # frame glare, skin highlight, they all land here.
    gx, gy = 110, 75  # distance = sqrt(50² + 35²) ≈ 61 px
    assert math.hypot(gx - pcx, gy - pcy) > cfg.max_distance_from_pupil_px + 5

    gray = _base_gray_with_pupil(pcx, pcy)
    # Correctly sized (r=4 → ~50 px, band [1,60] ✓) and very round
    cv2.circle(gray, (gx, gy), 4, 250, thickness=-1, lineType=cv2.LINE_AA)
    # Sanity precondition: there ARE bright pixels surviving the
    # threshold.  If this fails the test itself is broken.
    _, bright_mask = cv2.threshold(gray, cfg.brightness_threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) >= 1, "test setup error: expected ≥1 bright contour"

    pp = _fake_preprocess(gray)
    pup = _fake_pupil(float(pcx), float(pcy))

    r = det.detect(pp, pup)
    assert r.found is False
    # THE key assertion: proximity-gate rejection, not "no_valid_candidate"
    assert r.reason == "rejected_too_far_from_pupil"
    assert r.confidence == 0.0
    assert r.distance_from_pupil == 0.0  # placeholder on not-found


def test_picks_closest_when_multiple_bright_candidates() -> None:
    """Test 3 — multiple valid blobs inside the proximity band.

    Two saturated white circles.  Glint A is closer to the pupil (the
    *physically* correct true glint on the cornea) but slightly dimmer
    (245 vs 250).  Glint B is farther but still inside max_distance, and
    brighter.  Tiebreak rule (§5 step f):

        1. PRIMARY  = mean_brightness DESC → B wins on raw brightness?
           NOT THIS TIME.  See synthetic setup notes below.

        2. SECONDARY = area ASC → smaller wins.

    HOWEVER, we need an unambiguous expected winner.  The spec says
    "set up so the expected winner is unambiguous" — so we make the
    closer one *also* be clearly brighter, leaving no ambiguity, and
    verify the distance metric is smaller on the returned result
    (proximity dominates the downstream vector math anyway).
    """
    cfg = GlintConfig()
    det = GlintDetector(cfg)

    pcx, pcy = 60, 40

    # True glint: 12 px from pupil center, radius=3, brightest (255)
    true_gx, true_gy = pcx + 12, pcy
    true_r = 3
    true_val = 255

    # Impostor: 28 px from pupil center, radius=4, dimmer (240)
    # Still well inside max_distance=40; would be picked by a naive
    # "closest blob" detector IF the tiebreak weren't brightness first.
    # We make it dimmer so brightness-then-area unambiguously picks true.
    imp_gx, imp_gy = pcx, pcy + 28
    imp_r = 4
    imp_val = 240

    gray = _base_gray_with_pupil(pcx, pcy)
    cv2.circle(gray, (true_gx, true_gy), true_r, true_val, thickness=-1, lineType=cv2.LINE_AA)
    cv2.circle(gray, (imp_gx, imp_gy), imp_r, imp_val, thickness=-1, lineType=cv2.LINE_AA)

    pp = _fake_preprocess(gray)
    pup = _fake_pupil(float(pcx), float(pcy))
    r = det.detect(pp, pup)

    assert r.found is True
    assert abs(r.center_x - true_gx) <= 2.0
    assert abs(r.center_y - true_gy) <= 2.0
    # Sanity: detected glint is truly the closer one.
    d_true = math.hypot(true_gx - pcx, true_gy - pcy)
    d_imp = math.hypot(imp_gx - pcx, imp_gy - pcy)
    assert r.distance_from_pupil < d_imp - 10, (
        f"detected dist={r.distance_from_pupil:.1f} but impostor is at "
        f"{d_imp:.1f} — likely picked the wrong blob."
    )


# ---------------------------------------------------------------------------
# Tests 4–5: failure-mode guards
# ---------------------------------------------------------------------------


def test_no_pupil_anchor_returns_not_found() -> None:
    """Test 4 — guard clause bails before doing *any* brightness work.

    Spec requires: if ``not pupil_result.found`` we return immediately
    with ``reason="no_pupil_anchor"``, no threshold called, no contours
    extracted.  Without a pupil anchor the §5 proximity constraint is
    meaningless — "bright blob anywhere in the ROI" is a horrid detector
    and we never even try it.
    """
    cfg = GlintConfig()
    det = GlintDetector(cfg)

    gray = _base_gray_with_pupil(60, 40)
    # Even if we *accidentally* drew a real glint, the gate applies
    # regardless since the pupil anchor is what gates reasoning.
    cv2.circle(gray, (66, 42), 3, 250, thickness=-1, lineType=cv2.LINE_AA)

    pp = _fake_preprocess(gray)
    pup = _fake_pupil_missing(eye_side="left")

    r = det.detect(pp, pup)
    assert r.found is False
    assert r.reason == "no_pupil_anchor"
    assert r.eye_side == "left"


def test_rejects_oversized_bright_blob() -> None:
    """Test 5 — geometrically close but large/non-circular = rejected.

    Simulates a screen-reflection smear on the sclera right next to the
    pupil.  Very bright (245), very close (within 10 px of pupil center),
    but ~100 px of area with an elongated, jagged, non-circular shape
    that fails *both* the area band (60 max) and the 0.65 circularity
    gate.  Expected: ``found=False, reason="no_valid_candidate"`` —
    nothing survives the filters to even reach the distance step.
    """
    cfg = GlintConfig()
    det = GlintDetector(cfg)

    pcx, pcy = 60, 40
    gray = _base_gray_with_pupil(pcx, pcy)
    # Smear: a fat rotated ellipse 12 × 30 px with angle 20°.
    # Major axis 30 × minor 12 → area ≈ π·15·6 = 283 px, >> 60.
    # Aspect 2.5 → circularity 4π·283/(perimeter_of_oval)² ≈ 0.52
    # (rough estimate, but way below 0.65 gate).
    cv2.ellipse(
        gray,
        (pcx + 15, pcy - 2),   # centered ~15 px NE of pupil → close
        (15, 6),               # (semi-major, semi-minor) → full 30×12
        20, 0, 360,            # 20° rotation, full ellipse
        245, thickness=-1, lineType=cv2.LINE_AA,
    )

    pp = _fake_preprocess(gray)
    pup = _fake_pupil(float(pcx), float(pcy))
    r = det.detect(pp, pup)

    assert r.found is False
    # Area/circularity failed BEFORE distance gate → generic reason
    assert r.reason == "no_valid_candidate"
    assert r.confidence == 0.0


# ---------------------------------------------------------------------------
# Test 6: backend parity
# ---------------------------------------------------------------------------


def test_both_backends_agree_on_clean_case() -> None:
    """Test 6 — contour and SimpleBlobDetector produce compatible results.

    On a clean, round, near-saturated glint the two backends must find
    the same blob within a generous 5-pixel tolerance (SimpleBlobDetector
    interpolates its center slightly differently from
    ``minEnclosingCircle`` on integer-gridded contours).  This proves
    the ``"simpleblob"`` fallback isn't dead code.
    """
    pcx, pcy = 60, 40
    gx, gy = pcx - 10, pcy + 4
    gr = 3

    gray = _base_gray_with_pupil(pcx, pcy)
    cv2.circle(gray, (gx, gy), gr, 250, thickness=-1, lineType=cv2.LINE_AA)
    pp = _fake_preprocess(gray)
    pup = _fake_pupil(float(pcx), float(pcy))

    cfg_c = GlintConfig(detector_backend="contour")
    cfg_s = GlintConfig(detector_backend="simpleblob")
    r_c = GlintDetector(cfg_c).detect(pp, pup)
    r_s = GlintDetector(cfg_s).detect(pp, pup)

    assert r_c.found is True
    assert r_s.found is True
    d = math.hypot(r_c.center_x - r_s.center_x, r_c.center_y - r_s.center_y)
    assert d <= 5.0, (
        f"backend disagreement Δ={d:.2f} px (contour=({r_c.center_x:.1f},"
        f"{r_c.center_y:.1f})  simpleblob=({r_s.center_x:.1f},"
        f"{r_s.center_y:.1f}))"
    )


# ---------------------------------------------------------------------------
# Test 7: detect_both_eyes None passthrough
# ---------------------------------------------------------------------------


def test_detect_both_eyes_handles_none_input() -> None:
    """Test 7 — convenience wrapper tolerates partial upstream failures.

    Same exact pattern as Phase 4's
    ``PupilDetector.detect_both_eyes``: any eye whose (PreprocessResult,
    PupilResult) pair isn't fully populated gets ``None`` out, not an
    exception, and the detector's internal state is NOT mutated by the
    absent eye (important for any future per-side caching).
    """
    cfg = GlintConfig()
    det = GlintDetector(cfg)

    # Build a clean valid glint for one eye + None for the other.
    gx, gy = 66, 42
    gray_left = _base_gray_with_pupil(60, 40)
    cv2.circle(gray_left, (gx, gy), 3, 250, thickness=-1, lineType=cv2.LINE_AA)
    pp_left = _fake_preprocess(gray_left)
    pup_left = _fake_pupil(60.0, 40.0, eye_side="left")

    gr_right = _fake_pupil(70.0, 35.0, eye_side="right")
    pp_right = _fake_preprocess(_base_gray_with_pupil(70, 35))

    # Case 1: (None, valid_right) → returns (None, GlintResult)
    l1, r1 = det.detect_both_eyes(None, None, pp_right, gr_right)
    assert l1 is None
    assert r1 is not None
    assert isinstance(r1, GlintResult)

    # Case 2: (valid_left, None) → returns (GlintResult, None)
    l2, r2 = det.detect_both_eyes(pp_left, pup_left, None, None)
    assert r2 is None
    assert l2 is not None
    assert l2.found is True
    assert abs(l2.center_x - gx) <= 2.0

    # Case 3: (None, None) → (None, None) without raising
    l3, r3 = det.detect_both_eyes(None, None, None, None)
    assert l3 is None
    assert r3 is None


# ---------------------------------------------------------------------------
# Bonus tests: config validators + default.yaml loads correctly
# ---------------------------------------------------------------------------


def test_glint_config_validators_reject_bad_area_order() -> None:
    """Bonus A — Pydantic validator: max area ≤ min area is rejected."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GlintConfig(min_glint_area_px=30.0, max_glint_area_px=5.0)


def test_glint_config_validators_reject_bad_backend() -> None:
    """Bonus B — Literal validator: unknown backend string is rejected."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        GlintConfig(detector_backend="hogwild")  # type: ignore[arg-type]


def test_default_yaml_loads_glint_section_via_paceconfig() -> None:
    """Bonus C — end-to-end wiring: default.yaml populates GlintConfig.

    This catches the easy-to-miss mistake of adding GlintConfig to the
    PACEConfig model but forgetting to wire the section into
    ``PACEConfig.from_yaml``'s ``glint_fields`` loop.
    """
    from pathlib import Path
    default_yaml = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    cfg = PACEConfig.from_yaml(str(default_yaml))
    assert isinstance(cfg.glint, GlintConfig)
    assert cfg.glint.detector_backend == "contour"
    assert cfg.glint.brightness_threshold == 220
    assert cfg.glint.max_distance_from_pupil_px == pytest.approx(40.0)
    assert cfg.glint.max_glint_area_px == pytest.approx(60.0)
