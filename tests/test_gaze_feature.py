from __future__ import annotations

import math

import pytest

from app.config import GazeFeatureConfig
from app.detection.data_types import GlintResult, PupilResult
from app.tracking.data_types import EyeVector, GazeFeature
from app.tracking.gaze_feature import GazeFeatureExtractor, _geometric_mean


def _make_pupil(
    *,
    found: bool = True,
    confidence: float = 0.9,
    center_x: float = 60.0,
    center_y: float = 40.0,
    radius: float = 15.0,
    eye_side: str = "left",
) -> PupilResult:
    return PupilResult(
        found=found,
        confidence=confidence,
        center_x=center_x,
        center_y=center_y,
        radius=radius,
        ellipse_angle=0.0,
        major_axis=radius * 2.0,
        minor_axis=radius * 2.0,
        used_prediction_fallback=False,
        eye_side=eye_side,
    )


def _make_glint(
    *,
    found: bool = True,
    confidence: float = 0.9,
    center_x: float = 65.0,
    center_y: float = 38.0,
    eye_side: str = "left",
) -> GlintResult:
    return GlintResult(
        found=found,
        confidence=confidence,
        center_x=center_x,
        center_y=center_y,
        area=4.0,
        circularity=0.9,
        distance_from_pupil=math.hypot(60.0 - center_x, 40.0 - center_y) if found else 0.0,
        reason=None if found else "no_pupil_anchor",
        eye_side=eye_side,
    )


class TestGeometricMean:
    def test_positive_values(self):
        assert _geometric_mean(0.64, 0.36) == pytest.approx(0.48)

    def test_zero_returns_zero(self):
        assert _geometric_mean(0.0, 0.9) == 0.0
        assert _geometric_mean(0.9, 0.0) == 0.0

    def test_negative_returns_zero(self):
        assert _geometric_mean(-0.1, 0.9) == 0.0


class TestNormalizesByPupilRadius:
    def test_same_ratio_different_scales(self):
        cfg = GazeFeatureConfig(
            eye_scale_source="pupil_radius",
            min_pupil_confidence=0.1,
            min_glint_confidence=0.1,
            fusion_mode="confidence_weighted",
            smoothing_window=1,
        )
        extractor = GazeFeatureExtractor(cfg)

        pupil_close = _make_pupil(center_x=60.0, center_y=40.0, radius=30.0)
        glint_close = _make_glint(center_x=75.0, center_y=25.0)

        pupil_far = _make_pupil(center_x=60.0, center_y=40.0, radius=15.0)
        glint_far = _make_glint(center_x=67.5, center_y=32.5)

        vec_close = extractor.compute_eye_vector(pupil_close, glint_close)
        vec_far = extractor.compute_eye_vector(pupil_far, glint_far)

        assert vec_close is not None
        assert vec_far is not None
        assert vec_close.dx == pytest.approx(vec_far.dx, abs=1e-6)
        assert vec_close.dy == pytest.approx(vec_far.dy, abs=1e-6)
        assert vec_close.dx == pytest.approx(-0.5, abs=1e-6)
        assert vec_close.dy == pytest.approx(0.5, abs=1e-6)


class TestMissingGlintReturnsNone:
    def test_glint_not_found(self):
        cfg = GazeFeatureConfig()
        extractor = GazeFeatureExtractor(cfg)
        pupil = _make_pupil()
        glint = _make_glint(found=False, confidence=0.0)
        result = extractor.compute_eye_vector(pupil, glint)
        assert result is None

    def test_pupil_not_found(self):
        cfg = GazeFeatureConfig()
        extractor = GazeFeatureExtractor(cfg)
        pupil = _make_pupil(found=False, confidence=0.0)
        glint = _make_glint()
        result = extractor.compute_eye_vector(pupil, glint)
        assert result is None


class TestLowConfidenceInputRejected:
    def test_low_pupil_confidence(self):
        cfg = GazeFeatureConfig(min_pupil_confidence=0.5, min_glint_confidence=0.1)
        extractor = GazeFeatureExtractor(cfg)
        pupil = _make_pupil(confidence=0.3)
        glint = _make_glint(confidence=0.9)
        assert extractor.compute_eye_vector(pupil, glint) is None

    def test_low_glint_confidence(self):
        cfg = GazeFeatureConfig(min_pupil_confidence=0.1, min_glint_confidence=0.5)
        extractor = GazeFeatureExtractor(cfg)
        pupil = _make_pupil(confidence=0.9)
        glint = _make_glint(confidence=0.3)
        assert extractor.compute_eye_vector(pupil, glint) is None

    def test_both_at_threshold_pass(self):
        cfg = GazeFeatureConfig(min_pupil_confidence=0.4, min_glint_confidence=0.4)
        extractor = GazeFeatureExtractor(cfg)
        pupil = _make_pupil(confidence=0.4)
        glint = _make_glint(confidence=0.4)
        result = extractor.compute_eye_vector(pupil, glint)
        assert result is not None


class TestFusionConfidenceWeighted:
    def test_weighted_sits_closer_to_high_confidence(self):
        cfg = GazeFeatureConfig(fusion_mode="confidence_weighted", smoothing_window=1)
        extractor = GazeFeatureExtractor(cfg)

        left = EyeVector(
            dx=0.0, dy=0.0, raw_dx=0.0, raw_dy=0.0,
            scale_used=15.0, confidence=0.9, eye_side="left",
        )
        right = EyeVector(
            dx=1.0, dy=1.0, raw_dx=15.0, raw_dy=15.0,
            scale_used=15.0, confidence=0.3, eye_side="right",
        )

        fused = extractor.fuse(left, right)
        assert fused is not None

        w_sum = 0.9 + 0.3
        expected_x = (0.0 * 0.9 + 1.0 * 0.3) / w_sum
        expected_y = (0.0 * 0.9 + 1.0 * 0.3) / w_sum
        assert fused.vector_x == pytest.approx(expected_x)
        assert fused.vector_y == pytest.approx(expected_y)

        plain_avg = 0.5
        assert abs(fused.vector_x - 0.0) < abs(plain_avg - 0.0)


class TestFusionSingleEyeFallback:
    def test_left_only(self):
        cfg = GazeFeatureConfig(fusion_mode="confidence_weighted", smoothing_window=1)
        extractor = GazeFeatureExtractor(cfg)
        left = EyeVector(
            dx=0.5, dy=-0.3, raw_dx=7.5, raw_dy=-4.5,
            scale_used=15.0, confidence=0.8, eye_side="left",
        )
        fused = extractor.fuse(left, None)
        assert fused is not None
        assert fused.vector_x == pytest.approx(0.5)
        assert fused.vector_y == pytest.approx(-0.3)
        assert fused.confidence == pytest.approx(0.8)
        assert fused.eyes_used == "left"

    def test_right_only(self):
        cfg = GazeFeatureConfig(fusion_mode="confidence_weighted", smoothing_window=1)
        extractor = GazeFeatureExtractor(cfg)
        right = EyeVector(
            dx=0.2, dy=0.7, raw_dx=3.0, raw_dy=10.5,
            scale_used=15.0, confidence=0.7, eye_side="right",
        )
        fused = extractor.fuse(None, right)
        assert fused is not None
        assert fused.vector_x == pytest.approx(0.2)
        assert fused.vector_y == pytest.approx(0.7)
        assert fused.eyes_used == "right"


class TestFusionBothEyesNone:
    def test_returns_none_no_raise(self):
        cfg = GazeFeatureConfig()
        extractor = GazeFeatureExtractor(cfg)
        result = extractor.fuse(None, None)
        assert result is None


class TestBestEyeFusionMode:
    def test_uses_higher_confidence_exactly(self):
        cfg = GazeFeatureConfig(fusion_mode="best_eye", smoothing_window=1)
        extractor = GazeFeatureExtractor(cfg)

        left = EyeVector(
            dx=1.0, dy=2.0, raw_dx=15.0, raw_dy=30.0,
            scale_used=15.0, confidence=0.4, eye_side="left",
        )
        right = EyeVector(
            dx=-0.5, dy=0.25, raw_dx=-7.5, raw_dy=3.75,
            scale_used=15.0, confidence=0.9, eye_side="right",
        )

        fused = extractor.fuse(left, right)
        assert fused is not None
        assert fused.vector_x == pytest.approx(-0.5)
        assert fused.vector_y == pytest.approx(0.25)
        assert fused.confidence == pytest.approx(0.9)

    def test_tie_prefers_left(self):
        cfg = GazeFeatureConfig(fusion_mode="best_eye", smoothing_window=1)
        extractor = GazeFeatureExtractor(cfg)
        left = EyeVector(
            dx=1.0, dy=0.0, raw_dx=15.0, raw_dy=0.0,
            scale_used=15.0, confidence=0.7, eye_side="left",
        )
        right = EyeVector(
            dx=0.0, dy=1.0, raw_dx=0.0, raw_dy=15.0,
            scale_used=15.0, confidence=0.7, eye_side="right",
        )
        fused = extractor.fuse(left, right)
        assert fused is not None
        assert fused.vector_x == pytest.approx(1.0)
        assert fused.vector_y == pytest.approx(0.0)


class TestSmoothingReducesNoise:
    def test_outlier_is_pulled_toward_stable_values(self):
        cfg = GazeFeatureConfig(smoothing_window=3, fusion_mode="average")
        extractor = GazeFeatureExtractor(cfg)

        left_pupil_stable = _make_pupil(center_x=60.0, center_y=40.0, radius=15.0, eye_side="left")
        left_glint_stable = _make_glint(center_x=60.0, center_y=40.0, eye_side="left")
        right_pupil_stable = _make_pupil(center_x=60.0, center_y=40.0, radius=15.0, eye_side="right")
        right_glint_stable = _make_glint(center_x=60.0, center_y=40.0, eye_side="right")

        f1 = extractor.compute(left_pupil_stable, left_glint_stable, right_pupil_stable, right_glint_stable, 1.0)
        f2 = extractor.compute(left_pupil_stable, left_glint_stable, right_pupil_stable, right_glint_stable, 2.0)

        left_pupil_out = _make_pupil(center_x=60.0, center_y=40.0, radius=15.0, eye_side="left")
        left_glint_out = _make_glint(center_x=30.0, center_y=10.0, eye_side="left")
        right_pupil_out = _make_pupil(center_x=60.0, center_y=40.0, radius=15.0, eye_side="right")
        right_glint_out = _make_glint(center_x=30.0, center_y=10.0, eye_side="right")

        f3 = extractor.compute(left_pupil_out, left_glint_out, right_pupil_out, right_glint_out, 3.0)

        assert f3 is not None
        raw_outlier_dx = 30.0 / 15.0
        raw_outlier_dy = 30.0 / 15.0
        assert abs(f3.vector_x) < abs(raw_outlier_dx) - 0.1
        assert abs(f3.vector_y) < abs(raw_outlier_dy) - 0.1


class TestResetClearsSmoothingHistory:
    def test_reset_removes_pre_reset_influence(self):
        cfg = GazeFeatureConfig(smoothing_window=3, fusion_mode="average")
        extractor_old = GazeFeatureExtractor(cfg)
        extractor_fresh = GazeFeatureExtractor(cfg)

        left_pupil_old = _make_pupil(center_x=60.0, center_y=40.0, radius=15.0, eye_side="left")
        left_glint_old = _make_glint(center_x=45.0, center_y=25.0, eye_side="left")
        right_pupil_old = _make_pupil(center_x=60.0, center_y=40.0, radius=15.0, eye_side="right")
        right_glint_old = _make_glint(center_x=45.0, center_y=25.0, eye_side="right")

        for t in (1.0, 2.0):
            extractor_old.compute(left_pupil_old, left_glint_old, right_pupil_old, right_glint_old, t)

        extractor_old.reset()

        left_pupil_new = _make_pupil(center_x=60.0, center_y=40.0, radius=15.0, eye_side="left")
        left_glint_new = _make_glint(center_x=75.0, center_y=55.0, eye_side="left")
        right_pupil_new = _make_pupil(center_x=60.0, center_y=40.0, radius=15.0, eye_side="right")
        right_glint_new = _make_glint(center_x=75.0, center_y=55.0, eye_side="right")

        result_old = extractor_old.compute(left_pupil_new, left_glint_new, right_pupil_new, right_glint_new, 5.0)
        result_fresh = extractor_fresh.compute(left_pupil_new, left_glint_new, right_pupil_new, right_glint_new, 5.0)

        assert result_old is not None
        assert result_fresh is not None
        assert result_old.vector_x == pytest.approx(result_fresh.vector_x, abs=1e-9)
        assert result_old.vector_y == pytest.approx(result_fresh.vector_y, abs=1e-9)


class TestConfigValidators:
    def test_bad_scale_source_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GazeFeatureConfig(eye_scale_source="iris_diameter")

    def test_bad_fusion_mode_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GazeFeatureConfig(fusion_mode="median")

    def test_smoothing_window_zero_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GazeFeatureConfig(smoothing_window=0)

    def test_fixed_eye_scale_non_positive_rejected(self):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            GazeFeatureConfig(eye_scale_source="fixed", fixed_eye_scale_px=0.0)


class TestComputeEndToEnd:
    def test_full_chain_returns_gaze_feature(self):
        cfg = GazeFeatureConfig(smoothing_window=1)
        extractor = GazeFeatureExtractor(cfg)

        lp = _make_pupil(eye_side="left", center_x=60.0, center_y=40.0, radius=15.0)
        lg = _make_glint(eye_side="left", center_x=67.5, center_y=32.5)
        rp = _make_pupil(eye_side="right", center_x=60.0, center_y=40.0, radius=15.0)
        rg = _make_glint(eye_side="right", center_x=67.5, center_y=32.5)

        feat = extractor.compute(lp, lg, rp, rg, 123.456)
        assert feat is not None
        assert isinstance(feat, GazeFeature)
        assert feat.timestamp == pytest.approx(123.456)
        assert feat.eyes_used == "both"
        assert feat.fusion_mode_used == "confidence_weighted"
        assert feat.vector_x == pytest.approx(-0.5)
        assert feat.vector_y == pytest.approx(0.5)

    def test_all_none_inputs(self):
        cfg = GazeFeatureConfig()
        extractor = GazeFeatureExtractor(cfg)
        result = extractor.compute(None, None, None, None, 1.0)
        assert result is None


class TestFixedScaleSource:
    def test_uses_configured_fixed_scale(self):
        cfg = GazeFeatureConfig(
            eye_scale_source="fixed",
            fixed_eye_scale_px=20.0,
            min_pupil_confidence=0.1,
            min_glint_confidence=0.1,
            smoothing_window=1,
        )
        extractor = GazeFeatureExtractor(cfg)
        pupil = _make_pupil(center_x=60.0, center_y=40.0, radius=999.0)
        glint = _make_glint(center_x=80.0, center_y=30.0)

        vec = extractor.compute_eye_vector(pupil, glint)
        assert vec is not None
        assert vec.scale_used == pytest.approx(20.0)
        assert vec.dx == pytest.approx(-1.0)
        assert vec.dy == pytest.approx(0.5)


class TestEpsilonGuardRadius:
    def test_zero_radius_uses_epsilon(self):
        cfg = GazeFeatureConfig(
            eye_scale_source="pupil_radius",
            min_pupil_confidence=0.1,
            min_glint_confidence=0.1,
            smoothing_window=1,
        )
        extractor = GazeFeatureExtractor(cfg)
        pupil = _make_pupil(center_x=60.0, center_y=40.0, radius=0.0)
        glint = _make_glint(center_x=61.0, center_y=40.0)

        vec = extractor.compute_eye_vector(pupil, glint)
        assert vec is not None
        assert vec.scale_used == pytest.approx(1e-3)
        assert vec.dx == pytest.approx(-1.0 / 1e-3)
