"""Typed data carriers for the gaze-feature computation pipeline.

Phase 6 does not own any image buffers — its inputs are the *numeric*
detection results from Phase 4/5 and its outputs are the *numeric*
gaze features consumed by Phase 7 (calibration) and Phase 8 (gaze →
screen mapping).  Keeping everything in separate ``@dataclass(frozen=True)``
objects means calibration and cursor-smoothing can be re-tuned or
replaced without re-running the OpenCV detection stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EyeVector:
    """Per-eye pupil→glint vector before fusion, raw AND normalized.

    Both the normalised (``dx``/``dy``) and the raw-pixel
    (``raw_dx``/``raw_dy``) values are preserved so downstream stages
    have a choice:

    * **Phase 7/8 (calibration + gaze → screen mapping):** ALWAYS use
      ``dx``/``dy`` — the normalised form is what cancels camera-distance
      head movements per §12 of the project roadmap.
    * **Phase 13 GUI overlay debug panels + Phase 12 error analysis:**
      ``raw_dx``/``raw_dy``/``scale_used`` are useful to show the user
      *why* a vector was scaled a particular way and to check if
      ``PupilResult.radius`` (the default scale source) is drifting
      because of a partial eyelid occlusion.

    Attributes:
        dx: Normalized (scale-invariant) horizontal component of the
            pupil→glint vector, in units of *pupil-radii*.  0.0 = glint
            and pupil share the same X coordinate; +1.0 = glint is
            exactly one pupil-radius to the left of the pupil center.
        dy: Normalized (scale-invariant) vertical component.  Sign
            follows image-space convention (+Y = down on screen).
        raw_dx: Un-normalized horizontal component in the original
            resized-ROI pixel space (match Phase 4/5's coordinate frame).
        raw_dy: Un-normalized vertical component in ROI pixels.
        scale_used: The scale denominator actually applied —
            ``PupilResult.radius`` or ``GazeFeatureConfig.fixed_eye_scale_px``
            depending on ``eye_scale_source``.  Epsilon-guarded ≥ 1e-3
            so division-by-zero is impossible.
        confidence: Per-eye vector confidence, derived as the geometric
            mean of ``PupilResult.confidence`` and
            ``GlintResult.confidence``.  ``sqrt(pup·gnt)`` = the product
            is the honest combined score and the square root returns it
            to the same 0.0–1.0 band the input confidences live in.
        eye_side: ``"left"`` or ``"right"`` — needed so :meth:`fuse`
            can populate ``GazeFeature.eyes_used`` correctly and so the
            Phase 13 debug panel can colour each eye separately.
    """

    dx: float
    dy: float
    raw_dx: float
    raw_dy: float
    scale_used: float
    confidence: float
    eye_side: str


@dataclass(frozen=True)
class GazeFeature:
    """Fused, smoothed, single gaze feature per video frame.

    This is the *only* object Phase 7 (calibration) and Phase 8
    (gaze→screen mapping) consume — they do NOT see individual
    ``EyeVector`` objects or the Phase 4/5 detection outputs.

    Attributes:
        vector_x: Fused+smoothed horizontal gaze feature.  Same
            "pupil-radius units" as ``EyeVector.dx``.  This value is
            *approximately* proportional to the saccade angle — a 20%
            head-distance shift leaves it unchanged if gaze direction
            is held constant (that is the whole point of §12).
        vector_y: Fused+smoothed vertical gaze feature.
        confidence: Combined fused-feature confidence in [0, 1].  0.0
            → one of the two-eye inputs was terrible; ≥ 0.75 → both
            eyes agreed and were well above the confidence gate.
        timestamp: ``time.monotonic()`` reading captured inside
            :meth:`GazeFeatureExtractor.compute`.  Used by Phase 9's
            One-Euro filter to convert frame-integer indices to real
            time.
        fusion_mode_used: Literal string copy of
            ``GazeFeatureConfig.fusion_mode`` the instant this feature
            was produced.  Allows Phase 12 to detect config changes
            mid-experiment.
        eyes_used: Literal ``"left"`` / ``"right"`` / ``"both"``.  The
            degradation path (1 of 2 eyes lost) is an intentional
            feature, not a bug — the evaluation module compares
            "both eyes" vs "only left/right" detection rates as a
            robustness metric.
    """

    vector_x: float
    vector_y: float
    confidence: float
    timestamp: float
    fusion_mode_used: str
    eyes_used: str
