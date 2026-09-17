from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np


@dataclass(frozen=True)
class EyeROI:
    """Normalized eye Region-of-Interest crop.

    Attributes:
        image: Resized BGR uint8 numpy array of shape (H, W, 3), where
            (H, W) = config.eye_roi_output_size.
        original_bbox: (x1, y1, x2, y2) pixel coordinates in the original
            unresized frame from which this ROI was cropped (after margin
            expansion and frame clamping).
        scale_x: Horizontal scale factor applied during resize
            (= output_width / original_width). Phase 4+ uses this to map
            pupil detections back to original frame coordinates.
        scale_y: Vertical scale factor applied during resize
            (= output_height / original_height).
        iris_hint: Optional (x, y) normalized iris center in the *original
            full frame* (not the ROI), provided only by the MediaPipe
            backend (iris landmark indices 468-477). Haar backend sets None.
    """

    image: np.ndarray
    original_bbox: Tuple[int, int, int, int]
    scale_x: float
    scale_y: float
    iris_hint: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class DetectionResult:
    """Result of a single face + eye detection pass over one frame.

    This is the *only* interface Phase 3 (preprocessing) consumes — all
    MediaPipe / Haar implementation details are encapsulated here.

    Attributes:
        found: True iff a face and both usable eye ROIs were detected.
        confidence: Backend-specific confidence score in [0, 1].
            MediaPipe: 0.9 (fixed high, iris landmarks confirmed present).
            Haar: 0.6 (fixed moderate, no native score available).
            Not-found: 0.0.
        face_bbox: (x1, y1, x2, y2) pixel bounding box of the detected
            face, or None when found=False.
        left_eye: Normalized ROI crop for the subject's left eye (right
            side of the image), or None when not found.
        right_eye: Normalized ROI crop for the subject's right eye (left
            side of the image), or None when not found.
        backend_used: Literal "mediapipe" or "haar" — useful for metrics
            and for choosing per-backend thresholds downstream.
        frame_timestamp: time.monotonic() value captured at the start of
            the detect() call; lines up with FrameData.timestamp.
        consecutive_misses: Count of consecutive found=False results
            ending with this one (0 when found=True; increments while
            found=False). Phase 14 compares this against
            DetectionConfig.max_consecutive_misses to trigger user-facing
            "face lost" states.
    """

    found: bool
    confidence: float
    face_bbox: Optional[Tuple[int, int, int, int]]
    left_eye: Optional[EyeROI]
    right_eye: Optional[EyeROI]
    backend_used: str
    frame_timestamp: float
    consecutive_misses: int = 0


@dataclass(frozen=True)
class PreprocessResult:
    """Result of running the deterministic preprocessing pipeline on a
    single EyeROI.

    This is the *only* interface Phase 4 (pupil detection) and Phase 5
    (glint detection) consume.  Keep intermediate buffers (gray, blurred)
    because Phase 5's glint detector operates on the non-inverted
    grayscale image, NOT the pupil-oriented binary mask.

    Attributes:
        mask: Final binary pupil mask, uint8 shape (H, W), values in
            {0, 255}, with 255 = foreground (pupil candidate pixels)
            after morphological open/close cleanup.
        gray: Single-channel luminance image (H, W) uint8 BEFORE any
            blur or thresholding — used by Phase 5 glint detection.
        blurred: Gaussian-smoothed version of ``gray`` immediately prior
            to thresholding.  Useful debug panel; downstream code can
            re-read this instead of re-blurring internally.
        mean_brightness: Arithmetic mean of ``gray`` BEFORE blur, in
            [0, 255].  Lighting classification gate value.
        lighting_ok: True iff min_valid_brightness <= mean_brightness
            <= max_valid_brightness; False means Phase 14 should surface
            a "Lighting may be insufficient" warning to the user while
            downstream stages still attempt to detect.
        threshold_mode_used: Literal "otsu" | "adaptive" | "manual" —
            matches config.threshold_mode (useful for metrics and for
            Phase 13 debug GUI panels).
    """

    mask: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    mean_brightness: float
    lighting_ok: bool
    threshold_mode_used: str


@dataclass(frozen=True)
class PupilResult:
    """Result of running PupilDetector on a single PreprocessResult.mask.

    This is the *only* interface Phase 5 (glint detection, which needs
    center_x / center_y as the geometric anchor for its bright-blob
    proximity filter) and Phase 6 (pupil→glint gaze vector) consume.

    All coordinates are in the *resized ROI's* pixel space (matches the
    EyeROI.image dimensions).  Mapping back to the original full frame
    via EyeROI.scale_x / scale_y / original_bbox is the overlay-drawing
    GUI's job in Phase 13, NOT this module's.

    Attributes:
        found: True iff a pupil was detected (or plausibly predicted via
            Kalman fallback within kalman_max_prediction_frames).
        confidence: Real 0.0–1.0 score combining circularity, area
            typicality, ellipse/moments agreement, and a penalty when
            ``used_prediction_fallback`` is True.  Documented in detail
            in :meth:`~app.detection.pupil.PupilDetector.detect`.
        center_x: Sub-pixel pupil center X in ROI-local pixels.
        center_y: Sub-pixel pupil center Y in ROI-local pixels.
        radius: Effective pupil radius (= (major_axis + minor_axis) / 4,
            i.e. the mean of the two semi-axes of the fitted ellipse).
            Used by Phase 13 overlay drawing and as a sanity hint by
            Phase 5's glint proximity filter.
        ellipse_angle: Rotation angle in degrees of the fitted ellipse
            (as returned by ``cv2.fitEllipse``: 0 = vertical major axis,
            increasing counterclockwise in OpenCV's image-coordinate
            convention where Y points down).
        major_axis: Full major-axis length of the fitted ellipse (not
            the semi-axis) in ROI-local pixels.
        minor_axis: Full minor-axis length of the fitted ellipse in
            ROI-local pixels.
        used_prediction_fallback: True when the raw contour measurement
            this frame was rejected (implausible center jump OR no
            contour found at all) and the PupilDetector fell back to
            the Kalman filter's constant-velocity *prediction* (with
            an appropriate confidence penalty and max-frames guard).
        eye_side: Literal ``"left"`` or ``"right"`` — mirrors the
            ``eye_side`` argument passed into ``detect()`` so callers
            can disambiguate tuple results from ``detect_both_eyes``.
    """

    found: bool
    confidence: float
    center_x: float
    center_y: float
    radius: float
    ellipse_angle: float
    major_axis: float
    minor_axis: float
    used_prediction_fallback: bool
    eye_side: str


@dataclass(frozen=True)
class GlintResult:
    """Result of running GlintDetector on a single eye ROI.

    This is the *only* interface Phase 6 (pupil→glint gaze vector)
    consumes alongside :class:`PupilResult`.  All coordinates are in
    the *resized ROI's* pixel space — matches ``PupilResult`` exactly
    (both come from the same ``EyeROI``, so no scale mismatch exists).

    The ``reason`` field distinguishes *why* a detection failed so
    Phase 12's evaluation module can report meaningful statistics:

    * ``None`` – ``found=True``, no failure reason to report.
    * ``"no_pupil_anchor"`` – :class:`PupilResult.found` was False;
      the geometric-distance gate is meaningless without a pupil
      center to measure distance *from*, so we bail early.
    * ``"no_valid_candidate"`` – the brightness/area/circularity
      filters eliminated every bright blob before we even got to
      the geometric gate.
    * ``"rejected_too_far_from_pupil"`` – ≥1 candidate survived the
      first three filters but ALL failed the
      ``max_distance_from_pupil_px`` proximity check (the *key*
      §5 anti-false-positive rule, distinguishable from generic
      misses so the evaluation can separate "pipeline blind in this
      lighting" from "glasses glare correctly rejected").

    Attributes:
        found: True iff a glint was located inside the proximity band
            and with acceptable area/circularity.
        confidence: 0.0–1.0 multiplicative score (circularity · area
            typicality · inverse-normalised pupil-distance),
            documented in detail in
            :meth:`~app.detection.glint.GlintDetector.detect`.
        center_x: Sub-pixel glint center X in ROI-local pixels.
        center_y: Sub-pixel glint center Y in ROI-local pixels.
        area: Contour/keypoint-derived blob area in square ROI
            pixels.  Used by Phase 12 metrics to flag large "glint"
            candidates that are really smears.
        circularity: Computed value 4·π·A/P² in [0, 1].  1.0 = perfect
            circle.  Retained alongside confidence so downstream
            callers can re-tune the gate without re-running detection.
        distance_from_pupil: Euclidean distance in ROI-local pixels
            from ``(center_x, center_y)`` to
            ``(PupilResult.center_x, PupilResult.center_y)``.  When
            ``found=False`` this is 0.0; use ``reason`` to tell
            whether the distance even *mattered* for the decision.
        reason: Optional failure-reason tag listed above.  ``None``
            iff ``found=True``.
        eye_side: Literal ``"left"`` or ``"right"`` – propagated from
            :class:`PupilResult` so callers can disambiguate tuples
            from :meth:`detect_both_eyes`.
    """

    found: bool
    confidence: float
    center_x: float
    center_y: float
    area: float
    circularity: float
    distance_from_pupil: float
    reason: Optional[str]
    eye_side: str
