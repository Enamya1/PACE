from typing import Tuple, Optional

import cv2
import numpy as np

from app.config import PreprocessConfig
from app.detection.data_types import DetectionResult, EyeROI, PreprocessResult


class PreprocessPipeline:
    """Deterministic, config-driven image preprocessing for eye ROIs.

    Pipeline (executed by :meth:`process` in this exact order):

        1. BGR -> grayscale (skip if input is already single-channel)
        2. Lighting assessment  (mean gray, compare to [min, max] valid)
        3. GaussianBlur         (odd kernel size + sigma from config)
        4. Threshold dispatch   (otsu | adaptive_gaussian | manual)
              * output is always ``THRESH_BINARY_INV`` so dark pupil
                pixels become foreground (255) — this is what Phase 4's
                contour detection expects.
        5. Morphological open   (remove small noise specks)
        6. Morphological close  (fill small holes in pupil blob)

    **Only** :class:`~app.detection.data_types.PreprocessResult` leaves
    this module.  Phase 4 (pupil contours) and Phase 5 (glint detection)
    read directly from ``.mask`` and ``.gray`` respectively.
    """

    def __init__(self, config: PreprocessConfig) -> None:
        # All value validation is performed by pydantic's PreprocessConfig:
        #   * threshold_mode ∈ {"otsu", "adaptive", "manual"}
        #   * blur_kernel_size odd & >= 3
        #   * adaptive_block_size odd & >= 3
        #   * morph_kernel_size odd & >= 1
        #   * max_valid_brightness > min_valid_brightness
        self._config = config

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def process(self, eye_roi: EyeROI) -> PreprocessResult:
        image = eye_roi.image
        cfg = self._config

        # --- (a) Grayscale ---
        if image.ndim == 2 or (image.ndim == 3 and image.shape[2] == 1):
            gray = np.ascontiguousarray(
                image if image.ndim == 2 else image[:, :, 0], dtype=np.uint8
            )
        else:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # --- (b) Lighting assessment BEFORE any blur/threshold ---
        mean_brightness = float(np.mean(gray))
        lighting_ok = (
            cfg.min_valid_brightness <= mean_brightness <= cfg.max_valid_brightness
        )

        # --- (c) Gaussian blur ---
        k = cfg.blur_kernel_size
        blurred = cv2.GaussianBlur(gray, (k, k), cfg.blur_sigma)

        # --- (d) Threshold ---
        mode = cfg.threshold_mode
        if mode == "otsu":
            _, mask = cv2.threshold(
                blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        elif mode == "adaptive":
            mask = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                cfg.adaptive_block_size,
                cfg.adaptive_c,
            )
        elif mode == "manual":
            _, mask = cv2.threshold(
                blurred,
                cfg.manual_threshold_value,
                255,
                cv2.THRESH_BINARY_INV,
            )
        else:  # pragma: no cover - prevented by Literal[pydantic] + init
            # Left here purely as a belt-and-braces guard in case someone
            # constructs the class bypassing Pydantic.
            raise ValueError(
                f"Unknown threshold_mode={mode!r}. Expected one of "
                "'otsu', 'adaptive', 'manual'."
            )

        # --- (e) Morphological cleanup ---
        mk = cfg.morph_kernel_size
        if mk >= 3:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (mk, mk)
            )
        else:
            # kernel of size 1 is a no-op; still construct it so iteration
            # counts of 0 + non-trivial kernels behave identically.
            kernel = None

        if kernel is not None and cfg.morph_open_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_OPEN,
                kernel,
                iterations=cfg.morph_open_iterations,
            )
        if kernel is not None and cfg.morph_close_iterations > 0:
            mask = cv2.morphologyEx(
                mask,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=cfg.morph_close_iterations,
            )

        return PreprocessResult(
            mask=mask,
            gray=gray,
            blurred=blurred,
            mean_brightness=mean_brightness,
            lighting_ok=lighting_ok,
            threshold_mode_used=mode,
        )

    def process_both_eyes(
        self, detection_result: DetectionResult
    ) -> Tuple[Optional[PreprocessResult], Optional[PreprocessResult]]:
        """Convenience wrapper used by Phase 4.

        Returns:
            (left_preprocess, right_preprocess).  Either (or both) may be
            ``None`` when the corresponding EyeROI was missing on the
            input DetectionResult.
        """
        left_pp: Optional[PreprocessResult] = None
        right_pp: Optional[PreprocessResult] = None
        if detection_result.left_eye is not None:
            left_pp = self.process(detection_result.left_eye)
        if detection_result.right_eye is not None:
            right_pp = self.process(detection_result.right_eye)
        return left_pp, right_pp
