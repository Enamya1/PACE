from __future__ import annotations

import math
from collections import deque
from typing import Deque, Optional, Tuple

import numpy as np

from app.config import GazeFeatureConfig
from app.detection.data_types import GlintResult, PupilResult
from app.tracking.data_types import EyeVector, GazeFeature


def _geometric_mean(a: float, b: float) -> float:
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return math.sqrt(a * b)


class GazeFeatureExtractor:
    def __init__(self, config: GazeFeatureConfig) -> None:
        self._config = config
        self._smooth_buffer: Deque[Tuple[float, float]] = deque(maxlen=config.smoothing_window)

    def compute_eye_vector(
        self,
        pupil: PupilResult,
        glint: GlintResult,
    ) -> Optional[EyeVector]:
        cfg = self._config

        if not pupil.found or not glint.found:
            return None

        if pupil.confidence < cfg.min_pupil_confidence:
            return None
        if glint.confidence < cfg.min_glint_confidence:
            return None

        raw_dx = float(pupil.center_x - glint.center_x)
        raw_dy = float(pupil.center_y - glint.center_y)

        if cfg.eye_scale_source == "pupil_radius":
            scale = max(float(pupil.radius), 1e-3)
        else:
            scale = float(cfg.fixed_eye_scale_px)

        norm_dx = raw_dx / scale
        norm_dy = raw_dy / scale

        vector_confidence = _geometric_mean(pupil.confidence, glint.confidence)

        return EyeVector(
            dx=norm_dx,
            dy=norm_dy,
            raw_dx=raw_dx,
            raw_dy=raw_dy,
            scale_used=scale,
            confidence=vector_confidence,
            eye_side=pupil.eye_side,
        )

    def fuse(
        self,
        left: Optional[EyeVector],
        right: Optional[EyeVector],
    ) -> Optional[GazeFeature]:
        cfg = self._config

        if left is None and right is None:
            return None

        if left is not None and right is None:
            return GazeFeature(
                vector_x=left.dx,
                vector_y=left.dy,
                confidence=left.confidence,
                timestamp=0.0,
                fusion_mode_used=cfg.fusion_mode,
                eyes_used="left",
            )

        if left is None and right is not None:
            return GazeFeature(
                vector_x=right.dx,
                vector_y=right.dy,
                confidence=right.confidence,
                timestamp=0.0,
                fusion_mode_used=cfg.fusion_mode,
                eyes_used="right",
            )

        assert left is not None and right is not None
        mode = cfg.fusion_mode

        if mode == "best_eye":
            if left.confidence >= right.confidence:
                best = left
                used_side = "left"
            else:
                best = right
                used_side = "right"
            fused_dx = best.dx
            fused_dy = best.dy
            fused_conf = best.confidence
        elif mode == "average":
            fused_dx = (left.dx + right.dx) * 0.5
            fused_dy = (left.dy + right.dy) * 0.5
            fused_conf = (left.confidence + right.confidence) * 0.5
            used_side = "both"
        else:
            w_l = left.confidence
            w_r = right.confidence
            w_sum = w_l + w_r
            if w_sum <= 1e-9:
                fused_dx = (left.dx + right.dx) * 0.5
                fused_dy = (left.dy + right.dy) * 0.5
                fused_conf = 0.0
            else:
                fused_dx = (left.dx * w_l + right.dx * w_r) / w_sum
                fused_dy = (left.dy * w_l + right.dy * w_r) / w_sum
                fused_conf = max(left.confidence, right.confidence)
            used_side = "both"

        return GazeFeature(
            vector_x=fused_dx,
            vector_y=fused_dy,
            confidence=fused_conf,
            timestamp=0.0,
            fusion_mode_used=cfg.fusion_mode,
            eyes_used=used_side,
        )

    def _smooth(self, raw_x: float, raw_y: float) -> Tuple[float, float]:
        self._smooth_buffer.append((raw_x, raw_y))
        n = len(self._smooth_buffer)
        if n == 0:
            return raw_x, raw_y
        xs = [p[0] for p in self._smooth_buffer]
        ys = [p[1] for p in self._smooth_buffer]
        return (sum(xs) / n, sum(ys) / n)

    def compute(
        self,
        left_pupil: Optional[PupilResult],
        left_glint: Optional[GlintResult],
        right_pupil: Optional[PupilResult],
        right_glint: Optional[GlintResult],
        timestamp: float,
    ) -> Optional[GazeFeature]:
        left_vec: Optional[EyeVector] = None
        if left_pupil is not None and left_glint is not None:
            left_vec = self.compute_eye_vector(left_pupil, left_glint)

        right_vec: Optional[EyeVector] = None
        if right_pupil is not None and right_glint is not None:
            right_vec = self.compute_eye_vector(right_pupil, right_glint)

        fused = self.fuse(left_vec, right_vec)
        if fused is None:
            return None

        smoothed_x, smoothed_y = self._smooth(fused.vector_x, fused.vector_y)

        return GazeFeature(
            vector_x=smoothed_x,
            vector_y=smoothed_y,
            confidence=fused.confidence,
            timestamp=float(timestamp),
            fusion_mode_used=fused.fusion_mode_used,
            eyes_used=fused.eyes_used,
        )

    def reset(self) -> None:
        self._smooth_buffer.clear()
