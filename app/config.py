from pathlib import Path
from typing import Tuple, Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class CameraConfig(BaseModel):
    camera_index: int = Field(default=0, ge=0)
    resolution_width: int = Field(default=1280, ge=320, le=4096)
    resolution_height: int = Field(default=720, ge=240, le=2160)
    fps: int = Field(default=30, ge=1, le=120)
    frame_width: int = Field(default=640, ge=320, le=4096)
    frame_height: int = Field(default=480, ge=240, le=2160)
    target_fps: int = Field(default=30, ge=1, le=120)
    max_consecutive_failures: int = Field(default=15, ge=1, le=1000)
    backend: Literal["auto", "dshow", "avfoundation", "v4l2"] = "auto"

    @property
    def resolution(self) -> Tuple[int, int]:
        return (self.resolution_width, self.resolution_height)


class TrackingConfig(BaseModel):
    jitter_smoothing: float = Field(default=0.6, ge=0.0, le=1.0)
    sensitivity: float = Field(default=1.0, gt=0.0, le=10.0)
    cursor_delay_ms: int = Field(default=0, ge=0, le=500)


class BlinkConfig(BaseModel):
    blink_threshold: float = Field(default=0.25, gt=0.0, lt=1.0)
    blink_click_count: int = Field(default=2, ge=1, le=10)


class DetectionConfig(BaseModel):
    backend: Literal["mediapipe", "haar"] = "mediapipe"
    min_detection_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    min_tracking_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    eye_roi_margin_px: int = Field(default=8, ge=0, le=200)
    eye_roi_output_size: Tuple[int, int] = Field(default=(120, 80))
    max_consecutive_misses: int = Field(default=10, ge=1, le=1000)


ThresholdMode = Literal["otsu", "adaptive", "manual"]


class PreprocessConfig(BaseModel):
    blur_kernel_size: int = Field(default=5, ge=3)
    blur_sigma: float = Field(default=0.0, ge=0.0)
    threshold_mode: ThresholdMode = "otsu"
    manual_threshold_value: int = Field(default=60, ge=0, le=255)
    adaptive_block_size: int = Field(default=25, ge=3)
    adaptive_c: int = Field(default=5)
    morph_kernel_size: int = Field(default=3, ge=1)
    morph_open_iterations: int = Field(default=1, ge=0)
    morph_close_iterations: int = Field(default=2, ge=0)
    min_valid_brightness: float = Field(default=15.0, ge=0.0, le=255.0)
    max_valid_brightness: float = Field(default=240.0, ge=0.0, le=255.0)

    @field_validator("blur_kernel_size", "adaptive_block_size", "morph_kernel_size")
    @classmethod
    def _ensure_odd_kernel(cls, v: int) -> int:
        if v % 2 == 0:
            raise ValueError("Kernel/block size must be odd (OpenCV GaussianBlur, adaptiveThreshold, and MORPH_ELLIPSE all require odd dimensions for symmetry).")
        return v

    @field_validator("max_valid_brightness")
    @classmethod
    def _brightness_range_valid(cls, v: float, info) -> float:
        min_v = info.data.get("min_valid_brightness")
        if isinstance(min_v, (int, float)) and v <= float(min_v):
            raise ValueError("max_valid_brightness must be greater than min_valid_brightness.")
        return v


class PupilConfig(BaseModel):
    min_pupil_area_ratio: float = Field(default=0.01, gt=0.0, le=1.0)
    max_pupil_area_ratio: float = Field(default=0.35, gt=0.0, le=1.0)
    min_circularity: float = Field(default=0.55, ge=0.0, le=1.0)
    use_kalman_tracking: bool = True
    kalman_process_noise: float = Field(default=1e-2, gt=0.0)
    kalman_measurement_noise: float = Field(default=1e-1, gt=0.0)
    max_center_jump_px: float = Field(default=25.0, gt=0.0)
    min_confidence_to_accept: float = Field(default=0.4, ge=0.0, le=1.0)
    kalman_max_prediction_frames: int = Field(default=5, ge=1, le=120)

    @field_validator("max_pupil_area_ratio")
    @classmethod
    def _area_ratio_order(cls, v: float, info) -> float:
        min_v = info.data.get("min_pupil_area_ratio")
        if isinstance(min_v, (int, float)) and v <= float(min_v):
            raise ValueError("max_pupil_area_ratio must be > min_pupil_area_ratio.")
        return v


GlintBackend = Literal["contour", "simpleblob"]


class GlintConfig(BaseModel):
    brightness_threshold: int = Field(default=220, ge=0, le=255)
    min_glint_area_px: float = Field(default=1.0, ge=0.0)
    max_glint_area_px: float = Field(default=60.0, gt=0.0)
    min_circularity: float = Field(default=0.65, ge=0.0, le=1.0)
    max_distance_from_pupil_px: float = Field(default=40.0, gt=0.0)
    detector_backend: GlintBackend = "contour"
    min_confidence_to_accept: float = Field(default=0.4, ge=0.0, le=1.0)

    @field_validator("max_glint_area_px")
    @classmethod
    def _area_order(cls, v: float, info) -> float:
        min_v = info.data.get("min_glint_area_px")
        if isinstance(min_v, (int, float)) and v <= float(min_v):
            raise ValueError("max_glint_area_px must be > min_glint_area_px.")
        return v


EyeScaleSource = Literal["pupil_radius", "fixed"]
FusionMode = Literal["confidence_weighted", "average", "best_eye"]


class GazeFeatureConfig(BaseModel):
    eye_scale_source: EyeScaleSource = "pupil_radius"
    fixed_eye_scale_px: float = Field(default=15.0, gt=0.0)
    min_pupil_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    min_glint_confidence: float = Field(default=0.4, ge=0.0, le=1.0)
    fusion_mode: FusionMode = "confidence_weighted"
    smoothing_window: int = Field(default=3, ge=1)


class StorageConfig(BaseModel):
    save_logs_to_aws: bool = False
    save_eye_frames: bool = False


class PACEConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    pupil: PupilConfig = Field(default_factory=PupilConfig)
    glint: GlintConfig = Field(default_factory=GlintConfig)
    gaze_feature: GazeFeatureConfig = Field(default_factory=GazeFeatureConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    blink: BlinkConfig = Field(default_factory=BlinkConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    @classmethod
    def _flatten_nested_keys(cls, raw: dict) -> dict:
        out: dict = {}
        for section_name, section_values in raw.items():
            if isinstance(section_values, dict):
                for k, v in section_values.items():
                    if k == "resolution" and isinstance(v, dict):
                        out["resolution_width"] = v.get("width")
                        out["resolution_height"] = v.get("height")
                    else:
                        out[k] = v
            else:
                out[section_name] = section_values
        return out

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PACEConfig":
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        flat = cls._flatten_nested_keys(raw)

        camera_raw = raw.get("camera", {}) if isinstance(raw, dict) else {}
        camera_keys = (
            "camera_index",
            "fps",
            "frame_width",
            "frame_height",
            "target_fps",
            "max_consecutive_failures",
        )
        camera_fields: dict = {k: flat[k] for k in camera_keys if k in flat}
        if isinstance(camera_raw, dict):
            if "backend" in camera_raw:
                camera_fields["backend"] = camera_raw["backend"]
            res = camera_raw.get("resolution")
            if isinstance(res, dict):
                if "width" in res:
                    camera_fields["resolution_width"] = res["width"]
                if "height" in res:
                    camera_fields["resolution_height"] = res["height"]

        detection_raw = raw.get("detection", {}) if isinstance(raw, dict) else {}
        detection_fields: dict = {}
        if isinstance(detection_raw, dict):
            for fld in (
                "backend",
                "min_detection_confidence",
                "min_tracking_confidence",
                "eye_roi_margin_px",
                "max_consecutive_misses",
            ):
                if fld in detection_raw:
                    detection_fields[fld] = detection_raw[fld]
            if "eye_roi_output_size" in detection_raw:
                sz = detection_raw["eye_roi_output_size"]
                if isinstance(sz, (list, tuple)) and len(sz) == 2:
                    detection_fields["eye_roi_output_size"] = (int(sz[0]), int(sz[1]))

        preprocess_raw = raw.get("preprocess", {}) if isinstance(raw, dict) else {}
        preprocess_fields: dict = {}
        if isinstance(preprocess_raw, dict):
            for fld in (
                "blur_kernel_size",
                "blur_sigma",
                "threshold_mode",
                "manual_threshold_value",
                "adaptive_block_size",
                "adaptive_c",
                "morph_kernel_size",
                "morph_open_iterations",
                "morph_close_iterations",
                "min_valid_brightness",
                "max_valid_brightness",
            ):
                if fld in preprocess_raw:
                    preprocess_fields[fld] = preprocess_raw[fld]

        pupil_raw = raw.get("pupil", {}) if isinstance(raw, dict) else {}
        pupil_fields: dict = {}
        if isinstance(pupil_raw, dict):
            for fld in (
                "min_pupil_area_ratio",
                "max_pupil_area_ratio",
                "min_circularity",
                "use_kalman_tracking",
                "kalman_process_noise",
                "kalman_measurement_noise",
                "max_center_jump_px",
                "min_confidence_to_accept",
                "kalman_max_prediction_frames",
            ):
                if fld in pupil_raw:
                    pupil_fields[fld] = pupil_raw[fld]

        glint_raw = raw.get("glint", {}) if isinstance(raw, dict) else {}
        glint_fields: dict = {}
        if isinstance(glint_raw, dict):
            for fld in (
                "brightness_threshold",
                "min_glint_area_px",
                "max_glint_area_px",
                "min_circularity",
                "max_distance_from_pupil_px",
                "detector_backend",
                "min_confidence_to_accept",
            ):
                if fld in glint_raw:
                    glint_fields[fld] = glint_raw[fld]

        gaze_feature_raw = raw.get("gaze_feature", {}) if isinstance(raw, dict) else {}
        gaze_feature_fields: dict = {}
        if isinstance(gaze_feature_raw, dict):
            for fld in (
                "eye_scale_source",
                "fixed_eye_scale_px",
                "min_pupil_confidence",
                "min_glint_confidence",
                "fusion_mode",
                "smoothing_window",
            ):
                if fld in gaze_feature_raw:
                    gaze_feature_fields[fld] = gaze_feature_raw[fld]

        tracking_fields = {
            k: flat[k] for k in ("jitter_smoothing", "sensitivity", "cursor_delay_ms")
            if k in flat
        }
        blink_fields = {
            k: flat[k] for k in ("blink_threshold", "blink_click_count")
            if k in flat
        }
        storage_fields = {
            k: flat[k] for k in ("save_logs_to_aws", "save_eye_frames")
            if k in flat
        }

        return cls(
            camera=CameraConfig(**camera_fields),
            detection=DetectionConfig(**detection_fields),
            preprocess=PreprocessConfig(**preprocess_fields),
            pupil=PupilConfig(**pupil_fields),
            glint=GlintConfig(**glint_fields),
            gaze_feature=GazeFeatureConfig(**gaze_feature_fields),
            tracking=TrackingConfig(**tracking_fields),
            blink=BlinkConfig(**blink_fields),
            storage=StorageConfig(**storage_fields),
        )

    @field_validator("*", mode="before")
    @classmethod
    def accept_none_as_default(cls, v):
        return v


def load_config(config_path: str | Path | None = None) -> PACEConfig:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    return PACEConfig.from_yaml(config_path)
