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


class StorageConfig(BaseModel):
    save_logs_to_aws: bool = False
    save_eye_frames: bool = False


class PACEConfig(BaseModel):
    camera: CameraConfig = Field(default_factory=CameraConfig)
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

        camera_keys = (
            "camera_index",
            "resolution_width",
            "resolution_height",
            "fps",
            "frame_width",
            "frame_height",
            "target_fps",
            "max_consecutive_failures",
            "backend",
        )
        camera_fields = {k: flat[k] for k in camera_keys if k in flat}
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
