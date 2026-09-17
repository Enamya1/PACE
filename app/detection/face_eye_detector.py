import logging
import os
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.config import DetectionConfig
from app.detection.data_types import DetectionResult, EyeROI


logger = logging.getLogger(__name__)


LEFT_EYE_LANDMARKS = [
    33, 7, 163, 144, 145, 153, 154, 155,
    133, 173, 157, 158, 159, 160, 161, 246,
]
RIGHT_EYE_LANDMARKS = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398,
]
LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473


MEDIAPIPE_FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)

OPENCV_HAAR_FACE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/"
    "master/data/haarcascades/haarcascade_frontalface_default.xml"
)
OPENCV_HAAR_EYE_URL = (
    "https://raw.githubusercontent.com/opencv/opencv/"
    "master/data/haarcascades/haarcascade_eye.xml"
)


def _pace_cache_dir() -> Path:
    home = Path(os.path.expanduser("~"))
    cache = home / ".cache" / "pace"
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def _download_if_missing(
    dest_rel: str,
    url: str,
    description: str,
) -> Path:
    dest = _pace_cache_dir() / dest_rel
    if dest.exists() and dest.stat().st_size > 0:
        return dest

    logger.info(
        "PACE Phase 2 — one-time download of %s (%s) -> %s",
        description,
        url,
        dest,
    )
    print(
        f"[detection] First-run: downloading {description} "
        f"(internet required once; cached at {dest})."
    )
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        urllib.request.urlretrieve(url, str(tmp))
    except Exception as exc:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise RuntimeError(
            f"Failed to download {description} from {url}: {exc}. "
            f"Check your internet connection, or manually place the file at {dest}."
        ) from exc
    tmp.replace(dest)
    return dest


@dataclass
class _FaceEyeDetectorState:
    consecutive_misses: int = 0


class FaceEyeDetector:
    """Locate a face and extract normalized left/right eye ROI crops.

    Two backends are supported (chosen via DetectionConfig.backend):
      - "mediapipe" (PRIMARY): Per spec,
        ``mediapipe.solutions.face_mesh.FaceMesh(static_image_mode=False,
        max_num_faces=1, refine_landmarks=True, ...)`` which activates the
        10 iris landmarks (indices 468–477) on top of the standard 468
        face mesh.  On MediaPipe builds >= 1.0 that no longer ship the
        legacy ``solutions`` submodule, we transparently fall back to
        ``mp.tasks.vision.FaceLandmarker`` which produces the *exact* same
        landmark set, indices, and normalized coordinates — the only
        difference is the constructor.
      - "haar" (FALLBACK): OpenCV Haar cascade classifier pairs
        (frontalface + eye).  If the XMLs aren't shipped inside the
        installed opencv-python wheel (varies by build / OpenCV version),
        they are fetched once from opencv's GitHub mirror and cached.
        Lower precision, no iris hints, pure OpenCV path kept for judges.

    The only public entry point is ``detect(frame) -> DetectionResult``.
    Implementation details of each backend are private helpers.
    """

    # Which MP implementation is actually active after init:
    # "solutions" = spec path (mp.solutions.face_mesh),
    # "tasks"     = transparent fallback (mp.tasks.vision.FaceLandmarker),
    # None        = not mediapipe backend (haar).
    _mp_impl: Optional[str] = None

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._state = _FaceEyeDetectorState()

        self._face_cascade: Optional[cv2.CascadeClassifier] = None
        self._eye_cascade: Optional[cv2.CascadeClassifier] = None

        self._face_mesh_solutions = None
        self._face_mesh_tasks = None
        self._mp_Image = None
        self._mp_ImageFormat_SRGB = None

        if config.backend == "mediapipe":
            self._init_mediapipe()
        elif config.backend == "haar":
            self._init_haar()
        else:
            raise ValueError(
                f"Unknown detection backend: {config.backend!r}. "
                "Expected 'mediapipe' or 'haar'."
            )

    # ------------------------------------------------------------------ #
    #  Backend initialisation
    # ------------------------------------------------------------------ #
    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise ImportError(
                "MediaPipe is required for the 'mediapipe' detection backend. "
                "Install it with: pip install mediapipe  (or switch config "
                "detection.backend='haar')."
            ) from exc

        # --------------------------------------------------------------- #
        # PRIMARY path, per Phase 2 spec:
        #   mp.solutions.face_mesh.FaceMesh(static_image_mode=False,
        #       max_num_faces=1, refine_landmarks=True,
        #       min_detection_confidence=..., min_tracking_confidence=...)
        # --------------------------------------------------------------- #
        mp_solutions = getattr(mp, "solutions", None)
        if mp_solutions is not None and hasattr(mp_solutions, "face_mesh"):
            self._face_mesh_solutions = mp_solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=self._config.min_detection_confidence,
                min_tracking_confidence=self._config.min_tracking_confidence,
            )
            self._mp_impl = "solutions"
            return

        # --------------------------------------------------------------- #
        # FALLBACK: MediaPipe >= 1.0 dropped mp.solutions on some wheels
        # (notably the Windows 1.0.x wheel).  Use mp.tasks.vision.FaceLandmarker
        # which yields the *identical* 478-landmark set (468 face + 10 iris,
        # indices 468–477), same normalized [0,1] coordinate space, and the
        # same landmark semantic per index.  Phase 4 cannot tell the difference.
        # --------------------------------------------------------------- #
        logger.warning(
            "mp.solutions.face_mesh not available in this MediaPipe build; "
            "falling back to mp.tasks.vision.FaceLandmarker (identical "
            "landmark output)."
        )
        model_path = _download_if_missing(
            "face_landmarker.task",
            MEDIAPIPE_FACE_LANDMARKER_URL,
            "MediaPipe FaceLandmarker model (478 landmarks + iris)",
        )
        VisionRunningMode = mp.tasks.vision.RunningMode
        self._mp_Image = mp.Image
        self._mp_ImageFormat_SRGB = mp.ImageFormat.SRGB

        base_opts = mp.tasks.BaseOptions(model_asset_path=str(model_path))
        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=VisionRunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self._config.min_detection_confidence,
            min_face_presence_confidence=self._config.min_tracking_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._face_mesh_tasks = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )
        self._mp_impl = "tasks"

    def _init_haar(self) -> None:
        cv2_dir = Path(cv2.data.haarcascades)
        face_xml = cv2_dir / "haarcascade_frontalface_default.xml"
        eye_xml = cv2_dir / "haarcascade_eye.xml"

        if not face_xml.exists() or face_xml.stat().st_size < 1024:
            face_xml = _download_if_missing(
                "haarcascade_frontalface_default.xml",
                OPENCV_HAAR_FACE_URL,
                "OpenCV Haar frontal-face cascade",
            )
        if not eye_xml.exists() or eye_xml.stat().st_size < 1024:
            eye_xml = _download_if_missing(
                "haarcascade_eye.xml",
                OPENCV_HAAR_EYE_URL,
                "OpenCV Haar eye cascade",
            )

        face_cascade = cv2.CascadeClassifier(str(face_xml))
        if face_cascade.empty():
            raise FileNotFoundError(
                f"Haar face cascade failed to load from: {face_xml}. "
                "The file exists but CascadeClassifier.empty() returned True; "
                "it may be corrupt — delete it and the next init will re-fetch."
            )
        self._face_cascade = face_cascade

        eye_cascade = cv2.CascadeClassifier(str(eye_xml))
        if eye_cascade.empty():
            raise FileNotFoundError(
                f"Haar eye cascade failed to load from: {eye_xml}. "
                "The file exists but CascadeClassifier.empty() returned True; "
                "it may be corrupt — delete it and the next init will re-fetch."
            )
        self._eye_cascade = eye_cascade

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Run face + eye detection on a single BGR frame.

        Args:
            frame: HxWx3 uint8 BGR image (as returned by CameraStream).

        Returns:
            DetectionResult with found flag, confidence, ROIs, and the
            running consecutive_misses counter.
        """
        ts = time.monotonic()
        backend = self._config.backend

        if backend == "mediapipe":
            raw = self._detect_mediapipe(frame, ts)
        else:
            raw = self._detect_haar(frame)

        found = raw["found"]
        if found:
            self._state.consecutive_misses = 0
            cm = 0
        else:
            self._state.consecutive_misses += 1
            cm = self._state.consecutive_misses

        return DetectionResult(
            found=found,
            confidence=raw.get("confidence", 0.0),
            face_bbox=raw.get("face_bbox"),
            left_eye=raw.get("left_eye"),
            right_eye=raw.get("right_eye"),
            backend_used=backend,
            frame_timestamp=ts,
            consecutive_misses=cm,
        )

    @property
    def consecutive_misses(self) -> int:
        return self._state.consecutive_misses

    def reset_misses(self) -> None:
        self._state.consecutive_misses = 0

    # ------------------------------------------------------------------ #
    #  MediaPipe backend — dispatches to either solutions or tasks impl.
    #  Both produce identical landmark lists (NormalizedLandmark .x/.y/.z
    #  in [0,1], same indices 0..477, iris at 468 / 473).
    # ------------------------------------------------------------------ #
    def _detect_mediapipe(self, frame: np.ndarray, frame_ts_s: float) -> dict:
        if self._mp_impl == "solutions":
            return self._detect_mp_solutions(frame)
        if self._mp_impl == "tasks":
            return self._detect_mp_tasks(frame, frame_ts_s)
        return {"found": False, "confidence": 0.0}

    # ------ spec implementation (MediaPipe < 1.0 / legacy builds) ----- #
    def _detect_mp_solutions(self, frame: np.ndarray) -> dict:
        if self._face_mesh_solutions is None:
            return {"found": False, "confidence": 0.0}

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self._face_mesh_solutions.process(rgb)

        if not results.multi_face_landmarks:
            return {"found": False, "confidence": 0.0}

        lm = results.multi_face_landmarks[0].landmark
        return self._assemble_from_landmark_list(lm, frame, h, w)

    # ------------ transparent fallback (MediaPipe >= 1.0) ------------- #
    def _detect_mp_tasks(self, frame: np.ndarray, frame_ts_s: float) -> dict:
        if self._face_mesh_tasks is None or self._mp_Image is None:
            return {"found": False, "confidence": 0.0}

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        timestamp_ms = int(round(frame_ts_s * 1000))
        mp_img = self._mp_Image(
            image_format=self._mp_ImageFormat_SRGB,
            data=rgb,
        )
        result = self._face_mesh_tasks.detect_for_video(mp_img, timestamp_ms)

        if not result.face_landmarks:
            return {"found": False, "confidence": 0.0}

        lm = result.face_landmarks[0]
        return self._assemble_from_landmark_list(lm, frame, h, w)

    # ------ shared assembly: landmark list -> face_bbox + eye ROIs ---- #
    def _assemble_from_landmark_list(
        self, lm, frame: np.ndarray, h: int, w: int
    ) -> dict:
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        fx1, fx2 = int(min(xs) * w), int(max(xs) * w)
        fy1, fy2 = int(min(ys) * h), int(max(ys) * h)
        fx1, fy1 = max(0, fx1), max(0, fy1)
        fx2, fy2 = min(w, fx2), min(h, fy2)
        if fx2 - fx1 < 10 or fy2 - fy1 < 10:
            return {"found": False, "confidence": 0.0}
        face_bbox = (fx1, fy1, fx2, fy2)

        left_bbox = self._bbox_from_landmarks(
            lm, LEFT_EYE_LANDMARKS, w, h, self._config.eye_roi_margin_px
        )
        right_bbox = self._bbox_from_landmarks(
            lm, RIGHT_EYE_LANDMARKS, w, h, self._config.eye_roi_margin_px
        )
        if left_bbox is None or right_bbox is None:
            return {"found": False, "confidence": 0.0}

        left_iris = (lm[LEFT_IRIS_CENTER].x * w, lm[LEFT_IRIS_CENTER].y * h)
        right_iris = (lm[RIGHT_IRIS_CENTER].x * w, lm[RIGHT_IRIS_CENTER].y * h)

        out_sz = self._config.eye_roi_output_size
        left_roi = self._extract_roi(frame, left_bbox, out_sz, iris_hint=left_iris)
        right_roi = self._extract_roi(frame, right_bbox, out_sz, iris_hint=right_iris)
        if left_roi is None or right_roi is None:
            return {"found": False, "confidence": 0.0}

        return {
            "found": True,
            "confidence": 0.9,
            "face_bbox": face_bbox,
            "left_eye": left_roi,
            "right_eye": right_roi,
        }

    @staticmethod
    def _bbox_from_landmarks(
        landmarks,
        indices: List[int],
        width: int,
        height: int,
        margin_px: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        xs = [landmarks[i].x for i in indices]
        ys = [landmarks[i].y for i in indices]
        x1 = int(min(xs) * width) - margin_px
        y1 = int(min(ys) * height) - margin_px
        x2 = int(max(xs) * width) + margin_px
        y2 = int(max(ys) * height) + margin_px
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return (x1, y1, x2, y2)

    # ------------------------------------------------------------------ #
    #  Haar backend
    # ------------------------------------------------------------------ #
    def _detect_haar(self, frame: np.ndarray) -> dict:
        assert self._face_cascade is not None
        assert self._eye_cascade is not None

        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)
        )
        if len(faces) == 0:
            return {"found": False, "confidence": 0.0}

        areas = faces[:, 2] * faces[:, 3]
        fx, fy, fw, fh = faces[int(np.argmax(areas))]
        face_bbox = (int(fx), int(fy), int(fx + fw), int(fy + fh))

        upper_h = fh // 2
        if upper_h < 20:
            return {"found": False, "confidence": 0.0}
        face_roi_gray = gray[fy:fy + upper_h, fx:fx + fw]
        eyes = self._eye_cascade.detectMultiScale(
            face_roi_gray, scaleFactor=1.1, minNeighbors=8, minSize=(20, 20)
        )
        if len(eyes) < 2:
            return {"found": False, "confidence": 0.0}

        eye_list: List[Tuple[int, int, int, int]] = []
        for (ex, ey, ew, eh) in eyes:
            margin = self._config.eye_roi_margin_px
            ax1 = max(0, fx + ex - margin)
            ay1 = max(0, fy + ey - margin)
            ax2 = min(w, fx + ex + ew + margin)
            ay2 = min(h, fy + ey + eh + margin)
            if ax2 - ax1 > 4 and ay2 - ay1 > 4:
                eye_list.append((ax1, ay1, ax2, ay2))

        if len(eye_list) < 2:
            return {"found": False, "confidence": 0.0}

        eye_list.sort(key=lambda b: b[0])

        # NOTE: Haar eye detection — leftmost box in the *image* belongs to the
        # *subject's right eye* (camera is mirrored like a selfie).  We therefore
        # assign: eye_list[0] -> subject's right eye (right_eye field), and
        # eye_list[-1] -> subject's left eye (left_eye field). Phase 4 pupil
        # detection and downstream gaze code rely on this assignment.
        right_eye_bbox = eye_list[0]
        left_eye_bbox = eye_list[-1]

        out_sz = self._config.eye_roi_output_size
        left_roi = self._extract_roi(frame, left_eye_bbox, out_sz, iris_hint=None)
        right_roi = self._extract_roi(frame, right_eye_bbox, out_sz, iris_hint=None)
        if left_roi is None or right_roi is None:
            return {"found": False, "confidence": 0.0}

        return {
            "found": True,
            "confidence": 0.6,
            "face_bbox": face_bbox,
            "left_eye": left_roi,
            "right_eye": right_roi,
        }

    # ------------------------------------------------------------------ #
    #  Shared helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _extract_roi(
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        output_size: Tuple[int, int],
        iris_hint: Optional[Tuple[float, float]] = None,
    ) -> Optional[EyeROI]:
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
        orig_w = x2 - x1
        orig_h = y2 - y1
        if orig_w <= 0 or orig_h <= 0:
            return None
        resized = cv2.resize(roi, output_size, interpolation=cv2.INTER_LINEAR)
        return EyeROI(
            image=resized,
            original_bbox=bbox,
            scale_x=output_size[0] / orig_w,
            scale_y=output_size[1] / orig_h,
            iris_hint=iris_hint,
        )

    def close(self) -> None:
        if self._face_mesh_solutions is not None:
            try:
                self._face_mesh_solutions.close()
            except Exception:
                pass
            self._face_mesh_solutions = None
        if self._face_mesh_tasks is not None:
            try:
                self._face_mesh_tasks.close()
            except Exception:
                pass
            self._face_mesh_tasks = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
