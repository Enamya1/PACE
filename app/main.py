import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from app.config import load_config


def _run_camera_test(cfg) -> int:
    from app.camera.camera_stream import CameraStream
    from app.camera.exceptions import CameraError

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    stream = CameraStream(cfg.camera.camera_index, cfg.camera)
    try:
        try:
            stream.start()
        except CameraError as exc:
            print(f"[camera-test] FATAL: {exc}")
            return 1

        duration_s = 5.0
        frames_total = 0
        prev_fn = 0
        start_time = time.monotonic()
        last_tick = start_time

        while time.monotonic() - start_time < duration_s:
            fd = stream.get_frame(timeout=0.05)
            if fd is not None:
                frames_total += 1
                prev_fn = fd.frame_number

            now = time.monotonic()
            if now - last_tick >= 1.0:
                elapsed = now - start_time
                print(
                    f"[camera-test] t={elapsed:.1f}s | "
                    f"connected={stream.is_connected()} | "
                    f"frame_number={prev_fn} | "
                    f"avg_fps={frames_total / max(elapsed, 1e-6):.1f}"
                )
                last_tick = now

        total = time.monotonic() - start_time
        print(
            f"[camera-test] DONE after {total:.1f}s | "
            f"frames={frames_total} | "
            f"effective_fps={frames_total / max(total, 1e-6):.1f} | "
            f"connected={stream.is_connected()}"
        )
        return 0

    except KeyboardInterrupt:
        print("[camera-test] Interrupted by user.")
        return 130
    finally:
        stream.stop()


def _run_detection_test(cfg) -> int:
    from app.camera.camera_stream import CameraStream
    from app.camera.exceptions import CameraError
    from app.detection.face_eye_detector import FaceEyeDetector

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "phase2_sample_detection.png"

    stream = CameraStream(cfg.camera.camera_index, cfg.camera)
    detector = FaceEyeDetector(cfg.detection)
    sample_saved = False
    try:
        try:
            stream.start()
        except CameraError as exc:
            print(f"[detection-test] FATAL: {exc}")
            return 1

        duration_s = 5.0
        detections_total = 0
        hits_total = 0
        start_time = time.monotonic()
        last_tick = start_time
        last_status = ("", 0.0, "", (0, 0))

        while time.monotonic() - start_time < duration_s:
            fd = stream.get_frame(timeout=0.05)
            if fd is None:
                time.sleep(0.005)
                continue

            detections_total += 1
            frame = fd.frame
            res = detector.detect(frame)
            if res.found:
                hits_total += 1

            status = (
                "FOUND" if res.found else "NOT-FOUND",
                res.confidence,
                res.backend_used,
                (
                    res.left_eye.image.shape[:2][::-1]
                    if (res.found and res.left_eye is not None)
                    else (0, 0)
                ),
            )
            last_status = status

            if not sample_saved and res.found and res.face_bbox is not None:
                annotated = frame.copy()
                fx1, fy1, fx2, fy2 = res.face_bbox
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 255, 0), 2)
                for roi, label in ((res.left_eye, "L"), (res.right_eye, "R")):
                    if roi is None:
                        continue
                    ex1, ey1, ex2, ey2 = roi.original_bbox
                    cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (255, 0, 0), 2)
                    cv2.putText(
                        annotated,
                        label,
                        (ex1, max(0, ey1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    if roi.iris_hint is not None:
                        ix, iy = int(round(roi.iris_hint[0])), int(round(roi.iris_hint[1]))
                        cv2.circle(annotated, (ix, iy), 3, (0, 0, 255), -1)
                if cv2.imwrite(str(out_path), annotated):
                    sample_saved = True
                    print(f"[detection-test] saved annotated sample -> {out_path}")

            now = time.monotonic()
            if now - last_tick >= 1.0:
                elapsed = now - start_time
                found_str, conf, backend, roi_sz = last_status
                roi_str = f"eye_roi={roi_sz[0]}x{roi_sz[1]}" if roi_sz[0] else "eye_roi=N/A"
                print(
                    f"[detection-test] t={elapsed:.1f}s | "
                    f"status={found_str} | "
                    f"confidence={conf:.2f} | "
                    f"backend={backend} | "
                    f"{roi_str} | "
                    f"hit_rate={hits_total}/{detections_total} "
                    f"({100*hits_total/max(detections_total,1):.0f}%)"
                )
                last_tick = now

        total = time.monotonic() - start_time
        hit_rate_pct = 100.0 * hits_total / max(detections_total, 1)
        print(
            f"[detection-test] DONE after {total:.1f}s | "
            f"frames_processed={detections_total} | "
            f"hits={hits_total} ({hit_rate_pct:.1f}%) | "
            f"sample_saved={sample_saved} | "
            f"path={out_path if sample_saved else 'n/a'}"
        )
        return 0

    except KeyboardInterrupt:
        print("[detection-test] Interrupted by user.")
        return 130
    finally:
        detector.close()
        stream.stop()


def _run_preprocess_test(cfg) -> int:
    from app.camera.camera_stream import CameraStream
    from app.camera.exceptions import CameraError
    from app.detection.face_eye_detector import FaceEyeDetector
    from app.detection.preprocess import PreprocessPipeline

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "phase3_sample_preprocess.png"

    stream = CameraStream(cfg.camera.camera_index, cfg.camera)
    detector = FaceEyeDetector(cfg.detection)
    pipeline = PreprocessPipeline(cfg.preprocess)
    sample_saved = False
    try:
        try:
            stream.start()
        except CameraError as exc:
            print(f"[preprocess-test] FATAL: {exc}")
            return 1

        duration_s = 5.0
        frames_total = 0
        hits = 0
        start_time = time.monotonic()
        last_tick = start_time

        while time.monotonic() - start_time < duration_s:
            fd = stream.get_frame(timeout=0.05)
            if fd is None:
                continue
            frames_total += 1

            det = detector.detect(fd.frame)
            if det.found and det.left_eye is not None:
                hits += 1
                pp_left, pp_right = pipeline.process_both_eyes(det)

                now = time.monotonic()
                if now - last_tick >= 1.0:
                    elapsed = now - start_time
                    def _fmt(pp):
                        if pp is None:
                            return "N/A"
                        ok = "OK" if pp.lighting_ok else "LOW"
                        return (
                            f"bri={pp.mean_brightness:5.1f}({ok}) "
                            f"mode={pp.threshold_mode_used}"
                        )
                    print(
                        f"[preprocess-test] t={elapsed:.1f}s | "
                        f"L=[{_fmt(pp_left)}] | "
                        f"R=[{_fmt(pp_right)}] | "
                        f"hit_rate={hits}/{frames_total} ({100 * hits / max(1, frames_total):.0f}%)"
                    )
                    last_tick = now

                if not sample_saved and pp_left is not None:
                    orig = det.left_eye.image
                    gray_bgr = cv2.cvtColor(pp_left.gray, cv2.COLOR_GRAY2BGR)
                    blur_bgr = cv2.cvtColor(pp_left.blurred, cv2.COLOR_GRAY2BGR)
                    mask_bgr = cv2.cvtColor(pp_left.mask, cv2.COLOR_GRAY2BGR)
                    panel = np.hstack([orig, gray_bgr, blur_bgr, mask_bgr])
                    cv2.imwrite(str(out_path), panel)
                    sample_saved = True
                    print(f"[preprocess-test] saved 4-panel preprocess debug -> {out_path}")
            else:
                now = time.monotonic()
                if now - last_tick >= 1.0:
                    elapsed = now - start_time
                    print(
                        f"[preprocess-test] t={elapsed:.1f}s | "
                        f"L=[not-detected] | "
                        f"R=[not-detected] | "
                        f"hit_rate={hits}/{frames_total} ({100 * hits / max(1, frames_total):.0f}%)"
                    )
                    last_tick = now

        total = time.monotonic() - start_time
        path_str = str(out_path) if sample_saved else "n/a"
        print(
            f"[preprocess-test] DONE after {total:.1f}s | "
            f"frames_processed={frames_total} | "
            f"hits={hits} ({100 * hits / max(1, frames_total):.1f}%) | "
            f"sample_saved={sample_saved} | path={path_str}"
        )
        return 0

    except KeyboardInterrupt:
        print("[preprocess-test] Interrupted by user.")
        return 130
    finally:
        detector.close()
        stream.stop()


def _run_pupil_test(cfg) -> int:
    from app.camera.camera_stream import CameraStream
    from app.camera.exceptions import CameraError
    from app.detection.face_eye_detector import FaceEyeDetector
    from app.detection.preprocess import PreprocessPipeline
    from app.detection.pupil import PupilDetector

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "phase4_sample_pupil.png"

    stream = CameraStream(cfg.camera.camera_index, cfg.camera)
    detector = FaceEyeDetector(cfg.detection)
    preprocess = PreprocessPipeline(cfg.preprocess)
    pupil = PupilDetector(cfg.pupil)
    sample_saved = False

    try:
        try:
            stream.start()
        except CameraError as exc:
            print(f"[pupil-test] FATAL: {exc}")
            return 1

        duration_s = 5.0
        frames_total = 0
        hits = 0
        pupil_hits = 0
        start_time = time.monotonic()
        last_tick = start_time

        while time.monotonic() - start_time < duration_s:
            fd = stream.get_frame(timeout=0.05)
            if fd is None:
                continue
            frames_total += 1

            det = detector.detect(fd.frame)
            if not det.found or det.left_eye is None:
                now = time.monotonic()
                if now - last_tick >= 1.0:
                    elapsed = now - start_time
                    print(
                        f"[pupil-test] t={elapsed:.1f}s | "
                        f"L=[not-detected] | "
                        f"R=[not-detected] | "
                        f"roi_hit={hits}/{frames_total} ({100 * hits / max(1, frames_total):.0f}%)"
                    )
                    last_tick = now
                continue

            hits += 1
            pp_left, pp_right = preprocess.process_both_eyes(det)
            pup_left, pup_right = pupil.detect_both_eyes(pp_left, pp_right)

            if pup_left is not None and pup_left.found:
                pupil_hits += 1

            now = time.monotonic()
            if now - last_tick >= 1.0:
                elapsed = now - start_time
                def _fmtp(pr):
                    if pr is None:
                        return "N/A"
                    if not pr.found:
                        return "not-found"
                    fb = " [FB]" if pr.used_prediction_fallback else ""
                    return (
                        f"found={pr.found} conf={pr.confidence:.2f} "
                        f"ctr=({pr.center_x:5.1f},{pr.center_y:5.1f}){fb}"
                    )
                print(
                    f"[pupil-test] t={elapsed:.1f}s | "
                    f"L=[{_fmtp(pup_left)}] | "
                    f"R=[{_fmtp(pup_right)}] | "
                    f"roi_hit={hits}/{frames_total} ({100 * hits / max(1, frames_total):.0f}%) "
                    f"pupil_ok={pupil_hits}/{hits} ({100 * pupil_hits / max(1, hits):.0f}%)"
                )
                last_tick = now

            # Save annotated sample: draw fitted ellipse + center dot onto
            # the left-eye ROI and write out.
            if not sample_saved and pup_left is not None and pup_left.found and det.left_eye is not None:
                roi_img = det.left_eye.image.copy()
                (ecx, ecy) = (float(pup_left.center_x), float(pup_left.center_y))
                (dmaj, dmin) = (float(pup_left.major_axis), float(pup_left.minor_axis))
                ang = float(pup_left.ellipse_angle)
                # Fitted ellipse outline in YELLOW
                cv2.ellipse(roi_img, ((ecx, ecy), (dmin, dmaj), ang), (0, 255, 255), 1, cv2.LINE_AA)
                # Fitted center dot in RED (thick, for visibility)
                cr = max(1, int(round(pup_left.radius)))
                cv2.circle(roi_img, (int(round(ecx)), int(round(ecy))), cr, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(roi_img, (int(round(ecx)), int(round(ecy))), 2, (0, 0, 255), -1, cv2.LINE_AA)
                # Small label text
                cv2.putText(
                    roi_img,
                    f"conf={pup_left.confidence:.2f}",
                    (2, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                if cv2.imwrite(str(out_path), roi_img):
                    sample_saved = True
                    print(f"[pupil-test] saved annotated pupil sample -> {out_path}")

        total = time.monotonic() - start_time
        path_str = str(out_path) if sample_saved else "n/a"
        print(
            f"[pupil-test] DONE after {total:.1f}s | "
            f"frames={frames_total} roi_hits={hits} ({100 * hits / max(1, frames_total):.1f}%) | "
            f"pupil_hits={pupil_hits} ({100 * pupil_hits / max(1, hits):.1f}%) | "
            f"sample_saved={sample_saved} | path={path_str}"
        )
        return 0

    except KeyboardInterrupt:
        print("[pupil-test] Interrupted by user.")
        return 130
    finally:
        detector.close()
        stream.stop()


def _run_glint_test(cfg) -> int:
    from app.camera.camera_stream import CameraStream
    from app.camera.exceptions import CameraError
    from app.detection.face_eye_detector import FaceEyeDetector
    from app.detection.preprocess import PreprocessPipeline
    from app.detection.pupil import PupilDetector
    from app.detection.glint import GlintDetector

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / "phase5_sample_glint.png"

    stream = CameraStream(cfg.camera.camera_index, cfg.camera)
    detector = FaceEyeDetector(cfg.detection)
    preprocess = PreprocessPipeline(cfg.preprocess)
    pupil = PupilDetector(cfg.pupil)
    glint = GlintDetector(cfg.glint)
    sample_saved = False

    try:
        try:
            stream.start()
        except CameraError as exc:
            print(f"[glint-test] FATAL: {exc}")
            return 1

        duration_s = 5.0
        frames_total = 0
        roi_hits = 0
        pupil_hits = 0
        glint_hits = 0
        start_time = time.monotonic()
        last_tick = start_time

        while time.monotonic() - start_time < duration_s:
            fd = stream.get_frame(timeout=0.05)
            if fd is None:
                continue
            frames_total += 1

            det = detector.detect(fd.frame)
            if not det.found or det.left_eye is None:
                now = time.monotonic()
                if now - last_tick >= 1.0:
                    elapsed = now - start_time
                    print(
                        f"[glint-test] t={elapsed:.1f}s | "
                        f"L=[not-detected] | R=[not-detected] | "
                        f"roi_hit={roi_hits}/{frames_total} "
                        f"({100 * roi_hits / max(1, frames_total):.0f}%)"
                    )
                    last_tick = now
                continue

            roi_hits += 1
            pp_left, pp_right = preprocess.process_both_eyes(det)
            pup_left, pup_right = pupil.detect_both_eyes(pp_left, pp_right)
            gl_left, gl_right = glint.detect_both_eyes(
                pp_left, pup_left, pp_right, pup_right
            )

            if pup_left is not None and pup_left.found:
                pupil_hits += 1
            if gl_left is not None and gl_left.found:
                glint_hits += 1

            now = time.monotonic()
            if now - last_tick >= 1.0:
                elapsed = now - start_time
                def _fmtg(gr):
                    if gr is None:
                        return "N/A"
                    if not gr.found:
                        return f"miss reason={gr.reason or '?'}"
                    return (
                        f"found conf={gr.confidence:.2f} "
                        f"ctr=({gr.center_x:5.1f},{gr.center_y:5.1f}) "
                        f"d_pup={gr.distance_from_pupil:4.1f}px"
                    )
                print(
                    f"[glint-test] t={elapsed:.1f}s | "
                    f"L=[{_fmtg(gl_left)}] | "
                    f"R=[{_fmtg(gl_right)}] | "
                    f"roi={roi_hits}/{frames_total} "
                    f"({100 * roi_hits / max(1, frames_total):.0f}%) "
                    f"pupil_ok={pupil_hits}/{roi_hits} "
                    f"({100 * pupil_hits / max(1, roi_hits):.0f}%) "
                    f"glint_ok={glint_hits}/{pupil_hits} "
                    f"({100 * glint_hits / max(1, pupil_hits):.0f}%)"
                )
                last_tick = now

            if (
                not sample_saved
                and pup_left is not None and pup_left.found
                and det.left_eye is not None
                and pp_left is not None
            ):
                roi_img = det.left_eye.image.copy()
                # Draw pupil ellipse + center dot (from Phase 4 style)
                pcx, pcy = float(pup_left.center_x), float(pup_left.center_y)
                pma, pmi = float(pup_left.major_axis), float(pup_left.minor_axis)
                pang = float(pup_left.ellipse_angle)
                cv2.ellipse(roi_img, ((pcx, pcy), (pmi, pma), pang), (0, 255, 255), 1, cv2.LINE_AA)
                cr = max(1, int(round(pup_left.radius)))
                cv2.circle(roi_img, (int(round(pcx)), int(round(pcy))), cr, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.circle(roi_img, (int(round(pcx)), int(round(pcy))), 2, (0, 0, 255), -1, cv2.LINE_AA)

                # Draw glint markers: small white dot + yellow line to pupil
                if gl_left is not None and gl_left.found:
                    gx, gy = float(gl_left.center_x), float(gl_left.center_y)
                    # Yellow pupil→glint vector line
                    cv2.line(
                        roi_img,
                        (int(round(pcx)), int(round(pcy))),
                        (int(round(gx)), int(round(gy))),
                        (0, 255, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # Small white glint dot
                    cv2.circle(
                        roi_img,
                        (int(round(gx)), int(round(gy))),
                        2,
                        (255, 255, 255),
                        -1,
                        cv2.LINE_AA,
                    )
                    cv2.circle(
                        roi_img,
                        (int(round(gx)), int(round(gy))),
                        3,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
                    status = f"p={pup_left.confidence:.2f} g={gl_left.confidence:.2f}"
                else:
                    gr_str = gl_left.reason if (gl_left is not None and gl_left.reason) else "none"
                    status = f"p={pup_left.confidence:.2f} g:no({gr_str})"
                cv2.putText(
                    roi_img,
                    status,
                    (2, 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.32,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
                if cv2.imwrite(str(out_path), roi_img):
                    sample_saved = True
                    print(f"[glint-test] saved annotated glint sample -> {out_path}")

        total = time.monotonic() - start_time
        path_str = str(out_path) if sample_saved else "n/a"
        print(
            f"[glint-test] DONE after {total:.1f}s | "
            f"frames={frames_total} roi_hits={roi_hits} "
            f"({100 * roi_hits / max(1, frames_total):.1f}%) | "
            f"pupil_hits={pupil_hits} ({100 * pupil_hits / max(1, roi_hits):.1f}%) | "
            f"glint_hits={glint_hits} ({100 * glint_hits / max(1, pupil_hits):.1f}%) | "
            f"sample_saved={sample_saved} | path={path_str}"
        )
        return 0

    except KeyboardInterrupt:
        print("[glint-test] Interrupted by user.")
        return 130
    finally:
        detector.close()
        stream.stop()


def _run_feature_test(cfg) -> int:
    from app.camera.camera_stream import CameraStream
    from app.camera.exceptions import CameraError
    from app.detection.face_eye_detector import FaceEyeDetector
    from app.detection.preprocess import PreprocessPipeline
    from app.detection.pupil import PupilDetector
    from app.detection.glint import GlintDetector
    from app.tracking.gaze_feature import GazeFeatureExtractor

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

    print("=" * 72)
    print("[feature-test] PHASE 6 HEAD-MOVEMENT ROBUSTNESS DEMO")
    print("  Step 1 (0-3s):  Hold your gaze STEADY on one fixed spot.")
    print("  Step 2 (3-8s):  While still looking at the SAME spot, gently")
    print("                   move your HEAD left / right / forward / back.")
    print("  Observe: vector_x / vector_y should stay far more stable than")
    print("           the raw (pupil.center_x, pupil.center_y) pixel coords.")
    print("=" * 72)

    stream = CameraStream(cfg.camera.camera_index, cfg.camera)
    detector = FaceEyeDetector(cfg.detection)
    preprocess = PreprocessPipeline(cfg.preprocess)
    pupil = PupilDetector(cfg.pupil)
    glint = GlintDetector(cfg.glint)
    feature = GazeFeatureExtractor(cfg.gaze_feature)

    try:
        try:
            stream.start()
        except CameraError as exc:
            print(f"[feature-test] FATAL: {exc}")
            return 1

        duration_s = 8.0
        frames_total = 0
        roi_hits = 0
        pupil_hits = 0
        glint_hits = 0
        feature_hits = 0
        start_time = time.monotonic()
        last_tick = start_time

        while time.monotonic() - start_time < duration_s:
            fd = stream.get_frame(timeout=0.05)
            if fd is None:
                continue
            frames_total += 1

            det = detector.detect(fd.frame)
            if not det.found or det.left_eye is None:
                now = time.monotonic()
                if now - last_tick >= 1.0:
                    elapsed = now - start_time
                    print(
                        f"[feature-test] t={elapsed:.1f}s | "
                        f"feature=N/A (no face/ROI) | "
                        f"roi={roi_hits}/{frames_total} "
                        f"({100 * roi_hits / max(1, frames_total):.0f}%)"
                    )
                    last_tick = now
                continue

            roi_hits += 1
            pp_left, pp_right = preprocess.process_both_eyes(det)
            pup_left, pup_right = pupil.detect_both_eyes(pp_left, pp_right)
            gl_left, gl_right = glint.detect_both_eyes(
                pp_left, pup_left, pp_right, pup_right
            )

            if pup_left is not None and pup_left.found:
                pupil_hits += 1
            if gl_left is not None and gl_left.found:
                glint_hits += 1

            ts = time.monotonic()
            feat = feature.compute(pup_left, gl_left, pup_right, gl_right, ts)

            now = time.monotonic()
            if now - last_tick >= 1.0:
                elapsed = now - start_time
                if feat is None:
                    print(
                        f"[feature-test] t={elapsed:.1f}s | "
                        f"vector_x=N/A vector_y=N/A "
                        f"conf=N/A eyes_used=NONE "
                        f"fusion={cfg.gaze_feature.fusion_mode} | "
                        f"roi={roi_hits}/{frames_total} "
                        f"pupil={pupil_hits}/{roi_hits} "
                        f"glint={glint_hits}/{max(1, pupil_hits)}"
                    )
                else:
                    feature_hits += 1
                    raw_px = ""
                    if pup_left is not None and pup_left.found:
                        raw_px = (
                            f"  [raw L pupil=({pup_left.center_x:5.1f},"
                            f"{pup_left.center_y:5.1f}) r={pup_left.radius:4.1f}]"
                        )
                    print(
                        f"[feature-test] t={elapsed:.1f}s | "
                        f"vector_x={feat.vector_x:7.3f} "
                        f"vector_y={feat.vector_y:7.3f} "
                        f"conf={feat.confidence:.3f} "
                        f"eyes_used={feat.eyes_used:4s} "
                        f"fusion={feat.fusion_mode_used}{raw_px}"
                    )
                last_tick = now

        total = time.monotonic() - start_time
        print(
            f"[feature-test] DONE after {total:.1f}s | "
            f"frames={frames_total} "
            f"roi={roi_hits} ({100 * roi_hits / max(1, frames_total):.1f}%) | "
            f"pupil={pupil_hits} ({100 * pupil_hits / max(1, roi_hits):.1f}%) | "
            f"glint={glint_hits} ({100 * glint_hits / max(1, pupil_hits):.1f}%) | "
            f"feature_frames_with_1s_report={feature_hits}"
        )
        return 0

    except KeyboardInterrupt:
        print("[feature-test] Interrupted by user.")
        return 130
    finally:
        detector.close()
        stream.stop()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="PACE",
        description="Pupil Assistive Calibrated Interface for Hands-Free Cursor Control",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Start up, load config, print OK, and exit.",
    )
    parser.add_argument(
        "--camera-test",
        action="store_true",
        help="Run a 5-second live webcam capture smoke test: fps + status per second, then exit 0.",
    )
    parser.add_argument(
        "--detection-test",
        action="store_true",
        help="Run 5s of FaceEyeDetector against live webcam: status/confidence/backend per second, "
             "save 1 annotated sample frame to results/phase2_sample_detection.png.",
    )
    parser.add_argument(
        "--preprocess-test",
        action="store_true",
        help="Run 5s of Camera → FaceEyeDetector → PreprocessPipeline live: brightness/lighting_ok/"
             "threshold_mode per second, save 1 4-panel debug image (orig|gray|blur|mask) to "
             "results/phase3_sample_preprocess.png.",
    )
    parser.add_argument(
        "--pupil-test",
        action="store_true",
        help="Run 5s of Camera → FaceEyeDetector → PreprocessPipeline → PupilDetector live: "
             "found/confidence/center per eye per second, save 1 annotated ROI with fitted ellipse "
             "and red center dot to results/phase4_sample_pupil.png.",
    )
    parser.add_argument(
        "--glint-test",
        action="store_true",
        help="Run 5s of Camera → FaceEyeDetector → PreprocessPipeline → PupilDetector → GlintDetector "
             "live: per-eye found/confidence/distance_from_pupil/reason per second, save 1 ROI with "
             "pupil ellipse + glint dot + yellow pupil→glint vector line to results/phase5_sample_glint.png.",
    )
    parser.add_argument(
        "--feature-test",
        action="store_true",
        help="Run 8s of Camera → FaceEyeDetector → PreprocessPipeline → PupilDetector → GlintDetector → "
             "GazeFeatureExtractor LIVE. Prints once/sec: vector_x, vector_y, confidence, eyes_used, fusion_mode. "
             "Hold gaze steady then gently move your head — normalized vectors stay stable vs raw pixel coords.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (defaults to configs/default.yaml).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.dry_run:
        print(
            f"PACE foundation OK (camera_index={cfg.camera.camera_index}, "
            f"fps={cfg.camera.fps}, resolution={cfg.camera.resolution}, "
            f"detection.backend={cfg.detection.backend}, "
            f"preprocess.threshold_mode={cfg.preprocess.threshold_mode}, "
            f"pupil.min_circularity={cfg.pupil.min_circularity}, "
            f"glint.backend={cfg.glint.detector_backend}, "
            f"gaze_feature.fusion={cfg.gaze_feature.fusion_mode}, "
            f"eye_scale_source={cfg.gaze_feature.eye_scale_source})"
        )
        return 0

    if args.camera_test:
        return _run_camera_test(cfg)

    if args.detection_test:
        return _run_detection_test(cfg)

    if args.preprocess_test:
        return _run_preprocess_test(cfg)

    if args.pupil_test:
        return _run_pupil_test(cfg)

    if args.glint_test:
        return _run_glint_test(cfg)

    if args.feature_test:
        return _run_feature_test(cfg)

    print("PACE: Full pipeline not implemented yet. Use --dry-run, --camera-test, --detection-test, --preprocess-test, --pupil-test, --glint-test, or --feature-test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
