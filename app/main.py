import argparse
import logging
import sys
import time

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
                inst_fps = (fd.frame_number - prev_fn) if fd else 0
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
            f"fps={cfg.camera.fps}, resolution={cfg.camera.resolution})"
        )
        return 0

    if args.camera_test:
        return _run_camera_test(cfg)

    print("PACE: Full pipeline not implemented yet. Use --dry-run or --camera-test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
