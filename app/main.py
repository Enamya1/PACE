import argparse
import sys

from app.config import load_config


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
        "--config",
        type=str,
        default=None,
        help="Path to config YAML (defaults to configs/default.yaml).",
    )
    args = parser.parse_args()

    if args.dry_run:
        cfg = load_config(args.config)
        print(f"PACE foundation OK (camera_index={cfg.camera.camera_index}, "
              f"fps={cfg.camera.fps}, resolution={cfg.camera.resolution})")
        return 0

    print("PACE: Full pipeline not implemented yet. Use --dry-run for foundation check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
