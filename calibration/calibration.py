"""Stereo camera calibration CLI.

Intrinsics come from each camera's mono clip, extrinsics from the synchronized
stereo pair. The work happens in :mod:`pose3d.calibrate`; this file is argument
handling and reporting.

    python calibration/calibration.py --task_number 30 --trial_number 1
    python calibration/calibration.py --task_number 30 --trial_number 1 --force

Intrinsics are cached in ``camera_parameters/`` and reused across trials of the
same task, since they depend on the camera and not on where it was standing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from tqdm import tqdm

from pose3d.calibrate import (
    calibrate_extrinsics,
    calibrate_intrinsics,
    detect_boards,
)
from pose3d.camera import CM_PER_M, load_camera, save_extrinsics, save_intrinsics
from pose3d.synth_board import BoardSpec
from pose3d.video import default_workers, probe


def load_settings(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    required = ("checkerboard_rows", "checkerboard_columns", "checkerboard_box_size_scale")
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(f"{path}: missing settings {missing}")
    return data


def board_from_settings(settings: dict) -> BoardSpec:
    """Build a board spec, converting the file's centimetres into meters.

    ``checkerboard_box_size_scale`` is documented as centimetres. The old code
    carried that unit straight through into the extrinsics and every downstream
    distance, which is why the 3D stage needed a mystery scale factor of ~106.
    """
    return BoardSpec(
        rows=int(settings["checkerboard_rows"]),
        cols=int(settings["checkerboard_columns"]),
        square_size_m=float(settings["checkerboard_box_size_scale"]) / CM_PER_M,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task_number", type=int, required=True)
    ap.add_argument("--trial_number", type=int, required=True)
    ap.add_argument("--settings", type=Path,
                    default=Path(__file__).with_name("calibration_settings.yaml"))
    ap.add_argument("--sample_fps", type=float, default=5.0,
                    help="frames per second to sample from each clip")
    ap.add_argument("--max_views", type=int, default=60,
                    help="board views kept for intrinsics, after diversity selection")
    ap.add_argument("--max_pairs", type=int, default=60,
                    help="stereo pairs kept for extrinsics")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="parallel board-detection workers")
    ap.add_argument("--force", action="store_true",
                    help="recalibrate even if parameter files already exist")
    args = ap.parse_args()

    settings = load_settings(args.settings)
    board = board_from_settings(settings)

    task_dir = ROOT / "project" / f"task{args.task_number}"
    trial_dir = task_dir / f"trial{args.trial_number}"
    params_dir = task_dir / "camera_parameters"
    params_dir.mkdir(parents=True, exist_ok=True)

    mono = [task_dir / "mono0.mp4", task_dir / "mono1.mp4"]
    stereo = [trial_dir / "synchronized" / "stereo0.mp4",
              trial_dir / "synchronized" / "stereo1.mp4"]

    print(f"Board: {board.cols}x{board.rows} inner corners, "
          f"{board.square_size_m * 1000:.1f} mm squares")
    print(f"Detection workers: {args.workers}\n")

    # ---------------- intrinsics ---------------- #
    intrinsics = []
    for i in range(2):
        intr_path = params_dir / f"camera{i}_intrinsics.json"
        if intr_path.is_file() and not args.force:
            data = json.loads(intr_path.read_text(encoding="utf-8"))
            intrinsics.append(
                (np.array(data["intrinsic"], float), np.array(data["distortion"], float))
            )
            print(f"[camera{i}] intrinsics loaded from {intr_path.name} (use --force to redo)")
            continue

        if not mono[i].is_file():
            print(f"error: {mono[i]} not found", file=sys.stderr)
            return 1
        info = probe(mono[i])
        print(f"[camera{i}] scanning {mono[i].name} "
              f"({info.n_frames} frames, {info.width}x{info.height})")
        t0 = time.perf_counter()
        detections = detect_boards(
            str(mono[i]), board,
            sample_fps=args.sample_fps,
            sharpness_threshold=float(settings.get("sharpness_threshold", 45)),
            rotate=not info.is_portrait,
            workers=args.workers, progress=tqdm, desc=f"camera{i} boards",
        )
        size = (info.width, info.height) if info.is_portrait else (info.height, info.width)
        print(f"  found the board in {len(detections)} frames "
              f"in {time.perf_counter() - t0:.1f}s")
        result = calibrate_intrinsics(detections, board, size, max_views=args.max_views)
        print(f"  RMS {result.rms_px:.4f} px over {result.n_views} views | "
              f"f = ({result.K[0, 0]:.1f}, {result.K[1, 1]:.1f}) px, "
              f"c = ({result.K[0, 2]:.1f}, {result.K[1, 2]:.1f}) px")
        save_intrinsics(intr_path, result.K, result.dist)
        intrinsics.append((result.K, result.dist))
        print(f"  saved to {intr_path}")

    # ---------------- extrinsics ---------------- #
    extr1_path = params_dir / "camera1_extrinsics.json"
    if extr1_path.is_file() and not args.force:
        print(f"\n[stereo] extrinsics already exist at {extr1_path.name} "
              f"(use --force to redo)")
        return 0

    for path in stereo:
        if not path.is_file():
            print(f"\nerror: {path} not found. Run the synchronisation stage first.",
                  file=sys.stderr)
            return 1

    print("\n[stereo] scanning both clips")
    info = probe(stereo[0])
    size = (info.width, info.height) if info.is_portrait else (info.height, info.width)
    t0 = time.perf_counter()
    det = [
        detect_boards(
            str(stereo[i]), board,
            sample_fps=args.sample_fps,
            sharpness_threshold=float(settings.get("sharpness_threshold", 45)),
            rotate=not info.is_portrait,
            workers=args.workers, progress=tqdm, desc=f"stereo{i} boards",
        )
        for i in range(2)
    ]
    print(f"  camera0: {len(det[0])} detections | camera1: {len(det[1])} detections "
          f"({time.perf_counter() - t0:.1f}s)")

    result = calibrate_extrinsics(
        det[0], det[1], board,
        intrinsics[0][0], intrinsics[0][1],
        intrinsics[1][0], intrinsics[1][1],
        size, offsets=(-1, 0, 1), max_pairs=args.max_pairs,
    )
    print(f"  chosen frame offset {result.frame_offset:+d} "
          f"(RMS per offset: "
          f"{', '.join(f'{k:+d}:{v:.3f}' for k, v in sorted(result.per_offset_rms.items()))})")
    print(f"  stereo RMS {result.rms_px:.4f} px over {result.n_pairs} pairs")
    print(f"  baseline {result.baseline_m * 100:.1f} cm")
    if result.rms_px > 1.0:
        print("  warning: RMS above 1 px. A low RMS does not prove the calibration is "
              "correct, but a high one does mean something is wrong -- check that the "
              "board is sharp and that both cameras see it from varied distances.")

    save_extrinsics(params_dir / "camera0_extrinsics.json", np.eye(3), np.zeros((3, 1)))
    save_extrinsics(extr1_path, result.R, result.t * CM_PER_M,
                    best_offset=result.frame_offset)
    print(f"  saved to {params_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
