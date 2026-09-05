"""Video synchronization CLI.

Aligns the two stereo clips using the green-flash signal, then splits them into
the calibration segment and the pose-estimation segment.

    # 1) preview: align the clips and write a side-by-side montage
    python synchronize/synchronizevideo.py --task_number 30 --trial_number 1 --stage 1

    # 2) split: everything from the checkerboard onset to +N seconds is
    #    calibration footage, the rest is pose-estimation footage
    python synchronize/synchronizevideo.py --task_number 30 --trial_number 1 \
        --stage 2 --skip_start 30

Stage 1 exists so you can watch the montage and pick ``--skip_start``.

The old version cached flash detections with joblib keyed only on the filename,
which meant re-recording a trial while keeping its number silently reused the
previous video's flash timings. The cache here keys on the file's size and
modification time as well, so a changed file invalidates it automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from tqdm import tqdm

from pose3d.sync import (
    detect_flashes,
    find_first_board_frame,
    green_area_series,
    match_flash_sequences,
)
from pose3d.video import default_workers, probe, read_frames, write_video


def cache_key(path: Path) -> str:
    """Identity that changes when the file does."""
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def cached_green_series(path: Path, cache_dir: Path, workers: int, scale: float):
    """Green-pixel series, memoised on the file's content identity."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"green_{cache_key(path)}.npz"
    if cache_file.is_file():
        data = np.load(cache_file)
        print(f"  {path.name}: loaded cached flash scan")
        return data["series"], float(data["fps"])
    series, fps = green_area_series(str(path), scale=scale, workers=workers,
                                    progress=tqdm, desc=f"{path.name} flash scan")
    np.savez_compressed(cache_file, series=series, fps=fps)
    return series, fps


def montage(frame0: np.ndarray, frame1: np.ndarray, height: int = 720) -> np.ndarray:
    """Two frames side by side, scaled to a common height."""
    import cv2

    def fit(frame):
        h, w = frame.shape[:2]
        return cv2.resize(frame, (int(round(w * height / h)), height))

    a, b = fit(frame0), fit(frame1)
    return np.hstack([a, b])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task_number", type=int, required=True)
    ap.add_argument("--trial_number", type=int, required=True)
    ap.add_argument("--stage", type=int, choices=[1, 2], required=True,
                    help="1 = align and preview, 2 = align and split")
    ap.add_argument("--skip_start", type=float, default=0.0,
                    help="seconds of calibration footage after the checkerboard "
                         "appears (required for stage 2)")
    ap.add_argument("--scale", type=float, default=0.125,
                    help="downscale factor for the flash scan")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--max_board_search", type=int, default=3600,
                    help="frames to scan for the checkerboard onset")
    ap.add_argument("--settings", type=Path,
                    default=ROOT / "calibration" / "calibration_settings.yaml")
    ap.add_argument("--no_cache", action="store_true")
    args = ap.parse_args()

    if args.stage == 2 and args.skip_start <= 0:
        ap.error("--skip_start must be greater than 0 for stage 2")

    trial_dir = ROOT / "project" / f"task{args.task_number}" / f"trial{args.trial_number}"
    videos = [trial_dir / "stereo0.mp4", trial_dir / "stereo1.mp4"]
    for path in videos:
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    sync_dir = trial_dir / "synchronized"
    est_dir = trial_dir / "Estimation"
    cache_dir = ROOT / "synchronize" / "cache" / f"task{args.task_number}" / f"trial{args.trial_number}"

    settings = yaml.safe_load(args.settings.read_text(encoding="utf-8")) or {}
    pattern = (int(settings.get("checkerboard_columns", 8)),
               int(settings.get("checkerboard_rows", 5)))

    # ---------------- flash alignment ---------------- #
    print("Scanning for green flashes...")
    series, fps = [], []
    for path in videos:
        if args.no_cache:
            s, f = green_area_series(str(path), scale=args.scale, workers=args.workers,
                                     progress=tqdm, desc=f"{path.name} flash scan")
        else:
            s, f = cached_green_series(path, cache_dir, args.workers, args.scale)
        series.append(s)
        fps.append(f)

    events = [detect_flashes(s, f) for s, f in zip(series, fps)]
    print(f"  camera0: {len(events[0])} flashes | camera1: {len(events[1])} flashes")
    result = match_flash_sequences(events[0], events[1], fps[0])
    print(f"  {result.summary()}")
    if result.matched < 2:
        print("error: could not match flashes between the two clips. Check that the "
              "green light is visible in both.", file=sys.stderr)
        return 1
    if result.confidence < 0.5:
        print("  warning: fewer than half the flashes matched; the alignment may be wrong.")

    # Positive offset means camera1 lags, so camera1 starts later in its own file.
    starts = [max(0, -result.offset_frames), max(0, result.offset_frames)]
    print(f"  aligned start frames: camera0 {starts[0]}, camera1 {starts[1]}")

    # ---------------- checkerboard onset ---------------- #
    print("\nLocating the checkerboard...")
    onsets = []
    for i, path in enumerate(videos):
        found = find_first_board_frame(
            str(path), pattern, start=starts[i],
            max_frames=args.max_board_search, workers=args.workers, progress=tqdm,
        )
        onsets.append(found)
        print(f"  camera{i}: {'frame ' + str(found) if found is not None else 'not found'}")

    advance = 0
    if all(o is not None for o in onsets):
        # Advance both by the same amount so they stay aligned.
        advance = max(o - s for o, s in zip(onsets, starts))
        print(f"  advancing both clips by {advance} frames to the checkerboard")
    else:
        print("  checkerboard not found in both clips; starting from the flash alignment")

    calib_start = [s + advance for s in starts]
    info = [probe(p) for p in videos]
    remaining = min(i.n_frames - s for i, s in zip(info, calib_start))
    if remaining <= 0:
        print("error: no frames left after alignment", file=sys.stderr)
        return 1
    print(f"  {remaining} aligned frames available")

    # ---------------- stage 1: preview ---------------- #
    if args.stage == 1:
        sync_dir.mkdir(parents=True, exist_ok=True)
        out = sync_dir / f"task{args.task_number}_trial{args.trial_number}_preview.mp4"
        limit = min(remaining, int(fps[0] * 60))
        print(f"\nWriting {limit} montage frames to {out.name}...")
        gen0 = read_frames(videos[0], start=calib_start[0], stop=calib_start[0] + limit)
        gen1 = read_frames(videos[1], start=calib_start[1], stop=calib_start[1] + limit)
        frames = (montage(a, b) for (_, a), (_, b) in zip(gen0, gen1))
        written = write_video(out, tqdm(frames, total=limit, desc="preview"), fps[0])
        print(f"  wrote {written} frames to {out}")
        print("\nWatch it, decide how many seconds of checkerboard footage to keep, "
              "then run stage 2 with --skip_start.")
        return 0

    # ---------------- stage 2: split ---------------- #
    calib_frames = min(int(args.skip_start * fps[0]), remaining)
    est_frames = remaining - calib_frames
    if est_frames <= 0:
        print(f"error: --skip_start {args.skip_start}s consumes the whole clip",
              file=sys.stderr)
        return 1

    sync_dir.mkdir(parents=True, exist_ok=True)
    est_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSplitting: {calib_frames} calibration frames, {est_frames} estimation frames")

    for i, path in enumerate(videos):
        rotate = not info[i].is_portrait

        def clip(start, count, desc):
            for _, frame in tqdm(
                read_frames(path, start=start, stop=start + count), total=count, desc=desc
            ):
                if rotate:
                    import cv2
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                yield frame

        out_calib = sync_dir / f"stereo{i}.mp4"
        write_video(out_calib, clip(calib_start[i], calib_frames, f"cam{i} calib"), fps[i])

        out_est = est_dir / f"cam{i}.mp4"
        write_video(out_est, clip(calib_start[i] + calib_frames, est_frames, f"cam{i} est"),
                    fps[i])
        print(f"  camera{i}: {out_calib.name} + {out_est.name}")

    meta = {
        "flash_offset_frames": result.offset_frames,
        "matched_flashes": result.matched,
        "residual_ms": result.residual_ms,
        "confidence": result.confidence,
        "calibration_start": calib_start,
        "calibration_frames": calib_frames,
        "estimation_frames": est_frames,
        "fps": fps,
    }
    (sync_dir / "sync_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nWritten to {sync_dir} and {est_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
