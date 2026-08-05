"""Inspect calibration footage before committing to a full calibration run.

Reports how detectable the checkerboard actually is, and how the sharpness
threshold in ``calibration_settings.yaml`` would filter the clip. Running this
first is much cheaper than discovering after a 25-minute calibration that half
the frames were motion-blurred.

    python tools/inspect_footage.py --video project/task30/mono0.mp4
    python tools/inspect_footage.py --video a.mp4 --compare b.mp4 --plot out.png

Replaces ``sharp.py`` and ``thresh.py``, which between them opened GUI windows,
blocked on key presses and wrote a PNG to the current working directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from tqdm import tqdm

from pose3d.synth_board import BoardSpec
from pose3d.video import default_workers, map_frames, probe


class _Inspector:
    """Picklable per-frame measurement."""

    def __init__(self, pattern, rotate: bool):
        self.pattern = pattern
        self.rotate = rotate

    def __call__(self, index: int, frame: np.ndarray):
        import cv2

        if self.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        found, corners = cv2.findChessboardCornersSB(
            gray, self.pattern,
            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY,
        )
        area = 0.0
        if found and corners is not None:
            hull = cv2.convexHull(corners.reshape(-1, 2).astype(np.float32))
            area = float(cv2.contourArea(hull)) / (gray.shape[0] * gray.shape[1])
        return {"sharpness": sharpness, "found": bool(found), "area": area}


def inspect(video: Path, board: BoardSpec, step: int, workers: int) -> Dict[str, object]:
    info = probe(video)
    indices = list(range(0, info.n_frames, max(1, step)))
    results = map_frames(video, _Inspector(board.pattern, not info.is_portrait),
                         indices=indices, workers=workers,
                         progress=tqdm, desc=video.name)
    sharp = np.array([r["sharpness"] for _, r in results])
    found = np.array([r["found"] for _, r in results])
    area = np.array([r["area"] for _, r in results])
    return {"info": info, "sharpness": sharp, "found": found, "area": area,
            "sampled": len(indices)}


def report(name: str, data: Dict[str, object], thresholds: List[float]) -> None:
    info, sharp, found, area = (data["info"], data["sharpness"],
                                data["found"], data["area"])
    print(f"\n--- {name} ---")
    print(f"  {info.n_frames} frames, {info.width}x{info.height} @ {info.fps:.0f} fps "
          f"({info.duration_s:.0f}s); sampled {data['sampled']}")
    if sharp.size == 0:
        print("  no frames read")
        return
    print(f"  board detected in {found.mean() * 100:.1f}% of sampled frames")
    if found.any():
        visible = area[found]
        print(f"  board covers {np.median(visible) * 100:.2f}% of the frame "
              f"(min {visible.min() * 100:.2f}%, max {visible.max() * 100:.2f}%)")
        if np.median(visible) < 0.01:
            print("  warning: the board is very small in frame. Move closer or print "
                  "a bigger board -- extrinsic accuracy depends on it directly.")
    print(f"  sharpness: median {np.median(sharp):.1f}, "
          f"p10 {np.percentile(sharp, 10):.1f}, p90 {np.percentile(sharp, 90):.1f}")
    print(f"\n  {'threshold':>10s}{'frames kept':>13s}{'of those, board found':>24s}")
    for t in thresholds:
        keep = sharp >= t
        n = int(keep.sum())
        rate = float(found[keep].mean() * 100) if n else 0.0
        print(f"  {t:>10.0f}{n:>8d} ({n / len(sharp) * 100:4.0f}%){rate:>21.1f}%")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None,
                    help="a second clip to inspect alongside, e.g. the other camera")
    ap.add_argument("--settings", type=Path,
                    default=ROOT / "calibration" / "calibration_settings.yaml")
    ap.add_argument("--step", type=int, default=5, help="sample every Nth frame")
    ap.add_argument("--workers", type=int, default=default_workers())
    ap.add_argument("--plot", type=Path, default=None, help="write a histogram PNG here")
    args = ap.parse_args()

    settings = yaml.safe_load(args.settings.read_text(encoding="utf-8")) or {}
    board = BoardSpec(rows=int(settings.get("checkerboard_rows", 5)),
                      cols=int(settings.get("checkerboard_columns", 8)),
                      square_size_m=float(settings.get("checkerboard_box_size_scale", 2.7)) / 100)
    configured = float(settings.get("sharpness_threshold", 45))
    thresholds = sorted({configured * f for f in (0.5, 0.75, 1.0, 1.5, 2.0)})

    videos = [args.video] + ([args.compare] if args.compare else [])
    collected = {}
    for video in videos:
        if not video.is_file():
            print(f"error: {video} not found", file=sys.stderr)
            return 1
        collected[video.name] = inspect(video, board, args.step, args.workers)
        report(video.name, collected[video.name], thresholds)

    print(f"\nConfigured sharpness_threshold is {configured:.0f}. Pick the lowest "
          "threshold whose detection rate is still high: filtering harder than that "
          "throws away usable views without improving the calibration.")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(collected), figsize=(7 * len(collected), 4.5),
                                 squeeze=False)
        for ax, (name, data) in zip(axes[0], collected.items()):
            ax.hist(data["sharpness"], bins=60, alpha=0.75, label="all sampled")
            if data["found"].any():
                ax.hist(data["sharpness"][data["found"]], bins=60, alpha=0.75,
                        label="board detected")
            ax.axvline(configured, color="r", ls="--", label=f"threshold {configured:.0f}")
            ax.set_title(name)
            ax.set_xlabel("Laplacian variance")
            ax.set_ylabel("frames")
            ax.legend()
        fig.tight_layout()
        args.plot.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot, dpi=130)
        print(f"histogram written to {args.plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
