"""Cross-trial calibration validation.

Calibration is fitted on one trial. This checks it against the *other* trials of
the same task, which is the only way to tell a calibration that generalizes from
one that has memorised its own frames.

    python calibration/validate_calibration.py --task_number 30 --exclude_trial 1

What it reports and why
-----------------------
The reprojection RMS a calibration run prints about itself is not evidence: it
is the residual of the fit that produced the numbers, and a calibration with a
systematically wrong focal length drives it just as low. Held-out trials give an
honest figure. The Sampson error adds a check that is independent of the board:
it measures whether corresponding points in the two views actually satisfy the
epipolar geometry the extrinsics claim.

For an absolute check against known truth rather than a held-out consistency
check, see ``validation/validate_calibration_synthetic.py``, which renders a
board through a camera whose parameters are known and reports the error in the
recovered focal length, principal point and baseline.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import yaml
from tqdm import tqdm

from pose3d.calibrate import detect_boards, sampson_errors
from pose3d.camera import load_camera_pair
from pose3d.geometry import triangulate_points
from pose3d.video import default_workers, probe

sys.path.insert(0, str(Path(__file__).resolve().parent))
from calibration import board_from_settings, load_settings  # noqa: E402


def validate_trial(
    trial_dir: Path, cameras, board, settings: dict, workers: int, sample_fps: float
) -> Optional[Dict[str, float]]:
    """Reprojection, epipolar and depth statistics for one held-out trial."""
    import cv2

    videos = [trial_dir / "synchronized" / f"stereo{i}.mp4" for i in range(2)]
    if not all(v.is_file() for v in videos):
        return None

    info = probe(videos[0])
    detections = [
        detect_boards(str(v), board, sample_fps=sample_fps,
                      sharpness_threshold=float(settings.get("sharpness_threshold", 45)),
                      rotate=not info.is_portrait, workers=workers,
                      progress=tqdm, desc=f"{trial_dir.name}/{v.stem}")
        for v in videos
    ]
    by_frame = {d.frame: d for d in detections[1]}
    pairs = [(a, by_frame[a.frame]) for a in detections[0] if a.frame in by_frame]
    if len(pairs) < 5:
        return {"pairs": len(pairs)}

    obj = board.object_points()
    reproj: List[float] = []
    depths: List[float] = []
    pts0, pts1 = [], []

    for a, b in pairs:
        for cam, det in ((cameras.cam0, a), (cameras.cam1, b)):
            ok, rvec, tvec = cv2.solvePnP(obj, det.corners.reshape(-1, 1, 2),
                                          cam.K, cam.dist)
            if not ok:
                continue
            projected, _ = cv2.projectPoints(obj, rvec, tvec, cam.K, cam.dist)
            reproj.append(float(np.linalg.norm(
                projected.reshape(-1, 2) - det.corners, axis=1).mean()))

        u0 = cameras.cam0.undistort(a.corners)
        u1 = cameras.cam1.undistort(b.corners)
        pts0.append(a.corners.reshape(-1, 1, 2).astype(np.float32))
        pts1.append(b.corners.reshape(-1, 1, 2).astype(np.float32))
        points = triangulate_points(cameras.cam0.P, cameras.cam1.P, u0, u1)
        finite = points[np.isfinite(points).all(axis=1)]
        if finite.size:
            depths.extend(finite[:, 2].tolist())

    sampson = sampson_errors(pts0, pts1, cameras.cam0.K, cameras.cam0.dist,
                             cameras.cam1.K, cameras.cam1.dist,
                             cameras.cam1.R, cameras.cam1.t)
    # Sampson error is in normalized units; scale by focal length for pixels.
    focal = float(cameras.cam0.K[0, 0])

    return {
        "pairs": len(pairs),
        "reprojection_px": float(np.median(reproj)) if reproj else float("nan"),
        "sampson_px": float(np.median(sampson)) * focal if len(sampson) else float("nan"),
        "depth_min_m": float(np.percentile(depths, 1)) if depths else float("nan"),
        "depth_max_m": float(np.percentile(depths, 99)) if depths else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task_number", type=int, required=True)
    ap.add_argument("--exclude_trial", "--exclude_trial_number", type=int, required=True,
                    dest="exclude_trial",
                    help="the trial the calibration was fitted on")
    ap.add_argument("--settings", type=Path,
                    default=Path(__file__).with_name("calibration_settings.yaml"))
    ap.add_argument("--sample_fps", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=default_workers())
    args = ap.parse_args()

    task_dir = ROOT / "project" / f"task{args.task_number}"
    if not task_dir.is_dir():
        print(f"error: {task_dir} not found", file=sys.stderr)
        return 1

    settings = load_settings(args.settings)
    board = board_from_settings(settings)
    cameras = load_camera_pair(task_dir / "camera_parameters")

    print(f"Validating task {args.task_number} calibration "
          f"(fitted on trial {args.exclude_trial})")
    print(f"  baseline {cameras.baseline_m * 100:.1f} cm, "
          f"frame offset {cameras.frame_offset:+.0f}\n")

    excluded = f"trial{args.exclude_trial}"
    results: Dict[str, Dict[str, float]] = {}
    for trial_dir in sorted(task_dir.glob("trial*")):
        if not trial_dir.is_dir() or not re.fullmatch(r"trial\d+", trial_dir.name):
            continue
        if trial_dir.name == excluded:
            continue
        stats = validate_trial(trial_dir, cameras, board, settings,
                               args.workers, args.sample_fps)
        if stats is None:
            print(f"  {trial_dir.name}: no synchronised stereo videos, skipping")
            continue
        results[trial_dir.name] = stats

    if not results:
        print("\nNo held-out trials with synchronised footage were found. Record a "
              "second trial of the same task to validate against, or use "
              "validation/validate_calibration_synthetic.py for an absolute check.")
        return 1

    print(f"\n{'trial':<10s}{'pairs':>7s}{'reproj px':>12s}{'sampson px':>12s}"
          f"{'depth span m':>14s}")
    print("-" * 55)
    for name, s in results.items():
        if s.get("pairs", 0) < 5:
            print(f"{name:<10s}{s['pairs']:>7d}{'too few pairs':>26s}")
            continue
        span = f"{s['depth_min_m']:.2f}-{s['depth_max_m']:.2f}"
        print(f"{name:<10s}{s['pairs']:>7d}{s['reprojection_px']:>12.3f}"
              f"{s['sampson_px']:>12.3f}{span:>14s}")

    usable = [s for s in results.values() if s.get("pairs", 0) >= 5]
    if usable:
        worst = max(s["sampson_px"] for s in usable if np.isfinite(s["sampson_px"]))
        print()
        if worst > 2.0:
            print(f"Sampson error reaches {worst:.2f} px on held-out footage. The "
                  "extrinsics do not generalise -- most likely the cameras moved "
                  "between trials, or the calibration board never varied enough in "
                  "depth. Recalibrate on the trial you actually want to use.")
        else:
            print(f"Held-out Sampson error stays under {worst:.2f} px; the extrinsics "
                  "generalise across trials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
