"""Compare reconstruction quality across trials and camera placements.

    python estimation/compare.py --tasks 30 45 60 90
    python estimation/compare.py --tasks 30 --gt project/task30/gt.json

Three fixes over the previous version:

*Index space.* It used the full 26-joint BODY_25B map to index files the 3D
stage writes with 18 joints, so asking for ``LShoulder`` returned ``LWrist`` and
``LAnkle`` returned ``LBigToe``. Every joint lookup here goes through
:data:`pose3d.skeleton.BODY25B`, which is the same object the 3D stage used to
write the file.

*Discarded results.* It computed the inter-trial Procrustes and DTW metrics and
then overwrote both with NaN two loops later, so the two metrics the README
documents as inter-trial consistency were never reported.

*Foot slide.* ``pred[:, feet, [0, 2]]`` broadcasts the two index arrays together
and yields ``(F, 2)``, not the intended ``(F, 2, 2)``; the following
``norm(..., axis=2)`` then raised. It now selects the axes explicitly.
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from pose3d.kpio import load_json, load_keypoints_3d
from pose3d.metrics import (
    composite_score,
    foot_slide_rate_percent,
    mpjpe_mm,
    mpjve_mm,
    pa_mpjpe_mm,
    pck3d,
    sequence_distance_mm,
    static_pose_rms_mm,
)
from pose3d.pipeline import find_t_pose_frames
from pose3d.config import ReconstructionConfig
from pose3d.skeleton import BODY25B


def reconstruction_path(task_dir: Path, task: int, trial: int) -> Path:
    return (task_dir / f"trial{trial}" / "3D"
            / f"task{task}_trial{trial}_kpts_3d_final_processed.json")


def load_trial(task_dir: Path, task: int, trial: int) -> Optional[np.ndarray]:
    path = reconstruction_path(task_dir, task, trial)
    if not path.is_file():
        return None
    try:
        return load_keypoints_3d(path)
    except Exception as exc:                     # noqa: BLE001 - report and skip
        print(f"  warning: could not read {path.name}: {exc}")
        return None


def representative_pose(points: np.ndarray) -> Optional[np.ndarray]:
    """The clip's best T-pose frame, used for inter-trial pose comparison."""
    cfg = ReconstructionConfig(verbose=False)
    frames = find_t_pose_frames(points, BODY25B, cfg)
    if not frames:
        return None
    # Average the candidates: a single frame carries the detector's noise.
    stack = points[list(frames)]
    with np.errstate(invalid="ignore"):
        return np.nanmean(stack, axis=0)


def inter_trial_metrics(
    reconstructions: Dict[int, np.ndarray]
) -> Dict[int, Dict[str, float]]:
    """Pairwise consistency, averaged per trial."""
    out: Dict[int, Dict[str, List[float]]] = {t: {"static": [], "dtw": []} for t in reconstructions}
    poses = {t: representative_pose(p) for t, p in reconstructions.items()}

    for a, b in combinations(sorted(reconstructions), 2):
        if poses[a] is not None and poses[b] is not None:
            value = static_pose_rms_mm(poses[a], poses[b])
            if np.isfinite(value):
                out[a]["static"].append(value)
                out[b]["static"].append(value)
        value = sequence_distance_mm(reconstructions[a], reconstructions[b])
        if np.isfinite(value):
            out[a]["dtw"].append(value)
            out[b]["dtw"].append(value)

    return {
        t: {
            "StaticPoseRMS_inter_mm": float(np.mean(v["static"])) if v["static"] else np.nan,
            "DTW_RMS_inter_mm": float(np.mean(v["dtw"])) if v["dtw"] else np.nan,
        }
        for t, v in out.items()
    }


def ground_truth_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    n = min(pred.shape[0], gt.shape[0])
    if pred.shape[1] != gt.shape[1]:
        print(f"  warning: ground truth has {gt.shape[1]} joints, "
              f"reconstruction has {pred.shape[1]}; skipping GT metrics")
        return {}
    pred, gt = pred[:n], gt[:n]
    return {
        "MPJPE_mm": mpjpe_mm(pred, gt),
        "PA_MPJPE_mm": pa_mpjpe_mm(pred, gt),
        "PCK3D_150mm": pck3d(pred, gt, 150.0),
        "MPJVE_mm": mpjve_mm(pred, gt),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=int, nargs="+", required=True)
    ap.add_argument("--trials", type=int, nargs="*", default=None,
                    help="restrict to these trial numbers")
    ap.add_argument("--gt", type=Path, default=None,
                    help="ground-truth 3D keypoints JSON, enabling accuracy metrics")
    ap.add_argument("--csv", type=Path, default=None, help="write the table as CSV")
    args = ap.parse_args()

    project = ROOT / "project"
    gt = None
    if args.gt:
        if args.gt.is_file():
            gt = load_keypoints_3d(args.gt)
            print(f"Loaded ground truth: {gt.shape[0]} frames, {gt.shape[1]} joints")
        else:
            print(f"warning: ground truth not found at {args.gt}")

    rows: List[Dict[str, object]] = []

    for task in args.tasks:
        task_dir = project / f"task{task}"
        if not task_dir.is_dir():
            print(f"\nTask {task}: directory not found, skipping")
            continue

        metrics_file = load_json(task_dir / "evaluation_metrics.json", default={}) or {}
        trials = args.trials or sorted(
            int(k.replace("trial", "")) for k in metrics_file if k.startswith("trial")
        )
        if not trials:
            trials = sorted(
                int(p.name.replace("trial", ""))
                for p in task_dir.glob("trial*") if p.is_dir() and p.name[5:].isdigit()
            )
        if not trials:
            print(f"\nTask {task}: no trials found, skipping")
            continue

        print(f"\n--- Task {task} ---")
        reconstructions = {}
        for trial in trials:
            points = load_trial(task_dir, task, trial)
            if points is not None:
                reconstructions[trial] = points

        inter = inter_trial_metrics(reconstructions) if len(reconstructions) > 1 else {}

        for trial in trials:
            row: Dict[str, object] = {"Task": task, "Trial": trial}
            row.update(metrics_file.get(f"trial{trial}", {}))
            row.update(inter.get(trial, {}))

            points = reconstructions.get(trial)
            if points is not None:
                if "FootSlideRate_percent" not in row or not np.isfinite(
                    row.get("FootSlideRate_percent", np.nan)
                ):
                    row["FootSlideRate_percent"] = foot_slide_rate_percent(points, BODY25B)
                if gt is not None:
                    row.update(ground_truth_metrics(points, gt))
            row["Score"] = composite_score(
                {k: v for k, v in row.items() if isinstance(v, (int, float))}
            )
            rows.append(row)

    if not rows:
        print("\nNothing to compare.")
        return 1

    df = pd.DataFrame(rows)
    display_columns = [
        "Task", "Trial", "MPJPE_mm", "PA_MPJPE_mm", "PCK3D_150mm",
        "ReprojectionError_px", "BoneLengthCV_percent", "JerkRMS",
        "FootSlideRate_percent", "StaticPoseRMS_inter_mm", "DTW_RMS_inter_mm", "Score",
    ]
    present = [c for c in display_columns if c in df.columns and df[c].notna().any()]

    print("\n=== Per-trial ===")
    print(df[present].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    if len(args.tasks) > 1:
        print("\n=== Per-task means ===")
        numeric = [c for c in present if c not in ("Task", "Trial")]
        summary = df.groupby("Task")[numeric].mean()
        print(summary.to_string(float_format=lambda v: f"{v:.3f}"))
        best = summary["Score"].idxmin() if "Score" in summary else None
        if best is not None and np.isfinite(summary.loc[best, "Score"]):
            print(f"\nBest camera placement: task {best} "
                  f"(score {summary.loc[best, 'Score']:.3f}; lower is better)")

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nWritten to {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
