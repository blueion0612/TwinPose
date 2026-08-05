"""3D pose reconstruction CLI.

Reads the 2D keypoints written by ``estimation/Openpose.py`` plus the camera
parameters from calibration, and writes the reconstructed 3D pose, the wrist
kinematics and the evaluation metrics.

    python estimation/3D_estimation.py --task_number 30 --trial_number 1
    python estimation/3D_estimation.py --task_number 30 --trial_number 1 --subject_height 1.78

The reconstruction itself lives in :mod:`pose3d.pipeline`; this file is argument
handling and file layout. That split is what makes the pipeline testable -- see
``validation/run_benchmark.py``, which drives the same code on synthetic motion
with known ground truth.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from pose3d import BODY25B, ReconstructionConfig, load_camera_pair, reconstruct
from pose3d.hands import compute_wrist_kinematics, reconstruct_hands
from pose3d.kpio import (
    load_body_keypoints,
    load_hand_keypoints,
    merge_metrics,
    save_json,
    save_keypoints_3d,
)
from pose3d.skeleton import HANDS_N_JOINTS, height_from_bone_lengths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task_number", type=int, required=True)
    ap.add_argument("--trial_number", type=int, required=True)
    ap.add_argument("--subject_height", type=float, default=None,
                    help="subject standing height in metres; anchors the bone prior")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--use_calibration_offset", action="store_true",
                    help="trust the frame offset calibration chose instead of "
                         "re-estimating it from the keypoints")
    ap.add_argument("--refine_passes", type=int, default=2)
    ap.add_argument("--no_intermediate", action="store_true",
                    help="skip writing the per-step JSON files")
    ap.add_argument("--config", type=Path, default=None,
                    help="JSON file of ReconstructionConfig overrides")
    args = ap.parse_args()

    task_dir = ROOT / "project" / f"task{args.task_number}"
    trial_dir = task_dir / f"trial{args.trial_number}"
    dir_2d = trial_dir / "2D"
    dir_3d = trial_dir / "3D"
    dir_3d.mkdir(parents=True, exist_ok=True)
    prefix = f"task{args.task_number}_trial{args.trial_number}"

    body_paths = [dir_2d / f"kpts_cam{i}_all.json" for i in range(2)]
    hand_paths = [dir_2d / f"hands_cam{i}_all.json" for i in range(2)]
    params_dir = task_dir / "camera_parameters"

    for path in body_paths:
        if not path.is_file():
            print(f"error: {path} not found. Run estimation/Openpose.py first.",
                  file=sys.stderr)
            return 1
    if not (params_dir / "camera1_extrinsics.json").is_file():
        print(f"error: no calibration in {params_dir}. Run calibration first.",
              file=sys.stderr)
        return 1

    # ---------------- config ---------------- #
    overrides = {}
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
    cfg = ReconstructionConfig(
        fps=args.fps,
        subject_height_m=args.subject_height,
        sync_use_calibration_offset=args.use_calibration_offset,
        refine_passes=args.refine_passes,
        save_intermediate=not args.no_intermediate,
    )
    if overrides:
        cfg = cfg.replace(**overrides)
    cfg.validate()

    # ---------------- load ---------------- #
    cameras = load_camera_pair(params_dir)
    n_raw = len(BODY25B.raw_names)
    kpts, confs = [], []
    for path in body_paths:
        k, c = load_body_keypoints(path, n_raw)
        kpts.append(BODY25B.select(k))
        confs.append(BODY25B.select(c))

    print(f"Task {args.task_number} trial {args.trial_number}: "
          f"{kpts[0].shape[0]} frames, {BODY25B.n_joints} joints")
    print(f"Cameras {cameras.baseline_m * 100:.1f} cm apart; "
          f"calibration frame offset {cameras.frame_offset:+.0f}\n")

    # ---------------- reconstruct ---------------- #
    result = reconstruct(
        kpts[0], confs[0], kpts[1], confs[1], cameras, BODY25B, cfg,
        progress=lambda it, **kw: tqdm(it, **kw),
    )

    if cfg.save_intermediate:
        for name, array in result.stages.items():
            save_keypoints_3d(dir_3d / f"{prefix}_kpts_3d_{name}.json", array)
    save_keypoints_3d(dir_3d / f"{prefix}_kpts_3d_final_processed.json", result.points3d)

    # ---------------- hands ---------------- #
    hand_kpts, hand_confs = [], []
    for path in hand_paths:
        k, c = load_hand_keypoints(path, result.n_frames)
        hand_kpts.append(k)
        hand_confs.append(c)

    have_hands = any(np.isfinite(c).any() for c in hand_confs)
    if have_hands:
        print("\nReconstructing hands and wrist kinematics...")
        hands3d = reconstruct_hands(
            hand_kpts[0], hand_confs[0], hand_kpts[1], hand_confs[1],
            cameras, result.points3d, BODY25B, cfg,
        )
        merged_conf = np.nanmean(np.stack(hand_confs), axis=0)
        merged_conf = np.where(np.isfinite(merged_conf), merged_conf, 0.0)
        kin = compute_wrist_kinematics(hands3d, merged_conf, result.points3d, BODY25B, cfg)
        save_keypoints_3d(dir_3d / f"{prefix}_hands_3d.json", hands3d)
        save_json(dir_3d / f"{prefix}_wrist_kinematics.json", kin.angles)
        save_json(dir_3d / f"{prefix}_hand_frames_3d.json", kin.frames)
        print(f"  wrist angles resolved on {kin.valid_left} left / "
              f"{kin.valid_right} right frames out of {result.n_frames}")
    else:
        print("\nNo hand keypoints found; skipping wrist kinematics.")
        save_json(dir_3d / f"{prefix}_wrist_kinematics.json", [])
        save_json(dir_3d / f"{prefix}_hand_frames_3d.json", [])

    # ---------------- metrics ---------------- #
    merge_metrics(task_dir / "evaluation_metrics.json",
                  f"trial{args.trial_number}", result.metrics)
    save_json(dir_3d / f"{prefix}_run_report.json",
              {"metrics": result.metrics,
               "frame_offset": result.frame_offset,
               "bone_lengths_m": result.bone_lengths,
               "t_pose_frames": result.t_pose_frames[:50],
               "stats": {k: v for k, v in result.stats.items() if k != "offset_log"}})

    height = height_from_bone_lengths(result.bone_lengths)
    print("\n--- run summary ---")
    print(f"  frame offset used     {result.frame_offset:+.2f}")
    print(f"  T-pose frames         {len(result.t_pose_frames)}")
    if np.isfinite(height):
        print(f"  implied height        {height:.2f} m")
    print(f"  elapsed               {result.stats.get('elapsed_seconds')} s")
    print("\n--- metrics ---")
    for key, value in result.metrics.items():
        print(f"  {key:28s} {value:12.4f}" if np.isfinite(value)
              else f"  {key:28s} {'n/a':>12s}")
    print(f"\nWritten to {dir_3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
