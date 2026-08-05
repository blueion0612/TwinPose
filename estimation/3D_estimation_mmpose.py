"""3D reconstruction from MMPose COCO-WholeBody 2D results.

Same pipeline as ``estimation/3D_estimation.py``; only the 2D input format
differs, and that difference is handled by :mod:`pose3d.adapters`.

    python estimation/3D_estimation_mmpose.py --task_number 30 --trial_number 1 \
        --subject_height 1.72

Expects ``project/task<N>/trial<M>/2D/results_cam0.json`` and ``results_cam1.json``
in MMPose prediction format. Generate them with any WholeBody model::

    from mmpose.apis import MMPoseInferencer
    inferencer = MMPoseInferencer('wholebody')
    for _ in inferencer('project/task30/trial1/Estimation/cam0.mp4',
                        pred_out_dir='project/task30/trial1/2D'):
        pass
    # rename the produced JSON to results_cam0.json, and repeat for cam1

This used to be a 1895-line copy of the OpenPose pipeline. Keeping two copies
meant fixes landed in one and not the other -- the copy read the inter-camera
frame offset under a key calibration never wrote, so it always ran unaligned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from pose3d import BODY25B, ReconstructionConfig, load_camera_pair, reconstruct
from pose3d.adapters import load_mmpose_as_body25b
from pose3d.hands import compute_wrist_kinematics, reconstruct_hands
from pose3d.kpio import merge_metrics, save_json, save_keypoints_3d
from pose3d.skeleton import height_from_bone_lengths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task_number", type=int, required=True)
    ap.add_argument("--trial_number", type=int, required=True)
    ap.add_argument("--subject_height", type=float, default=None,
                    help="subject standing height in metres")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--min_confidence", type=float, default=0.35,
                    help="MMPose scores run higher than OpenPose's; raise this if "
                         "the reconstruction is picking up bad detections")
    ap.add_argument("--use_calibration_offset", action="store_true")
    ap.add_argument("--refine_passes", type=int, default=2)
    args = ap.parse_args()

    task_dir = ROOT / "project" / f"task{args.task_number}"
    trial_dir = task_dir / f"trial{args.trial_number}"
    dir_3d = trial_dir / "3D"
    dir_3d.mkdir(parents=True, exist_ok=True)
    prefix = f"task{args.task_number}_trial{args.trial_number}"

    sources = [trial_dir / "2D" / f"results_cam{i}.json" for i in range(2)]
    for path in sources:
        if not path.is_file():
            print(f"error: {path} not found", file=sys.stderr)
            return 1

    cameras = load_camera_pair(task_dir / "camera_parameters")
    body, conf, hands, hand_conf = [], [], [], []
    for path in sources:
        bk, bc, hk, hc = load_mmpose_as_body25b(path)
        body.append(BODY25B.select(bk))
        conf.append(BODY25B.select(bc))
        hands.append(hk)
        hand_conf.append(hc)

    n = min(b.shape[0] for b in body)
    body = [b[:n] for b in body]
    conf = [c[:n] for c in conf]
    hands = [h[:n] for h in hands]
    hand_conf = [c[:n] for c in hand_conf]

    print(f"MMPose input: {n} frames, mapped to {BODY25B.n_joints} body joints")
    print(f"Cameras {cameras.baseline_m * 100:.1f} cm apart; "
          f"calibration frame offset {cameras.frame_offset:+.0f}\n")

    cfg = ReconstructionConfig(
        fps=args.fps,
        subject_height_m=args.subject_height,
        min_confidence=args.min_confidence,
        sync_use_calibration_offset=args.use_calibration_offset,
        refine_passes=args.refine_passes,
    )
    result = reconstruct(
        body[0], conf[0], body[1], conf[1], cameras, BODY25B, cfg,
        progress=lambda it, **kw: tqdm(it, **kw),
    )

    for name, array in result.stages.items():
        save_keypoints_3d(dir_3d / f"{prefix}_kpts_3d_{name}.json", array)
    save_keypoints_3d(dir_3d / f"{prefix}_kpts_3d_final_processed.json", result.points3d)

    hands3d = reconstruct_hands(
        hands[0], hand_conf[0], hands[1], hand_conf[1],
        cameras, result.points3d, BODY25B, cfg,
    )
    merged = np.nanmean(np.stack(hand_conf), axis=0)
    merged = np.where(np.isfinite(merged), merged, 0.0)
    kin = compute_wrist_kinematics(hands3d, merged, result.points3d, BODY25B, cfg)
    save_keypoints_3d(dir_3d / f"{prefix}_hands_3d.json", hands3d)
    save_json(dir_3d / f"{prefix}_wrist_kinematics.json", kin.angles)
    save_json(dir_3d / f"{prefix}_hand_frames_3d.json", kin.frames)

    merge_metrics(task_dir / f"evaluation_metrics_trial{args.trial_number}.json",
                  f"trial{args.trial_number}", result.metrics)

    height = height_from_bone_lengths(result.bone_lengths)
    print("\n--- run summary ---")
    print(f"  frame offset used   {result.frame_offset:+.2f}")
    print(f"  T-pose frames       {len(result.t_pose_frames)}")
    if np.isfinite(height):
        print(f"  implied height      {height:.2f} m")
    print(f"  wrist angles        {kin.valid_left} left / {kin.valid_right} right frames")
    print("\n--- metrics ---")
    for key, value in result.metrics.items():
        print(f"  {key:28s} {value:12.4f}" if np.isfinite(value)
              else f"  {key:28s} {'n/a':>12s}")
    print(f"\nWritten to {dir_3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
