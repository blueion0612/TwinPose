"""End-to-end accuracy benchmark on synthetic motion with known ground truth.

No videos, model weights or keypoint files ship with this repository, so the
pipeline cannot be re-run on the original recordings and the numbers quoted in
older versions of the README cannot be reproduced or checked. (They also
disagree with each other: the README reports a reprojection error of 9.66 px in
one section and 9.19 px in another, while the committed
``evaluation_metrics.json`` says 20.35 px for the same task and trial.)

This benchmark closes that gap. It builds a scripted motion by forward
kinematics, so bone lengths are exact; projects it through the **real**
calibrated cameras from ``project/task30``; corrupts the projections with a
detector noise model; and runs the real reconstruction pipeline. Because the
ground truth is known, it reports true accuracy -- MPJPE and PA-MPJPE -- rather
than only self-consistency.

    python validation/run_benchmark.py
    python validation/run_benchmark.py --seeds 5 --json results.json
    python validation/run_benchmark.py --ablation

Every number in the README's evaluation section comes from this script.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pose3d import BODY25B, ReconstructionConfig, load_camera_pair, reconstruct
from pose3d.geometry import triangulate_frames
from pose3d.metrics import (
    bone_length_cv_percent,
    jerk_rms,
    mpjpe_mm,
    pa_mpjpe_mm,
    pck3d,
    reprojection_error_px,
)
from pose3d.pipeline import bootstrap_single_view, synthesise_midhip
from pose3d.skeleton import scaled_bone_lengths
from pose3d.synth import DetectorNoise, make_sequence, perfect_observations, project_sequence

CAMERA_DIR = ROOT / "project" / "task30" / "camera_parameters"
SUBJECT_HEIGHT_M = 1.72


def base_config(**overrides) -> ReconstructionConfig:
    cfg = ReconstructionConfig(
        subject_height_m=SUBJECT_HEIGHT_M,
        verbose=False,
        sigma_bone_m=0.012,
        sigma_accel_m=0.005,
        refine_passes=2,
    )
    return cfg.replace(**overrides) if overrides else cfg


def prepare(observations: Dict[str, np.ndarray], cfg: ReconstructionConfig):
    """Condition raw observations exactly as the pipeline does, for stage metrics."""
    k0, c0 = synthesise_midhip(observations["kpts0"].copy(),
                               observations["conf0"].copy(), BODY25B)
    k1, c1 = synthesise_midhip(observations["kpts1"].copy(),
                               observations["conf1"].copy(), BODY25B)
    for k, c in ((k0, c0), (k1, c1)):
        k[np.nan_to_num(c, nan=-1.0) < cfg.min_confidence] = np.nan
    cams = load_camera_pair(CAMERA_DIR)
    return k0, c0, k1, c1, cams.cam0.undistort(k0), cams.cam1.undistort(k1), cams


def score(points: np.ndarray, gt: np.ndarray, u0, u1, cams, fps=30.0) -> Dict[str, float]:
    return {
        "MPJPE_mm": mpjpe_mm(points, gt),
        "PA_MPJPE_mm": pa_mpjpe_mm(points, gt),
        "PCK3D_50mm": pck3d(points, gt, 50.0),
        "PCK3D_150mm": pck3d(points, gt, 150.0),
        "Reprojection_px": reprojection_error_px(points, u0, u1, cams),
        "BoneLengthCV_percent": bone_length_cv_percent(points, BODY25B),
        "JerkRMS": jerk_rms(points, fps),
        "Coverage_percent": float(np.isfinite(points[..., 0]).mean() * 100.0),
    }


def run_seed(seed: int, cfg: ReconstructionConfig, noise: Optional[DetectorNoise] = None):
    """One full pipeline run, plus the intermediate stages for comparison."""
    cams = load_camera_pair(CAMERA_DIR)
    sequence = make_sequence(cameras=cams, subject_height_m=SUBJECT_HEIGHT_M)
    gt = sequence.points3d
    obs = (perfect_observations(sequence, cams) if noise == "perfect"
           else project_sequence(sequence, cams, noise, seed=seed))

    k0, c0, k1, c1, u0, u1, cams = prepare(obs, cfg)

    stages: Dict[str, Dict[str, float]] = {}
    triangulated = triangulate_frames(u0, u1, c0, c1, cams,
                                      min_confidence=cfg.min_confidence, undistort=False)
    stages["1_triangulation"] = score(triangulated, gt, u0, u1, cams, cfg.fps)

    prior = scaled_bone_lengths(BODY25B, cfg.subject_height_m)
    boot, _ = bootstrap_single_view(triangulated, (u0, u1), (c0, c1), cams,
                                    BODY25B, prior, cfg)
    stages["2_bootstrapped"] = score(boot, gt, u0, u1, cams, cfg.fps)

    started = time.perf_counter()
    result = reconstruct(obs["kpts0"], obs["conf0"], obs["kpts1"], obs["conf1"],
                         cams, BODY25B, cfg, ground_truth=gt)
    elapsed = time.perf_counter() - started
    stages["3_refined"] = score(result.points3d, gt, u0, u1, cams, cfg.fps)

    return {
        "seed": seed,
        "frames": int(gt.shape[0]),
        "elapsed_s": round(elapsed, 2),
        "frame_offset": result.frame_offset,
        "t_pose_frames": len(result.t_pose_frames),
        "stages": stages,
        "gt_jerk": jerk_rms(gt, cfg.fps),
    }


def aggregate(runs: List[Dict], key: str) -> Dict[str, Dict[str, float]]:
    """Mean and standard deviation of each metric, per stage."""
    out: Dict[str, Dict[str, float]] = {}
    for stage in runs[0]["stages"]:
        metrics = runs[0]["stages"][stage].keys()
        out[stage] = {}
        for metric in metrics:
            values = [r["stages"][stage][metric] for r in runs
                      if np.isfinite(r["stages"][stage][metric])]
            if values:
                out[stage][metric] = float(np.mean(values))
                out[stage][metric + "_sd"] = float(np.std(values))
    return out


#: (metric key, column heading) for the summary table.
_TABLE_COLUMNS = [
    ("MPJPE_mm", "MPJPE mm"),
    ("PA_MPJPE_mm", "PA-MPJPE mm"),
    ("PCK3D_50mm", "PCK@50mm %"),
    ("PCK3D_150mm", "PCK@150mm %"),
    ("Reprojection_px", "reproj px"),
    ("BoneLengthCV_percent", "boneCV %"),
    ("JerkRMS", "jerk RMS"),
    ("Coverage_percent", "coverage %"),
]


def print_table(summary: Dict[str, Dict[str, float]], title: str) -> None:
    print(f"\n=== {title} ===")
    header = f"{'stage':<18s}" + "".join(f"{label:>14s}" for _, label in _TABLE_COLUMNS)
    print(header)
    print("-" * len(header))
    for stage, metrics in summary.items():
        row = f"{stage:<18s}"
        for key, _ in _TABLE_COLUMNS:
            value = metrics.get(key)
            row += f"{value:>14.2f}" if value is not None else f"{'-':>14s}"
        print(row)


def _rotate_about(axis: np.ndarray, degrees: float) -> np.ndarray:
    a = np.asarray(axis, float)
    a = a / max(np.linalg.norm(a), 1e-12)
    th = np.deg2rad(degrees)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)


def perturbed_cameras(rotation_deg: float, baseline_percent: float, seed: int):
    """The camera pair as a *slightly wrong* calibration would have estimated it."""
    from pose3d.camera import Camera, CameraPair

    truth = load_camera_pair(CAMERA_DIR)
    rng = np.random.default_rng(seed + 7000)
    R = _rotate_about(rng.normal(size=3), rotation_deg) @ truth.cam1.R
    t = truth.cam1.t * (1.0 + baseline_percent / 100.0)
    cam1 = Camera(K=truth.cam1.K, dist=truth.cam1.dist, R=R, t=t, name="camera1")
    return CameraPair(truth.cam0, cam1, truth.frame_offset)


def joint_bias(n_joints: int, magnitude_mm: float, seed: int) -> np.ndarray:
    """A fixed per-joint offset, constant over time.

    Models the error this benchmark otherwise cannot see: a detector's notion of
    "hip" is not the anatomical hip joint centre, and that discrepancy is
    systematic rather than noise. The synthetic detector projects the true
    joints, so nothing else here represents it.
    """
    rng = np.random.default_rng(seed + 9000)
    return rng.normal(0.0, magnitude_mm / 1000.0 / np.sqrt(3), size=(n_joints, 3))


def run_error_budget(cfg: ReconstructionConfig, seeds: int) -> Dict[str, Dict[str, float]]:
    """How much does each departure from ideal conditions actually cost?

    The headline number assumes perfect calibration and zero-mean pixel noise.
    Neither holds in a real recording, and the differences are not small.
    """
    from dataclasses import replace as _replace

    truth = load_camera_pair(CAMERA_DIR)

    def measure(cameras_for_reconstruction, noiseless: bool, bias_mm: float):
        mpjpe, pa = [], []
        for seed in range(seeds):
            sequence = make_sequence(cameras=truth, subject_height_m=SUBJECT_HEIGHT_M)
            gt = sequence.points3d
            source = sequence
            if bias_mm > 0:
                offsets = joint_bias(gt.shape[1], bias_mm, seed)
                source = _replace(sequence, points3d=gt + offsets[None])
            obs = (perfect_observations(source, truth) if noiseless
                   else project_sequence(source, truth, seed=seed))
            cams = cameras_for_reconstruction(seed)
            result = reconstruct(obs["kpts0"], obs["conf0"], obs["kpts1"], obs["conf1"],
                                 cams, BODY25B, cfg)
            mpjpe.append(mpjpe_mm(result.points3d, gt))
            pa.append(pa_mpjpe_mm(result.points3d, gt))
        return {"MPJPE_mm": float(np.mean(mpjpe)), "PA_MPJPE_mm": float(np.mean(pa))}

    exact = lambda _seed: truth  # noqa: E731
    conditions = [
        ("perfect 2D, perfect calibration", exact, True, 0.0),
        ("detector noise, perfect calibration", exact, False, 0.0),
        ("+ rotation error 0.13 deg (measured)",
         lambda s: perturbed_cameras(0.13, 0.0, s), True, 0.0),
        ("+ rotation error 0.50 deg", lambda s: perturbed_cameras(0.50, 0.0, s), True, 0.0),
        ("+ rotation error 2.00 deg", lambda s: perturbed_cameras(2.00, 0.0, s), True, 0.0),
        ("+ baseline error 0.14% (measured)",
         lambda s: perturbed_cameras(0.0, 0.14, s), True, 0.0),
        ("+ baseline error 1.0%", lambda s: perturbed_cameras(0.0, 1.0, s), True, 0.0),
        ("+ baseline error 5.4% (old README)",
         lambda s: perturbed_cameras(0.0, 5.4, s), True, 0.0),
        ("realistic: measured calib + noise",
         lambda s: perturbed_cameras(0.13, 0.14, s), False, 0.0),
        ("realistic + 10 mm anatomical bias",
         lambda s: perturbed_cameras(0.13, 0.14, s), False, 10.0),
        ("realistic + 20 mm anatomical bias",
         lambda s: perturbed_cameras(0.13, 0.14, s), False, 20.0),
        ("realistic + 30 mm anatomical bias",
         lambda s: perturbed_cameras(0.13, 0.14, s), False, 30.0),
    ]

    print(f"\n=== Error budget ({seeds} seeds) ===")
    print(f"{'condition':<40s}{'MPJPE mm':>12s}{'PA-MPJPE mm':>14s}")
    print("-" * 66)
    out: Dict[str, Dict[str, float]] = {}
    for label, cameras, noiseless, bias in conditions:
        values = measure(cameras, noiseless, bias)
        out[label] = values
        print(f"{label:<40s}{values['MPJPE_mm']:>12.1f}{values['PA_MPJPE_mm']:>14.1f}")

    print("\nMPJPE keeps absolute scale; PA-MPJPE aligns each frame first and so "
          "measures pose *shape* only.\nA baseline error is a pure scale error: it "
          "wrecks MPJPE and leaves PA-MPJPE untouched.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3,
                    help="independent noise realisations to average over")
    ap.add_argument("--json", type=Path, default=None, help="write full results as JSON")
    ap.add_argument("--ablation", action="store_true",
                    help="also run with pieces of the pipeline disabled")
    ap.add_argument("--noiseless", action="store_true",
                    help="also run on noise-free projections, as an upper bound")
    ap.add_argument("--error-budget", action="store_true", dest="error_budget",
                    help="also measure what calibration error and anatomical bias cost")
    args = ap.parse_args()

    if not CAMERA_DIR.is_dir():
        print(f"error: {CAMERA_DIR} not found", file=sys.stderr)
        return 1

    cfg = base_config()
    cams = load_camera_pair(CAMERA_DIR)
    print(f"Cameras: baseline {cams.baseline_m * 100:.1f} cm, "
          f"calibration offset {cams.frame_offset:+.0f} frames")
    print(f"Subject: {SUBJECT_HEIGHT_M} m, motion = T-pose / squat / 4 steps / raise arm")
    print(f"Averaging over {args.seeds} noise seeds\n")

    runs = [run_seed(seed, cfg) for seed in range(args.seeds)]
    summary = aggregate(runs, "stages")
    print_table(summary, f"Accuracy against ground truth ({args.seeds} seeds)")
    print(f"\nground-truth jerk RMS for reference: {runs[0]['gt_jerk']:.1f}")
    print(f"mean runtime: {np.mean([r['elapsed_s'] for r in runs]):.1f}s "
          f"for {runs[0]['frames']} frames")
    print(f"frame offset recovered: "
          f"{[r['frame_offset'] for r in runs]} (true 0)")

    report: Dict[str, object] = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "config": cfg.to_dict(),
        "seeds": args.seeds,
        "summary": summary,
        "runs": runs,
    }

    if args.noiseless:
        clean = [run_seed(0, cfg, noise="perfect")]
        clean_summary = aggregate(clean, "stages")
        print_table(clean_summary, "Noise-free upper bound (geometry only)")
        report["noiseless"] = clean_summary

    if args.ablation:
        print("\n=== Ablation: what each part of the refinement contributes ===")
        variants = {
            "full": {},
            "no bone prior": {"sigma_bone_m": 10.0},
            "no smoothness prior": {"sigma_accel_m": 10.0},
            "no joint limits": {"sigma_limit": 100.0},
            "no refinement": {"refine_passes": 0},
            "1 pass": {"refine_passes": 1},
            # The subject is 1.72 m, the reference height, so a "generic" prior
            # would be identical. Use a wrong height instead, to show how much
            # the bone prior's accuracy actually matters.
            "bone prior 15% too tall": {"subject_height_m": 1.98},
        }
        rows = {}
        for name, overrides in variants.items():
            variant_cfg = base_config(**overrides) if overrides else cfg
            if overrides.get("refine_passes") == 0:
                variant_cfg = base_config(refine_passes=1, max_nfev=1)
            variant_runs = [run_seed(s, variant_cfg) for s in range(args.seeds)]
            final = aggregate(variant_runs, "stages")["3_refined"]
            rows[name] = final
            print(f"  {name:<22s} MPJPE {final['MPJPE_mm']:7.2f} mm  "
                  f"PA {final['PA_MPJPE_mm']:7.2f} mm  "
                  f"boneCV {final['BoneLengthCV_percent']:6.2f}%  "
                  f"jerk {final['JerkRMS']:8.1f}")
        report["ablation"] = rows

    if args.error_budget:
        report["error_budget"] = run_error_budget(cfg, args.seeds)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
        print(f"\nFull results written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
