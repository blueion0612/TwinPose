"""Validate calibration against a camera whose parameters are known exactly.

A reprojection RMS says the optimiser found a self-consistent answer. It does
not say the answer is right: a calibration with a 10% focal-length error fits
its own detections just as well, and every downstream distance inherits that
error. This renders a checkerboard through a known camera, calibrates from the
render, and reports the error in the quantities that actually propagate --
focal length, principal point, baseline and relative pose.

Usage::

    python validation/validate_calibration_synthetic.py
    python validation/validate_calibration_synthetic.py --frames 400 --workers 8
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pose3d.calibrate import (
    calibrate_extrinsics,
    calibrate_intrinsics,
    compare_cameras,
    detect_boards,
)
from pose3d.camera import Camera, CameraPair, load_camera_pair
from pose3d.synth_board import BoardSpec, render_mono_video, render_stereo_videos
from pose3d.video import default_workers


def _fmt(value: float, width: int = 10, digits: int = 3) -> str:
    return f"{value:>{width}.{digits}f}"


def run(frames: int, workers: int, keep: Path | None, quiet: bool = False) -> Dict[str, object]:
    truth = load_camera_pair(ROOT / "project" / "task30" / "camera_parameters")
    board = BoardSpec()
    size = (1080, 1920)

    workdir = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="pose3d_calib_"))
    workdir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, object] = {"frames": frames, "workers": workers}

    try:
        # ---------------------------------------------------------------- #
        print(f"Rendering synthetic calibration footage into {workdir} ...")
        t0 = time.perf_counter()
        mono0 = workdir / "mono0.mp4"
        mono1 = workdir / "mono1.mp4"
        render_mono_video(mono0, truth.cam0, board, size, n_frames=frames, seed=1)
        render_mono_video(mono1, truth.cam1, board, size, n_frames=frames, seed=2)
        stereo0 = workdir / "stereo0.mp4"
        stereo1 = workdir / "stereo1.mp4"
        render_stereo_videos(stereo0, stereo1, truth, board, size, n_frames=frames, seed=3)
        report["render_seconds"] = round(time.perf_counter() - t0, 1)
        print(f"  rendered in {report['render_seconds']}s")

        # ---------------------------------------------------------------- #
        print("\nDetecting boards ...")
        t0 = time.perf_counter()
        det = {
            name: detect_boards(
                str(path), board, sample_fps=30.0, sharpness_threshold=0.0,
                workers=workers, desc=name,
            )
            for name, path in (
                ("mono0", mono0), ("mono1", mono1),
                ("stereo0", stereo0), ("stereo1", stereo1),
            )
        }
        report["detect_seconds"] = round(time.perf_counter() - t0, 1)
        for name, d in det.items():
            print(f"  {name:8s} {len(d):4d}/{frames} frames with a detected board")
        print(f"  detection took {report['detect_seconds']}s "
              f"({4 * frames / max(report['detect_seconds'], 1e-9):.0f} frames/s across 4 clips)")

        # ---------------------------------------------------------------- #
        print("\nIntrinsics")
        print(f"  {'camera':8s} {'fx err%':>10s} {'fy err%':>10s} {'cx err px':>10s} "
              f"{'cy err px':>10s} {'rms px':>8s} {'views':>6s}")
        intrinsics = {}
        for idx, (name, cam) in enumerate((("camera0", truth.cam0), ("camera1", truth.cam1))):
            result = calibrate_intrinsics(det[f"mono{idx}"], board, size, max_views=60)
            intrinsics[name] = result
            est = Camera(K=result.K, dist=result.dist, R=np.eye(3), t=np.zeros((3, 1)), name=name)
            metrics = compare_cameras(est, Camera(K=cam.K, dist=cam.dist, R=np.eye(3),
                                                  t=np.zeros((3, 1)), name=name))
            report[f"{name}_intrinsics"] = {k: round(v, 4) for k, v in metrics.items()
                                            if "rotation" not in k and "translation" not in k
                                            and "baseline" not in k}
            report[f"{name}_intrinsics"]["rms_px"] = round(result.rms_px, 4)
            print(f"  {name:8s} {_fmt(metrics['fx_error_percent'])} "
                  f"{_fmt(metrics['fy_error_percent'])} {_fmt(metrics['cx_error_px'])} "
                  f"{_fmt(metrics['cy_error_px'])} {result.rms_px:8.4f} {result.n_views:6d}")

        # ---------------------------------------------------------------- #
        print("\nExtrinsics (using the recovered intrinsics, not the true ones)")
        ext = calibrate_extrinsics(
            det["stereo0"], det["stereo1"], board,
            intrinsics["camera0"].K, intrinsics["camera0"].dist,
            intrinsics["camera1"].K, intrinsics["camera1"].dist,
            size, offsets=(-1, 0, 1), max_pairs=60,
        )
        est_cam1 = Camera(K=intrinsics["camera1"].K, dist=intrinsics["camera1"].dist,
                          R=ext.R, t=ext.t, name="camera1")
        metrics = compare_cameras(est_cam1, truth.cam1)
        true_baseline = float(np.linalg.norm(truth.cam1.t))
        print(f"  rotation error      {metrics['rotation_error_deg']:8.4f} deg")
        print(f"  translation error   {metrics['translation_error_mm']:8.2f} mm")
        print(f"  baseline            {ext.baseline_m * 1000:8.1f} mm  "
              f"(true {true_baseline * 1000:.1f} mm, error "
              f"{metrics['baseline_error_mm']:.1f} mm = "
              f"{metrics['baseline_error_mm'] / (true_baseline * 1000) * 100:.2f}%)")
        print(f"  stereo RMS          {ext.rms_px:8.4f} px over {ext.n_pairs} pairs")
        print(f"  chosen frame offset {ext.frame_offset:+d} "
              f"(per-offset RMS: "
              f"{', '.join(f'{k:+d}:{v:.3f}' for k, v in sorted(ext.per_offset_rms.items()))})")

        report["extrinsics"] = {k: round(v, 4) for k, v in metrics.items()}
        report["extrinsics"].update({
            "rms_px": round(ext.rms_px, 4),
            "n_pairs": ext.n_pairs,
            "chosen_offset": ext.frame_offset,
            "baseline_mm": round(ext.baseline_m * 1000, 2),
            "true_baseline_mm": round(true_baseline * 1000, 2),
        })
        return report
    finally:
        if keep is None:
            shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=240,
                    help="frames to render per clip (default 240)")
    ap.add_argument("--workers", type=int, default=default_workers(),
                    help="parallel detection workers")
    ap.add_argument("--keep", type=Path, default=None,
                    help="keep the rendered videos in this directory")
    ap.add_argument("--json", type=Path, default=None, help="write the report as JSON")
    args = ap.parse_args()

    report = run(args.frames, args.workers, args.keep)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
