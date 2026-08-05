"""Pipeline runner.

Chains the individual stages so a trial can be processed with one command.

    # 0) check what inference will actually run on before committing an hour
    python main.py check

    # 1) align the clips and write a preview, to choose --skip_start
    python main.py preview --task 30 --trial 1

    # 2) the whole pipeline: sync -> calibrate -> 2D -> 3D
    python main.py run --task 30 --trial 1 --skip_start 30 --subject_height 1.72

    # validation and reporting
    python main.py validate --task 30 --exclude_trial 1
    python main.py benchmark
    python main.py plot --task 30 --trial 1
    python main.py compare --tasks 30 45 60 90

Each stage is also a standalone script; this only sequences them.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

ROOT = Path(__file__).resolve().parent
SCRIPTS = {
    "sync": ROOT / "synchronize" / "synchronizevideo.py",
    "calibrate": ROOT / "calibration" / "calibration.py",
    "validate": ROOT / "calibration" / "validate_calibration.py",
    "openpose": ROOT / "estimation" / "Openpose.py",
    "estimate": ROOT / "estimation" / "3D_estimation.py",
    "estimate_mmpose": ROOT / "estimation" / "3D_estimation_mmpose.py",
    "plot": ROOT / "estimation" / "plot.py",
    "compare": ROOT / "estimation" / "compare.py",
    "benchmark": ROOT / "validation" / "run_benchmark.py",
    "calib_check": ROOT / "validation" / "validate_calibration_synthetic.py",
}


def run_step(name: str, script: Path, arguments: Sequence[object]) -> None:
    command = [sys.executable, str(script)] + [str(a) for a in arguments]
    print(f"\n{'=' * 62}\n  {name}\n{'=' * 62}")
    print(f"$ {' '.join(command)}\n", flush=True)
    result = subprocess.run(command)
    if result.returncode != 0:
        print(f"\n[{name}] failed with exit code {result.returncode}", file=sys.stderr)
        raise SystemExit(result.returncode)
    print(f"\n[{name}] done")


def check_inputs(task: int, trial: int) -> None:
    """Fail early and specifically, rather than an hour into the run."""
    task_dir = ROOT / "project" / f"task{task}"
    trial_dir = task_dir / f"trial{trial}"
    required = {
        "task directory": task_dir,
        "trial directory": trial_dir,
        "intrinsics clip, camera 0": task_dir / "mono0.mp4",
        "intrinsics clip, camera 1": task_dir / "mono1.mp4",
        "stereo clip, camera 0": trial_dir / "stereo0.mp4",
        "stereo clip, camera 1": trial_dir / "stereo1.mp4",
    }
    missing = [f"  {label}: {path}" for label, path in required.items() if not path.exists()]
    if missing:
        print("Missing inputs:\n" + "\n".join(missing), file=sys.stderr)
        print("\nSee project/README.md for the expected layout.", file=sys.stderr)
        raise SystemExit(1)
    print("Input check passed.")


def cmd_check(args) -> None:
    run_step("Inference device", SCRIPTS["openpose"],
             ["--task_number", 0, "--trial_number", 0, "--check_only"])


def cmd_preview(args) -> None:
    check_inputs(args.task, args.trial)
    run_step("Synchronisation preview", SCRIPTS["sync"],
             ["--task_number", args.task, "--trial_number", args.trial, "--stage", 1])
    print("\nWatch the montage in project/task*/trial*/synchronized/, decide how many "
          "seconds of checkerboard footage to keep, then run:\n"
          f"  python main.py run --task {args.task} --trial {args.trial} --skip_start <seconds>")


def cmd_run(args) -> None:
    check_inputs(args.task, args.trial)
    run_step("1/4 Synchronise and split", SCRIPTS["sync"],
             ["--task_number", args.task, "--trial_number", args.trial,
              "--stage", 2, "--skip_start", args.skip_start])
    run_step("2/4 Calibrate", SCRIPTS["calibrate"],
             ["--task_number", args.task, "--trial_number", args.trial]
             + (["--force"] if args.recalibrate else []))
    run_step("3/4 2D pose estimation", SCRIPTS["openpose"],
             ["--task_number", args.task, "--trial_number", args.trial]
             + (["--cpu"] if args.cpu else []))
    estimate_args: List[object] = ["--task_number", args.task, "--trial_number", args.trial]
    if args.subject_height:
        estimate_args += ["--subject_height", args.subject_height]
    run_step("4/4 3D reconstruction", SCRIPTS["estimate"], estimate_args)
    print(f"\nFinished. Results in project/task{args.task}/trial{args.trial}/3D/")


def cmd_validate(args) -> None:
    run_step("Cross-trial calibration validation", SCRIPTS["validate"],
             ["--task_number", args.task, "--exclude_trial", args.exclude_trial])


def cmd_benchmark(args) -> None:
    extra: List[object] = ["--seeds", args.seeds]
    if args.ablation:
        extra.append("--ablation")
    if args.error_budget:
        extra.append("--error-budget")
    if args.json:
        extra += ["--json", args.json]
    run_step("Synthetic accuracy benchmark", SCRIPTS["benchmark"], extra)


def cmd_calib_check(args) -> None:
    run_step("Synthetic calibration validation", SCRIPTS["calib_check"],
             ["--frames", args.frames])


def cmd_plot(args) -> None:
    extra: List[object] = ["--task_number", args.task]
    if args.trial is not None:
        extra += ["--trial_number", args.trial]
    if args.stages:
        extra.append("--stages")
    if args.all_trials:
        extra.append("--all_trials")
    if args.save:
        extra += ["--save", args.save]
    run_step("Visualise", SCRIPTS["plot"], extra)


def cmd_compare(args) -> None:
    extra: List[object] = ["--tasks"] + list(args.tasks)
    if args.csv:
        extra += ["--csv", args.csv]
    run_step("Compare", SCRIPTS["compare"], extra)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="report which device 2D inference will use")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("preview", help="align the clips and write a montage")
    p.add_argument("--task", "--task_number", type=int, required=True, dest="task")
    p.add_argument("--trial", "--trial_number", type=int, required=True, dest="trial")
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("run", help="run the full pipeline for one trial")
    p.add_argument("--task", "--task_number", type=int, required=True, dest="task")
    p.add_argument("--trial", "--trial_number", type=int, required=True, dest="trial")
    p.add_argument("--skip_start", type=float, required=True,
                   help="seconds of calibration footage after the checkerboard appears")
    p.add_argument("--subject_height", type=float, default=None,
                   help="subject standing height in metres; sharpens the bone prior")
    p.add_argument("--recalibrate", action="store_true",
                   help="redo calibration even if parameters already exist")
    p.add_argument("--cpu", action="store_true", help="force CPU 2D inference")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("validate", help="validate calibration on held-out trials")
    p.add_argument("--task", "--task_number", type=int, required=True, dest="task")
    p.add_argument("--exclude_trial", "--exclude_trial_number", type=int, required=True,
                   dest="exclude_trial")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("benchmark", help="synthetic accuracy benchmark with known truth")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--ablation", action="store_true",
                   help="measure what each refinement term contributes")
    p.add_argument("--error-budget", action="store_true", dest="error_budget",
                   help="measure what calibration error and anatomical bias cost")
    p.add_argument("--json", type=Path, default=None)
    p.set_defaults(func=cmd_benchmark)

    p = sub.add_parser("calib-check", help="validate calibration against known cameras")
    p.add_argument("--frames", type=int, default=240)
    p.set_defaults(func=cmd_calib_check)

    p = sub.add_parser("plot", help="visualise 3D results")
    p.add_argument("--task", "--task_number", type=int, required=True, dest="task")
    p.add_argument("--trial", "--trial_number", type=int, default=None, dest="trial")
    p.add_argument("--stages", action="store_true")
    p.add_argument("--all_trials", action="store_true")
    p.add_argument("--save", type=Path, default=None)
    p.set_defaults(func=cmd_plot)

    p = sub.add_parser("compare", help="compare metrics across tasks")
    p.add_argument("--tasks", type=int, nargs="+", required=True)
    p.add_argument("--csv", type=Path, default=None)
    p.set_defaults(func=cmd_compare)

    return ap


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    parsed.func(parsed)
