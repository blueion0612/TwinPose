"""Visualise 3D reconstructions.

    # final result for one trial
    python estimation/plot.py --task_number 30 --trial_number 1

    # every pipeline stage side by side, to see what each one changed
    python estimation/plot.py --task_number 30 --trial_number 1 --stages

    # all trials of a task together
    python estimation/plot.py --task_number 30 --all_trials

    # export instead of showing (needs ffmpeg)
    python estimation/plot.py --task_number 30 --trial_number 1 --save out.mp4

Replaces the previous ``plot.py`` and ``plot_mmpose.py``, which were two
divergent copies with different axis conventions. There is one convention here:
the world's up axis is measured from the reconstruction (see
:func:`pose3d.pipeline.up_direction`) rather than assumed, because which axis
points up depends on how the checkerboard sat during calibration.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from pose3d.kpio import load_json, load_keypoints_3d
from pose3d.pipeline import up_direction
from pose3d.skeleton import BODY25B

STAGE_ORDER = [
    ("step1_triangulated", "1. Triangulation"),
    ("step2_bootstrapped", "2. Single-view bootstrap"),
    ("step3_refined", "3. Bundle adjustment"),
    ("step4_final", "4. Final"),
]


def world_to_plot(points: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Rotate the world so matplotlib's z axis is the subject's up axis."""
    z = up / max(np.linalg.norm(up), 1e-9)
    helper = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(helper, z)
    x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)
    return points @ np.column_stack([x, y, z])


def equalise(ax, points: np.ndarray) -> None:
    """Give all three axes the same scale, so the skeleton is not distorted."""
    finite = points[np.isfinite(points).all(axis=-1)]
    if finite.size == 0:
        return
    centre = finite.reshape(-1, 3).mean(axis=0)
    span = float(np.max(finite.reshape(-1, 3).ptp(axis=0))) or 1.0
    radius = span * 0.6
    ax.set_xlim(centre[0] - radius, centre[0] + radius)
    ax.set_ylim(centre[1] - radius, centre[1] + radius)
    ax.set_zlim(centre[2] - radius, centre[2] + radius)


class Panel:
    """One 3D subplot tracking one reconstruction."""

    def __init__(self, ax, points: np.ndarray, title: str,
                 hand_frames: Optional[List] = None,
                 kinematics: Optional[List] = None):
        self.ax = ax
        self.title = title
        self.hand_frames = hand_frames
        self.kinematics = kinematics

        up = up_direction(points, BODY25B)
        self.points = world_to_plot(points, up)
        self.up = up

        first = self.points[0]
        self.scatter = ax.scatter(first[:, 0], first[:, 1], first[:, 2], s=14, c="tab:blue")
        self.lines = [
            ax.plot(*[[first[a, k], first[b, k]] for k in range(3)],
                    color="0.25", lw=1.5)[0]
            for a, b in BODY25B.edges
        ]
        self.hand_lines = [
            ax.plot([], [], [], lw=2, color=c)[0]
            for c in ("r", "g", "b", "r", "g", "b")
        ]
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Up (m)")
        ax.view_init(elev=12, azim=-70)
        equalise(ax, self.points)

    def update(self, frame: int) -> None:
        if frame >= self.points.shape[0]:
            return
        pts = self.points[frame]
        self.scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        for line, (a, b) in zip(self.lines, BODY25B.edges):
            if np.isfinite(pts[[a, b]]).all():
                line.set_data_3d(pts[[a, b], 0], pts[[a, b], 1], pts[[a, b], 2])
            else:
                line.set_data_3d([], [], [])

        for line in self.hand_lines:
            line.set_data_3d([], [], [])
        if self.hand_frames and frame < len(self.hand_frames):
            entry = self.hand_frames[frame] or {}
            i = 0
            for hand in ("left", "right"):
                data = entry.get(hand)
                if not data:
                    i += 3
                    continue
                origin = world_to_plot(np.array(data["origin"])[None], self.up)[0]
                for axis_name in ("x_axis", "y_axis", "z_axis"):
                    vec = world_to_plot(np.array(data[axis_name])[None], self.up)[0] * 0.08
                    if i < len(self.hand_lines):
                        self.hand_lines[i].set_data_3d(
                            [origin[0], origin[0] + vec[0]],
                            [origin[1], origin[1] + vec[1]],
                            [origin[2], origin[2] + vec[2]],
                        )
                    i += 1

        subtitle = ""
        if self.kinematics and frame < len(self.kinematics):
            entry = self.kinematics[frame] or {}
            parts = [
                f"{hand[0].upper()}: FE {v['FE']:+.0f} RU {v['RU']:+.0f} PS {v['PS']:+.0f}"
                for hand, v in entry.items()
                if isinstance(v, dict) and np.isfinite(list(v.values())).all()
            ]
            subtitle = "\n" + "  |  ".join(parts) if parts else ""
        self.ax.set_title(f"{self.title}\nframe {frame + 1}/{self.points.shape[0]}{subtitle}",
                          fontsize=9)


def gather(task: int, trial: int, stages: bool) -> List[Tuple[str, Path, Path, Path]]:
    """Build the list of (title, points, hand frames, kinematics) paths."""
    trial_dir = ROOT / "project" / f"task{task}" / f"trial{trial}"
    d3 = trial_dir / "3D"
    prefix = f"task{task}_trial{trial}_"
    hands = d3 / f"{prefix}hand_frames_3d.json"
    kin = d3 / f"{prefix}wrist_kinematics.json"

    if not stages:
        return [(f"task{task}/trial{trial}",
                 d3 / f"{prefix}kpts_3d_final_processed.json", hands, kin)]

    out = []
    for key, label in STAGE_ORDER:
        path = d3 / f"{prefix}kpts_3d_{key}.json"
        if path.is_file():
            out.append((label, path, hands if "Final" in label else None,
                        kin if "Final" in label else None))
    if not out:
        out = [(f"task{task}/trial{trial}",
                d3 / f"{prefix}kpts_3d_final_processed.json", hands, kin)]
    return out


def main() -> int:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task_number", type=int, required=True)
    ap.add_argument("--trial_number", type=int, default=None)
    ap.add_argument("--stages", action="store_true",
                    help="show every pipeline stage side by side")
    ap.add_argument("--all_trials", action="store_true",
                    help="show every trial of the task side by side")
    ap.add_argument("--save", type=Path, default=None, help="write an MP4 instead of showing")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--step", type=int, default=1, help="plot every Nth frame")
    args = ap.parse_args()

    task_dir = ROOT / "project" / f"task{args.task_number}"
    entries: List[Tuple[str, Path, Optional[Path], Optional[Path]]] = []

    if args.all_trials:
        for trial_dir in sorted(task_dir.glob("trial*")):
            if not trial_dir.is_dir() or not trial_dir.name[5:].isdigit():
                continue
            trial = int(trial_dir.name[5:])
            entries.extend(gather(args.task_number, trial, stages=False))
    else:
        if args.trial_number is None:
            ap.error("--trial_number is required unless --all_trials is given")
        entries = gather(args.task_number, args.trial_number, args.stages)

    panels_data = []
    for title, points_path, hands_path, kin_path in entries:
        if not points_path.is_file():
            print(f"skipping {title}: {points_path.name} not found")
            continue
        points = load_keypoints_3d(points_path)
        panels_data.append((
            title, points,
            load_json(hands_path) if hands_path and hands_path.is_file() else None,
            load_json(kin_path) if kin_path and kin_path.is_file() else None,
        ))

    if not panels_data:
        print("Nothing to plot. Run the 3D stage first.")
        return 1

    n = len(panels_data)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(6.5 * cols, 6.0 * rows))
    panels = [
        Panel(fig.add_subplot(rows, cols, i + 1, projection="3d"), pts, title, hands, kin)
        for i, (title, pts, hands, kin) in enumerate(panels_data)
    ]
    total = max(p.points.shape[0] for p in panels)
    frames = range(0, total, max(1, args.step))
    fig.tight_layout()

    def update(frame: int):
        for panel in panels:
            panel.update(frame)
        return []

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / args.fps, repeat=False)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        print(f"Writing {args.save} ...")
        anim.save(str(args.save), writer="ffmpeg", fps=args.fps, dpi=150)
        print("done")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
