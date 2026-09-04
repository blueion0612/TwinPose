"""Render the stage figure used at the top of the README.

Every number is read from validation/benchmark_results.json, the file the
benchmark writes, so the figure cannot drift from the reported results.

    python docs/figures/make_hero.py

Writes hero_stages.png and hero_stages-dark.png.
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RESULTS = ROOT / "validation" / "benchmark_results.json"

THEMES = {
    "light": dict(bg="white", ink="#1c2530", muted="#5b6875", grid="#e2e7ec",
                  bars=["#9fb0c0", "#c8683f", "#3f7d5a"]),
    "dark": dict(bg="#0d1117", ink="#e6edf3", muted="#9198a1", grid="#262c34",
                 bars=["#5b6875", "#e08a5c", "#5aa87a"]),
}

STAGES = [("1_triangulation", "triangulate"),
          ("2_bootstrapped", "bootstrap"),
          ("3_refined", "refine")]

PANELS = [
    ("MPJPE_mm", "Error", "mm", False),
    ("Coverage_percent", "Coverage", "%", True),
    ("BoneLengthCV_percent", "Bone length spread", "%", False),
]


def render(theme, out_path, summary, seeds):
    T = THEMES[theme]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.3), dpi=170)
    fig.patch.set_facecolor(T["bg"])

    for ax, (key, title, unit, higher_better) in zip(axes, PANELS):
        vals = [summary[s][key] for s, _ in STAGES]
        errs = [summary[s].get(f"{key}_sd", 0.0) for s, _ in STAGES]
        labels = [lab for _, lab in STAGES]

        ax.bar(labels, vals, yerr=errs, capsize=4, color=T["bars"],
               edgecolor="none", width=0.62, error_kw=dict(ecolor=T["muted"], lw=1.2))
        for i, v in enumerate(vals):
            ax.text(i, v + max(vals) * 0.045, f"{v:.1f}", ha="center",
                    fontsize=9.6, color=T["ink"], fontweight="bold")

        ax.set_title(f"{title}  ({unit})", fontsize=11, color=T["ink"],
                     fontweight="bold", pad=10)
        top = max(v + e for v, e in zip(vals, errs)) * 1.22
        ax.set_ylim(0, min(top, 106) if unit == "%" and max(vals) > 80 else top)
        ax.set_facecolor(T["bg"])
        ax.tick_params(colors=T["muted"], labelsize=9.4)
        ax.grid(axis="y", color=T["grid"], lw=0.9)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(T["grid"])
        arrow = "higher is better" if higher_better else "lower is better"
        ax.set_xlabel(arrow, fontsize=8.6, color=T["muted"], labelpad=6)

    fig.suptitle(f"Three stages, {seeds} seeds, against synthetic ground truth",
                 fontsize=10.4, color=T["muted"], y=1.02)
    fig.tight_layout(pad=0.5)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor=T["bg"])
    plt.close(fig)
    print("wrote", out_path)


if __name__ == "__main__":
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    render("light", HERE / "hero_stages.png", data["summary"], data["seeds"])
    render("dark", HERE / "hero_stages-dark.png", data["summary"], data["seeds"])
