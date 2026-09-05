"""Draw the README hero: the three pipeline stages against synthetic ground truth.

Every number is read from validation/benchmark_results.json, the file the
benchmark writes, so the figure cannot drift from the table in the README, which
tests/test_readme_numbers.py checks against the same file.

    python docs/figures/make_hero.py

Writes hero_stages.png and hero_stages-dark.png.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__)) + os.sep
sys.path.insert(0, HERE)

import figstyle  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

RESULTS = os.path.join(HERE, "..", "..", "validation", "benchmark_results.json")
with open(RESULTS, encoding="utf-8") as fh:
    DATA = json.load(fh)

STAGES = [("1_triangulation", "triangulate"),
          ("2_bootstrapped", "bootstrap"),
          ("3_refined", "refine")]
PANELS = [
    ("MPJPE_mm", "Error", "mm", False),
    ("Coverage_percent", "Coverage", "%", True),
    ("BoneLengthCV_percent", "Bone length spread", "%", False),
]


def stages(T):
    summary, seeds = DATA["summary"], DATA["seeds"]
    fig, axes = plt.subplots(1, 3, figsize=(figstyle.WIDTH, 3.4))
    colors = [T["muted"], T["gold"], T["green"]]      # the refined stage is the result
    for ax, (key, title, unit, higher_better) in zip(axes, PANELS):
        vals = [summary[s][key] for s, _ in STAGES]
        errs = [summary[s].get(f"{key}_sd", 0.0) for s, _ in STAGES]
        ax.yaxis.grid(True, color=T["line"], linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        ax.bar([lab for _, lab in STAGES], vals, yerr=errs, capsize=4, color=colors,
               width=0.62, error_kw=dict(ecolor=T["muted"], lw=1.1), zorder=3)
        top = max(v + e for v, e in zip(vals, errs))
        for i, v in enumerate(vals):
            ax.text(i, v + errs[i] + top * 0.03, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=figstyle.SMALL, fontfamily=figstyle.MONO,
                    color=T["ink"] if i == 2 else T["muted"],
                    fontweight="bold" if i == 2 else "normal", zorder=4)
        ax.set_title(f"{title}, {unit}", pad=8)
        ax.set_ylim(0, min(top * 1.25, 106) if unit == "%" and max(vals) > 80 else top * 1.25)
        ax.tick_params(axis="x", length=0, pad=5)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        figstyle.mono_ticks(ax)
        ax.set_xlabel("higher is better" if higher_better else "lower is better",
                      fontsize=figstyle.SMALL, color=T["muted"], labelpad=6)
    fig.suptitle(f"Three stages, {seeds} seeds, against synthetic ground truth",
                 fontsize=figstyle.BODY, color=T["muted"], fontweight="normal", y=1.03)
    fig.tight_layout(pad=0.6)
    return fig


if __name__ == "__main__":
    figstyle.save_both(stages, HERE + "hero_stages")
