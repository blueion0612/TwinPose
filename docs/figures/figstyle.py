"""House style for README figures. Copy this file next to make_hero.py.

One palette for every repository, taken from the portfolio site, so a README
figure, a social preview card and the site read as one hand. Light figures sit
on GitHub's white canvas, dark figures on its #0d1117 canvas. Both are saved
with a transparent background: GitHub paints its own canvas behind README
images, so the figure follows whichever theme the reader has, including the
dimmed dark theme, without a visible rectangle.

Type: if a fonts/ directory sits beside this file with Archivo-Regular.ttf,
Archivo-SemiBold.ttf and IBMPlexMono-Medium.ttf (built by
_standards/fonts/build_fonts.py), labels are set in Archivo and numbers in IBM
Plex Mono, the site's faces. Without it, DejaVu Sans, so nothing breaks.

    import figstyle

    def draw(T):                       # T is one entry of figstyle.PALETTE
        fig, ax = plt.subplots(figsize=(figstyle.WIDTH, 4.6))
        ax.text(..., fontfamily=figstyle.MONO)     # numbers
        return fig

    figstyle.save_both(draw, HERE + "hero_scenarios")
    # writes hero_scenarios.png and hero_scenarios-dark.png

Contrast on the canvas each theme is drawn for, WCAG ratio: light green 8.8,
light gold 7.0, light muted 5.4; dark green 9.8, dark gold 8.4, dark muted 8.2.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

PALETTE = {
    "light": dict(
        canvas="#ffffff",       # GitHub light canvas, for reference only, never painted
        ink="#241c10",          # site day ink
        soft="#332918",
        muted="#6b675c",
        line="#cdc8bb",
        fill="#f4f2ec",         # box fill, a warm paper tint on white
        green="#12570a",        # site day green
        gold="#6f5522",         # site day gold
        green_fill="#e6efe2",
        gold_fill="#f2ebdd",
    ),
    "dark": dict(
        canvas="#0d1117",       # GitHub dark canvas, for reference only, never painted
        ink="#ece7d9",          # site night ink, bone white
        soft="#ddd9cb",
        muted="#a9ac9b",
        line="#3a3c33",
        fill="#161811",
        green="#44d62c",        # site night green, phosphor
        gold="#c9a96a",         # site night gold, brass
        green_fill="#12200f",
        gold_fill="#26211a",
    ),
}

DPI = 170
WIDTH = 9.2      # inches. 900 / (9.2 * 170) = 0.575, so labels survive the README column.
TITLE = 11.5     # point sizes the standard fixes
BODY = 9.6
SMALL = 9.2

_FONTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts")
_have_fonts = False
for _p in glob.glob(os.path.join(_FONTS, "*.ttf")):
    font_manager.fontManager.addfont(_p)
    _have_fonts = True
_names = {f.name for f in font_manager.fontManager.ttflist}
SANS = "Archivo" if _have_fonts and "Archivo" in _names else "DejaVu Sans"
MONO = "IBM Plex Mono" if _have_fonts and "IBM Plex Mono" in _names else "DejaVu Sans Mono"


def apply(T):
    """Set matplotlib defaults for one theme. Called by save_both before draw()."""
    plt.rcParams.update({
        "font.family": SANS,
        "font.size": BODY,
        "text.color": T["ink"],
        "axes.labelcolor": T["muted"],
        "axes.edgecolor": T["line"],
        "axes.linewidth": 1.0,
        "axes.titlesize": TITLE,
        "axes.titleweight": "bold",
        "axes.titlecolor": T["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "xtick.color": T["muted"],
        "ytick.color": T["muted"],
        "xtick.labelsize": BODY,
        "ytick.labelsize": BODY,
        "legend.frameon": False,
        "legend.fontsize": BODY,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.edgecolor": "none",
    })


def mono_ticks(ax, axis="y"):
    """Set an axis's tick labels in the numeric face."""
    for label in (ax.get_yticklabels() if axis == "y" else ax.get_xticklabels()):
        label.set_fontfamily(MONO)


def save_both(draw, stem):
    """Render draw(T) once per theme and write <stem>.png and <stem>-dark.png."""
    for theme, suffix in (("light", ""), ("dark", "-dark")):
        T = PALETTE[theme]
        apply(T)
        fig = draw(T)
        out = f"{stem}{suffix}.png"
        fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.15, transparent=True)
        plt.close(fig)
        print("wrote", out)
