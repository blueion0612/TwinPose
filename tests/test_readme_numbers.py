"""The Accuracy table in the README states the numbers validation/benchmark_results.json holds.

The benchmark writes that file, the hero figure is drawn from it, and this test
fails the moment the README table drifts from it.
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# README row label -> (source block, stage key)
ROWS = {
    "1. Triangulation": ("summary", "1_triangulation"),
    "2. + single-view bootstrap": ("summary", "2_bootstrapped"),
    "3. + bundle adjustment": ("summary", "3_refined"),
    "noise-free upper bound": ("noiseless", "3_refined"),
}
# column order in the README and how each value is printed
COLUMNS = [("MPJPE_mm", "{:.2f}"), ("PA_MPJPE_mm", "{:.2f}"), ("PCK3D_50mm", "{:.1f}"),
           ("PCK3D_150mm", "{:.1f}"), ("BoneLengthCV_percent", "{:.2f}"), ("JerkRMS", "{:.0f}"),
           ("Coverage_percent", "{:.1f}")]


def _load():
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    with open(os.path.join(ROOT, "validation", "benchmark_results.json"), encoding="utf-8") as fh:
        data = json.load(fh)
    return readme, data


def _row(readme, label):
    for line in readme.splitlines():
        cells = [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]
        if cells and cells[0] == label:
            return cells[1:]
    raise AssertionError(f"README has no Accuracy row labelled {label!r}")


def test_accuracy_table_matches_benchmark():
    readme, data = _load()
    for label, (block, stage) in ROWS.items():
        cells = _row(readme, label)
        assert len(cells) == len(COLUMNS), (label, cells)
        for cell, (key, fmt) in zip(cells, COLUMNS):
            assert cell == fmt.format(data[block][stage][key]), (label, key, cell)


def test_protocol_numbers_match_benchmark():
    readme, data = _load()
    assert f"{data['seeds']} independent noise seeds" in readme
