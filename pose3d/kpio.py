"""Reading and writing keypoint JSON files.

Covers the three on-disk formats the project uses:

* body 2D  -- ``[frame][joint] = [x, y, confidence]`` from ``Openpose.py``
* hand 2D  -- ``[frame] = {"left": [[x, y, c], ...], "right": [...]}``
* 3D       -- ``[frame][joint] = [x, y, z]``

Missing detections are represented as NaN in memory and as ``null`` on disk;
the loaders also accept OpenPose's ``-1`` sentinel and zero-confidence entries.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .skeleton import HAND_N_JOINTS, HANDS_N_JOINTS, hand_slice

PathLike = Union[str, Path]


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_body_keypoints(
    path: PathLike, n_joints: int, *, default_confidence: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """Load 2D body keypoints.

    Parameters
    ----------
    n_joints
        Number of columns to allocate. Frames with fewer detections are padded
        with NaN, which is how the pipeline represents "MidHip not emitted by
        the network yet".

    Returns
    -------
    (keypoints, confidence)
        ``(F, n_joints, 2)`` float array and ``(F, n_joints)`` float array.
        Undetected joints are NaN in both.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"2D keypoint file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    n_frames = len(raw)
    kpts = np.full((n_frames, n_joints, 2), np.nan, dtype=float)
    conf = np.full((n_frames, n_joints), np.nan, dtype=float)

    for f, frame in enumerate(raw):
        if not frame:
            continue
        for j, entry in enumerate(frame):
            if j >= n_joints or entry is None:
                continue
            try:
                x, y = _as_float(entry[0]), _as_float(entry[1])
                c = _as_float(entry[2]) if len(entry) > 2 else default_confidence
            except (TypeError, IndexError):
                continue
            # OpenPose signals "not found" with -1 coordinates.
            if not (np.isfinite(x) and np.isfinite(y)) or x < 0 or y < 0:
                conf[f, j] = c if c > 0 else np.nan
                continue
            kpts[f, j] = (x, y)
            conf[f, j] = c
    return kpts, conf


def save_body_keypoints(path: PathLike, kpts: np.ndarray, conf: np.ndarray) -> None:
    """Write 2D body keypoints in the format ``load_body_keypoints`` reads."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    kpts = np.asarray(kpts, dtype=float)
    conf = np.asarray(conf, dtype=float)
    payload: List[List[List[float]]] = []
    for f in range(kpts.shape[0]):
        frame = []
        for j in range(kpts.shape[1]):
            x, y = kpts[f, j]
            c = conf[f, j]
            if not np.isfinite(x) or not np.isfinite(y):
                frame.append([-1.0, -1.0, 0.0 if not np.isfinite(c) else float(c)])
            else:
                frame.append([float(x), float(y), 0.0 if not np.isfinite(c) else float(c)])
        payload.append(frame)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def load_hand_keypoints(
    path: PathLike, n_frames: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Load 2D hand keypoints into a 42-wide array (21 per hand).

    Accepts the legacy 22-per-hand layout as well: the OpenPose hand model emits
    21 joints plus a background heatmap channel, and older runs stored all 22.
    The trailing background channel is discarded.

    Returns
    -------
    (keypoints, confidence)
        ``(n_frames, 42, 2)`` and ``(n_frames, 42)``, NaN where absent.
    """
    kpts = np.full((n_frames, HANDS_N_JOINTS, 2), np.nan, dtype=float)
    conf = np.full((n_frames, HANDS_N_JOINTS), np.nan, dtype=float)

    p = Path(path)
    if not p.is_file():
        return kpts, conf
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    for f, frame in enumerate(raw):
        if f >= n_frames or not frame:
            continue
        for hand in ("left", "right"):
            points = frame.get(hand)
            if not points:
                continue
            base = hand_slice(hand).start
            # Legacy files carry 22 entries; only the first 21 are joints.
            for j, entry in enumerate(points[:HAND_N_JOINTS]):
                if entry is None:
                    continue
                x, y = _as_float(entry[0]), _as_float(entry[1])
                c = _as_float(entry[2]) if len(entry) > 2 else 1.0
                if not (np.isfinite(x) and np.isfinite(y)) or x <= 0 or y <= 0:
                    continue
                kpts[f, base + j] = (x, y)
                conf[f, base + j] = c
    return kpts, conf


def save_hand_keypoints(path: PathLike, kpts: np.ndarray, conf: np.ndarray) -> None:
    """Write 2D hand keypoints from a 42-wide array."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: List[Dict[str, List[List[float]]]] = []
    for f in range(kpts.shape[0]):
        frame: Dict[str, List[List[float]]] = {}
        for hand in ("left", "right"):
            sl = hand_slice(hand)
            xy, c = kpts[f, sl], conf[f, sl]
            if not np.isfinite(c).any():
                continue
            frame[hand] = [
                [
                    float(xy[j, 0]) if np.isfinite(xy[j, 0]) else -1.0,
                    float(xy[j, 1]) if np.isfinite(xy[j, 1]) else -1.0,
                    float(c[j]) if np.isfinite(c[j]) else 0.0,
                ]
                for j in range(HAND_N_JOINTS)
            ]
        payload.append(frame)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def save_keypoints_3d(path: PathLike, points: np.ndarray) -> None:
    """Write an ``(F, J, 3)`` array, encoding NaN as JSON ``null``.

    ``json.dump`` would otherwise emit bare ``NaN``, which is invalid JSON and
    which ``json.load`` only accepts because of a CPython extension.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(points, dtype=float)
    payload = [
        [[None if not np.isfinite(v) else float(v) for v in joint] for joint in frame]
        for frame in arr
    ]
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def load_keypoints_3d(path: PathLike) -> np.ndarray:
    """Read an ``(F, J, 3)`` array written by :func:`save_keypoints_3d`."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"3D keypoint file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return np.array(
        [[[np.nan if v is None else float(v) for v in joint] for joint in frame]
         for frame in raw],
        dtype=float,
    )


def save_json(path: PathLike, payload: Any, *, indent: Optional[int] = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent)


def load_json(path: PathLike, default: Any = None) -> Any:
    p = Path(path)
    if not p.is_file():
        return default
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return default


def merge_metrics(path: PathLike, trial_key: str, metrics: Dict[str, Any]) -> None:
    """Merge one trial's metrics into a task-level metrics file."""
    existing = load_json(path, default={}) or {}
    if not isinstance(existing, dict):
        existing = {}
    existing[trial_key] = metrics
    save_json(path, existing, indent=4)
