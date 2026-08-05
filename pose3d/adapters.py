"""Adapters that map other 2D detectors onto this project's keypoint schema.

``estimation/3D_estimation_mmpose.py`` used to be a 1895-line near-copy of the
OpenPose pipeline whose only real difference was the input format. Every fix had
to be made twice and, predictably, was not: the MMPose copy read the
inter-camera offset under the key ``frame_offset`` while calibration wrote
``best_offset``, so it silently ran with an offset of zero forever.

Converting the input instead means one pipeline, one set of fixes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .skeleton import BODY25B_RAW_NAMES, HAND_N_JOINTS, HANDS_N_JOINTS, hand_slice

PathLike = Union[str, Path]

# --------------------------------------------------------------------------- #
# COCO-WholeBody (MMPose, 133 keypoints)
# --------------------------------------------------------------------------- #

#: COCO-WholeBody body indices, by this project's joint names. Joints COCO does
#: not have (Neck, Head, MidHip) are derived below.
_COCO_TO_BODY25B: Dict[str, int] = {
    "Nose": 0,
    "LEye": 1, "REye": 2, "LEar": 3, "REar": 4,
    "LShoulder": 5, "RShoulder": 6,
    "LElbow": 7, "RElbow": 8,
    "LWrist": 9, "RWrist": 10,
    "LHip": 11, "RHip": 12,
    "LKnee": 13, "RKnee": 14,
    "LAnkle": 15, "RAnkle": 16,
    "LBigToe": 17, "LSmallToe": 18, "LHeel": 19,
    "RBigToe": 20, "RSmallToe": 21, "RHeel": 22,
}

#: Hand keypoint ranges. Both models order the 21 joints identically --
#: wrist, then thumb through pinky, proximal to distal.
_COCO_LEFT_HAND = slice(91, 112)
_COCO_RIGHT_HAND = slice(112, 133)

COCO_WHOLEBODY_N_KEYPOINTS = 133


def load_mmpose_results(
    path: PathLike, *, n_expected: int = COCO_WHOLEBODY_N_KEYPOINTS
) -> Tuple[np.ndarray, np.ndarray]:
    """Read an MMPose prediction file into ``(F, 133, 2)`` and ``(F, 133)``.

    Accepts the ``{"instance_info": [{"instances": [...]}, ...]}`` layout the
    ``MMPoseInferencer`` writes. When a frame holds several people, the one with
    the highest mean keypoint score is taken -- the recording protocol has a
    single subject, so extra detections are bystanders or false positives.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"MMPose result file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    frames = data.get("instance_info", data if isinstance(data, list) else [])
    n_frames = len(frames)
    kpts = np.full((n_frames, n_expected, 2), np.nan)
    conf = np.full((n_frames, n_expected), np.nan)

    for f, entry in enumerate(frames):
        instances = entry.get("instances") if isinstance(entry, dict) else None
        if not instances:
            continue
        best = max(
            instances,
            key=lambda inst: float(np.mean(inst.get("keypoint_scores") or [-1.0])),
        )
        xy = np.asarray(best.get("keypoints", []), dtype=float)
        scores = np.asarray(best.get("keypoint_scores", []), dtype=float)
        n = min(len(xy), len(scores), n_expected)
        if n == 0:
            continue
        kpts[f, :n] = xy[:n, :2]
        conf[f, :n] = scores[:n]
    return kpts, conf


def coco_wholebody_to_body25b(
    kpts: np.ndarray, conf: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Re-index COCO-WholeBody body joints into the BODY_25B raw layout.

    Neck and Head do not exist in COCO and are synthesised: the neck as the
    shoulder midpoint, the head as the ear midpoint (falling back to the eyes).
    Their confidence is the weaker of the two contributing joints, so a derived
    joint never claims more certainty than its inputs.
    """
    n_frames = kpts.shape[0]
    out_k = np.full((n_frames, len(BODY25B_RAW_NAMES), 2), np.nan)
    out_c = np.full((n_frames, len(BODY25B_RAW_NAMES)), np.nan)

    for name, source in _COCO_TO_BODY25B.items():
        target = BODY25B_RAW_NAMES.index(name)
        if source < kpts.shape[1]:
            out_k[:, target] = kpts[:, source]
            out_c[:, target] = conf[:, source]

    def midpoint(a: int, b: int) -> Tuple[np.ndarray, np.ndarray]:
        pair_k = kpts[:, [a, b]]
        pair_c = conf[:, [a, b]]
        valid = np.isfinite(pair_k[..., 0]) & np.isfinite(pair_c)
        both = valid.all(axis=1)
        mid = np.full((n_frames, 2), np.nan)
        mid[both] = pair_k[both].mean(axis=1)
        strength = np.full(n_frames, np.nan)
        strength[both] = pair_c[both].min(axis=1)
        return mid, strength

    neck_k, neck_c = midpoint(5, 6)
    out_k[:, BODY25B_RAW_NAMES.index("Neck")] = neck_k
    out_c[:, BODY25B_RAW_NAMES.index("Neck")] = neck_c

    head_k, head_c = midpoint(3, 4)          # ears
    eyes_k, eyes_c = midpoint(1, 2)          # fallback
    missing = ~np.isfinite(head_k[:, 0])
    head_k[missing] = eyes_k[missing]
    head_c[missing] = eyes_c[missing]
    out_k[:, BODY25B_RAW_NAMES.index("Head")] = head_k
    out_c[:, BODY25B_RAW_NAMES.index("Head")] = head_c

    return out_k, out_c


def coco_wholebody_hands(
    kpts: np.ndarray, conf: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract the 42-column hand array from COCO-WholeBody keypoints."""
    n_frames = kpts.shape[0]
    out_k = np.full((n_frames, HANDS_N_JOINTS, 2), np.nan)
    out_c = np.full((n_frames, HANDS_N_JOINTS), np.nan)
    for hand, source in (("left", _COCO_LEFT_HAND), ("right", _COCO_RIGHT_HAND)):
        if source.stop > kpts.shape[1]:
            continue
        target = hand_slice(hand)
        out_k[:, target] = kpts[:, source][:, :HAND_N_JOINTS]
        out_c[:, target] = conf[:, source][:, :HAND_N_JOINTS]
    return out_k, out_c


def load_mmpose_as_body25b(
    path: PathLike,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One call from an MMPose result file to this pipeline's arrays.

    Returns
    -------
    (body_kpts, body_conf, hand_kpts, hand_conf)
        Body arrays are in the 26-column BODY_25B raw layout (MidHip left NaN
        for the pipeline to synthesise); hand arrays are 42 columns.
    """
    raw_k, raw_c = load_mmpose_results(path)
    body_k, body_c = coco_wholebody_to_body25b(raw_k, raw_c)
    hand_k, hand_c = coco_wholebody_hands(raw_k, raw_c)
    return body_k, body_c, hand_k, hand_c
