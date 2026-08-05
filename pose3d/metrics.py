"""Evaluation metrics.

Two families:

*Self-consistency* metrics need no ground truth and are what the pipeline
reports for real recordings -- reprojection error, bone-length variability,
jerk, hip depth stability.

*Ground-truth* metrics (MPJPE, PA-MPJPE, PCK, MPJVE, foot-slide) are what the
synthetic benchmark uses to actually measure accuracy.

The DTW used by the inter-trial metric is implemented here rather than pulled
from ``fastdtw``, whose last release predates Python 3.8 and no longer installs.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import CameraPair
from .geometry import rigid_align
from .skeleton import Skeleton

# --------------------------------------------------------------------------- #
# Self-consistency metrics
# --------------------------------------------------------------------------- #


def reprojection_error_px(
    points3d: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    cameras: CameraPair,
    *,
    statistic: str = "median",
) -> float:
    """Reprojection error against the 2D observations, in pixels.

    ``kpts0``/``kpts1`` must already be undistorted, matching the ideal-pinhole
    forward model used everywhere in the pipeline.
    """
    values: List[np.ndarray] = []
    for cam, obs in ((cameras.cam0, kpts0), (cameras.cam1, kpts1)):
        proj = cam.project(points3d)
        err = np.linalg.norm(proj - obs, axis=-1)
        values.append(err[np.isfinite(err)])
    if not values or all(v.size == 0 for v in values):
        return float("nan")
    allv = np.concatenate([v for v in values if v.size])
    if allv.size == 0:
        return float("nan")
    if statistic == "median":
        return float(np.median(allv))
    if statistic == "mean":
        return float(np.mean(allv))
    if statistic == "rms":
        return float(np.sqrt(np.mean(allv ** 2)))
    raise ValueError(f"unknown statistic {statistic!r}")


def bone_length_cv_percent(
    points3d: np.ndarray, skeleton: Skeleton, frames: Optional[Sequence[int]] = None
) -> float:
    """Median across bones of each bone's coefficient of variation, in percent.

    This measures whether the skeleton stays rigid over time. The original code
    computed something different under the same name: it pooled *all* bones'
    lengths into one distribution and took the CV of that, which mostly measures
    "femurs are longer than clavicles" and lands around 50% no matter how good
    the reconstruction is. That explains the README's 53.58% figure.
    """
    idx = np.arange(points3d.shape[0]) if frames is None else np.asarray(list(frames), dtype=int)
    if idx.size == 0:
        return float("nan")
    sub = points3d[idx]

    cvs: List[float] = []
    for p, c, _name in skeleton.bone_pairs:
        d = np.linalg.norm(sub[:, p] - sub[:, c], axis=-1)
        d = d[np.isfinite(d)]
        if d.size < 2:
            continue
        mean = float(np.mean(d))
        if mean > 1e-9:
            cvs.append(float(np.std(d)) / mean * 100.0)
    return float(np.median(cvs)) if cvs else float("nan")


def bone_length_error_percent(
    points3d: np.ndarray, skeleton: Skeleton, reference: Dict[str, float]
) -> float:
    """Median absolute deviation from the reference bone model, in percent."""
    errs: List[float] = []
    for p, c, name in skeleton.bone_pairs:
        target = reference.get(name)
        if not target or not np.isfinite(target) or target <= 1e-9:
            continue
        d = np.linalg.norm(points3d[:, p] - points3d[:, c], axis=-1)
        d = d[np.isfinite(d)]
        if d.size:
            errs.extend(np.abs(d - target) / target * 100.0)
    return float(np.median(errs)) if errs else float("nan")


def jerk_rms(points3d: np.ndarray, fps: float = 30.0) -> float:
    """RMS of the third time derivative, in m/s^3.

    Frames containing NaN propagate into the difference and are dropped, so a
    partially reconstructed sequence is scored on the parts that exist.
    """
    if points3d.shape[0] < 4:
        return float("nan")
    d3 = np.diff(points3d, n=3, axis=0) * (float(fps) ** 3)
    finite = d3[np.isfinite(d3)]
    return float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan")


def acceleration_rms(points3d: np.ndarray, fps: float = 30.0) -> float:
    """RMS acceleration, in m/s^2. Less scale-sensitive than jerk."""
    if points3d.shape[0] < 3:
        return float("nan")
    d2 = np.diff(points3d, n=2, axis=0) * (float(fps) ** 2)
    finite = d2[np.isfinite(d2)]
    return float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan")


def hip_depth_variance(points3d: np.ndarray, skeleton: Skeleton) -> float:
    """Variance of mean hip depth, in m^2 -- a proxy for depth-axis noise."""
    try:
        l, r = skeleton.index("LHip"), skeleton.index("RHip")
    except KeyError:
        return float("nan")
    # Mean of whichever hips are present, without nanmean's all-NaN warning.
    hips = points3d[:, [l, r], 2]
    present = np.isfinite(hips)
    count = present.sum(axis=1)
    total = np.where(present, hips, 0.0).sum(axis=1)
    z = total[count > 0] / count[count > 0]
    return float(np.var(z)) if z.size else float("nan")


def valid_fraction(points3d: np.ndarray) -> float:
    """Fraction of joint slots that hold a finite 3D position."""
    if points3d.size == 0:
        return 0.0
    return float(np.isfinite(points3d[..., 0]).mean())


# --------------------------------------------------------------------------- #
# Ground-truth metrics
# --------------------------------------------------------------------------- #


def mpjpe_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint position error in millimetres, no alignment."""
    err = np.linalg.norm(pred - gt, axis=-1)
    err = err[np.isfinite(err)]
    return float(np.mean(err) * 1000.0) if err.size else float("nan")


def pa_mpjpe_mm(pred: np.ndarray, gt: np.ndarray, *, per_frame: bool = True) -> float:
    """MPJPE after per-frame similarity (Procrustes) alignment, in millimetres.

    Aligning per frame is the standard protocol: it removes the global rotation,
    translation and scale ambiguity that any two-view reconstruction has, and so
    isolates pose accuracy from placement accuracy.
    """
    if not per_frame:
        aligned, _, _, _ = rigid_align(pred.reshape(-1, 3), gt.reshape(-1, 3))
        return mpjpe_mm(aligned.reshape(pred.shape), gt)

    errors: List[np.ndarray] = []
    for f in range(pred.shape[0]):
        aligned, _, _, _ = rigid_align(pred[f], gt[f])
        e = np.linalg.norm(aligned - gt[f], axis=-1)
        errors.append(e[np.isfinite(e)])
    allv = np.concatenate(errors) if errors else np.empty(0)
    return float(np.mean(allv) * 1000.0) if allv.size else float("nan")


def pck3d(pred: np.ndarray, gt: np.ndarray, threshold_mm: float = 150.0) -> float:
    """Percentage of joints within ``threshold_mm`` of ground truth."""
    err = np.linalg.norm(pred - gt, axis=-1)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return float("nan")
    return float(np.mean(err * 1000.0 < threshold_mm) * 100.0)


def mpjve_mm(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean per-joint velocity error, in millimetres per frame."""
    if pred.shape[0] < 2:
        return float("nan")
    err = np.linalg.norm(np.diff(pred, axis=0) - np.diff(gt, axis=0), axis=-1)
    err = err[np.isfinite(err)]
    return float(np.mean(err) * 1000.0) if err.size else float("nan")


def foot_slide_rate_percent(
    points3d: np.ndarray,
    skeleton: Skeleton,
    *,
    up_axis: int = 1,
    contact_height_m: float = 0.05,
    slide_speed_m: float = 0.02,
) -> float:
    """Percentage of ground-contact frames in which a foot slides.

    The previous implementation indexed ``pred[:, foot_indices, [0, 2]]``, where
    NumPy broadcasts the two index arrays together and yields ``(F, 2)`` rather
    than the intended ``(F, 2, 2)``; the following ``norm(..., axis=2)`` then
    raised. Here the axes are selected explicitly.
    """
    names = [n for n in ("LAnkle", "RAnkle") if n in skeleton]
    if not names:
        return float("nan")
    feet = skeleton.indices(*names)
    ground_axes = [a for a in range(3) if a != up_axis]

    pos = points3d[:, feet, :]                       # (F, n_feet, 3)
    height = pos[:-1, :, up_axis]                    # (F-1, n_feet)
    horiz = pos[:, :, ground_axes]                   # (F, n_feet, 2)
    speed = np.linalg.norm(np.diff(horiz, axis=0), axis=-1)  # (F-1, n_feet)

    floor = np.nanmin(pos[..., up_axis]) if np.isfinite(pos[..., up_axis]).any() else 0.0
    contact = np.isfinite(height) & ((height - floor) < contact_height_m)
    if not contact.any():
        return float("nan")
    sliding = contact & np.isfinite(speed) & (speed > slide_speed_m)
    return float(sliding.sum() / contact.sum() * 100.0)


# --------------------------------------------------------------------------- #
# Sequence comparison (inter-trial consistency)
# --------------------------------------------------------------------------- #


def dtw_distance(
    a: np.ndarray, b: np.ndarray, *, radius: int = 20
) -> Tuple[float, int]:
    """Sakoe-Chiba band-limited DTW between two ``(T, D)`` sequences.

    Replaces the ``fastdtw`` dependency. The band of width ``2*radius+1`` around
    the diagonal makes this O(T * radius) instead of O(T^2), which is both
    faster than ``fastdtw``'s multilevel approximation and exact within the band
    -- appropriate here because the sequences being compared are repetitions of
    the same scripted motion and so are already roughly aligned.

    Returns
    -------
    (total_distance, path_length)
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("nan"), 0

    # Widen the band so it can always span the length difference.
    radius = max(int(radius), abs(n - m) + 1)
    inf = float("inf")
    prev = np.full(m + 1, inf)
    prev[0] = 0.0
    # Backtracking is unnecessary; we only need the path length, so carry it.
    prev_len = np.zeros(m + 1, dtype=np.int64)

    for i in range(1, n + 1):
        cur = np.full(m + 1, inf)
        cur_len = np.zeros(m + 1, dtype=np.int64)
        centre = (i - 1) * m // n + 1
        lo, hi = max(1, centre - radius), min(m, centre + radius)
        if lo > hi:
            lo, hi = 1, m
        diff = a[i - 1][None, :] - b[lo - 1:hi]
        costs = np.sqrt(np.einsum("ij,ij->i", diff, diff))

        for offset, j in enumerate(range(lo, hi + 1)):
            candidates = (prev[j], cur[j - 1], prev[j - 1])
            lengths = (prev_len[j], cur_len[j - 1], prev_len[j - 1])
            k = int(np.argmin(candidates))
            best = candidates[k]
            if best == inf:
                continue
            cur[j] = costs[offset] + best
            cur_len[j] = lengths[k] + 1
        prev, prev_len = cur, cur_len

    total = float(prev[m])
    return (total, int(prev_len[m])) if np.isfinite(total) else (float("nan"), 0)


def sequence_distance_mm(
    seq_a: np.ndarray, seq_b: np.ndarray, *, radius: int = 20
) -> float:
    """Mean per-frame DTW distance between two pose sequences, in millimetres.

    Each frame is centred on its own centroid first, so the metric measures
    posture similarity rather than where in the room the subject stood.
    """
    def prepare(seq: np.ndarray) -> np.ndarray:
        ok = np.isfinite(seq).all(axis=(1, 2))
        s = seq[ok]
        if s.shape[0] == 0:
            return s.reshape(0, 0)
        s = s - s.mean(axis=1, keepdims=True)
        return s.reshape(s.shape[0], -1)

    a, b = prepare(seq_a), prepare(seq_b)
    if a.shape[0] < 2 or b.shape[0] < 2 or a.shape[1] != b.shape[1]:
        return float("nan")
    total, length = dtw_distance(a, b, radius=radius)
    if not np.isfinite(total) or length == 0:
        return float("nan")
    n_joints = seq_a.shape[1]
    # Distance is an L2 norm over all joints; normalise to per-joint millimetres.
    return float(total / length / np.sqrt(n_joints) * 1000.0)


def static_pose_rms_mm(pose_a: np.ndarray, pose_b: np.ndarray) -> float:
    """Procrustes-aligned RMS distance between two single poses, in mm."""
    aligned, _, _, _ = rigid_align(pose_a, pose_b)
    d = np.linalg.norm(aligned - pose_b, axis=-1)
    d = d[np.isfinite(d)]
    return float(np.sqrt(np.mean(d ** 2)) * 1000.0) if d.size else float("nan")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #

#: Value each metric is divided by before entering the composite score, chosen
#: so that 1.0 is "acceptable" and 0.0 is perfect.
_SCORE_SCALES: Dict[str, float] = {
    "PA_MPJPE_mm": 100.0,
    "MPJVE_mm": 50.0,
    "FootSlideRate_percent": 20.0,
    "ReprojectionError_px": 10.0,
    "BoneLengthCV_percent": 5.0,
    "JerkRMS": 300.0,
}
_SCORE_WEIGHTS: Dict[str, float] = {
    "PA_MPJPE_mm": 0.25,
    "MPJVE_mm": 0.15,
    "FootSlideRate_percent": 0.15,
    "ReprojectionError_px": 0.15,
    "BoneLengthCV_percent": 0.15,
    "JerkRMS": 0.15,
}


def composite_score(metrics: Dict[str, float]) -> float:
    """Weighted composite of whichever metrics are present. Lower is better."""
    total, weight_used = 0.0, 0.0
    for key, weight in _SCORE_WEIGHTS.items():
        value = metrics.get(key)
        if value is None or not np.isfinite(value):
            continue
        total += min(float(value) / _SCORE_SCALES[key], 2.0) * weight
        weight_used += weight
    return total / weight_used if weight_used > 0 else float("nan")


def summarise(
    points3d: np.ndarray,
    skeleton: Skeleton,
    cameras: CameraPair,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    *,
    fps: float = 30.0,
    bone_reference: Optional[Dict[str, float]] = None,
    t_pose_frames: Optional[Sequence[int]] = None,
    ground_truth: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute the full metric set for one reconstruction."""
    out: Dict[str, float] = {
        "ReprojectionError_px": reprojection_error_px(points3d, kpts0, kpts1, cameras),
        "BoneLengthCV_percent": bone_length_cv_percent(points3d, skeleton),
        "JerkRMS": jerk_rms(points3d, fps),
        "AccelRMS": acceleration_rms(points3d, fps),
        "HipZVariance": hip_depth_variance(points3d, skeleton),
        "ValidFraction_percent": valid_fraction(points3d) * 100.0,
        "FootSlideRate_percent": foot_slide_rate_percent(points3d, skeleton),
    }
    if bone_reference:
        out["BoneLengthError_percent"] = bone_length_error_percent(
            points3d, skeleton, bone_reference
        )
    if t_pose_frames is not None and len(t_pose_frames):
        out["BoneLengthCV_tpose_percent"] = bone_length_cv_percent(
            points3d, skeleton, t_pose_frames
        )
    if ground_truth is not None:
        out["MPJPE_mm"] = mpjpe_mm(points3d, ground_truth)
        out["PA_MPJPE_mm"] = pa_mpjpe_mm(points3d, ground_truth)
        out["PCK3D_150mm"] = pck3d(points3d, ground_truth, 150.0)
        out["PCK3D_50mm"] = pck3d(points3d, ground_truth, 50.0)
        out["MPJVE_mm"] = mpjve_mm(points3d, ground_truth)
    out["Score"] = composite_score(out)
    return out
