"""The 3D reconstruction pipeline, as a function over arrays.

``estimation/3D_estimation.py`` used to be 1943 lines with everything inside
``if __name__ == '__main__'``: no part of it could be imported, called or tested,
and the only way to find out whether a change helped was to run the whole thing
on a video and read the console. Here the stages are functions and
:func:`reconstruct` operates on plain arrays, which is what makes the synthetic
benchmark in ``validation/`` possible.

Stages
------
1. estimate the inter-camera frame offset (coarse integer, then sub-frame)
2. synthesise MidHip, gate on confidence, undistort
3. triangulate both-view joints
4. bootstrap single-view joints along their camera ray
5. measure the subject's bones from T-pose frames
6. windowed spatio-temporal bundle adjustment  (:mod:`pose3d.refine`)
7. fill short gaps, optionally smooth
8. metrics
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import CameraPair
from .config import ReconstructionConfig
from .geometry import back_project_ray, triangulate_frames
from .metrics import summarise
from .refine import RefinementStats, refine_sequence
from .skeleton import BODY25B, Skeleton, height_from_bone_lengths, scaled_bone_lengths


@dataclass
class ReconstructionResult:
    """Everything one run produces."""

    points3d: np.ndarray
    bone_lengths: Dict[str, float]
    metrics: Dict[str, float]
    frame_offset: float
    t_pose_frames: List[int] = field(default_factory=list)
    stages: Dict[str, np.ndarray] = field(default_factory=dict)
    stats: Dict[str, object] = field(default_factory=dict)
    undistorted: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def n_frames(self) -> int:
        return int(self.points3d.shape[0])


def _log(cfg: ReconstructionConfig, message: str) -> None:
    if cfg.verbose:
        print(message, flush=True)


# --------------------------------------------------------------------------- #
# Stage 1 -- inter-camera synchronization
# --------------------------------------------------------------------------- #


def shift_series(
    kpts: np.ndarray, conf: np.ndarray, shift: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample a keypoint track by a fractional number of frames.

    Vectorized over joints; the previous implementation built a pandas Series
    per joint per axis inside the offset loop.
    """
    n = kpts.shape[0]
    if shift == 0 or n == 0:
        return kpts.copy(), conf.copy()
    src = np.arange(n, dtype=float)
    dst = src + float(shift)

    flat = kpts.reshape(n, -1)
    out = np.empty_like(flat)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c], left=np.nan, right=np.nan)

    conf_out = np.empty_like(conf)
    for c in range(conf.shape[1]):
        conf_out[:, c] = np.interp(dst, src, conf[:, c], left=np.nan, right=np.nan)
    return out.reshape(kpts.shape), conf_out


def _resampling_variance_ratio(offset: float) -> float:
    """Variance of a linearly resampled sample, relative to an original one.

    Resampling at a fractional offset forms ``(1-w)*x[t] + w*x[t+1]``, whose
    variance is ``(1-w)^2 + w^2`` times the original: 1.0 on an integer, 0.5 at
    a half frame. Only camera 1 is resampled, so the triangulation residual --
    which sees both views -- shrinks by ``(1 + ratio) / 2``.
    """
    w = float(offset) - np.floor(float(offset))
    return (1.0 - w) ** 2 + w ** 2


def _debias_residual(residual_px: float, offset: float, noise_floor_px: float) -> float:
    """Undo the noise reduction that fractional resampling introduces.

    Averaging two noisy detections always lowers reprojection error, whether or
    not the timing is right, so an uncorrected search is biased toward
    half-integer offsets. The correction has to apply to the *noise* part of the
    residual only: scaling the whole residual also suppresses the sync signal,
    which made genuine half-frame offsets unrecoverable even from noise-free
    input.

    Modeling the residual as ``signal^2 + noise^2 * (1 + r) / 2`` and adding
    back what resampling removed gives::

        corrected^2 = observed^2 + noise_floor^2 * (1 - r) / 2

    which is a no-op on integers and on noise-free data, and grows with the
    measured noise level otherwise.
    """
    r = _resampling_variance_ratio(offset)
    penalty = max(0.0, noise_floor_px) ** 2 * (1.0 - r) / 2.0
    return float(np.sqrt(max(residual_px, 0.0) ** 2 + penalty))


def motion_mask(kpts: np.ndarray, percentile: float = 80.0) -> np.ndarray:
    """Mark the (frame, joint) samples that are moving fastest in 2D.

    Synchronization error shows up as a position error proportional to joint
    speed, so a joint that is not moving carries no information about the
    offset. Scoring on the whole clip therefore dilutes the signal badly here:
    the recording protocol opens with a five-second static T-pose, during which
    every candidate offset looks equally good.
    """
    speed = np.full(kpts.shape[:2], np.nan)
    if kpts.shape[0] > 1:
        step = np.linalg.norm(np.diff(kpts, axis=0), axis=-1)
        speed[:-1] = step
    finite = speed[np.isfinite(speed)]
    if finite.size == 0:
        return np.isfinite(kpts[..., 0])
    threshold = float(np.percentile(finite, percentile))
    return np.isfinite(speed) & (speed >= threshold)


def _offset_score(
    kpts0: np.ndarray,
    conf0: np.ndarray,
    kpts1: np.ndarray,
    conf1: np.ndarray,
    cameras: CameraPair,
    skeleton: Skeleton,
    cfg: ReconstructionConfig,
    offset: float,
    sample: np.ndarray,
    moving: np.ndarray,
) -> Tuple[float, Dict[str, float]]:
    """Score one candidate offset. Lower is better.

    The signal is the reprojection error of the triangulated points, restricted
    to fast-moving joints and corrected for the noise reduction that fractional
    resampling introduces. A small coverage penalty discourages offsets that
    simply discard frames.

    An earlier version of this scoring function also included bone-length
    variability. That term was ~11 while the reprojection differences between
    candidate offsets were ~0.1, so it decided the answer by itself and the
    search returned essentially arbitrary offsets.

    Returns the *raw* residual; the caller debiases it once a noise floor is
    known (see :func:`_debias_residual`).
    """
    k1, c1 = shift_series(kpts1, conf1, -offset)
    k0s, c0s = kpts0[sample], conf0[sample]
    k1s, c1s = k1[sample], c1[sample]
    moving_s = moving[sample]

    X = triangulate_frames(
        k0s, k1s, c0s, c1s, cameras, min_confidence=cfg.sync_min_confidence
    )
    if not np.isfinite(X).any():
        return float("inf"), {}

    u0 = cameras.cam0.undistort(k0s)
    u1 = cameras.cam1.undistort(k1s)

    residuals = []
    for cam, obs in ((cameras.cam0, u0), (cameras.cam1, u1)):
        err = np.linalg.norm(cam.project(X) - obs, axis=-1)
        residuals.append(err[moving_s & np.isfinite(err)])
    pooled = np.concatenate([r for r in residuals if r.size]) if residuals else np.empty(0)
    if pooled.size < 20:
        return float("inf"), {}

    # Trimmed mean rather than a median. The median is robust but flat: the
    # differences between candidate offsets are a few percent, and discarding
    # all but the middle sample throws away most of that signal. Trimming the
    # bottom 10% and top 20% keeps the sensitivity while still rejecting the
    # mis-detections that make a plain mean useless here. Measured on the
    # synthetic benchmark: 86.7% exact integer recovery versus 73.3% for the
    # median.
    lo, hi = np.percentile(pooled, [10.0, 80.0])
    core = pooled[(pooled >= lo) & (pooled <= hi)]
    value = float(core.mean()) if core.size else float(np.median(pooled))

    return value, {
        "reproj_px": value,
        "coverage": float(np.isfinite(X[..., 0]).mean()),
        "samples": int(pooled.size),
    }


def estimate_frame_offset(
    kpts0: np.ndarray,
    conf0: np.ndarray,
    kpts1: np.ndarray,
    conf1: np.ndarray,
    cameras: CameraPair,
    skeleton: Skeleton,
    cfg: ReconstructionConfig,
) -> Tuple[float, Dict[float, Dict[str, float]]]:
    """Coarse-to-fine search for the inter-camera frame offset.

    The old code evaluated ``np.arange(-3, 3.05, 0.05)`` -- 121 candidates --
    and triangulated every frame for each, which dominated the stage's runtime
    for a result that a two-stage search finds in about a tenth of the work.
    """
    n = kpts0.shape[0]
    if n == 0:
        return 0.0, {}
    step = max(1, n // max(1, cfg.sync_sample_frames))
    sample = np.arange(0, n, step)
    moving = motion_mask(kpts0)

    log: Dict[float, Dict[str, float]] = {}

    def measure(off: float) -> float:
        off = float(round(off, 4))
        if off not in log:
            raw, detail = _offset_score(
                kpts0, conf0, kpts1, conf1, cameras, skeleton, cfg, off, sample, moving
            )
            detail["raw_px"] = raw
            log[off] = detail
        return log[off]["raw_px"]

    def scored(off: float, noise_floor: float) -> float:
        off = float(round(off, 4))
        raw = measure(off)
        detail = log[off]
        if not np.isfinite(raw):
            detail["score"] = float("inf")
            return float("inf")
        value = _debias_residual(raw, off, noise_floor)
        value += 2.0 * (1.0 - detail.get("coverage", 1.0))
        detail["score"] = value
        return value

    # Pass 1: integers only. They all share the same resampling variance, so
    # they are directly comparable without any debiasing.
    coarse = np.arange(
        -cfg.sync_max_offset, cfg.sync_max_offset + 1e-9, cfg.sync_coarse_step
    )
    best = min(coarse, key=lambda o: scored(o, 0.0))

    # The winning integer's residual is dominated by detector noise rather than
    # by timing error, which makes it a usable estimate of the noise floor.
    noise_floor = measure(best)

    best_integer = float(best)
    for off in coarse:
        scored(off, noise_floor)

    # Pass 2: sub-frame, debiased against that floor -- and only accepted if it
    # wins clearly. Two reasons for the guard. The debias needs a noise-floor
    # estimate, and the one available here is itself contaminated when the true
    # offset is fractional. And the payoff is small: on the synthetic benchmark
    # a half-frame sync error costs about 0.5 mm of MPJPE, against roughly 3 mm
    # for a whole-frame error. Getting the integer right is what matters; going
    # sub-frame on thin evidence risks losing that.
    if cfg.sync_fine_step > 0:
        fine = np.arange(
            best_integer - cfg.sync_coarse_step,
            best_integer + cfg.sync_coarse_step + 1e-9,
            cfg.sync_fine_step,
        )
        candidate = float(round(min(fine, key=lambda o: scored(o, noise_floor)), 4))
        integer_score = scored(best_integer, noise_floor)
        candidate_score = scored(candidate, noise_floor)
        improvement = (
            (integer_score - candidate_score) / integer_score if integer_score > 0 else 0.0
        )
        best = candidate if improvement >= cfg.sync_fine_min_gain else best_integer
    else:
        best = best_integer

    best = float(round(best, 4))
    best_score = log[best]["score"]

    # How decisively did the winner win? Sync is only observable through motion,
    # so a clip of someone standing still gives a flat score curve and any
    # answer is guesswork. Recording that beats reporting a confident number
    # that happens to be noise.
    others = [
        d["score"] for off, d in log.items()
        if abs(off - best) >= 0.5 and np.isfinite(d.get("score", np.inf))
    ]
    margin = (min(others) - best_score) / best_score if others and best_score > 0 else 0.0
    log[best]["margin"] = float(margin)
    log[best]["noise_floor_px"] = float(noise_floor)
    return best, log


# --------------------------------------------------------------------------- #
# Stage 2 -- conditioning
# --------------------------------------------------------------------------- #


def synthesise_midhip(
    kpts: np.ndarray, conf: np.ndarray, skeleton: Skeleton
) -> Tuple[np.ndarray, np.ndarray]:
    """Fill the MidHip column as the confidence-weighted mean of the two hips."""
    if "MidHip" not in skeleton:
        return kpts, conf
    l, r, m = skeleton.indices("LHip", "RHip", "MidHip")

    pts = kpts[:, [l, r], :]
    w = np.nan_to_num(conf[:, [l, r]], nan=0.0)
    # A hip with no detection must not contribute even if its confidence is NaN.
    w = np.where(np.isfinite(pts[..., 0]), w, 0.0)
    total = w.sum(axis=1, keepdims=True)

    weights = np.divide(w, total, out=np.full_like(w, 0.5), where=total > 0)
    mid = np.nansum(np.nan_to_num(pts, nan=0.0) * weights[..., None], axis=1)

    none_valid = ~np.isfinite(pts[..., 0]).any(axis=1)
    mid[none_valid] = np.nan

    kpts = kpts.copy()
    conf = conf.copy()
    kpts[:, m, :] = mid

    # MidHip is only as trustworthy as the weaker hip. Computed without nanmin
    # so an all-missing frame does not raise "All-NaN slice encountered".
    hip_conf = np.where(np.isfinite(conf[:, [l, r]]), conf[:, [l, r]], np.inf)
    weakest = hip_conf.min(axis=1)
    conf[:, m] = np.where(np.isfinite(weakest) & ~none_valid, weakest, np.nan)
    return kpts, conf


def _fill_short_gaps(values: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate runs of NaN no longer than ``max_gap``.

    ``values`` is ``(F, ...)``; interpolation runs along axis 0. Leading and
    trailing gaps are not extrapolated.
    """
    out = values.copy()
    n = out.shape[0]
    if n < 2 or max_gap <= 0:
        return out
    flat = out.reshape(n, -1)

    for c in range(flat.shape[1]):
        col = flat[:, c]
        valid = np.isfinite(col)
        if valid.all() or not valid.any():
            continue
        idx = np.flatnonzero(valid)
        # Walk the gaps between consecutive valid samples.
        for a, b in zip(idx[:-1], idx[1:]):
            gap = b - a - 1
            if 0 < gap <= max_gap:
                col[a + 1:b] = np.interp(
                    np.arange(a + 1, b), [a, b], [col[a], col[b]]
                )
    return flat.reshape(values.shape)


# --------------------------------------------------------------------------- #
# Stage 4 -- single-view bootstrapping
# --------------------------------------------------------------------------- #


def bootstrap_single_view(
    points3d: np.ndarray,
    kpts_und: Tuple[np.ndarray, np.ndarray],
    conf: Tuple[np.ndarray, np.ndarray],
    cameras: CameraPair,
    skeleton: Skeleton,
    bone_lengths: Dict[str, float],
    cfg: ReconstructionConfig,
) -> Tuple[np.ndarray, int]:
    """Place joints only one camera can see, using the parent joint's depth.

    A joint seen by a single camera is confined to a ray; the missing degree of
    freedom is resolved by requiring the bone to its already-reconstructed
    parent to have the expected length. Where the ray and the bone sphere miss
    each other entirely, the closest point on the ray is used.

    Returns ``(points, n_bootstrapped)``.
    """
    out = points3d.copy()
    count = 0

    for _ in range(2):   # a second sweep picks up children of joints just filled
        progressed = False
        for b in skeleton.bones:
            p_idx, c_idx = skeleton.index(b.parent), skeleton.index(b.child)
            target = bone_lengths.get(b.name)
            if not target or not np.isfinite(target):
                continue

            missing = ~np.isfinite(out[:, c_idx, 0]) & np.isfinite(out[:, p_idx, 0])
            if not missing.any():
                continue

            for c in range(2):
                cam = cameras[c]
                seen = (
                    missing
                    & np.isfinite(kpts_und[c][:, c_idx, 0])
                    & (np.nan_to_num(conf[c][:, c_idx], nan=-1.0) >= cfg.min_confidence)
                )
                frames = np.flatnonzero(seen)
                if frames.size == 0:
                    continue

                uv = kpts_und[c][frames, c_idx]
                rays = np.concatenate(
                    [uv, np.ones((uv.shape[0], 1))], axis=1
                ) @ cam.K_inv.T
                rays /= np.linalg.norm(rays, axis=1, keepdims=True)
                dirs = rays @ cam.R                      # camera -> world
                origin = cam.center

                parent = out[frames, p_idx]
                # Where the ray meets the sphere of radius `target` centered on
                # the parent joint. Two roots when it cuts through; the tangent
                # point when it misses.
                oc = parent - origin
                t_closest = np.einsum("ij,ij->i", oc, dirs)
                perp_sq = np.einsum("ij,ij->i", oc, oc) - t_closest ** 2
                disc = target ** 2 - perp_sq
                root = np.sqrt(np.maximum(disc, 0.0))
                t_near = np.maximum(t_closest - root, 1e-3)
                t_far = np.maximum(t_closest + root, 1e-3)

                near = origin + dirs * t_near[:, None]
                far = origin + dirs * t_far[:, None]

                # Pick the root that continues the joint's own trajectory. The
                # previous version always took the far root, which flips the
                # limb toward the camera whenever the near root was correct and
                # produces the depth pops that used to need "torso linearity"
                # and acceleration damping to hide.
                reference = np.full((frames.size, 3), np.nan)
                prev_frames = frames - 1
                valid_prev = prev_frames >= 0
                if valid_prev.any():
                    cand = out[prev_frames[valid_prev], c_idx]
                    reference[valid_prev] = cand
                # Fall back to extending the previous bone direction, or to the
                # parent itself for the very first frame.
                fallback = parent
                reference = np.where(np.isfinite(reference), reference, fallback)

                choose_far = (
                    np.linalg.norm(far - reference, axis=1)
                    < np.linalg.norm(near - reference, axis=1)
                )
                candidate = np.where(choose_far[:, None], far, near)

                ok = np.isfinite(candidate).all(axis=1)
                if not ok.any():
                    continue
                out[frames[ok], c_idx] = candidate[ok]
                count += int(ok.sum())
                progressed = True
                missing[frames[ok]] = False
        if not progressed:
            break
    return out, count


# --------------------------------------------------------------------------- #
# Stage 5 -- subject bone model
# --------------------------------------------------------------------------- #


def find_t_pose_frames(
    points3d: np.ndarray, skeleton: Skeleton, cfg: ReconstructionConfig
) -> List[int]:
    """Frames where the subject is closest to a T-pose.

    Scores vertical alignment of shoulder-elbow-wrist against the shoulder line
    plus a penalty for hips sitting above shoulders. The vertical axis is
    detected from the data rather than assumed, because which world axis points
    up depends on how the checkerboard happened to be oriented during
    calibration.
    """
    needed = ("LShoulder", "RShoulder", "LElbow", "RElbow", "LWrist", "RWrist", "LHip", "RHip")
    if any(name not in skeleton for name in needed):
        return []
    ids = {name: skeleton.index(name) for name in needed}

    horizon = min(int(cfg.t_pose_search_seconds * cfg.fps), points3d.shape[0])
    if horizon <= 0:
        return []
    window = points3d[:horizon]

    up = up_direction(points3d, skeleton)
    scores: Dict[int, float] = {}

    for f in range(horizon):
        frame = window[f]
        if any(not np.isfinite(frame[i]).all() for i in ids.values()):
            continue
        h = frame @ up                     # signed height of every joint

        shoulder_level = abs(h[ids["LShoulder"]] - h[ids["RShoulder"]])
        arms_level = (
            abs(h[ids["LShoulder"]] - h[ids["LElbow"]])
            + abs(h[ids["LElbow"]] - h[ids["LWrist"]])
            + abs(h[ids["RShoulder"]] - h[ids["RElbow"]])
            + abs(h[ids["RElbow"]] - h[ids["RWrist"]])
        )
        # Standing upright: hips below shoulders.
        hips_above = max(
            0.0,
            float(0.5 * (h[ids["LHip"]] + h[ids["RHip"]])
                  - 0.5 * (h[ids["LShoulder"]] + h[ids["RShoulder"]])),
        )
        # Arms out: wrist span should be well beyond shoulder width.
        span = float(np.linalg.norm(frame[ids["LWrist"]] - frame[ids["RWrist"]]))
        shoulder_width = float(
            np.linalg.norm(frame[ids["LShoulder"]] - frame[ids["RShoulder"]])
        )
        not_spread = max(0.0, 2.5 * shoulder_width - span)

        scores[f] = float(
            shoulder_level + arms_level + 5.0 * hips_above + 2.0 * not_spread
        )

    if not scores:
        return []
    best = min(scores.values())
    # A perfect score is 0, so fall back to an absolute slack when best ~ 0.
    threshold = best * cfg.t_pose_score_tolerance if best > 1e-6 else 1e-3
    return sorted(f for f, s in scores.items() if s <= threshold)


def up_direction(points3d: np.ndarray, skeleton: Skeleton) -> np.ndarray:
    """Unit vector pointing along the subject's head-up axis, in world frame.

    Which world axis is "up" depends on how the checkerboard happened to sit
    during calibration, so it is measured rather than assumed. The mean
    MidHip -> Neck direction over the clip is a robust estimate: whatever the
    subject does, their torso spends the recording pointing broadly upward.
    """
    default = np.array([0.0, -1.0, 0.0])
    if "Neck" not in skeleton or "MidHip" not in skeleton:
        return default
    neck, hip = skeleton.indices("Neck", "MidHip")
    d = points3d[:, neck] - points3d[:, hip]
    d = d[np.isfinite(d).all(axis=1)]
    if d.size == 0:
        return default
    mean = d.mean(axis=0)
    norm = float(np.linalg.norm(mean))
    return mean / norm if norm > 1e-9 else default


def measure_bone_lengths(
    points3d: np.ndarray,
    skeleton: Skeleton,
    frames: Sequence[int],
    prior: Dict[str, float],
    cfg: ReconstructionConfig,
) -> Dict[str, float]:
    """Personalise the bone model from T-pose frames.

    Falls back to the prior for any bone whose measurement is missing or
    implausible. The old version accepted every measurement unconditionally,
    so a mis-triangulated T-pose could hand the refinement stage a bone model
    that was wrong by a factor of two and then hold the whole sequence to it.
    """
    lengths = dict(prior)
    if not frames:
        return lengths

    idx = np.asarray(list(frames), dtype=int)
    for p, c, name in skeleton.bone_pairs:
        d = np.linalg.norm(points3d[idx, p] - points3d[idx, c], axis=-1)
        d = d[np.isfinite(d)]
        if d.size == 0:
            continue
        if d.size >= 10 and cfg.t_pose_trim_fraction > 0:
            k = int(d.size * cfg.t_pose_trim_fraction)
            d = np.sort(d)[k: d.size - k] if k > 0 else np.sort(d)
        if d.size == 0:
            continue
        measured = float(np.mean(d))

        reference = prior.get(name)
        if reference and np.isfinite(reference) and reference > 0:
            ratio = measured / reference
            if not (1.0 / cfg.t_pose_max_deviation <= ratio <= cfg.t_pose_max_deviation):
                continue
        lengths[name] = measured
    return lengths


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def reconstruct(
    kpts0: np.ndarray,
    conf0: np.ndarray,
    kpts1: np.ndarray,
    conf1: np.ndarray,
    cameras: CameraPair,
    skeleton: Skeleton = BODY25B,
    cfg: Optional[ReconstructionConfig] = None,
    *,
    ground_truth: Optional[np.ndarray] = None,
    progress: Optional[Callable] = None,
) -> ReconstructionResult:
    """Reconstruct a 3D pose sequence from synchronized 2D detections.

    Parameters
    ----------
    kpts0, kpts1
        ``(F, J, 2)`` pixel detections in each view, NaN where absent.
    conf0, conf1
        ``(F, J)`` detector confidences.
    ground_truth
        Optional ``(F, J, 3)`` reference, which switches on the GT metrics.

    Returns
    -------
    ReconstructionResult
    """
    cfg = (cfg or ReconstructionConfig()).validate()
    started = time.perf_counter()
    stages: Dict[str, np.ndarray] = {}
    stats: Dict[str, object] = {}

    kpts0 = np.asarray(kpts0, dtype=float).copy()
    kpts1 = np.asarray(kpts1, dtype=float).copy()
    conf0 = np.asarray(conf0, dtype=float).copy()
    conf1 = np.asarray(conf1, dtype=float).copy()

    n = min(kpts0.shape[0], kpts1.shape[0])
    kpts0, kpts1, conf0, conf1 = kpts0[:n], kpts1[:n], conf0[:n], conf1[:n]

    # -- 1. synchronization ------------------------------------------------ #
    if cfg.sync_use_calibration_offset:
        offset = float(cameras.frame_offset)
        _log(cfg, f"[1/8] Using calibration frame offset: {offset:+.2f}")
        stats["offset_log"] = {}
    else:
        _log(cfg, "[1/8] Searching for inter-camera frame offset...")
        offset, offset_log = estimate_frame_offset(
            kpts0, conf0, kpts1, conf1, cameras, skeleton, cfg
        )
        stats["offset_log"] = {f"{k:+.2f}": v for k, v in sorted(offset_log.items())}
        _log(cfg, f"      -> best offset {offset:+.2f} frames "
                  f"(evaluated {len(offset_log)} candidates)")
    kpts1, conf1 = shift_series(kpts1, conf1, -offset)

    # -- 2. conditioning --------------------------------------------------- #
    _log(cfg, "[2/8] Synthesising MidHip and gating on confidence...")
    kpts0, conf0 = synthesise_midhip(kpts0, conf0, skeleton)
    kpts1, conf1 = synthesise_midhip(kpts1, conf1, skeleton)

    gate0 = np.nan_to_num(conf0, nan=-1.0) < cfg.min_confidence
    gate1 = np.nan_to_num(conf1, nan=-1.0) < cfg.min_confidence
    kpts0[gate0] = np.nan
    kpts1[gate1] = np.nan

    und0 = cameras.cam0.undistort(kpts0)
    und1 = cameras.cam1.undistort(kpts1)

    # -- 3. triangulation -------------------------------------------------- #
    _log(cfg, "[3/8] Triangulating two-view joints...")
    points = triangulate_frames(
        und0, und1, conf0, conf1, cameras,
        min_confidence=cfg.min_confidence, undistort=False,
    )
    stages["step1_triangulated"] = points.copy()
    two_view = int(np.isfinite(points[..., 0]).sum())

    # -- 4. bone prior and single-view bootstrapping ----------------------- #
    prior = scaled_bone_lengths(skeleton, cfg.subject_height_m)
    _log(cfg, "[4/8] Bootstrapping single-view joints...")
    points, n_boot = bootstrap_single_view(
        points, (und0, und1), (conf0, conf1), cameras, skeleton, prior, cfg
    )
    stages["step2_bootstrapped"] = points.copy()
    _log(cfg, f"      -> {two_view} two-view, {n_boot} bootstrapped")

    # -- 5. subject bone model --------------------------------------------- #
    _log(cfg, "[5/8] Measuring bone lengths from T-pose...")
    t_pose = find_t_pose_frames(points, skeleton, cfg)
    bone_lengths = measure_bone_lengths(points, skeleton, t_pose, prior, cfg)
    height = height_from_bone_lengths(bone_lengths, skeleton)
    height_note = f", implied height {height:.2f} m" if np.isfinite(height) else ""
    _log(cfg, f"      -> {len(t_pose)} T-pose frames{height_note}")

    # -- 6. refinement ----------------------------------------------------- #
    _log(cfg, "[6/8] Windowed spatio-temporal bundle adjustment...")
    filled = _fill_short_gaps(points, cfg.max_interpolation_gap)
    obs = np.stack([und0, und1])
    weights = np.stack([
        np.clip(np.nan_to_num(conf0, nan=0.0), 0.0, 1.0),
        np.clip(np.nan_to_num(conf1, nan=0.0), 0.0, 1.0),
    ])
    refined, refine_stats = refine_sequence(
        filled, obs, weights, cameras, skeleton, bone_lengths, cfg, progress=progress
    )
    stages["step3_refined"] = refined.copy()
    stats["refinement"] = refine_stats.as_dict()
    _log(cfg, f"      -> reprojection RMS {refine_stats.reprojection_before_px:.2f} px"
              f" -> {refine_stats.reprojection_after_px:.2f} px")

    # -- 7. gap filling and optional smoothing ----------------------------- #
    _log(cfg, "[7/8] Filling gaps...")
    final = _fill_short_gaps(refined, cfg.max_interpolation_gap)
    if cfg.savgol_after_refine and final.shape[0] >= cfg.savgol_window:
        from scipy.signal import savgol_filter

        for j in range(final.shape[1]):
            col = final[:, j, :]
            if np.isfinite(col).all():
                final[:, j, :] = savgol_filter(
                    col, cfg.savgol_window, cfg.savgol_poly, axis=0, mode="interp"
                )
    stages["step4_final"] = final.copy()

    # -- 8. metrics -------------------------------------------------------- #
    _log(cfg, "[8/8] Computing metrics...")
    metrics = summarise(
        final, skeleton, cameras, und0, und1,
        fps=cfg.fps,
        bone_reference=bone_lengths,
        t_pose_frames=t_pose,
        ground_truth=ground_truth,
    )
    stats["elapsed_seconds"] = round(time.perf_counter() - started, 2)
    stats["two_view_points"] = two_view
    stats["bootstrapped_points"] = n_boot
    stats["config"] = cfg.to_dict()

    return ReconstructionResult(
        points3d=final,
        bone_lengths=bone_lengths,
        metrics=metrics,
        frame_offset=offset,
        t_pose_frames=t_pose,
        stages=stages,
        stats=stats,
        undistorted=(und0, und1),
    )
