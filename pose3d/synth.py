"""Synthetic motion with exact ground truth, for measuring the pipeline.

Why this exists
---------------
No videos, model weights or keypoint files ship with this repository -- they are
all excluded by ``.gitignore`` -- so there is no way to re-run the pipeline on
the original recordings and no way to check whether a change to the
reconstruction code helps or hurts. The numbers quoted in the README came from
runs whose inputs no longer exist, and they disagree with each other and with
the committed ``evaluation_metrics.json``.

This module closes that loop. It builds a scripted motion by forward kinematics
(so bone lengths are exact by construction), projects it through the **real**
calibrated cameras from ``project/task30/camera_parameters``, and corrupts the
2D projections with a detector noise model. Running the pipeline on that gives
true MPJPE against known ground truth.

The motion follows the protocol the README describes for the recorded trials:
T-pose, slow squat, four walking steps, then raising one arm.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import CameraPair
from .skeleton import BODY25B, Skeleton, scaled_bone_lengths


# --------------------------------------------------------------------------- #
# Motion authoring
# --------------------------------------------------------------------------- #


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _smoothstep(t: float) -> float:
    """C1-continuous ease in/out on [0, 1]; keeps synthetic jerk realistic."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


@dataclass
class MotionSegment:
    """One scripted phase of the routine."""

    name: str
    seconds: float
    #: Called with a phase in [0, 1]; returns the pose parameter dict.
    pose: object


@dataclass
class SyntheticSequence:
    """Ground-truth motion plus everything needed to reproduce it."""

    points3d: np.ndarray                   # (F, J, 3) world metres
    bone_lengths: Dict[str, float]
    fps: float
    segments: List[Tuple[str, int, int]] = field(default_factory=list)
    subject_height_m: float = 1.72

    @property
    def n_frames(self) -> int:
        return int(self.points3d.shape[0])


def _pose_parameters(t_global: float, segments: Sequence[MotionSegment], fps: float):
    """Interpolate the scripted parameters at a given time, in seconds."""
    elapsed = 0.0
    for seg in segments:
        if t_global < elapsed + seg.seconds or seg is segments[-1]:
            phase = (t_global - elapsed) / max(seg.seconds, 1e-9)
            return seg.name, seg.pose(min(max(phase, 0.0), 1.0))
        elapsed += seg.seconds
    return segments[-1].name, segments[-1].pose(1.0)


def _default_script() -> List[MotionSegment]:
    """T-pose -> slow squat -> four steps -> raise right arm.

    Every entry is an angle in radians. ``arm_abduction`` at pi/2 is a T-pose
    (arms straight out); 0 is arms at the sides.
    """
    rest = dict(
        knee_bend=0.0, hip_drop=0.0, arm_abduction=np.pi / 2, arm_forward=0.0,
        right_arm_raise=0.0, step_phase=0.0, walk=0.0, yaw=0.0, forward=0.0,
        elbow_bend=0.0,
    )

    def t_pose(_: float) -> Dict[str, float]:
        return dict(rest)

    def arms_down(p: float) -> Dict[str, float]:
        d = dict(rest)
        d["arm_abduction"] = _lerp(np.pi / 2, 0.20, _smoothstep(p))
        return d

    def squat(p: float) -> Dict[str, float]:
        # Down for the first half, back up for the second.
        depth = _smoothstep(p * 2.0) if p < 0.5 else _smoothstep(2.0 - p * 2.0)
        d = dict(rest)
        d["arm_abduction"] = 0.20
        d["knee_bend"] = 1.30 * depth
        d["hip_drop"] = 0.42 * depth
        d["arm_forward"] = 0.9 * depth
        return d

    # Four steps cover roughly 0.9 m here rather than a full stride length: the
    # cameras are ~1.9 m apart and ~4 m away, so a longer walk leaves one view.
    _WALK_DISTANCE_M = 0.9

    def walk(p: float) -> Dict[str, float]:
        d = dict(rest)
        d["arm_abduction"] = 0.18
        d["walk"] = 1.0
        d["step_phase"] = p * 4.0 * np.pi        # four steps = two full cycles
        d["forward"] = p * _WALK_DISTANCE_M
        return d

    def raise_arm(p: float) -> Dict[str, float]:
        s = _smoothstep(min(p * 1.6, 1.0))
        d = dict(rest)
        d["arm_abduction"] = 0.18
        d["forward"] = _WALK_DISTANCE_M
        d["right_arm_raise"] = 2.55 * s
        d["elbow_bend"] = 0.35 * s
        return d

    # Durations matter: four steps spread over nine seconds is a quarter of real
    # walking cadence, and a synchronisation estimator has almost nothing to work
    # with when nothing moves. These give ~1 step/s, which is a normal cadence,
    # and limb-tip speeds around 1-2 m/s.
    return [
        MotionSegment("t_pose", 4.0, t_pose),
        MotionSegment("arms_down", 1.0, arms_down),
        MotionSegment("squat", 4.0, squat),
        MotionSegment("walk", 4.5, walk),
        MotionSegment("raise_arm", 3.5, raise_arm),
    ]


def suggest_subject_placement(
    cameras: CameraPair, image_size: Tuple[int, int] = (1080, 1920)
) -> np.ndarray:
    """A pelvis position both cameras can see comfortably.

    Takes the ray through each camera's principal point and returns the midpoint
    of the two rays' closest approach, which is the world point best centred in
    both views. Derived from the calibration rather than hard-coded, so the
    synthetic subject lands sensibly for any camera geometry.
    """
    w, h = image_size
    origins, directions = [], []
    for cam in (cameras.cam0, cameras.cam1):
        centre_px = (float(cam.K[0, 2]), float(cam.K[1, 2]))
        # Guard against a principal point far outside the frame.
        if not (0 < centre_px[0] < w and 0 < centre_px[1] < h):
            centre_px = (w / 2.0, h / 2.0)
        origins.append(cam.center)
        directions.append(cam.ray_through(centre_px))

    o1, o2 = origins
    d1, d2 = directions
    # Closest approach between two skew lines.
    w0 = o1 - o2
    a, b, c = float(d1 @ d1), float(d1 @ d2), float(d2 @ d2)
    d, e = float(d1 @ w0), float(d2 @ w0)
    denom = a * c - b * b
    if abs(denom) < 1e-9:
        t1 = t2 = 3.0
    else:
        t1 = (b * e - c * d) / denom
        t2 = (a * e - b * d) / denom
    # Keep the subject at a sane distance even for near-parallel cameras.
    t1 = float(np.clip(t1, 1.5, 6.0))
    t2 = float(np.clip(t2, 1.5, 6.0))
    return 0.5 * ((o1 + d1 * t1) + (o2 + d2 * t2))


def _forward_kinematics(
    params: Dict[str, float],
    bones: Dict[str, float],
    skeleton: Skeleton,
    origin: np.ndarray,
    walk_direction: np.ndarray,
) -> np.ndarray:
    """Place every joint from the scripted parameters.

    Coordinates follow the convention the calibrated cameras produce: +Y is
    **down** (it is the image's down axis, and camera 0 defines the world
    frame), so world "up" is -Y. Building the pose directly in that frame avoids
    an extra transform that would be easy to get silently wrong.
    """
    P = np.full((skeleton.n_joints, 3), np.nan)
    idx = skeleton.index
    B = bones

    up = np.array([0.0, -1.0, 0.0])          # world "up" is -Y
    fwd = np.asarray(walk_direction, dtype=float)
    right = np.cross(up, fwd)
    right = right / max(np.linalg.norm(right), 1e-9)

    knee = params["knee_bend"]
    walking = params["walk"] > 0.5
    phase = params["step_phase"]

    leg_len = B["femur_l"] + B["tibia_l"]
    # Squatting lowers the pelvis: the legs fold, so the hip drops toward the floor.
    hip_height = leg_len * np.cos(knee * 0.5)
    root = origin + up * (hip_height - leg_len) + fwd * params["forward"]
    # A little vertical bob while walking, as the stance leg passes under.
    if walking:
        root = root + up * (0.018 * np.cos(2.0 * phase))

    P[idx("MidHip")] = root
    P[idx("LHip")] = root - right * B["midhip_to_lhip"]
    P[idx("RHip")] = root + right * B["midhip_to_rhip"]

    torso_tilt = 0.42 * knee                  # lean forward when squatting
    torso_dir = up * np.cos(torso_tilt) + fwd * np.sin(torso_tilt)
    P[idx("Neck")] = root + torso_dir * B["neck_to_midhip"]
    P[idx("Head")] = P[idx("Neck")] + torso_dir * B["neck_to_head"]
    # Direction must be normalised before scaling by the bone length, or the
    # generated skeleton is not rigid and every bone-length metric measures the
    # generator instead of the reconstruction.
    nose_dir = fwd * 0.85 + up * 0.2
    nose_dir = nose_dir / np.linalg.norm(nose_dir)
    P[idx("Nose")] = P[idx("Head")] + nose_dir * B["head_to_nose"]

    P[idx("LShoulder")] = P[idx("Neck")] - right * B["clavicle_l_to_neck"]
    P[idx("RShoulder")] = P[idx("Neck")] + right * B["clavicle_r_to_neck"]

    # -- legs -------------------------------------------------------------- #
    for side, sign in (("L", -1.0), ("R", 1.0)):
        hip = P[idx(f"{side}Hip")]
        swing = 0.0
        knee_side = knee
        if walking:
            # Opposite legs are half a cycle apart.
            side_phase = phase + (0.0 if side == "L" else np.pi)
            swing = 0.42 * np.sin(side_phase)
            knee_side = knee + 0.55 * max(0.0, np.sin(side_phase + np.pi / 2)) ** 2

        thigh_dir = -up * np.cos(knee_side * 0.5 + swing * 0.0) + fwd * np.sin(swing)
        thigh_dir = thigh_dir / np.linalg.norm(thigh_dir)
        # Squat rotates the thigh forward about the hip.
        thigh_dir = _rotate_about(thigh_dir, right, -knee_side * 0.5)
        knee_pos = hip + thigh_dir * B[f"femur_{side.lower()}"]
        P[idx(f"{side}Knee")] = knee_pos

        shank_dir = _rotate_about(thigh_dir, right, knee_side)
        ankle = knee_pos + shank_dir * B[f"tibia_{side.lower()}"]
        P[idx(f"{side}Ankle")] = ankle
        # Normalised before scaling, so the foot segment is exactly rigid.
        toe_dir = fwd * 0.97 - up * 0.24
        toe_dir = toe_dir / np.linalg.norm(toe_dir)
        P[idx(f"{side}BigToe")] = ankle + toe_dir * B[f"ankle_to_bigtoe_{side.lower()}"]

    # -- arms -------------------------------------------------------------- #
    for side, sign in (("L", -1.0), ("R", 1.0)):
        shoulder = P[idx(f"{side}Shoulder")]
        abduction = params["arm_abduction"]
        forward = params["arm_forward"]
        if side == "R":
            abduction = max(abduction, params["right_arm_raise"])

        if walking:
            side_phase = phase + (np.pi if side == "L" else 0.0)
            forward = forward + 0.30 * np.sin(side_phase)

        # Abduction rotates the arm away from the body in the frontal plane.
        upper = (
            right * sign * np.sin(min(abduction, np.pi / 2))
            - up * np.cos(abduction)
            + fwd * np.sin(forward) * 0.9
        )
        # Beyond 90 degrees the arm keeps rising rather than crossing over.
        if abduction > np.pi / 2:
            extra = abduction - np.pi / 2
            upper = right * sign * np.cos(extra) + up * np.sin(extra) + fwd * np.sin(forward) * 0.4
        upper = upper / np.linalg.norm(upper)

        elbow = shoulder + upper * B[f"humerus_{side.lower()}"]
        P[idx(f"{side}Elbow")] = elbow

        bend = params["elbow_bend"] + (0.25 if walking else 0.05)
        fore = _rotate_about(upper, right * sign, bend)
        P[idx(f"{side}Wrist")] = elbow + fore * B[f"radius_ulna_{side.lower()}"]

    return P


def _rotate_about(vector: np.ndarray, axis: np.ndarray, angle: float) -> np.ndarray:
    """Rodrigues rotation of ``vector`` about a unit ``axis``."""
    axis = axis / max(np.linalg.norm(axis), 1e-12)
    c, s = np.cos(angle), np.sin(angle)
    out = vector * c + np.cross(axis, vector) * s + axis * np.dot(axis, vector) * (1.0 - c)
    n = np.linalg.norm(out)
    return out / n if n > 1e-12 else vector


def make_sequence(
    *,
    fps: float = 30.0,
    subject_height_m: float = 1.72,
    skeleton: Skeleton = BODY25B,
    script: Optional[Sequence[MotionSegment]] = None,
    origin: Optional[np.ndarray] = None,
    cameras: Optional[CameraPair] = None,
    walk_direction: Optional[np.ndarray] = None,
    time_offset_frames: float = 0.0,
) -> SyntheticSequence:
    """Generate the ground-truth 3D motion.

    Bone lengths are exact by construction, so a reconstruction's bone-length
    CV measures reconstruction error and nothing else.

    Parameters
    ----------
    origin
        Where the pelvis starts, in world metres. Defaults to
        :func:`suggest_subject_placement` when ``cameras`` is given, otherwise
        3 m along +Z.
    walk_direction
        Horizontal direction the subject walks in. Defaults to the direction
        that keeps them roughly equidistant from both cameras -- walking
        straight at a camera would make the projection degenerate.
    time_offset_frames
        Shift the sampling instants by this many frames. Used by
        :func:`make_desynchronised_pair` to simulate unsynchronised shutters.
    """
    segments = list(script) if script is not None else _default_script()
    bones = scaled_bone_lengths(skeleton, subject_height_m)
    total_seconds = sum(s.seconds for s in segments)
    n_frames = int(round(total_seconds * fps))

    if origin is None:
        origin = (
            suggest_subject_placement(cameras)
            if cameras is not None
            else np.array([0.0, 0.0, 3.0])
        )
    origin = np.asarray(origin, dtype=float).reshape(3)

    if walk_direction is None:
        if cameras is not None:
            # Walk along the camera baseline: both views keep good parallax.
            baseline = cameras.cam1.center - cameras.cam0.center
            baseline[1] = 0.0            # stay horizontal (world up is -Y)
            norm = np.linalg.norm(baseline)
            walk_direction = baseline / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
        else:
            walk_direction = np.array([1.0, 0.0, 0.0])
    walk_direction = np.asarray(walk_direction, dtype=float).reshape(3)
    walk_direction = walk_direction / max(np.linalg.norm(walk_direction), 1e-9)

    points = np.empty((n_frames, skeleton.n_joints, 3))
    spans: List[Tuple[str, int, int]] = []
    current_name, start = None, 0

    for f in range(n_frames):
        t = (f + time_offset_frames) / fps
        name, params = _pose_parameters(t, segments, fps)
        points[f] = _forward_kinematics(params, bones, skeleton, origin, walk_direction)
        if name != current_name:
            if current_name is not None:
                spans.append((current_name, start, f))
            current_name, start = name, f
    if current_name is not None:
        spans.append((current_name, start, n_frames))

    return SyntheticSequence(
        points3d=points,
        bone_lengths=bones,
        fps=fps,
        segments=spans,
        subject_height_m=subject_height_m,
    )


def make_desynchronised_pair(
    *,
    offset_frames: float,
    fps: float = 30.0,
    subject_height_m: float = 1.72,
    skeleton: Skeleton = BODY25B,
    cameras: Optional[CameraPair] = None,
    **kwargs,
) -> Tuple[SyntheticSequence, SyntheticSequence]:
    """Two views of the same motion, sampled ``offset_frames`` apart in time.

    This is what unsynchronised phone shutters actually do: camera 1 samples the
    *scene* at a different instant, rather than sampling a linear interpolation
    of camera 0's frames. Simulating it by resampling camera 0's keypoints
    would low-pass the motion and make sub-frame offsets look unrecoverable
    even when they are not, so the second view is generated from the kinematic
    model at the shifted time base instead.

    Returns
    -------
    (sequence_for_cam0, sequence_for_cam1)
    """
    common = dict(
        fps=fps, subject_height_m=subject_height_m, skeleton=skeleton,
        cameras=cameras, **kwargs,
    )
    base = make_sequence(time_offset_frames=0.0, **common)
    shifted = make_sequence(time_offset_frames=float(offset_frames), **common)
    return base, shifted


# --------------------------------------------------------------------------- #
# Detector simulation
# --------------------------------------------------------------------------- #


@dataclass
class DetectorNoise:
    """Model of what a real 2D detector does to a perfect projection.

    Defaults are calibrated to BODY_25B behaviour at 2-3 m on 1080x1920 phone
    video: a few pixels of jitter, occasional dropouts, and rarer bursts where a
    limb is occluded for a fraction of a second.
    """

    pixel_sigma: float = 2.5
    #: Extra jitter on the joints BODY_25B is worst at.
    hard_joint_sigma: float = 4.5
    hard_joints: Tuple[str, ...] = ("LBigToe", "RBigToe", "LAnkle", "RAnkle", "Nose")
    #: Probability a given joint is missing in a given frame.
    dropout_rate: float = 0.04
    #: Probability an occlusion burst starts on a joint in a given frame.
    occlusion_rate: float = 0.0025
    occlusion_length: Tuple[int, int] = (5, 25)
    #: Probability of a gross mis-detection (limb swap, background match).
    outlier_rate: float = 0.004
    outlier_pixels: float = 45.0
    #: Confidence is sampled around this mean for visible joints.
    confidence_mean: float = 0.78
    confidence_sigma: float = 0.12


def project_sequence(
    sequence: SyntheticSequence,
    cameras: CameraPair,
    noise: Optional[DetectorNoise] = None,
    *,
    skeleton: Skeleton = BODY25B,
    seed: int = 0,
    image_size: Optional[Tuple[int, int]] = (1080, 1920),
) -> Dict[str, np.ndarray]:
    """Project ground truth into both cameras and apply the detector model.

    Returns
    -------
    dict with ``kpts0``, ``conf0``, ``kpts1``, ``conf1`` and ``clean0``/``clean1``
    (the noise-free projections, useful for isolating geometry errors from
    detector errors).
    """
    rng = np.random.default_rng(seed)
    noise = noise or DetectorNoise()
    F, J = sequence.points3d.shape[:2]

    out: Dict[str, np.ndarray] = {}
    hard = {skeleton.get(n) for n in noise.hard_joints}
    hard.discard(None)

    for c, cam in enumerate((cameras.cam0, cameras.cam1)):
        clean = cam.project(sequence.points3d)          # (F, J, 2)
        kpts = clean.copy()
        conf = np.clip(
            rng.normal(noise.confidence_mean, noise.confidence_sigma, size=(F, J)),
            0.05, 1.0,
        )

        sigma = np.full(J, noise.pixel_sigma)
        for j in hard:
            sigma[j] = noise.hard_joint_sigma
        kpts += rng.normal(0.0, 1.0, size=kpts.shape) * sigma[None, :, None]

        # Gross mis-detections: large offset, and the detector stays confident.
        outliers = rng.random((F, J)) < noise.outlier_rate
        kpts[outliers] += rng.normal(0.0, noise.outlier_pixels, size=(int(outliers.sum()), 2))

        # Independent per-frame dropouts.
        drops = rng.random((F, J)) < noise.dropout_rate

        # Occlusion bursts: a joint disappears for a run of frames.
        for j in range(J):
            starts = np.flatnonzero(rng.random(F) < noise.occlusion_rate)
            for s in starts:
                length = int(rng.integers(*noise.occlusion_length))
                drops[s: s + length, j] = True

        kpts[drops] = np.nan
        conf[drops] = np.nan

        # Anything outside the sensor is not detected either.
        if image_size is not None:
            w, h = image_size
            off = (
                (kpts[..., 0] < 0) | (kpts[..., 0] >= w)
                | (kpts[..., 1] < 0) | (kpts[..., 1] >= h)
            )
            kpts[off] = np.nan
            conf[off] = np.nan

        behind = ~np.isfinite(clean[..., 0])
        kpts[behind] = np.nan
        conf[behind] = np.nan

        out[f"kpts{c}"] = kpts
        out[f"conf{c}"] = conf
        out[f"clean{c}"] = clean
    return out


def perfect_observations(
    sequence: SyntheticSequence, cameras: CameraPair
) -> Dict[str, np.ndarray]:
    """Noise-free projections with unit confidence -- an upper bound on accuracy."""
    F, J = sequence.points3d.shape[:2]
    out: Dict[str, np.ndarray] = {}
    for c, cam in enumerate((cameras.cam0, cameras.cam1)):
        proj = cam.project(sequence.points3d)
        conf = np.where(np.isfinite(proj[..., 0]), 1.0, np.nan)
        out[f"kpts{c}"] = proj
        out[f"conf{c}"] = conf
    return out
