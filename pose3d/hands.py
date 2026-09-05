"""Hand reconstruction and wrist kinematics.

The old implementation had two index-space defects that combined to make this
whole stage produce nothing:

1. ``calculate_hand_frame`` indexed joints 1, 2, 5, 9, 13 and 17 of its input,
   documented as "the 22 hand keypoints", but the call site handed it a
   five-row array holding only the wrist and four MCP joints. Every call raised
   ``IndexError``, and the call site's bare ``except Exception`` turned that
   into ``None``, so ``*_wrist_kinematics.json`` came out as a list of empty
   dicts on every run.
2. The depth-backup logic indexed the 44-column *hand* array with *body* joint
   indices, so "left wrist depth" read the left index-finger MCP.

Both disappear here because every lookup goes through
:func:`pose3d.skeleton.hand_joint_index`, which takes a hand and a joint name.

Angle convention
----------------
Wrist angles are reported in degrees relative to the forearm frame, signed so
that the same number means the same anatomical motion on both hands:

* ``FE``  flexion (+) / extension (-)
* ``RU``  radial (+) / ulnar (-) deviation
* ``PS``  pronation (+) / supination (-)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from .camera import CameraPair
from .config import ReconstructionConfig
from .geometry import back_project_ray, orthonormal_basis, triangulate_frames
from .skeleton import (
    HAND_N_JOINTS,
    HANDS_N_JOINTS,
    PALM_JOINTS,
    Skeleton,
    hand_joint_index,
    hand_slice,
)

#: Triangles across the palm whose normals are averaged to get the palm plane.
#: Averaging several beats fitting one because the MCP joints are the noisiest
#: part of a hand detection.
_PALM_TRIANGLES: Tuple[Tuple[str, str, str], ...] = (
    ("THUMB_CMC", "PINKY_MCP", "INDEX_MCP"),
    ("THUMB_CMC", "PINKY_MCP", "MIDDLE_MCP"),
    ("WRIST", "RING_MCP", "INDEX_MCP"),
    ("WRIST", "MIDDLE_MCP", "THUMB_MCP"),
)


@dataclass
class WristKinematics:
    """Per-frame wrist angles and hand frames for both hands."""

    angles: List[Dict[str, Dict[str, float]]]
    frames: List[Dict[str, Optional[Dict[str, List[float]]]]]
    valid_left: int = 0
    valid_right: int = 0

    def summary(self) -> Dict[str, int]:
        return {
            "frames": len(self.angles),
            "valid_left": self.valid_left,
            "valid_right": self.valid_right,
        }


def forearm_frame(
    elbow: np.ndarray, wrist: np.ndarray, hand: str
) -> Optional[np.ndarray]:
    """Right-handed orthonormal basis with its first column along elbow -> wrist.

    The basis is a proper rotation for both hands; the left/right mirroring is
    applied to the reported angles in :func:`wrist_angles` instead. Building a
    reflected basis for the right hand -- as an earlier version did -- makes it
    an improper transform that ``Rotation.from_matrix`` refuses, which silently
    emptied every right-hand angle.

    Returns ``None`` when either endpoint is missing or the segment collapses.
    """
    if not (np.isfinite(elbow).all() and np.isfinite(wrist).all()):
        return None
    axis = np.asarray(wrist, dtype=float) - np.asarray(elbow, dtype=float)
    if np.linalg.norm(axis) < 1e-6:
        return None
    return orthonormal_basis(axis, np.array([0.0, 0.0, 1.0]))


def hand_frame(
    hand_points: np.ndarray,
    hand_conf: np.ndarray,
    forearm_axis: np.ndarray,
    hand: str,
    *,
    min_confidence: float = 0.001,
) -> Optional[np.ndarray]:
    """Estimate the hand's orientation from its palm, after Rolley-Parnell (2018).

    Parameters
    ----------
    hand_points
        ``(21, 3)`` positions for *this* hand, indexed by
        :data:`pose3d.skeleton.HAND_JOINT_NAMES`.
    hand_conf
        ``(21,)`` confidences for the same joints.
    forearm_axis
        Elbow -> wrist direction, used only to disambiguate the palm normal's
        sign.

    Returns
    -------
    ``(3, 3)`` rotation matrix, or ``None`` if the palm cannot be resolved.
    """
    hand_points = np.asarray(hand_points, dtype=float)
    hand_conf = np.asarray(hand_conf, dtype=float)
    if hand_points.shape[0] < HAND_N_JOINTS:
        raise ValueError(
            f"hand_frame expects {HAND_N_JOINTS} joints, got {hand_points.shape[0]}. "
            "Pass the full per-hand slice, not just the palm."
        )

    from .skeleton import HAND_INDEX

    def usable(name: str) -> bool:
        i = HAND_INDEX[name]
        return bool(np.isfinite(hand_points[i]).all()) and float(
            np.nan_to_num(hand_conf[i], nan=0.0)
        ) >= min_confidence

    def P(name: str) -> np.ndarray:
        return hand_points[HAND_INDEX[name]]

    if not usable("WRIST"):
        return None

    # Palm normal: mean of several triangle normals, so one bad MCP cannot
    # dominate the plane estimate.
    normals: List[np.ndarray] = []
    for a, b, c in _PALM_TRIANGLES:
        if not (usable(a) and usable(b) and usable(c)):
            continue
        n = np.cross(P(b) - P(a), P(c) - P(a))
        if np.linalg.norm(n) > 1e-9:
            normals.append(n / np.linalg.norm(n))
    if not normals:
        return None
    z = np.sum(normals, axis=0)
    nz = np.linalg.norm(z)
    if nz < 1e-9:
        return None
    z /= nz
    if np.dot(z, forearm_axis) < 0:
        z = -z

    # Long axis: mean wrist -> MCP direction across whatever MCPs we trust.
    mcp = [P(name) - P("WRIST") for name in PALM_JOINTS if usable(name)]
    if len(mcp) < 2:
        return None
    x = np.sum(mcp, axis=0)
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return None
    x /= nx

    y = np.cross(z, x)
    ny = np.linalg.norm(y)
    if ny < 1e-9:
        return None
    y /= ny
    x = np.cross(y, z)          # re-orthogonalise against float drift
    nx = np.linalg.norm(x)
    if nx < 1e-9:
        return None
    x /= nx

    R = np.column_stack([x, y, z])
    det = float(np.linalg.det(R))
    if det < 0:
        R[:, 0] *= -1.0
        det = float(np.linalg.det(R))
    if not np.isfinite(R).all() or abs(det - 1.0) > 1e-3:
        return None
    return R


def wrist_angles(
    R_forearm: Optional[np.ndarray], R_hand: Optional[np.ndarray], hand: str
) -> Dict[str, float]:
    """Flexion/extension, radial/ulnar deviation and pronation/supination.

    Decomposes the forearm-to-hand rotation as an intrinsic y-x-z sequence. The
    right hand's deviation and rotation signs are mirrored so that a positive
    number means the same anatomical direction on both sides.
    """
    nan = {"FE": float("nan"), "RU": float("nan"), "PS": float("nan")}
    if R_forearm is None or R_hand is None:
        return nan
    try:
        rel = Rotation.from_matrix(np.asarray(R_forearm).T @ np.asarray(R_hand))
        y, x, z = rel.as_euler("yxz", degrees=True)
    except ValueError:
        return nan
    mirror = -1.0 if hand == "right" else 1.0
    return {"FE": float(-z), "RU": float(y * mirror), "PS": float(-x * mirror)}


def interpolate_rotations(
    rotations: Sequence[Optional[Rotation]], max_gap: int = 8
) -> List[Optional[Rotation]]:
    """SLERP across short runs of missing orientations."""
    filled: List[Optional[Rotation]] = list(rotations)
    valid = [(i, r) for i, r in enumerate(filled) if r is not None]
    if len(valid) < 2:
        return filled

    for (i0, r0), (i1, r1) in zip(valid[:-1], valid[1:]):
        gap = i1 - i0
        if not (1 < gap <= max_gap):
            continue
        slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
        times = np.linspace(0.0, 1.0, gap + 1)[1:-1]
        for k, rot in enumerate(slerp(times)):
            filled[i0 + k + 1] = rot
    return filled


def reconstruct_hands(
    kpts0: np.ndarray,
    conf0: np.ndarray,
    kpts1: np.ndarray,
    conf1: np.ndarray,
    cameras: CameraPair,
    body3d: np.ndarray,
    skeleton: Skeleton,
    cfg: ReconstructionConfig,
) -> np.ndarray:
    """Reconstruct 3D hand keypoints, falling back to single-view when needed.

    Two-view triangulation is used wherever both cameras see a joint. Otherwise
    the joint is placed on its camera ray at the depth of the corresponding body
    wrist -- crucially, at that wrist's depth *in the back-projecting camera's
    own frame*. The old code used the wrist's world-space Z for both cameras,
    which only happens to be right for camera 0 because camera 0 defines the
    world origin.

    Returns
    -------
    ``(F, 42, 3)`` array in meters, NaN where unreconstructible.
    """
    out = triangulate_frames(
        kpts0, kpts1, conf0, conf1, cameras, min_confidence=cfg.hand_min_confidence
    )

    wrist_body = {"left": skeleton.get("LWrist"), "right": skeleton.get("RWrist")}
    conf = (np.nan_to_num(conf0, nan=-1.0), np.nan_to_num(conf1, nan=-1.0))
    kpts = (kpts0, kpts1)

    for hand in ("left", "right"):
        body_idx = wrist_body[hand]
        if body_idx is None:
            continue
        sl = hand_slice(hand)
        anchor = body3d[:, body_idx]                       # (F, 3) world

        for c in range(2):
            cam = cameras[c]
            # Depth of the body wrist in *this* camera's frame, per frame.
            depth = (anchor @ cam.R.T + cam.t.ravel())[:, 2]        # (F,)
            usable_frame = np.isfinite(depth) & (depth > 1e-3)
            if not usable_frame.any():
                continue

            missing = ~np.isfinite(out[:, sl, 0])                   # (F, 21)
            have_obs = (
                np.isfinite(kpts[c][:, sl, 0])
                & np.isfinite(kpts[c][:, sl, 1])
                & (conf[c][:, sl] >= cfg.hand_min_confidence)
            )
            todo = missing & have_obs & usable_frame[:, None]
            if not todo.any():
                continue

            f_idx, j_local = np.nonzero(todo)
            uv = cam.undistort(kpts[c][f_idx, sl.start + j_local])  # (N, 2)
            ok = np.isfinite(uv).all(axis=1)
            if not ok.any():
                continue
            f_idx, j_local, uv = f_idx[ok], j_local[ok], uv[ok]

            rays = np.concatenate([uv, np.ones((uv.shape[0], 1))], axis=1) @ cam.K_inv.T
            scale = depth[f_idx] / np.where(np.abs(rays[:, 2]) < 1e-12, np.nan, rays[:, 2])
            pts_cam = rays * scale[:, None]
            world = (pts_cam - cam.t.ravel()) @ cam.R          # R.T @ v == v @ R

            good = np.isfinite(world).all(axis=1)
            out[f_idx[good], sl.start + j_local[good]] = world[good]
    return out


def compute_wrist_kinematics(
    hands3d: np.ndarray,
    hand_conf: np.ndarray,
    body3d: np.ndarray,
    skeleton: Skeleton,
    cfg: ReconstructionConfig,
) -> WristKinematics:
    """Full wrist kinematics track for both hands.

    Parameters
    ----------
    hands3d
        ``(F, 42, 3)`` reconstructed hand joints.
    hand_conf
        ``(F, 42)`` confidence, averaged across cameras.
    body3d
        ``(F, J, 3)`` reconstructed body joints, for the elbows.
    """
    F = hands3d.shape[0]
    if hands3d.shape[1] != HANDS_N_JOINTS:
        raise ValueError(
            f"expected {HANDS_N_JOINTS} hand joints, got {hands3d.shape[1]}"
        )

    elbow_of = {"left": skeleton.get("LElbow"), "right": skeleton.get("RElbow")}
    rotations: Dict[str, List[Optional[Rotation]]] = {"left": [], "right": []}
    forearms: Dict[str, List[Optional[np.ndarray]]] = {"left": [], "right": []}

    for hand in ("left", "right"):
        sl = hand_slice(hand)
        wrist_col = hand_joint_index(hand, "WRIST")
        elbow_idx = elbow_of[hand]

        for f in range(F):
            wrist = hands3d[f, wrist_col]
            elbow = body3d[f, elbow_idx] if elbow_idx is not None else np.full(3, np.nan)
            R_fore = forearm_frame(elbow, wrist, hand)
            forearms[hand].append(R_fore)

            if R_fore is None:
                rotations[hand].append(None)
                continue
            axis = wrist - elbow
            R_hand = hand_frame(
                hands3d[f, sl],
                hand_conf[f, sl],
                axis,
                hand,
                min_confidence=0.0,   # 3D points already passed the 2D gate
            )
            rotations[hand].append(Rotation.from_matrix(R_hand) if R_hand is not None else None)

    for hand in ("left", "right"):
        rotations[hand] = interpolate_rotations(
            rotations[hand], max_gap=cfg.hand_max_rotation_gap
        )

    angles: List[Dict[str, Dict[str, float]]] = []
    frames: List[Dict[str, Optional[Dict[str, List[float]]]]] = []
    counts = {"left": 0, "right": 0}

    for f in range(F):
        frame_angles: Dict[str, Dict[str, float]] = {}
        frame_axes: Dict[str, Optional[Dict[str, List[float]]]] = {"left": None, "right": None}

        for hand in ("left", "right"):
            rot, R_fore = rotations[hand][f], forearms[hand][f]
            if rot is None or R_fore is None:
                continue
            vals = wrist_angles(R_fore, rot.as_matrix(), hand)
            if not np.isfinite(list(vals.values())).all():
                continue
            frame_angles[hand] = vals
            counts[hand] += 1

            M = rot.as_matrix()
            origin = hands3d[f, hand_joint_index(hand, "WRIST")]
            frame_axes[hand] = {
                "origin": [float(v) for v in origin],
                "x_axis": [float(v) for v in M[:, 0]],
                "y_axis": [float(v) for v in M[:, 1]],
                "z_axis": [float(v) for v in M[:, 2]],
            }

        angles.append(frame_angles)
        frames.append(frame_axes)

    angles = _smooth_angle_track(angles, cfg.hand_angle_smooth_window)
    return WristKinematics(
        angles=angles,
        frames=frames,
        valid_left=counts["left"],
        valid_right=counts["right"],
    )


def _smooth_angle_track(
    angles: List[Dict[str, Dict[str, float]]], window: int
) -> List[Dict[str, Dict[str, float]]]:
    """Savitzky-Golay smoothing of each angle channel, gaps preserved.

    Angles are circular, so smoothing the raw degrees would corrupt any trace
    that crosses +/-180. Each channel is unwrapped first and re-wrapped after.
    """
    from scipy.signal import savgol_filter

    if window < 5 or len(angles) < window:
        return angles
    if window % 2 == 0:
        window += 1

    for hand in ("left", "right"):
        for channel in ("FE", "RU", "PS"):
            idx = [i for i, a in enumerate(angles) if hand in a]
            if len(idx) < window:
                continue
            series = np.array([angles[i][hand][channel] for i in idx], dtype=float)
            if not np.isfinite(series).all():
                continue
            unwrapped = np.unwrap(np.deg2rad(series))
            smoothed = savgol_filter(unwrapped, window, min(3, window - 1))
            rewrapped = np.rad2deg(np.arctan2(np.sin(smoothed), np.cos(smoothed)))
            for k, i in enumerate(idx):
                angles[i][hand][channel] = float(rewrapped[k])
    return angles
