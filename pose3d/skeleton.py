"""Keypoint schema, bone model and index remapping -- the single source of truth.

Historical note
---------------
``3D_estimation.py``, ``plot.py``, ``compare.py`` and ``3D_estimation_mmpose.py``
each used to re-derive the "drop the face + small-foot joints" remapping by hand.
Three of them agreed; ``compare.py`` did not, so it silently read ``LWrist``
whenever it asked for ``LShoulder``.  Everything now goes through
:data:`BODY25B`, so that class of bug cannot recur.

Units
-----
Bone lengths are in **metres** throughout this package.  Camera translations
coming out of calibration are in the checkerboard's unit (centimetres, because
``checkerboard_box_size_scale`` is given in cm); :mod:`pose3d.camera` converts
them to metres on load, so every 3D coordinate in the pipeline is in metres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Raw OpenPose BODY_25B layout, as produced by estimation/Openpose.py.
# Index 25 (MidHip) is not emitted by the network; the pipeline synthesises it
# from the two hip joints, so it is part of the schema from the start.
# --------------------------------------------------------------------------- #
BODY25B_RAW_NAMES: Tuple[str, ...] = (
    "Nose", "LEye", "REye", "LEar", "REar",
    "LShoulder", "RShoulder", "LElbow", "RElbow",
    "LWrist", "RWrist", "LHip", "RHip",
    "LKnee", "RKnee", "LAnkle", "RAnkle",
    "Neck", "Head", "LBigToe", "LSmallToe",
    "LHeel", "RBigToe", "RSmallToe", "RHeel",
    "MidHip",
)

#: Joints the 3D stage discards. The eyes/ears carry no useful 3D signal at
#: 2-3 m with two phone cameras, and the small toe / heel detections are the
#: noisiest keypoints BODY_25B produces.
BODY25B_DROPPED: Tuple[str, ...] = (
    "LEye", "REye", "LEar", "REar",
    "LSmallToe", "LHeel", "RSmallToe", "RHeel",
)


@dataclass(frozen=True)
class Bone:
    """A rigid segment between two joints.

    ``default_length`` is a population prior in metres for a 1.72 m adult; it is
    rescaled to the subject via :func:`scaled_bone_lengths` and then replaced
    outright by the T-pose measurement when one is available.
    """

    parent: str
    child: str
    name: str
    default_length: float
    #: Bones whose length is stable across a recording and therefore usable to
    #: anchor the global scale. Excludes anything that hinges or slides.
    scale_anchor: bool = False


# --------------------------------------------------------------------------- #
# Bone model.
#
# The `midhip_to_?hip` defaults deserve a comment: the original code called
# these `neck_to_lhip_diag` / `neck_to_rhip_diag` and gave them a default of
# 0.52 m, which is a torso length, not half a pelvis width.  Those defaults fed
# both the single-view bootstrapping and the first bundle-adjustment pass, so
# every bootstrapped hip was placed roughly five times too far from the pelvis
# centre.  The project's own T-pose measurement (README section 5-5-1) reports
# 0.0955 m, which matches the anatomy: a hip half-width is ~0.08-0.11 m.
# --------------------------------------------------------------------------- #
_BODY25B_BONES: Tuple[Bone, ...] = (
    Bone("LShoulder", "LElbow", "humerus_l", 0.307),
    Bone("LElbow", "LWrist", "radius_ulna_l", 0.265),
    Bone("RShoulder", "RElbow", "humerus_r", 0.307),
    Bone("RElbow", "RWrist", "radius_ulna_r", 0.265),
    Bone("LHip", "LKnee", "femur_l", 0.430, scale_anchor=True),
    Bone("LKnee", "LAnkle", "tibia_l", 0.390, scale_anchor=True),
    Bone("RHip", "RKnee", "femur_r", 0.430, scale_anchor=True),
    Bone("RKnee", "RAnkle", "tibia_r", 0.390, scale_anchor=True),
    Bone("Neck", "LShoulder", "clavicle_l_to_neck", 0.155, scale_anchor=True),
    Bone("Neck", "RShoulder", "clavicle_r_to_neck", 0.155, scale_anchor=True),
    Bone("Neck", "MidHip", "neck_to_midhip", 0.480, scale_anchor=True),
    Bone("MidHip", "LHip", "midhip_to_lhip", 0.100, scale_anchor=True),
    Bone("MidHip", "RHip", "midhip_to_rhip", 0.100, scale_anchor=True),
    Bone("Neck", "Head", "neck_to_head", 0.180),
    Bone("Head", "Nose", "head_to_nose", 0.090),
    Bone("LAnkle", "LBigToe", "ankle_to_bigtoe_l", 0.175),
    Bone("RAnkle", "RBigToe", "ankle_to_bigtoe_r", 0.175),
)

#: Three-joint chains used by the anatomical plausibility term. Each entry is
#: (proximal, middle, distal); the middle joint is the one that hinges.
_BODY25B_CHAINS: Tuple[Tuple[str, str, str], ...] = (
    ("LShoulder", "LElbow", "LWrist"),
    ("RShoulder", "RElbow", "RWrist"),
    ("LHip", "LKnee", "LAnkle"),
    ("RHip", "RKnee", "RAnkle"),
    ("Neck", "MidHip", "LHip"),
    ("Neck", "MidHip", "RHip"),
)

#: Kinematic tree used by the synthetic generator and by bootstrapping order.
_BODY25B_PARENTS: Dict[str, Optional[str]] = {
    "MidHip": None,
    "LHip": "MidHip",
    "RHip": "MidHip",
    "Neck": "MidHip",
    "LKnee": "LHip",
    "RKnee": "RHip",
    "LAnkle": "LKnee",
    "RAnkle": "RKnee",
    "LBigToe": "LAnkle",
    "RBigToe": "RAnkle",
    "LShoulder": "Neck",
    "RShoulder": "Neck",
    "LElbow": "LShoulder",
    "RElbow": "RShoulder",
    "LWrist": "LElbow",
    "RWrist": "RElbow",
    "Head": "Neck",
    "Nose": "Head",
}


@dataclass(frozen=True)
class Skeleton:
    """An ordered set of joints plus the bones and chains defined over them.

    Attributes
    ----------
    names
        Joint names in pipeline order. ``names[i]`` is the joint stored at
        column ``i`` of every ``(F, J, ...)`` array in the pipeline.
    raw_names
        The full upstream layout the pipeline slices from.
    source_indices
        For each pipeline column, its index in ``raw_names``. Use
        :meth:`select` to slice a raw array down to the pipeline layout.
    """

    names: Tuple[str, ...]
    raw_names: Tuple[str, ...]
    source_indices: np.ndarray
    bones: Tuple[Bone, ...]
    chains: Tuple[Tuple[int, int, int], ...]
    parents: Tuple[Optional[int], ...]
    _index: Dict[str, int] = field(repr=False, default_factory=dict)

    # -- construction ------------------------------------------------------ #
    @classmethod
    def build(
        cls,
        raw_names: Sequence[str],
        dropped: Sequence[str],
        bones: Sequence[Bone],
        chains: Sequence[Tuple[str, str, str]],
        parents: Dict[str, Optional[str]],
    ) -> "Skeleton":
        dropped_set = set(dropped)
        unknown = dropped_set - set(raw_names)
        if unknown:
            raise ValueError(f"cannot drop unknown joints: {sorted(unknown)}")

        kept = [n for n in raw_names if n not in dropped_set]
        index = {name: i for i, name in enumerate(kept)}
        source = np.array([raw_names.index(n) for n in kept], dtype=int)

        # A bone or chain survives only if every joint it touches survives.
        kept_bones = tuple(
            b for b in bones if b.parent in index and b.child in index
        )
        kept_chains = tuple(
            (index[p], index[c], index[g])
            for p, c, g in chains
            if p in index and c in index and g in index
        )
        parent_idx: List[Optional[int]] = []
        for name in kept:
            p = parents.get(name)
            parent_idx.append(index[p] if p is not None and p in index else None)

        return cls(
            names=tuple(kept),
            raw_names=tuple(raw_names),
            source_indices=source,
            bones=kept_bones,
            chains=kept_chains,
            parents=tuple(parent_idx),
            _index=index,
        )

    # -- lookups ----------------------------------------------------------- #
    def __len__(self) -> int:
        return len(self.names)

    @property
    def n_joints(self) -> int:
        return len(self.names)

    def __contains__(self, name: object) -> bool:
        return name in self._index

    def index(self, name: str) -> int:
        """Pipeline column for ``name``. Raises if the joint was dropped."""
        try:
            return self._index[name]
        except KeyError:
            raise KeyError(
                f"joint {name!r} is not part of this skeleton "
                f"(available: {', '.join(self.names)})"
            ) from None

    def indices(self, *names: str) -> List[int]:
        return [self.index(n) for n in names]

    def get(self, name: str, default: Optional[int] = None) -> Optional[int]:
        return self._index.get(name, default)

    def raw_index(self, name: str) -> int:
        return self.raw_names.index(name)

    # -- array helpers ----------------------------------------------------- #
    def select(self, array: np.ndarray, axis: int = 1) -> np.ndarray:
        """Slice a raw ``(F, len(raw_names), ...)`` array to pipeline layout."""
        if array.shape[axis] != len(self.raw_names):
            raise ValueError(
                f"expected {len(self.raw_names)} joints on axis {axis}, "
                f"got {array.shape[axis]}"
            )
        return np.take(array, self.source_indices, axis=axis)

    @property
    def bone_pairs(self) -> List[Tuple[int, int, str]]:
        """``(parent_idx, child_idx, bone_name)`` in pipeline index space."""
        return [(self.index(b.parent), self.index(b.child), b.name) for b in self.bones]

    @property
    def edges(self) -> List[Tuple[int, int]]:
        """Index pairs for drawing the skeleton."""
        return [(self.index(b.parent), self.index(b.child)) for b in self.bones]

    @property
    def default_lengths(self) -> Dict[str, float]:
        return {b.name: b.default_length for b in self.bones}

    @property
    def scale_anchor_bones(self) -> List[Tuple[int, int, str]]:
        return [
            (self.index(b.parent), self.index(b.child), b.name)
            for b in self.bones
            if b.scale_anchor
        ]

    def topological_order(self) -> List[int]:
        """Joint indices ordered so a parent always precedes its children."""
        order: List[int] = []
        seen = set()

        def visit(i: int) -> None:
            if i in seen:
                return
            p = self.parents[i]
            if p is not None:
                visit(p)
            seen.add(i)
            order.append(i)

        for i in range(self.n_joints):
            visit(i)
        return order


#: The schema the whole pipeline runs on: BODY_25B plus a synthesised MidHip,
#: minus the eyes/ears and the small-toe/heel joints. 18 joints.
BODY25B: Skeleton = Skeleton.build(
    raw_names=BODY25B_RAW_NAMES,
    dropped=BODY25B_DROPPED,
    bones=_BODY25B_BONES,
    chains=_BODY25B_CHAINS,
    parents=_BODY25B_PARENTS,
)


_REFERENCE_HEIGHT_M = 1.72

_DEFAULT_BY_NAME: Dict[str, float] = {b.name: b.default_length for b in _BODY25B_BONES}

#: Hip-joint-centre to ankle-joint-centre as a fraction of standing height.
#: Derived from this module's own femur+tibia defaults so the prior and its
#: inverse stay mutually consistent. Note this is the *joint centre* chain,
#: which sits below the 0.53 trochanter-height figure in the anthropometry
#: tables because the ankle joint centre is above the floor.
_HIP_HEIGHT_FRACTION = (
    _DEFAULT_BY_NAME["femur_l"] + _DEFAULT_BY_NAME["tibia_l"]
) / _REFERENCE_HEIGHT_M


def scaled_bone_lengths(
    skeleton: Skeleton, subject_height_m: Optional[float] = None
) -> Dict[str, float]:
    """Bone-length prior, optionally rescaled to a subject's standing height.

    The population defaults describe a 1.72 m adult.  Passing the real height
    turns a generic prior into a subject-specific one, which matters because
    these numbers seed the single-view bootstrapping and the first refinement
    pass long before any T-pose measurement is available.

    Parameters
    ----------
    subject_height_m
        Standing height in metres. ``None`` returns the unscaled defaults.
    """
    lengths = skeleton.default_lengths
    if subject_height_m is None:
        return lengths
    if not np.isfinite(subject_height_m) or subject_height_m <= 0:
        raise ValueError(f"subject_height_m must be positive, got {subject_height_m!r}")
    ratio = float(subject_height_m) / _REFERENCE_HEIGHT_M
    return {name: length * ratio for name, length in lengths.items()}


def height_from_bone_lengths(
    lengths: Dict[str, float], skeleton: Skeleton = BODY25B
) -> float:
    """Estimate standing height from a measured bone model.

    Uses the leg chain (hip -> knee -> ankle) plus the anthropometric hip-height
    fraction; returns NaN when the leg bones are unavailable.
    """
    legs = []
    for femur, tibia in (("femur_l", "tibia_l"), ("femur_r", "tibia_r")):
        f, t = lengths.get(femur), lengths.get(tibia)
        if f and t and np.isfinite(f) and np.isfinite(t):
            legs.append(f + t)
    if not legs:
        return float("nan")
    return float(np.mean(legs)) / _HIP_HEIGHT_FRACTION


# --------------------------------------------------------------------------- #
# Hand schema (OpenPose hand model: 21 keypoints per hand).
#
# The Caffe hand model emits 22 heatmap channels -- 21 joints plus a background
# channel.  The old code stored all 22 as if they were joints and laid the two
# hands out at stride 22, so "right wrist" landed at index 22 which is actually
# the left hand's background channel.  We keep 21 per hand and lay them out at
# stride 21; readers tolerate the legacy 22-wide format.
# --------------------------------------------------------------------------- #
HAND_JOINT_NAMES: Tuple[str, ...] = (
    "WRIST",
    "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_MCP", "INDEX_PIP", "INDEX_DIP", "INDEX_TIP",
    "MIDDLE_MCP", "MIDDLE_PIP", "MIDDLE_DIP", "MIDDLE_TIP",
    "RING_MCP", "RING_PIP", "RING_DIP", "RING_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
)
HAND_N_JOINTS = len(HAND_JOINT_NAMES)  # 21
HAND_INDEX: Dict[str, int] = {n: i for i, n in enumerate(HAND_JOINT_NAMES)}

#: Both hands concatenated: left occupies [0, 21), right occupies [21, 42).
HANDS_N_JOINTS = 2 * HAND_N_JOINTS

#: Finger bones, for drawing and for the hand-frame plane fit.
HAND_EDGES: Tuple[Tuple[int, int], ...] = tuple(
    (a, b)
    for a, b in (
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
    )
)

#: Metacarpophalangeal joints -- the rigid palm. These define the hand frame.
PALM_JOINTS: Tuple[str, ...] = ("INDEX_MCP", "MIDDLE_MCP", "RING_MCP", "PINKY_MCP")


def hand_slice(hand: str) -> slice:
    """Column range for ``'left'`` or ``'right'`` in a 42-wide hand array."""
    if hand == "left":
        return slice(0, HAND_N_JOINTS)
    if hand == "right":
        return slice(HAND_N_JOINTS, HANDS_N_JOINTS)
    raise ValueError(f"hand must be 'left' or 'right', got {hand!r}")


def hand_joint_index(hand: str, joint: str) -> int:
    """Absolute column of ``joint`` on ``hand`` in a 42-wide hand array."""
    try:
        local = HAND_INDEX[joint]
    except KeyError:
        raise KeyError(
            f"unknown hand joint {joint!r} (available: {', '.join(HAND_JOINT_NAMES)})"
        ) from None
    return hand_slice(hand).start + local
