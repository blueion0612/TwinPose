"""Low-cost 3D pose estimation from two smartphone videos.

This package holds the logic that used to be copy-pasted across the individual
pipeline scripts.  In particular the keypoint schema and its "drop the face and
the small foot joints" index remapping now live in exactly one place
(:mod:`pose3d.skeleton`); before, four scripts each re-derived it and one of them
(``estimation/compare.py``) got it wrong and silently read the wrong joints.

Typical use::

    from pose3d import BODY25B, load_camera_pair, reconstruct

    cams = load_camera_pair(task_dir / "camera_parameters")
    result = reconstruct(kpts0, conf0, kpts1, conf1, cams, BODY25B)
"""

from __future__ import annotations

from .skeleton import (
    BODY25B,
    BODY25B_RAW_NAMES,
    Bone,
    Skeleton,
    scaled_bone_lengths,
)
from .camera import Camera, CameraPair, load_camera, load_camera_pair
from .kpio import (
    load_body_keypoints,
    load_hand_keypoints,
    save_keypoints_3d,
    load_keypoints_3d,
)
from .geometry import (
    triangulate_frames,
    triangulate_points,
    project,
    back_project_ray,
    orthonormal_basis,
)
from .config import ReconstructionConfig
from .pipeline import ReconstructionResult, reconstruct

__all__ = [
    "BODY25B",
    "BODY25B_RAW_NAMES",
    "Bone",
    "Skeleton",
    "scaled_bone_lengths",
    "Camera",
    "CameraPair",
    "load_camera",
    "load_camera_pair",
    "load_body_keypoints",
    "load_hand_keypoints",
    "save_keypoints_3d",
    "load_keypoints_3d",
    "triangulate_frames",
    "triangulate_points",
    "project",
    "back_project_ray",
    "orthonormal_basis",
    "ReconstructionConfig",
    "ReconstructionResult",
    "reconstruct",
]

__version__ = "2.0.0"
