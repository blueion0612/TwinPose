"""Pinhole camera model and calibration-file IO.

Two long-standing problems are fixed here.

**Unit confusion.**  ``calibration.py`` writes translations in the checkerboard's
unit, which is centimetres (``checkerboard_box_size_scale: 2.7`` is a cm value).
``3D_estimation.py`` consumed them as-is and then invented a ``SCALE`` factor of
~106 to reconcile the resulting centimetre-ish world with metre-denominated bone
lengths, while ``3D_estimation_mmpose.py`` divided by 100 and expected a scale of
~1.0.  Loading now converts to metres exactly once, here, so downstream code has
a single unambiguous unit.

**Offset key mismatch.**  ``calibration.py`` stores the chosen inter-camera frame
offset under ``best_offset``; ``3D_estimation_mmpose.py`` read ``frame_offset``
and so silently always got 0.  :func:`load_camera` accepts either spelling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

#: Calibration distances follow ``checkerboard_box_size_scale``, which the
#: settings file documents as centimetres.
CM_PER_M = 100.0

#: Accepted spellings for the stored inter-camera frame offset, most-preferred
#: first. ``best_offset`` is what calibration.py writes.
_OFFSET_KEYS: Tuple[str, ...] = ("best_offset", "frame_offset", "offset")

PathLike = Union[str, Path]


@dataclass(frozen=True)
class Camera:
    """A calibrated pinhole camera with lens distortion.

    All translations and world coordinates are in **metres**.

    Attributes
    ----------
    K
        3x3 intrinsic matrix.
    dist
        Distortion coefficients in OpenCV order (4, 5, 8, 12 or 14 of them).
    R, t
        World-to-camera rotation and translation: ``x_cam = R @ x_world + t``.
    """

    K: np.ndarray
    dist: np.ndarray
    R: np.ndarray
    t: np.ndarray
    name: str = "camera"

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass the setattr guard for normalisation.
        object.__setattr__(self, "K", np.asarray(self.K, dtype=float).reshape(3, 3))
        object.__setattr__(self, "dist", np.asarray(self.dist, dtype=float).ravel())
        object.__setattr__(self, "R", np.asarray(self.R, dtype=float).reshape(3, 3))
        object.__setattr__(self, "t", np.asarray(self.t, dtype=float).reshape(3, 1))

        det = float(np.linalg.det(self.R))
        if not np.isclose(det, 1.0, atol=1e-3):
            raise ValueError(
                f"{self.name}: R is not a rotation matrix (det={det:.6f}). "
                "Check the extrinsics file."
            )

    # -- derived quantities ------------------------------------------------ #
    @property
    def P(self) -> np.ndarray:
        """3x4 projection matrix ``K [R|t]``."""
        return self.K @ np.hstack((self.R, self.t))

    @property
    def center(self) -> np.ndarray:
        """Camera centre in world coordinates, shape ``(3,)``."""
        return (-self.R.T @ self.t).ravel()

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    # -- projection -------------------------------------------------------- #
    def project(self, points_world: np.ndarray) -> np.ndarray:
        """Project ``(..., 3)`` world points to ``(..., 2)`` pixels.

        Ignores distortion: the pipeline undistorts observations up front, so
        the forward model is the ideal pinhole. Points at or behind the image
        plane come back as NaN rather than as a spurious finite pixel.
        """
        pts = np.asarray(points_world, dtype=float)
        cam = pts @ self.R.T + self.t.ravel()
        z = cam[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            xy = cam[..., :2] / z[..., None]
        uv = xy @ self.K[:2, :2].T + self.K[:2, 2]
        return np.where((z > 1e-9)[..., None], uv, np.nan)

    def undistort(self, points_px: np.ndarray) -> np.ndarray:
        """Undistort ``(..., 2)`` pixel observations, staying in pixel units."""
        import cv2  # local import: the geometry core stays importable without cv2

        pts = np.asarray(points_px, dtype=np.float64)
        shape = pts.shape
        flat = pts.reshape(-1, 1, 2)
        finite = np.isfinite(flat[:, 0, 0]) & np.isfinite(flat[:, 0, 1])
        out = np.full_like(flat, np.nan)
        if finite.any():
            out[finite] = cv2.undistortPoints(
                flat[finite], self.K, self.dist, P=self.K
            )
        return out.reshape(shape)

    def ray_through(self, point_px: Sequence[float]) -> np.ndarray:
        """Unit world-space direction of the ray through a pixel."""
        uv = np.array([point_px[0], point_px[1], 1.0], dtype=float)
        d = self.R.T @ (self.K_inv @ uv)
        n = np.linalg.norm(d)
        if n < 1e-12:
            raise ValueError("degenerate ray direction")
        return d / n


@dataclass(frozen=True)
class CameraPair:
    """The two cameras plus the frame offset calibration chose between them."""

    cam0: Camera
    cam1: Camera
    #: Frames to shift camera 1 by, relative to camera 0. Calibration stores
    #: this as ``best_offset`` in ``camera1_extrinsics.json``.
    frame_offset: float = 0.0

    def __iter__(self):
        return iter((self.cam0, self.cam1))

    def __getitem__(self, i: int) -> Camera:
        return (self.cam0, self.cam1)[i]

    def __len__(self) -> int:
        return 2

    @property
    def projections(self) -> np.ndarray:
        """Stacked ``(2, 3, 4)`` projection matrices."""
        return np.stack([self.cam0.P, self.cam1.P])

    @property
    def baseline_m(self) -> float:
        """Distance between the two camera centres, in metres."""
        return float(np.linalg.norm(self.cam0.center - self.cam1.center))


def _read_json(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"calibration file not found: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def extract_frame_offset(extrinsics: Dict[str, Any]) -> float:
    """Read the stored frame offset under any of its historical key names."""
    for key in _OFFSET_KEYS:
        if key in extrinsics:
            try:
                return float(extrinsics[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def load_camera(
    intrinsics_path: PathLike,
    extrinsics_path: PathLike,
    *,
    name: str = "camera",
    translation_units: str = "cm",
) -> Tuple[Camera, float]:
    """Load one camera from its two JSON files.

    Parameters
    ----------
    translation_units
        Unit of ``t`` on disk. ``"cm"`` matches what ``calibration.py`` writes
        (the checkerboard square size is specified in cm); ``"m"`` skips the
        conversion.

    Returns
    -------
    (camera, frame_offset)
        ``frame_offset`` is 0.0 for camera 0; camera 1 carries the value chosen
        by the calibration offset search.
    """
    intr = _read_json(intrinsics_path)
    extr = _read_json(extrinsics_path)

    for key, src in (("intrinsic", intrinsics_path), ("distortion", intrinsics_path)):
        if key not in intr:
            raise KeyError(f"{src}: missing required key {key!r}")
    for key in ("R", "t"):
        if key not in extr:
            raise KeyError(f"{extrinsics_path}: missing required key {key!r}")

    if translation_units == "cm":
        divisor = CM_PER_M
    elif translation_units == "m":
        divisor = 1.0
    else:
        raise ValueError(f"translation_units must be 'cm' or 'm', got {translation_units!r}")

    camera = Camera(
        K=np.array(intr["intrinsic"], dtype=float),
        dist=np.array(intr["distortion"], dtype=float),
        R=np.array(extr["R"], dtype=float),
        t=np.array(extr["t"], dtype=float).reshape(3, 1) / divisor,
        name=name,
    )
    return camera, extract_frame_offset(extr)


def load_camera_pair(
    camera_parameters_dir: PathLike, *, translation_units: str = "cm"
) -> CameraPair:
    """Load both cameras from a task's ``camera_parameters/`` directory."""
    d = Path(camera_parameters_dir)
    cam0, _ = load_camera(
        d / "camera0_intrinsics.json",
        d / "camera0_extrinsics.json",
        name="camera0",
        translation_units=translation_units,
    )
    cam1, offset = load_camera(
        d / "camera1_intrinsics.json",
        d / "camera1_extrinsics.json",
        name="camera1",
        translation_units=translation_units,
    )
    return CameraPair(cam0=cam0, cam1=cam1, frame_offset=offset)


def save_extrinsics(
    path: PathLike,
    R: np.ndarray,
    t: np.ndarray,
    *,
    best_offset: Optional[float] = None,
) -> None:
    """Write extrinsics in the layout the rest of the project expects."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "R": np.asarray(R, dtype=float).reshape(3, 3).tolist(),
        "t": np.asarray(t, dtype=float).reshape(3, 1).tolist(),
    }
    if best_offset is not None:
        payload["best_offset"] = float(best_offset)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def save_intrinsics(path: PathLike, K: np.ndarray, dist: np.ndarray) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "intrinsic": np.asarray(K, dtype=float).reshape(3, 3).tolist(),
        "distortion": np.asarray(dist, dtype=float).ravel().tolist(),
    }
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
