"""Camera IO, projection and triangulation.

The offset-key test pins down a bug that made the MMPose pipeline run unaligned
forever: calibration writes ``best_offset``, that pipeline read ``frame_offset``,
and ``dict.get(..., 0.0)`` turned the mismatch into a silent zero.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from pose3d.camera import (
    CM_PER_M,
    Camera,
    CameraPair,
    extract_frame_offset,
    load_camera,
    load_camera_pair,
    save_extrinsics,
    save_intrinsics,
)
from pose3d.geometry import (
    orthonormal_basis,
    rigid_align,
    triangulate_frames,
    triangulate_points,
)

K = np.array([[1400.0, 0.0, 540.0], [0.0, 1400.0, 960.0], [0.0, 0.0, 1.0]])


def make_pair(baseline_m: float = 1.0) -> CameraPair:
    cam0 = Camera(K=K, dist=np.zeros(5), R=np.eye(3), t=np.zeros((3, 1)), name="c0")
    angle = np.deg2rad(-20.0)
    R = np.array([[np.cos(angle), 0, np.sin(angle)],
                  [0, 1, 0],
                  [-np.sin(angle), 0, np.cos(angle)]])
    t = (-R @ np.array([[baseline_m], [0.0], [0.0]]))
    cam1 = Camera(K=K, dist=np.zeros(5), R=R, t=t, name="c1")
    return CameraPair(cam0, cam1)


# --------------------------------------------------------------------------- #
# Camera model
# --------------------------------------------------------------------------- #


def test_rejects_a_non_rotation_matrix():
    with pytest.raises(ValueError, match="not a rotation matrix"):
        Camera(K=K, dist=np.zeros(5), R=np.diag([1.0, 1.0, 2.0]), t=np.zeros((3, 1)))


def test_camera_centre_and_projection_agree():
    pair = make_pair(1.5)
    assert np.allclose(pair.cam0.center, [0, 0, 0], atol=1e-9)
    assert np.allclose(pair.cam1.center, [1.5, 0, 0], atol=1e-9)
    assert pair.baseline_m == pytest.approx(1.5)


def test_projection_puts_the_optical_axis_at_the_principal_point():
    cam = make_pair().cam0
    uv = cam.project(np.array([0.0, 0.0, 3.0]))
    assert uv == pytest.approx([540.0, 960.0])


def test_points_behind_the_camera_project_to_nan():
    cam = make_pair().cam0
    uv = cam.project(np.array([[0.0, 0.0, -2.0], [0.0, 0.0, 2.0]]))
    assert not np.isfinite(uv[0]).any()
    assert np.isfinite(uv[1]).all()


def test_ray_through_a_pixel_points_back_at_the_point():
    cam = make_pair().cam0
    point = np.array([0.4, -0.3, 2.5])
    uv = cam.project(point)
    ray = cam.ray_through(uv)
    assert np.allclose(ray, point / np.linalg.norm(point), atol=1e-9)


# --------------------------------------------------------------------------- #
# Calibration file IO
# --------------------------------------------------------------------------- #


def test_offset_is_read_under_every_historical_key():
    assert extract_frame_offset({"best_offset": -1}) == -1.0
    assert extract_frame_offset({"frame_offset": 2}) == 2.0
    assert extract_frame_offset({"offset": 3.5}) == 3.5
    assert extract_frame_offset({}) == 0.0
    # `best_offset` is what calibration writes, so it wins.
    assert extract_frame_offset({"best_offset": -1, "frame_offset": 9}) == -1.0


def test_translation_is_converted_from_centimetres(tmp_path):
    save_intrinsics(tmp_path / "camera0_intrinsics.json", K, np.zeros(5))
    save_extrinsics(tmp_path / "camera0_extrinsics.json", np.eye(3), np.zeros((3, 1)))
    save_intrinsics(tmp_path / "camera1_intrinsics.json", K, np.zeros(5))
    save_extrinsics(tmp_path / "camera1_extrinsics.json", np.eye(3),
                    np.array([[187.7], [0.0], [0.0]]), best_offset=-1)

    pair = load_camera_pair(tmp_path)
    assert pair.cam1.t.ravel()[0] == pytest.approx(1.877)
    assert pair.frame_offset == -1.0


def test_loading_the_real_task30_calibration():
    """The committed sample parameters must stay loadable."""
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    pair = load_camera_pair(root / "project" / "task30" / "camera_parameters")
    assert 1.5 < pair.baseline_m < 2.5, "baseline should be a plausible ~1.9 m"
    assert pair.frame_offset == -1.0


def test_missing_key_reports_which_file(tmp_path):
    (tmp_path / "bad.json").write_text(json.dumps({"intrinsic": K.tolist()}), encoding="utf-8")
    save_extrinsics(tmp_path / "ext.json", np.eye(3), np.zeros((3, 1)))
    with pytest.raises(KeyError, match="distortion"):
        load_camera(tmp_path / "bad.json", tmp_path / "ext.json")


# --------------------------------------------------------------------------- #
# Triangulation
# --------------------------------------------------------------------------- #


def test_triangulation_recovers_exact_points():
    pair = make_pair(1.2)
    points = np.array([[0.0, 0.0, 3.0], [0.5, -0.4, 2.5], [-0.7, 0.6, 4.0]])
    uv0 = pair.cam0.project(points)
    uv1 = pair.cam1.project(points)
    recovered = triangulate_points(pair.cam0.P, pair.cam1.P, uv0, uv1)
    assert np.allclose(recovered, points, atol=1e-6)


def test_triangulate_frames_masks_missing_and_low_confidence():
    pair = make_pair()
    points = np.array([[[0.0, 0.0, 3.0], [0.3, 0.1, 3.2]]])
    uv0 = pair.cam0.project(points)
    uv1 = pair.cam1.project(points)
    conf = np.array([[0.9, 0.05]])

    out = triangulate_frames(uv0, uv1, conf, conf, pair,
                             min_confidence=0.3, undistort=False)
    assert np.allclose(out[0, 0], points[0, 0], atol=1e-6)
    assert not np.isfinite(out[0, 1]).any(), "low-confidence joint must stay NaN"


def test_triangulation_is_accurate_under_pixel_noise():
    rng = np.random.default_rng(0)
    pair = make_pair(1.5)
    points = rng.uniform([-0.6, -0.8, 2.2], [0.6, 0.8, 3.4], size=(200, 3))
    uv0 = pair.cam0.project(points) + rng.normal(0, 1.0, (200, 2))
    uv1 = pair.cam1.project(points) + rng.normal(0, 1.0, (200, 2))
    recovered = triangulate_points(pair.cam0.P, pair.cam1.P, uv0, uv1)
    error = np.linalg.norm(recovered - points, axis=1)
    # 1 px at 3 m through a 1.5 m baseline is a few millimetres.
    assert np.median(error) < 0.01, f"median error {np.median(error) * 1000:.1f} mm"


def test_empty_input_is_handled():
    pair = make_pair()
    assert triangulate_points(pair.cam0.P, pair.cam1.P,
                              np.empty((0, 2)), np.empty((0, 2))).shape == (0, 3)


# --------------------------------------------------------------------------- #
# Frames and alignment
# --------------------------------------------------------------------------- #


def test_orthonormal_basis_is_orthonormal():
    R = orthonormal_basis(np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
    assert R is not None
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_orthonormal_basis_survives_a_parallel_hint():
    R = orthonormal_basis(np.array([1.0, 0.0, 0.0]), np.array([2.0, 0.0, 0.0]))
    assert R is not None and np.allclose(R.T @ R, np.eye(3), atol=1e-9)


def test_orthonormal_basis_rejects_a_zero_vector():
    assert orthonormal_basis(np.zeros(3), np.array([1.0, 0.0, 0.0])) is None


def test_rigid_align_undoes_a_similarity_transform():
    rng = np.random.default_rng(1)
    source = rng.normal(size=(20, 3))
    angle = 0.7
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    target = 2.5 * (source @ R.T) + np.array([1.0, -2.0, 0.5])

    aligned, scale, _, _ = rigid_align(source, target)
    assert scale == pytest.approx(2.5, rel=1e-6)
    assert np.allclose(aligned, target, atol=1e-8)


def test_rigid_align_ignores_nan_rows():
    rng = np.random.default_rng(2)
    source = rng.normal(size=(10, 3))
    target = source + 1.0
    source[3] = np.nan
    aligned, _, _, _ = rigid_align(source, target)
    assert not np.isfinite(aligned[3]).any()
    assert np.allclose(aligned[0], target[0], atol=1e-8)
