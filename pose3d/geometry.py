"""Multi-view geometry: triangulation, projection and orthonormal frames.

Everything here is vectorized over frames and joints. The old pipeline called
``cv2.triangulatePoints`` once per frame and then ran a separate
``scipy.optimize.least_squares`` per *point* (~50k solver invocations for a
30-second clip); the closed-form optimal correction used here is both faster and
better conditioned.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .camera import Camera, CameraPair


def project(camera: Camera, points_world: np.ndarray) -> np.ndarray:
    """Project ``(..., 3)`` world points through ``camera`` to ``(..., 2)`` px."""
    return camera.project(points_world)


def project_pair(cameras: CameraPair, points_world: np.ndarray) -> np.ndarray:
    """Project through both cameras, returning ``(2, ..., 2)``."""
    return np.stack([cameras.cam0.project(points_world),
                     cameras.cam1.project(points_world)])


def triangulate_points(
    P0: np.ndarray, P1: np.ndarray, uv0: np.ndarray, uv1: np.ndarray
) -> np.ndarray:
    """Linear (DLT) triangulation of corresponding undistorted pixels.

    Parameters
    ----------
    P0, P1
        ``(3, 4)`` projection matrices.
    uv0, uv1
        ``(N, 2)`` undistorted pixel coordinates.

    Returns
    -------
    ``(N, 3)`` world points; rows whose solution is degenerate come back NaN.
    """
    uv0 = np.asarray(uv0, dtype=float).reshape(-1, 2)
    uv1 = np.asarray(uv1, dtype=float).reshape(-1, 2)
    n = uv0.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=float)

    # Standard DLT: each view contributes two rows of A x = 0.
    A = np.empty((n, 4, 4), dtype=float)
    A[:, 0] = uv0[:, 0, None] * P0[2] - P0[0]
    A[:, 1] = uv0[:, 1, None] * P0[2] - P0[1]
    A[:, 2] = uv1[:, 0, None] * P1[2] - P1[0]
    A[:, 3] = uv1[:, 1, None] * P1[2] - P1[1]

    out = np.full((n, 3), np.nan, dtype=float)
    finite = np.isfinite(A).all(axis=(1, 2))
    if not finite.any():
        return out

    _, _, vh = np.linalg.svd(A[finite])
    X = vh[:, -1, :]
    w = X[:, 3]
    ok = np.abs(w) > 1e-12
    xyz = np.full((X.shape[0], 3), np.nan, dtype=float)
    xyz[ok] = X[ok, :3] / w[ok, None]

    out[finite] = xyz
    return out


def _optimal_correction(
    P0: np.ndarray, P1: np.ndarray, uv0: np.ndarray, uv1: np.ndarray, X: np.ndarray,
    iterations: int = 2,
) -> np.ndarray:
    """Gauss-Newton polish of DLT points against reprojection error.

    Two iterations is enough: DLT minimises an algebraic error that is already
    close to the geometric optimum for a well-conditioned stereo pair, and this
    removes the residual bias. Vectorized over all points at once.
    """
    X = X.copy()
    for _ in range(iterations):
        valid = np.isfinite(X).all(axis=1)
        if not valid.any():
            break
        Xv = X[valid]
        Xh = np.hstack([Xv, np.ones((Xv.shape[0], 1))])

        # Residual and Jacobian of (u,v) = (x/z, y/z) composed with P.
        JTJ = np.zeros((Xv.shape[0], 3, 3))
        JTr = np.zeros((Xv.shape[0], 3))
        for P, uv in ((P0, uv0[valid]), (P1, uv1[valid])):
            x = Xh @ P.T                     # (N, 3) homogeneous image points
            z = x[:, 2]
            good = np.abs(z) > 1e-9
            proj = np.full((Xv.shape[0], 2), np.nan)
            proj[good] = x[good, :2] / z[good, None]
            r = proj - uv                     # (N, 2)

            # d(proj)/dX = (P[:2,:3] - proj * P[2,:3]) / z
            dP = (P[None, :2, :3] - proj[:, :, None] * P[None, 2, :3]) / z[:, None, None]
            r_ok = np.isfinite(r).all(axis=1) & np.isfinite(dP).all(axis=(1, 2))
            dP = np.where(r_ok[:, None, None], dP, 0.0)
            r = np.where(r_ok[:, None], r, 0.0)

            JTJ += np.einsum("nki,nkj->nij", dP, dP)
            JTr += np.einsum("nki,nk->ni", dP, r)

        # Levenberg damping keeps the step sane when a point is near-degenerate.
        JTJ += 1e-9 * np.eye(3)[None]
        try:
            # NumPy 2 dropped the "b with one fewer dimension is a stack of
            # vectors" shorthand, so the column axis is explicit here.
            delta = np.linalg.solve(JTJ, -JTr[..., None])[..., 0]
        except np.linalg.LinAlgError:
            break
        step_ok = np.isfinite(delta).all(axis=1)
        Xv[step_ok] += delta[step_ok]
        X[valid] = Xv
    return X


def triangulate_frames(
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    conf0: np.ndarray,
    conf1: np.ndarray,
    cameras: CameraPair,
    *,
    min_confidence: float = 0.0,
    refine: bool = True,
    undistort: bool = True,
) -> np.ndarray:
    """Triangulate every frame/joint visible in both views.

    Parameters
    ----------
    kpts0, kpts1
        ``(F, J, 2)`` pixel observations, NaN where absent.
    conf0, conf1
        ``(F, J)`` confidences.
    min_confidence
        Observations below this are treated as missing.
    refine
        Run the Gauss-Newton polish after the DLT.
    undistort
        Undistort observations first. Pass ``False`` if they already are.

    Returns
    -------
    ``(F, J, 3)`` world points in meters, NaN where not reconstructible.
    """
    kpts0 = np.asarray(kpts0, dtype=float)
    kpts1 = np.asarray(kpts1, dtype=float)
    F, J = kpts0.shape[:2]
    out = np.full((F, J, 3), np.nan, dtype=float)

    visible = (
        np.isfinite(kpts0[..., 0]) & np.isfinite(kpts1[..., 0])
        & (np.nan_to_num(conf0, nan=-1.0) >= min_confidence)
        & (np.nan_to_num(conf1, nan=-1.0) >= min_confidence)
    )
    if not visible.any():
        return out

    f_idx, j_idx = np.nonzero(visible)
    uv0 = kpts0[f_idx, j_idx]
    uv1 = kpts1[f_idx, j_idx]

    if undistort:
        uv0 = cameras.cam0.undistort(uv0)
        uv1 = cameras.cam1.undistort(uv1)

    X = triangulate_points(cameras.cam0.P, cameras.cam1.P, uv0, uv1)
    if refine:
        X = _optimal_correction(cameras.cam0.P, cameras.cam1.P, uv0, uv1, X)

    out[f_idx, j_idx] = X
    return out


def reprojection_errors(
    points3d: np.ndarray,
    kpts0: np.ndarray,
    kpts1: np.ndarray,
    cameras: CameraPair,
) -> np.ndarray:
    """Per-camera L2 reprojection error, shape ``(2, F, J)``, NaN where absent.

    ``kpts0``/``kpts1`` must be in the same (undistorted) space as the forward
    model -- see :meth:`Camera.undistort`.
    """
    errs = []
    for cam, obs in ((cameras.cam0, kpts0), (cameras.cam1, kpts1)):
        proj = cam.project(points3d)
        errs.append(np.linalg.norm(proj - obs, axis=-1))
    return np.stack(errs)


def back_project_ray(
    camera: Camera, point_px: Sequence[float], depth_z: float
) -> np.ndarray:
    """World point on the ray through ``point_px`` at camera-frame depth ``z``.

    Used to place a joint that only one camera can see, given a depth guess
    borrowed from an anatomically adjacent joint.
    """
    uv = np.array([point_px[0], point_px[1], 1.0], dtype=float)
    ray_cam = camera.K_inv @ uv
    if abs(ray_cam[2]) < 1e-12:
        return np.full(3, np.nan)
    point_cam = ray_cam * (depth_z / ray_cam[2])
    return camera.R.T @ (point_cam - camera.t.ravel())


def orthonormal_basis(primary: np.ndarray, hint: np.ndarray) -> Optional[np.ndarray]:
    """Build a right-handed orthonormal basis whose first column follows ``primary``.

    Always a proper rotation (determinant +1). An earlier version took a
    ``right_handed`` flag and produced a *reflection* when it was false, which
    is not a rotation at all: ``scipy``'s ``Rotation.from_matrix`` rejected it,
    the caller's ``except ValueError`` swallowed the rejection, and every
    right-hand wrist angle in the pipeline came back empty. Left/right mirroring
    belongs in the reported angle signs, not in the basis.

    Returns ``None`` when the inputs are degenerate (zero-length or parallel),
    which callers treat as "no frame for this sample".
    """
    primary = np.asarray(primary, dtype=float)
    hint = np.asarray(hint, dtype=float)
    n = np.linalg.norm(primary)
    if not np.isfinite(n) or n < 1e-9:
        return None
    x = primary / n

    hn = np.linalg.norm(hint)
    if not np.isfinite(hn) or hn < 1e-9 or abs(float(np.dot(x, hint / hn))) > 0.999:
        # Pick any axis that is not nearly parallel to x.
        hint = np.array([1.0, 0.0, 0.0]) if abs(x[0]) < 0.9 else np.array([0.0, 1.0, 0.0])

    z = np.cross(x, hint)
    zn = np.linalg.norm(z)
    if zn < 1e-9:
        return None
    z /= zn
    y = np.cross(z, x)

    R = np.column_stack([x, y, z])
    if not np.isfinite(R).all() or abs(float(np.linalg.det(R)) - 1.0) > 1e-6:
        return None
    return R


def rigid_align(
    source: np.ndarray, target: np.ndarray, *, with_scale: bool = True
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Similarity Procrustes alignment of ``source`` onto ``target``.

    Both are ``(N, 3)``. Rows containing NaN in either input are ignored.

    Returns
    -------
    (aligned_source, scale, R, t)
        ``aligned_source`` has the same shape as ``source`` (NaN preserved).
    """
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    if source.shape != target.shape:
        raise ValueError(f"shape mismatch: {source.shape} vs {target.shape}")

    mask = np.isfinite(source).all(axis=-1) & np.isfinite(target).all(axis=-1)
    if mask.sum() < 3:
        return np.full_like(source, np.nan), float("nan"), np.eye(3), np.zeros(3)

    S, T = source[mask], target[mask]
    mu_s, mu_t = S.mean(axis=0), T.mean(axis=0)
    Sc, Tc = S - mu_s, T - mu_t

    H = Sc.T @ Tc
    U, sigma, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    if with_scale:
        var_s = float((Sc ** 2).sum())
        scale = float((sigma * np.array([1.0, 1.0, d])).sum() / var_s) if var_s > 1e-12 else 1.0
    else:
        scale = 1.0
    t = mu_t - scale * (R @ mu_s)

    aligned = np.full_like(source, np.nan)
    ok = np.isfinite(source).all(axis=-1)
    aligned[ok] = scale * (source[ok] @ R.T) + t
    return aligned, scale, R, t
