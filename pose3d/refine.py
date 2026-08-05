"""Windowed spatio-temporal bundle adjustment.

What this replaces
------------------
The previous pipeline ran six separate correction stages over the triangulated
points:

1. a global bundle adjustment whose result was never written back,
2. bone-length normalisation that lerped each child joint 20% toward its target,
3. a joint-angle nudge that moved hyperextended middle joints 10% toward the
   chord,
4. a torso-linearity nudge (20%) plus a MidHip re-projection (50%),
5. a second bundle adjustment, also never written back,
6. Savitzky-Golay smoothing followed by subtracting 25% of each point's
   acceleration.

Stages 2-4 and 6 are gradient steps on objectives that partly disagree with each
other, applied in a fixed order with hand-tuned step sizes and no convergence
check. Pulling a bone to its target length moves the child joint off its camera
ray; the next stage then moves it somewhere else again. Nothing measures whether
the result got better.

Here all of those objectives are residual blocks in one least-squares problem,
so the solver trades them off explicitly and the reported cost is monotone:

* **reprojection** -- the data term, one 2-vector per observed (frame, joint,
  camera), robustified with a Huber loss,
* **bone length** -- each bone toward its personalised length,
* **temporal acceleration** -- a second-difference prior, which does the job of
  the Savitzky-Golay pass and the acceleration damping without the phase
  distortion those introduce,
* **joint limits** -- a one-sided hinge that only activates on hyperextension,
  so normal poses are untouched.

Residuals are expressed in units of their own uncertainty (``residual / sigma``)
so a single Huber ``f_scale`` is meaningful across blocks and the weights have a
physical reading rather than being arbitrary gains.

The problem is solved in overlapping temporal windows and blended with a cosine
ramp. The temporal prior is local, so windowing barely changes the optimum while
keeping each solve small enough for an analytic sparse Jacobian.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from .camera import CameraPair
from .config import ReconstructionConfig
from .skeleton import Skeleton


@dataclass
class RefinementStats:
    """What the solver actually did, for the run report."""

    windows: int = 0
    parameters: int = 0
    residuals: int = 0
    cost_before: float = float("nan")
    cost_after: float = float("nan")
    reprojection_before_px: float = float("nan")
    reprojection_after_px: float = float("nan")
    limit_activations: int = 0
    passes: List[Dict[str, float]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "windows": self.windows,
            "parameters": self.parameters,
            "residuals": self.residuals,
            "cost_before": self.cost_before,
            "cost_after": self.cost_after,
            "reprojection_before_px": self.reprojection_before_px,
            "reprojection_after_px": self.reprojection_after_px,
            "limit_activations": self.limit_activations,
            "passes": self.passes,
        }


class _WindowProblem:
    """Residuals and analytic sparse Jacobian for one temporal window."""

    def __init__(
        self,
        points: np.ndarray,            # (W, J, 3) initial estimate
        obs: np.ndarray,               # (2, W, J, 2) undistorted px, NaN if absent
        weights: np.ndarray,           # (2, W, J) per-observation weight, 0 if absent
        cameras: CameraPair,
        skeleton: Skeleton,
        bone_lengths: Dict[str, float],
        cfg: ReconstructionConfig,
    ) -> None:
        self.W, self.J = points.shape[:2]
        self.cameras = cameras
        self.cfg = cfg

        active = np.isfinite(points).all(axis=-1)          # (W, J)
        self.active = active
        self.param_index = np.full((self.W, self.J), -1, dtype=np.int64)
        self.param_index[active] = np.arange(int(active.sum()))
        self.n_params = int(active.sum())
        self.x0 = points[active].ravel().copy()

        self._build_reprojection(obs, weights)
        self._build_bones(skeleton, bone_lengths)
        self._build_acceleration()
        self._build_limits(skeleton)

        self.n_residuals = (
            self._n_rep_rows + self.bone_p.size + self.acc_cur.size * 3 + self.lim_c.size
        )
        self._rows, self._cols = self._jacobian_structure()

    # -- block construction ------------------------------------------------ #
    def _build_reprojection(self, obs: np.ndarray, weights: np.ndarray) -> None:
        self.rep_param: List[np.ndarray] = []
        self.rep_obs: List[np.ndarray] = []
        self.rep_w: List[np.ndarray] = []
        offset = 0
        self.rep_offset: List[int] = []
        sigma = max(float(self.cfg.sigma_reproj_px), 1e-6)

        for c in range(2):
            ok = (
                self.active
                & np.isfinite(obs[c, ..., 0])
                & np.isfinite(obs[c, ..., 1])
                & (weights[c] > 0)
            )
            f_idx, j_idx = np.nonzero(ok)
            self.rep_param.append(self.param_index[f_idx, j_idx])
            self.rep_obs.append(obs[c][f_idx, j_idx])
            self.rep_w.append(weights[c][f_idx, j_idx] / sigma)
            self.rep_offset.append(offset)
            offset += f_idx.size * 2
        self._n_rep_rows = offset

    def _build_bones(self, skeleton: Skeleton, bone_lengths: Dict[str, float]) -> None:
        p_list, c_list, len_list = [], [], []
        for p, c, name in skeleton.bone_pairs:
            target = bone_lengths.get(name)
            if target is None or not np.isfinite(target) or target <= 1e-6:
                continue
            both = self.active[:, p] & self.active[:, c]
            frames = np.nonzero(both)[0]
            if frames.size == 0:
                continue
            p_list.append(self.param_index[frames, p])
            c_list.append(self.param_index[frames, c])
            len_list.append(np.full(frames.size, float(target)))
        self.bone_p = np.concatenate(p_list) if p_list else np.empty(0, dtype=np.int64)
        self.bone_c = np.concatenate(c_list) if c_list else np.empty(0, dtype=np.int64)
        self.bone_len = np.concatenate(len_list) if len_list else np.empty(0)
        self.bone_w = 1.0 / max(float(self.cfg.sigma_bone_m), 1e-9)

    def _build_acceleration(self) -> None:
        if self.W < 3:
            self.acc_prev = self.acc_cur = self.acc_next = np.empty(0, dtype=np.int64)
            self.acc_w = 0.0
            return
        prev = self.active[:-2] & self.active[1:-1] & self.active[2:]   # (W-2, J)
        f_idx, j_idx = np.nonzero(prev)
        self.acc_prev = self.param_index[f_idx, j_idx]
        self.acc_cur = self.param_index[f_idx + 1, j_idx]
        self.acc_next = self.param_index[f_idx + 2, j_idx]
        self.acc_w = 1.0 / max(float(self.cfg.sigma_accel_m), 1e-9)

    def _build_limits(self, skeleton: Skeleton) -> None:
        p_list, c_list, g_list = [], [], []
        for p, c, g in skeleton.chains:
            both = self.active[:, p] & self.active[:, c] & self.active[:, g]
            frames = np.nonzero(both)[0]
            if frames.size == 0:
                continue
            p_list.append(self.param_index[frames, p])
            c_list.append(self.param_index[frames, c])
            g_list.append(self.param_index[frames, g])
        self.lim_p = np.concatenate(p_list) if p_list else np.empty(0, dtype=np.int64)
        self.lim_c = np.concatenate(c_list) if c_list else np.empty(0, dtype=np.int64)
        self.lim_g = np.concatenate(g_list) if g_list else np.empty(0, dtype=np.int64)
        self.lim_w = 1.0 / max(float(self.cfg.sigma_limit), 1e-9)
        self.limit_hits = 0

    # -- residuals --------------------------------------------------------- #
    def residuals(self, x: np.ndarray) -> np.ndarray:
        X = x.reshape(-1, 3)
        out = np.empty(self.n_residuals, dtype=float)

        # Reprojection.
        for c in range(2):
            cam = self.cameras[c]
            idx, obs, w = self.rep_param[c], self.rep_obs[c], self.rep_w[c]
            base = self.rep_offset[c]
            if idx.size == 0:
                continue
            Xc = X[idx] @ cam.R.T + cam.t.ravel()
            # Points at or behind the pinhole would blow up; clamping keeps the
            # residual finite and large, which pushes them back in front.
            z = np.maximum(Xc[:, 2], 1e-3)
            uv = (Xc[:, :2] / z[:, None]) @ cam.K[:2, :2].T + cam.K[:2, 2]
            out[base:base + idx.size * 2] = ((uv - obs) * w[:, None]).ravel()

        # Bone length.
        base = self._n_rep_rows
        if self.bone_p.size:
            d = X[self.bone_c] - X[self.bone_p]
            length = np.linalg.norm(d, axis=1)
            out[base:base + self.bone_p.size] = (length - self.bone_len) * self.bone_w
        base += self.bone_p.size

        # Temporal acceleration.
        if self.acc_cur.size:
            acc = X[self.acc_prev] - 2.0 * X[self.acc_cur] + X[self.acc_next]
            out[base:base + acc.size] = (acc * self.acc_w).ravel()
        base += self.acc_cur.size * 3

        # Joint limits: a one-sided hinge against folding past what the joint
        # can do. Zero for every normal pose, including a fully straight limb.
        if self.lim_c.size:
            cos = self._chain_cosine(X)
            violation = np.maximum(0.0, cos - self.cfg.max_chain_cosine)
            self.limit_hits = int((violation > 0).sum())
            out[base:base + self.lim_c.size] = violation * self.lim_w
        return out

    def _chain_cosine(self, X: np.ndarray) -> np.ndarray:
        v1 = X[self.lim_p] - X[self.lim_c]
        v2 = X[self.lim_g] - X[self.lim_c]
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        denom = np.maximum(n1 * n2, 1e-12)
        return np.einsum("ij,ij->i", v1, v2) / denom

    # -- Jacobian ---------------------------------------------------------- #
    def _jacobian_structure(self) -> Tuple[np.ndarray, np.ndarray]:
        rows: List[np.ndarray] = []
        cols: List[np.ndarray] = []

        for c in range(2):
            idx = self.rep_param[c]
            if idx.size == 0:
                continue
            base = self.rep_offset[c]
            r = np.repeat(base + np.arange(idx.size * 2), 3)
            col = (idx[:, None] * 3 + np.arange(3)[None, :])       # (N, 3)
            cols.append(np.tile(col, (1, 2)).reshape(-1))          # u and v rows
            rows.append(r)

        base = self._n_rep_rows
        if self.bone_p.size:
            r = np.repeat(base + np.arange(self.bone_p.size), 6)
            col = np.concatenate(
                [
                    self.bone_p[:, None] * 3 + np.arange(3)[None, :],
                    self.bone_c[:, None] * 3 + np.arange(3)[None, :],
                ],
                axis=1,
            )
            rows.append(r)
            cols.append(col.reshape(-1))
        base += self.bone_p.size

        if self.acc_cur.size:
            n = self.acc_cur.size
            dims = np.arange(3)
            r = np.repeat(base + np.arange(n * 3), 3)
            col = np.stack(
                [
                    (self.acc_prev[:, None] * 3 + dims[None, :]),
                    (self.acc_cur[:, None] * 3 + dims[None, :]),
                    (self.acc_next[:, None] * 3 + dims[None, :]),
                ],
                axis=2,
            )                                                       # (n, 3dim, 3param)
            rows.append(r)
            cols.append(col.reshape(-1))
        base += self.acc_cur.size * 3

        if self.lim_c.size:
            r = np.repeat(base + np.arange(self.lim_c.size), 9)
            col = np.concatenate(
                [
                    self.lim_p[:, None] * 3 + np.arange(3)[None, :],
                    self.lim_c[:, None] * 3 + np.arange(3)[None, :],
                    self.lim_g[:, None] * 3 + np.arange(3)[None, :],
                ],
                axis=1,
            )
            rows.append(r)
            cols.append(col.reshape(-1))

        if not rows:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
        return np.concatenate(rows), np.concatenate(cols)

    def jacobian(self, x: np.ndarray) -> csr_matrix:
        X = x.reshape(-1, 3)
        data: List[np.ndarray] = []

        # d(reprojection)/dX = w * K2 @ d(xy/z)/dXc @ R
        for c in range(2):
            idx, w = self.rep_param[c], self.rep_w[c]
            if idx.size == 0:
                continue
            cam = self.cameras[c]
            Xc = X[idx] @ cam.R.T + cam.t.ravel()
            z = np.maximum(Xc[:, 2], 1e-3)
            inv_z = 1.0 / z
            # M = [[1/z, 0, -x/z^2], [0, 1/z, -y/z^2]]
            M = np.zeros((idx.size, 2, 3))
            M[:, 0, 0] = inv_z
            M[:, 1, 1] = inv_z
            M[:, 0, 2] = -Xc[:, 0] * inv_z * inv_z
            M[:, 1, 2] = -Xc[:, 1] * inv_z * inv_z
            KM = np.einsum("ab,nbc->nac", cam.K[:2, :2], M)         # (N, 2, 3)
            JR = np.einsum("nab,bc->nac", KM, cam.R)                # (N, 2, 3)
            data.append((JR * w[:, None, None]).reshape(-1))

        # d(bone)/dX = +/- unit vector
        if self.bone_p.size:
            d = X[self.bone_c] - X[self.bone_p]
            length = np.maximum(np.linalg.norm(d, axis=1), 1e-12)
            u = d / length[:, None]
            block = np.concatenate([-u, u], axis=1) * self.bone_w   # (N, 6)
            data.append(block.reshape(-1))

        # d(acceleration)/dX is constant: (+1, -2, +1) per dimension.
        if self.acc_cur.size:
            n = self.acc_cur.size
            block = np.tile(np.array([1.0, -2.0, 1.0]) * self.acc_w, (n * 3, 1))
            data.append(block.reshape(-1))

        # d(hinge)/dX, zero wherever the limit is satisfied.
        if self.lim_c.size:
            v1 = X[self.lim_p] - X[self.lim_c]
            v2 = X[self.lim_g] - X[self.lim_c]
            n1 = np.maximum(np.linalg.norm(v1, axis=1), 1e-12)
            n2 = np.maximum(np.linalg.norm(v2, axis=1), 1e-12)
            cos = np.einsum("ij,ij->i", v1, v2) / (n1 * n2)
            g1 = v2 / (n1 * n2)[:, None] - (cos / (n1 * n1))[:, None] * v1
            g2 = v1 / (n1 * n2)[:, None] - (cos / (n2 * n2))[:, None] * v2
            # residual = w * max(0, cos - cmax)  =>  d/dX = +w * dcos/dX
            # dcos/dXp = g1, dcos/dXg = g2, dcos/dXc = -(g1 + g2)
            gate = (cos > self.cfg.max_chain_cosine)[:, None].astype(float)
            block = np.concatenate([g1, -(g1 + g2), g2], axis=1) * self.lim_w * gate
            data.append(block.reshape(-1))

        values = np.concatenate(data) if data else np.empty(0)
        return csr_matrix(
            (values, (self._rows, self._cols)),
            shape=(self.n_residuals, self.n_params * 3),
        )

    # -- reporting --------------------------------------------------------- #
    def reprojection_rms_px(self, x: np.ndarray) -> float:
        X = x.reshape(-1, 3)
        errs: List[np.ndarray] = []
        for c in range(2):
            idx, obs = self.rep_param[c], self.rep_obs[c]
            if idx.size == 0:
                continue
            cam = self.cameras[c]
            Xc = X[idx] @ cam.R.T + cam.t.ravel()
            z = np.maximum(Xc[:, 2], 1e-3)
            uv = (Xc[:, :2] / z[:, None]) @ cam.K[:2, :2].T + cam.K[:2, 2]
            errs.append(np.linalg.norm(uv - obs, axis=1))
        if not errs:
            return float("nan")
        allv = np.concatenate(errs)
        return float(np.sqrt(np.mean(allv ** 2))) if allv.size else float("nan")

    def scatter(self, x: np.ndarray, out: np.ndarray) -> None:
        """Write optimised parameters back into a ``(W, J, 3)`` array."""
        out[self.active] = x.reshape(-1, 3)


def _window_bounds(n_frames: int, size: int, overlap: int) -> List[Tuple[int, int]]:
    """Window start/stop pairs covering ``[0, n_frames)`` with overlap."""
    if n_frames <= size:
        return [(0, n_frames)]
    stride = max(1, size - overlap)
    bounds: List[Tuple[int, int]] = []
    start = 0
    while start < n_frames:
        stop = min(start + size, n_frames)
        bounds.append((start, stop))
        if stop >= n_frames:
            break
        start += stride
    return bounds


def _blend_ramp(length: int) -> np.ndarray:
    """Cosine ramp from 0 to 1, used to cross-fade overlapping windows."""
    if length <= 1:
        return np.ones(max(length, 1))
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, length)))


def refine_sequence(
    points3d: np.ndarray,
    obs: np.ndarray,
    weights: np.ndarray,
    cameras: CameraPair,
    skeleton: Skeleton,
    bone_lengths: Dict[str, float],
    cfg: ReconstructionConfig,
    *,
    progress: Optional[object] = None,
) -> Tuple[np.ndarray, RefinementStats]:
    """Refine a whole sequence with overlapping windowed bundle adjustment.

    Parameters
    ----------
    points3d
        ``(F, J, 3)`` initial estimate. NaN entries are left untouched.
    obs
        ``(2, F, J, 2)`` undistorted pixel observations, NaN where absent.
    weights
        ``(2, F, J)`` per-observation weights (typically detector confidence);
        zero or NaN disables that observation.
    bone_lengths
        Target length in metres per bone name.

    Returns
    -------
    (refined_points, stats)
    """
    cfg.validate()
    points3d = np.asarray(points3d, dtype=float)
    F = points3d.shape[0]
    weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0)

    stats = RefinementStats()
    if F == 0 or not np.isfinite(points3d).any():
        return points3d.copy(), stats

    current = points3d.copy()
    bounds = _window_bounds(F, cfg.window_frames, cfg.window_overlap)
    stats.windows = len(bounds)

    for pass_i in range(max(1, cfg.refine_passes)):
        accum = np.zeros_like(current)
        weight_sum = np.zeros((F, 1, 1))
        touched = np.zeros(F, dtype=bool)

        pass_before: List[float] = []
        pass_after: List[float] = []
        limit_hits = 0
        n_params = n_res = 0

        iterator: Sequence[Tuple[int, int]] = bounds
        if progress is not None:
            iterator = progress(bounds, desc=f"  refine pass {pass_i + 1}/{cfg.refine_passes}")

        for (a, b) in iterator:
            problem = _WindowProblem(
                current[a:b],
                obs[:, a:b],
                weights[:, a:b],
                cameras,
                skeleton,
                bone_lengths,
                cfg,
            )
            if problem.n_params == 0 or problem.n_residuals == 0:
                continue

            n_params += problem.n_params
            n_res += problem.n_residuals
            pass_before.append(problem.reprojection_rms_px(problem.x0))

            result = least_squares(
                problem.residuals,
                problem.x0,
                jac=problem.jacobian,
                method="trf",
                loss="huber",
                f_scale=float(cfg.huber_f_scale),
                tr_solver="lsmr",
                max_nfev=int(cfg.max_nfev),
                ftol=1e-8,
                xtol=1e-8,
                gtol=1e-8,
                verbose=0,
            )
            pass_after.append(problem.reprojection_rms_px(result.x))
            limit_hits += problem.limit_hits

            window = current[a:b].copy()
            problem.scatter(result.x, window)

            # Cross-fade: ramp up over the leading overlap, down over the
            # trailing one, so neighbouring solutions blend instead of stepping.
            ramp = np.ones(b - a)
            lead = min(cfg.window_overlap, b - a)
            if a > 0 and lead > 1:
                ramp[:lead] = _blend_ramp(lead)
            if b < F and lead > 1:
                ramp[-lead:] = _blend_ramp(lead)[::-1]
            ramp = np.maximum(ramp, 1e-6)[:, None, None]

            accum[a:b] += np.nan_to_num(window, nan=0.0) * ramp
            weight_sum[a:b] += ramp
            touched[a:b] = True

        # Recombine. Frames no window could solve keep their previous value.
        blended = current.copy()
        safe = weight_sum[:, 0, 0] > 0
        blended[safe] = accum[safe] / weight_sum[safe]
        # Preserve the NaN mask: refinement never invents new joints.
        blended[~np.isfinite(current)] = np.nan
        current = blended

        stats.passes.append(
            {
                "pass": pass_i + 1,
                "reprojection_before_px": float(np.mean(pass_before)) if pass_before else float("nan"),
                "reprojection_after_px": float(np.mean(pass_after)) if pass_after else float("nan"),
                "limit_activations": limit_hits,
            }
        )
        if pass_i == 0:
            stats.reprojection_before_px = (
                float(np.mean(pass_before)) if pass_before else float("nan")
            )
            stats.parameters = n_params
            stats.residuals = n_res
        stats.reprojection_after_px = (
            float(np.mean(pass_after)) if pass_after else float("nan")
        )
        stats.limit_activations = limit_hits

    return current, stats
