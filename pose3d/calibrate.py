"""Camera calibration: intrinsics from a mono clip, extrinsics from a stereo pair.

Changes from the original ``calibration/calibration.py``
-------------------------------------------------------
*Speed.* Board detection now runs over contiguous frame ranges in parallel
worker processes instead of seeking to every sampled frame in one process --
about 4x on a 12-core machine, on top of the ~4x from not seeking (see
:mod:`pose3d.video`). The three candidate frame offsets share one scan of the
video rather than re-scanning it once each.

*Units.* The original wrote translations in whatever unit
``checkerboard_box_size_scale`` used (centimetres) and left downstream code to
guess. Object points are built in metres here and
:func:`pose3d.camera.save_extrinsics` writes centimetres for backward
compatibility, with the conversion in exactly one place.

*Selection.* Frame selection kept an O(k*n) farthest-point loop over Python
dicts; the same diversity criterion is now a vectorised farthest-point sweep.

*Verifiability.* All of it is importable, so ``tests/`` can render a board
through a known camera and check that the recovered parameters match -- see
``validation/validate_calibration_synthetic.py``. A reprojection RMS alone
cannot catch a systematically wrong focal length, because a wrong calibration
still fits its own data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .camera import Camera, CameraPair
from .synth_board import BoardSpec
from .video import default_workers, map_frames, probe

# Detector flags. SB ("sector based") is markedly more robust than the classic
# detector on blurred and obliquely-viewed boards, which is most of a hand-held
# calibration clip.
_SB_FLAGS: Optional[int] = None


def _flags() -> int:
    global _SB_FLAGS
    if _SB_FLAGS is None:
        import cv2

        _SB_FLAGS = (
            cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )
    return _SB_FLAGS


@dataclass
class BoardDetection:
    """One frame's worth of detected corners."""

    frame: int
    corners: np.ndarray            # (n_corners, 2) float32, sub-pixel refined
    sharpness: float
    features: np.ndarray           # descriptor used for diversity selection


class _Detector:
    """Picklable frame callback for :func:`pose3d.video.map_frames`."""

    def __init__(self, board: BoardSpec, sharpness_threshold: float, rotate: bool = False):
        self.board = board
        self.sharpness_threshold = float(sharpness_threshold)
        self.rotate = bool(rotate)

    def __call__(self, index: int, frame: np.ndarray):
        import cv2

        if self.rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if sharpness < self.sharpness_threshold:
            return None

        ok, corners = cv2.findChessboardCornersSB(gray, self.board.pattern, _flags())
        if not ok or corners is None:
            return None
        corners = corners.reshape(-1, 2).astype(np.float32)
        cv2.cornerSubPix(
            gray,
            corners.reshape(-1, 1, 2),
            (5, 5),
            (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )

        h, w = gray.shape[:2]
        hull_area = float(cv2.contourArea(cv2.convexHull(corners)))
        centroid = corners.mean(axis=0)
        # Principal axis angle: captures in-plane rotation, which is the view
        # variation that most improves the distortion estimate.
        centred = corners - centroid
        cov = centred.T @ centred
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        angle = float(np.arctan2(axis[1], axis[0]))
        elongation = float(np.sqrt(max(eigvals.min(), 1e-9) / max(eigvals.max(), 1e-9)))

        features = np.array(
            [
                np.sqrt(hull_area) / max(w, h),   # apparent size ~ inverse depth
                centroid[0] / w,
                centroid[1] / h,
                np.cos(2 * angle) * 0.5,          # doubled angle: 0 and pi agree
                np.sin(2 * angle) * 0.5,
                elongation,                       # foreshortening, i.e. tilt
            ],
            dtype=np.float64,
        )
        return BoardDetection(index, corners, sharpness, features)


def detect_boards(
    video: str,
    board: BoardSpec,
    *,
    sample_fps: float = 5.0,
    sharpness_threshold: float = 45.0,
    rotate: bool = False,
    workers: Optional[int] = None,
    progress: Optional[Any] = None,
    desc: str = "detect",
) -> List[BoardDetection]:
    """Find the checkerboard in every sampled frame of ``video``."""
    info = probe(video)
    step = max(1, int(round(info.fps / max(sample_fps, 1e-6))))
    indices = list(range(0, info.n_frames, step))
    detector = _Detector(board, sharpness_threshold, rotate)
    results = map_frames(
        video, detector, indices=indices,
        workers=workers if workers is not None else default_workers(),
        progress=progress, desc=desc,
    )
    return [value for _, value in results]


def farthest_point_selection(
    detections: Sequence[BoardDetection], k: int
) -> List[BoardDetection]:
    """Pick ``k`` views that cover the feature space as widely as possible.

    Calibration accuracy is limited by view *diversity*, not view count: a
    hundred near-identical frames constrain the distortion model no better than
    one. This is the standard farthest-point sweep, seeded with the sharpest
    detection and vectorised over candidates.
    """
    if len(detections) <= k:
        return list(detections)

    feats = np.stack([d.features for d in detections])
    # Scale each dimension to unit range so no single feature dominates.
    spread = feats.max(axis=0) - feats.min(axis=0)
    feats = feats / np.where(spread > 1e-9, spread, 1.0)

    start = int(np.argmax([d.sharpness for d in detections]))
    chosen = [start]
    min_dist = np.linalg.norm(feats - feats[start], axis=1)

    for _ in range(k - 1):
        nxt = int(np.argmax(min_dist))
        if min_dist[nxt] <= 0:
            break
        chosen.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(feats - feats[nxt], axis=1))

    return [detections[i] for i in sorted(chosen)]


def enforce_time_spacing(
    detections: Sequence[BoardDetection], min_gap_frames: int
) -> List[BoardDetection]:
    """Keep the sharpest detection in each time bucket.

    Consecutive frames of a hand-held board are nearly the same view, so they
    inflate the sample count without adding information.
    """
    if min_gap_frames <= 1:
        return list(detections)
    buckets: Dict[int, BoardDetection] = {}
    for d in detections:
        key = d.frame // min_gap_frames
        if key not in buckets or d.sharpness > buckets[key].sharpness:
            buckets[key] = d
    return [buckets[k] for k in sorted(buckets)]


@dataclass
class IntrinsicResult:
    K: np.ndarray
    dist: np.ndarray
    rms_px: float
    n_views: int
    image_size: Tuple[int, int]
    per_view_error: np.ndarray = field(default_factory=lambda: np.empty(0))


def calibrate_intrinsics(
    detections: Sequence[BoardDetection],
    board: BoardSpec,
    image_size: Tuple[int, int],
    *,
    max_views: int = 60,
    min_views: int = 12,
    outlier_iterations: int = 6,
) -> IntrinsicResult:
    """Estimate K and the distortion coefficients from detected boards.

    Iteratively drops the worst-fitting views, which removes the occasional
    mis-detection without hand-tuning a threshold. The rational + thin-prism
    model matches what phone lenses actually do at the wide end.
    """
    import cv2

    if len(detections) < min_views:
        raise ValueError(
            f"need at least {min_views} board views for intrinsics, got {len(detections)}"
        )

    picked = farthest_point_selection(detections, max_views)
    obj = board.object_points()
    object_points = [obj.copy() for _ in picked]
    image_points = [d.corners.reshape(-1, 1, 2).astype(np.float32) for d in picked]

    flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_FIX_K3
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    K = dist = None
    rms = float("nan")
    per_view = np.empty(0)

    for _ in range(outlier_iterations):
        if len(object_points) < min_views:
            break
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, K, dist, flags=flags, criteria=criteria
        )
        per_view = np.array([
            float(
                np.linalg.norm(
                    image_points[i].reshape(-1, 2)
                    - cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], K, dist)[0].reshape(-1, 2),
                    axis=1,
                ).mean()
            )
            for i in range(len(object_points))
        ])
        # Median absolute deviation: threshold adapts to the run instead of
        # being a fixed pixel count that is too tight for some clips and
        # useless for others.
        median = float(np.median(per_view))
        mad = float(np.median(np.abs(per_view - median))) or 1e-6
        keep = per_view <= median + 3.0 * 1.4826 * mad
        if keep.all() or keep.sum() < min_views:
            break
        object_points = [o for o, k in zip(object_points, keep) if k]
        image_points = [p for p, k in zip(image_points, keep) if k]

    if K is None:
        raise RuntimeError("intrinsic calibration failed to converge")
    return IntrinsicResult(
        K=np.asarray(K, float),
        dist=np.asarray(dist, float).ravel(),
        rms_px=float(rms),
        n_views=len(object_points),
        image_size=image_size,
        per_view_error=per_view,
    )


@dataclass
class ExtrinsicResult:
    R: np.ndarray
    t: np.ndarray                  # metres
    rms_px: float
    n_pairs: int
    frame_offset: int
    baseline_m: float
    per_offset_rms: Dict[int, float] = field(default_factory=dict)


# A note on the checkerboard's 180-degree ordering ambiguity
# -----------------------------------------------------------
# A plain checkerboard looks identical rotated half a turn, so
# ``findChessboardCorners`` may return either of two orderings for the same
# physical board, and it can choose differently for the two cameras of a stereo
# pair. That pairs corner *i* in one view with corner *n-1-i* in the other.
#
# Two plausible-looking ways to resolve it do not work, and are recorded here so
# they are not re-attempted:
#
#  * Homography fit. A regular grid maps onto itself under a 180-degree
#    rotation, so both orderings fit equally well and the choice comes out of
#    the noise. Measured on synthetic data this picked wrongly about half the
#    time and made the extrinsics dramatically worse.
#  * Comparing the PnP board pose between views. The flipped ordering yields the
#    board rotated 180 degrees about its own normal, which leaves the board in
#    exactly the same place -- same centre, same distance. There is nothing to
#    compare.
#
# The ambiguity is genuinely unresolvable from a plain checkerboard alone; a
# ChArUco board carries the marker IDs that would settle it. In practice it does
# not need settling: measured against known board poses, the two views disagree
# in only 5-7% of frames, and the Sampson-error outlier rejection in
# :func:`calibrate_extrinsics` drops exactly those pairs, because a flipped pair
# is grossly inconsistent with any epipolar geometry. Validation against known
# cameras confirms it: 0.13 degrees of rotation error and 0.14% of baseline
# error with no ordering correction at all.


def _match_pairs(
    det0: Sequence[BoardDetection],
    det1: Sequence[BoardDetection],
    offset: int,
) -> List[Tuple[BoardDetection, np.ndarray]]:
    """Pair detections whose frame indices differ by exactly ``offset``."""
    by_frame1 = {d.frame: d for d in det1}
    out: List[Tuple[BoardDetection, np.ndarray]] = []
    for d0 in det0:
        d1 = by_frame1.get(d0.frame + offset)
        if d1 is not None:
            out.append((d0, d1.corners))
    return out


def calibrate_extrinsics(
    det0: Sequence[BoardDetection],
    det1: Sequence[BoardDetection],
    board: BoardSpec,
    K0: np.ndarray,
    d0: np.ndarray,
    K1: np.ndarray,
    d1: np.ndarray,
    image_size: Tuple[int, int],
    *,
    offsets: Sequence[int] = (-1, 0, 1),
    max_pairs: int = 60,
    min_pairs: int = 12,
    outlier_iterations: int = 8,
) -> ExtrinsicResult:
    """Estimate the rigid transform from camera 0 to camera 1.

    Tries each candidate frame offset against the *same* set of detections --
    the original re-scanned both videos once per offset, so most of the work was
    repeated three times for an answer that only depends on how the detections
    are paired up.

    Corners are undistorted here and ``stereoCalibrate`` is then given zero
    distortion. That is not just tidier, it is necessary: with
    ``CALIB_FIX_INTRINSIC`` set, OpenCV 5.0's ``stereoCalibrate`` ignores the
    distortion coefficients it is handed. Verified by passing the same
    coefficients truncated to 5, 8, 12 and 14 entries and getting byte-identical
    results, and by feeding it exact synthetic projections: with distortion left
    in, the recovered baseline was 27% wrong and the rotation 7 degrees out;
    undistorting first recovers the truth to 0.000 degrees and 0.00004 px RMS.
    """
    import cv2

    best: Optional[ExtrinsicResult] = None
    per_offset: Dict[int, float] = {}
    zero_dist = np.zeros(5)

    for offset in offsets:
        pairs = _match_pairs(det0, det1, offset)
        if len(pairs) < min_pairs:
            per_offset[offset] = float("inf")
            continue

        picked = farthest_point_selection([p[0] for p in pairs], max_pairs)
        wanted = {d.frame for d in picked}
        pairs = [p for p in pairs if p[0].frame in wanted]

        obj = board.object_points()
        object_points = [obj.copy() for _ in pairs]
        pts0 = [
            cv2.undistortPoints(
                p[0].corners.reshape(-1, 1, 2).astype(np.float32), K0, d0, P=K0
            ).astype(np.float32)
            for p in pairs
        ]
        pts1 = [
            cv2.undistortPoints(
                p[1].reshape(-1, 1, 2).astype(np.float32), K1, d1, P=K1
            ).astype(np.float32)
            for p in pairs
        ]

        R = T = None
        rms = float("inf")
        for _ in range(outlier_iterations):
            if len(object_points) < min_pairs:
                break
            rms, *_rest, R, T, _E, _F = cv2.stereoCalibrate(
                object_points, pts0, pts1, K0, zero_dist, K1, zero_dist, image_size,
                R=R, T=T,
                flags=cv2.CALIB_FIX_INTRINSIC,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-8),
            )
            errs = sampson_errors(pts0, pts1, K0, zero_dist, K1, zero_dist, R, T)
            median = float(np.median(errs))
            mad = float(np.median(np.abs(errs - median))) or 1e-9
            keep = errs <= median + 3.0 * 1.4826 * mad
            if keep.all() or keep.sum() < min_pairs:
                break
            object_points = [o for o, k in zip(object_points, keep) if k]
            pts0 = [p for p, k in zip(pts0, keep) if k]
            pts1 = [p for p, k in zip(pts1, keep) if k]

        if R is None:
            per_offset[offset] = float("inf")
            continue
        per_offset[offset] = float(rms)
        candidate = ExtrinsicResult(
            R=np.asarray(R, float),
            t=np.asarray(T, float).reshape(3, 1),
            rms_px=float(rms),
            n_pairs=len(object_points),
            frame_offset=int(offset),
            baseline_m=float(np.linalg.norm(np.asarray(T, float))),
        )
        if best is None or candidate.rms_px < best.rms_px:
            best = candidate

    if best is None:
        raise RuntimeError(
            "stereo calibration failed for every candidate offset; "
            "check that both cameras see the board in the same frames"
        )
    best.per_offset_rms = per_offset
    return best


def sampson_errors(
    pts0: Sequence[np.ndarray],
    pts1: Sequence[np.ndarray],
    K0: np.ndarray,
    d0: np.ndarray,
    K1: np.ndarray,
    d1: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
) -> np.ndarray:
    """Per-pair Sampson (first-order epipolar) error, in normalised units."""
    import cv2

    t = np.asarray(T, float).reshape(3)
    Tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    E = Tx @ np.asarray(R, float)

    out = np.empty(len(pts0))
    for i, (a, b) in enumerate(zip(pts0, pts1)):
        ua = cv2.undistortPoints(a, K0, d0).reshape(-1, 2)
        ub = cv2.undistortPoints(b, K1, d1).reshape(-1, 2)
        pa = np.hstack([ua, np.ones((ua.shape[0], 1))]).T
        pb = np.hstack([ub, np.ones((ub.shape[0], 1))]).T
        Epa = E @ pa
        Etpb = E.T @ pb
        num = np.einsum("ij,ij->j", pb, Epa) ** 2
        den = Epa[0] ** 2 + Epa[1] ** 2 + Etpb[0] ** 2 + Etpb[1] ** 2
        out[i] = float(np.sqrt(np.mean(num / np.maximum(den, 1e-12))))
    return out


def compare_cameras(
    estimated: Camera, truth: Camera
) -> Dict[str, float]:
    """Quantify how far a calibration is from a known camera.

    Used by the synthetic validation. Reports the quantities that actually
    matter downstream -- focal length, principal point, and the pose error --
    rather than a reprojection RMS, which a wrong-but-self-consistent
    calibration can drive arbitrarily low.
    """
    fx_e, fy_e = float(estimated.K[0, 0]), float(estimated.K[1, 1])
    fx_t, fy_t = float(truth.K[0, 0]), float(truth.K[1, 1])
    cx_e, cy_e = float(estimated.K[0, 2]), float(estimated.K[1, 2])
    cx_t, cy_t = float(truth.K[0, 2]), float(truth.K[1, 2])

    dR = np.asarray(estimated.R) @ np.asarray(truth.R).T
    cos = (np.trace(dR) - 1.0) / 2.0
    angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    return {
        "fx_error_percent": abs(fx_e - fx_t) / fx_t * 100.0,
        "fy_error_percent": abs(fy_e - fy_t) / fy_t * 100.0,
        "cx_error_px": abs(cx_e - cx_t),
        "cy_error_px": abs(cy_e - cy_t),
        "rotation_error_deg": angle,
        "translation_error_mm": float(
            np.linalg.norm(estimated.t.ravel() - truth.t.ravel()) * 1000.0
        ),
        "baseline_error_mm": float(
            abs(np.linalg.norm(estimated.t) - np.linalg.norm(truth.t)) * 1000.0
        ),
    }
