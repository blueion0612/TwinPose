"""Synthetic checkerboard video, for validating and benchmarking calibration.

``calibration.py`` has never had a way to check its own answer. It reports a
reprojection RMS, but a low RMS only says the optimiser found a self-consistent
solution -- it says nothing about whether the recovered focal length or baseline
are right, and a systematically wrong calibration can fit its own data perfectly.

Rendering a board through a *known* camera closes that loop: run calibration on
the result and compare the recovered intrinsics and extrinsics against the truth
that generated them.

The renderer is deliberately simple -- flat-shaded quads, a little blur, a little
sensor noise -- because ``findChessboardCornersSB`` keys off the saddle points
between squares, which survive that treatment intact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .camera import Camera, CameraPair


@dataclass
class BoardSpec:
    """A checkerboard, described the way ``calibration_settings.yaml`` describes it."""

    rows: int = 5                 # inner corners
    cols: int = 8                 # inner corners
    square_size_m: float = 0.027  # 2.7 cm, the A4-printed board

    @property
    def pattern(self) -> Tuple[int, int]:
        """``(cols, rows)``, the order OpenCV wants."""
        return (self.cols, self.rows)

    @property
    def n_corners(self) -> int:
        return self.rows * self.cols

    @property
    def centre_offset(self) -> np.ndarray:
        """Board centre in board coordinates, so poses can be centre-relative.

        ``object_points`` puts the origin on the first inner corner, which is
        the convention OpenCV uses. Placing a pose by that corner would swing
        the board off-frame as it rotates.
        """
        return np.array(
            [
                (self.cols - 1) * self.square_size_m / 2.0,
                (self.rows - 1) * self.square_size_m / 2.0,
                0.0,
            ]
        )

    @property
    def drawn_size_m(self) -> Tuple[float, float]:
        """Physical width and height of the printed pattern, in metres."""
        return (
            (self.cols + 1) * self.square_size_m,
            (self.rows + 1) * self.square_size_m,
        )

    def object_points(self) -> np.ndarray:
        """Inner-corner positions in board coordinates, ``(rows*cols, 3)`` metres.

        Ordering matches ``findChessboardCorners``: x varies fastest.
        """
        obj = np.zeros((self.n_corners, 3), np.float32)
        obj[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        return obj * self.square_size_m

    def square_corners(self) -> np.ndarray:
        """Outer corners of every square, ``(rows+1, cols+1, 3)`` metres.

        The drawn board is one square larger than the inner-corner grid in each
        direction, which is what makes the outermost inner corners detectable.
        """
        ys, xs = np.mgrid[0:self.rows + 2, 0:self.cols + 2]
        grid = np.stack([xs, ys, np.zeros_like(xs)], axis=-1).astype(np.float64)
        # Shift so the inner-corner grid starts at the origin, matching
        # `object_points`.
        grid[..., :2] -= 1.0
        return grid * self.square_size_m


@dataclass
class BoardPose:
    """Board-to-world rigid transform for one frame."""

    R: np.ndarray
    t: np.ndarray

    def transform(self, points_board: np.ndarray) -> np.ndarray:
        return points_board @ self.R.T + self.t

    @classmethod
    def centred_at(
        cls, R: np.ndarray, centre_world: np.ndarray, board: "BoardSpec"
    ) -> "BoardPose":
        """Pose that puts the board's *centre* at ``centre_world``."""
        return cls(R=R, t=np.asarray(centre_world, float) - R @ board.centre_offset)


def _rotation(rx: float, ry: float, rz: float) -> np.ndarray:
    """Intrinsic x-y-z rotation, built without pulling in scipy."""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def board_trajectory(
    n_frames: int,
    *,
    centre: np.ndarray,
    board: BoardSpec,
    view_direction: Optional[np.ndarray] = None,
    distance_range: Tuple[float, float] = (0.9, 1.9),
    seed: int = 0,
) -> List[BoardPose]:
    """A slow sweep of board poses covering a range of distances and angles.

    Calibration quality depends on *diversity* of views far more than on their
    number, so the path deliberately spans depth, in-plane rotation and tilt
    rather than jiggling in one place. Motion is smooth, matching the recording
    instruction to move the board slowly enough that each frame stays sharp.
    """
    rng = np.random.default_rng(seed)
    poses: List[BoardPose] = []
    near, far = distance_range

    forward = np.array([0.0, 0.0, 1.0]) if view_direction is None else np.asarray(
        view_direction, float
    )
    forward = forward / max(np.linalg.norm(forward), 1e-9)
    # Any two directions orthogonal to `forward`, for the lateral sweep.
    helper = np.array([0.0, 1.0, 0.0]) if abs(forward[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    right = np.cross(helper, forward)
    right /= max(np.linalg.norm(right), 1e-9)
    down = np.cross(forward, right)

    for f in range(n_frames):
        u = f / max(n_frames - 1, 1)
        # Three incommensurate rates so the pose never repeats exactly.
        depth = near + (far - near) * (0.5 - 0.5 * np.cos(2 * np.pi * u * 1.7))
        rx = 0.42 * np.sin(2 * np.pi * u * 2.3)
        ry = 0.52 * np.sin(2 * np.pi * u * 1.3 + 0.7)
        rz = 0.85 * np.sin(2 * np.pi * u * 0.9 + 1.9)
        lateral = 0.16 * np.sin(2 * np.pi * u * 1.1)
        vertical = 0.12 * np.sin(2 * np.pi * u * 1.9 + 0.4)

        jitter = rng.normal(0.0, 0.0015, size=3)   # hand tremor
        centre_world = (
            centre + forward * depth + right * lateral + down * vertical + jitter
        )
        poses.append(
            BoardPose.centred_at(_rotation(rx, ry, rz), centre_world, board)
        )
    return poses


def render_frame(
    camera: Camera,
    board: BoardSpec,
    pose: BoardPose,
    size: Tuple[int, int],
    *,
    rng: Optional[np.random.Generator] = None,
    blur: bool = True,
    noise_sigma: float = 2.0,
    background: int = 110,
) -> Optional[np.ndarray]:
    """Render one view of the board, or ``None`` if it is not usable.

    Distortion is applied, so a calibration run on these frames has to recover
    the distortion coefficients to fit them.
    """
    import cv2

    width, height = size
    rng = rng or np.random.default_rng(0)

    grid_board = board.square_corners()                    # (R+2, C+2, 3)
    grid_world = pose.transform(grid_board.reshape(-1, 3)).reshape(grid_board.shape)

    # Project with distortion, which Camera.project deliberately does not apply.
    cam_pts = grid_world.reshape(-1, 3) @ camera.R.T + camera.t.ravel()
    if np.any(cam_pts[:, 2] <= 1e-3):
        return None
    projected, _ = cv2.projectPoints(
        grid_world.reshape(-1, 3).astype(np.float64),
        cv2.Rodrigues(camera.R)[0], camera.t.astype(np.float64),
        camera.K, camera.dist,
    )
    pts = projected.reshape(grid_board.shape[0], grid_board.shape[1], 2)
    if not np.isfinite(pts).all():
        return None

    # The board must be comfortably inside the frame, or the outer corners get
    # clipped and the detector fails.
    margin = 12
    if (pts[..., 0].min() < margin or pts[..., 0].max() > width - margin
            or pts[..., 1].min() < margin or pts[..., 1].max() > height - margin):
        return None
    # Reject views so oblique or so distant that the squares collapse. An A4
    # board at 2 m is only ~170 px across, so this threshold is in absolute
    # pixels-per-square rather than a fraction of the frame.
    hull = cv2.convexHull(pts.reshape(-1, 2).astype(np.float32))
    area = float(cv2.contourArea(hull))
    squares = (board.rows + 1) * (board.cols + 1)
    if area / squares < 36.0:          # fewer than ~6x6 px per square
        return None

    img = np.full((height, width, 3), background, np.uint8)
    # White quiet zone: the detector needs contrast around the outermost squares.
    outer = np.array([pts[0, 0], pts[0, -1], pts[-1, -1], pts[-1, 0]], np.float32)
    cv2.fillConvexPoly(img, outer.astype(np.int32), (245, 245, 245), lineType=cv2.LINE_AA)

    for r in range(grid_board.shape[0] - 1):
        for c in range(grid_board.shape[1] - 1):
            if (r + c) % 2 == 0:
                continue
            quad = np.array(
                [pts[r, c], pts[r, c + 1], pts[r + 1, c + 1], pts[r + 1, c]], np.float32
            )
            cv2.fillConvexPoly(img, quad.astype(np.int32), (18, 18, 18), lineType=cv2.LINE_AA)

    if blur:
        img = cv2.GaussianBlur(img, (3, 3), 0.7)
    if noise_sigma > 0:
        noise = rng.normal(0.0, noise_sigma, img.shape)
        img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    return img


def render_mono_video(
    path,
    camera: Camera,
    board: BoardSpec,
    size: Tuple[int, int],
    n_frames: int = 240,
    *,
    fps: float = 30.0,
    seed: int = 0,
    distance_range: Tuple[float, float] = (0.7, 1.9),
) -> int:
    """Render an intrinsics-calibration clip. Returns frames written."""
    from .video import write_video

    rng = np.random.default_rng(seed)
    poses = mono_board_poses(
        camera, board, size, n_frames, distance_range=distance_range, seed=seed
    )

    def frames() -> Iterator[np.ndarray]:
        blank = np.full((size[1], size[0], 3), 110, np.uint8)
        for pose in poses:
            img = render_frame(camera, board, pose, size, rng=rng)
            yield blank if img is None else img

    return write_video(path, frames(), fps, size=size)


def mono_board_poses(
    camera: Camera,
    board: BoardSpec,
    size: Tuple[int, int],
    n_frames: int,
    *,
    distance_range: Tuple[float, float] = (0.7, 1.9),
    seed: int = 0,
    max_attempts: int = 60,
) -> List["BoardPose"]:
    """Diverse, fully-visible board poses for intrinsic calibration.

    Intrinsics -- and the principal point especially -- are only well
    conditioned when the board visits the *corners* of the frame at a range of
    tilts and distances. A board that stays near the image centre leaves the
    principal point almost unconstrained, which is exactly the failure the
    reprojection RMS cannot see. Poses are therefore sampled to tile the frame
    rather than to follow a smooth path.
    """
    rng = np.random.default_rng(seed)
    width, height = size
    near, far = distance_range
    poses: List[BoardPose] = []

    for f in range(n_frames):
        u = f / max(n_frames - 1, 1)
        for attempt in range(max_attempts):
            depth = float(rng.uniform(near, far))
            # Aim at a target pixel, then back-project: this tiles image space
            # uniformly instead of world space.
            target_px = (
                float(rng.uniform(0.15, 0.85)) * width,
                float(rng.uniform(0.12, 0.88)) * height,
            )
            direction = camera.ray_through(target_px)
            centre = camera.center + direction * depth
            tilt = _rotation(
                float(rng.normal(0.0, 0.38)),
                float(rng.normal(0.0, 0.38)),
                float(rng.uniform(-np.pi / 2, np.pi / 2)),
            )
            pose = BoardPose.centred_at(_look_at_rotation(-camera.R[2]) @ tilt, centre, board)
            if board_visible(camera, board, pose, size):
                poses.append(pose)
                break
        else:
            poses.append(poses[-1] if poses else BoardPose(np.eye(3), camera.center + camera.R[2] * 1.2))
    return poses


def board_visible(
    camera: Camera, board: BoardSpec, pose: BoardPose, size: Tuple[int, int],
    *, margin: int = 12, min_px_per_square: float = 36.0,
) -> bool:
    """Whether the whole board projects inside the frame at a usable scale.

    Cheap enough to use for rejection sampling: it projects corners only, with
    no rendering.
    """
    import cv2

    width, height = size
    grid = pose.transform(board.square_corners().reshape(-1, 3))
    cam_pts = grid @ camera.R.T + camera.t.ravel()
    if np.any(cam_pts[:, 2] <= 1e-3):
        return False
    projected, _ = cv2.projectPoints(
        grid.astype(np.float64), cv2.Rodrigues(camera.R)[0],
        camera.t.astype(np.float64), camera.K, camera.dist,
    )
    pts = projected.reshape(-1, 2)
    if not np.isfinite(pts).all():
        return False
    if (pts[:, 0].min() < margin or pts[:, 0].max() > width - margin
            or pts[:, 1].min() < margin or pts[:, 1].max() > height - margin):
        return False
    hull = cv2.convexHull(pts.astype(np.float32))
    area = float(cv2.contourArea(hull))
    return area / ((board.rows + 1) * (board.cols + 1)) >= min_px_per_square


def _look_at_rotation(normal: np.ndarray) -> np.ndarray:
    """Board rotation whose +Z (the board normal) points along ``normal``."""
    z = normal / max(np.linalg.norm(normal), 1e-9)
    helper = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= max(np.linalg.norm(x), 1e-9)
    y = np.cross(z, x)
    return np.column_stack([x, y, z])


def stereo_board_poses(
    cameras: CameraPair,
    board: BoardSpec,
    size: Tuple[int, int],
    n_frames: int,
    *,
    seed: int = 0,
    max_attempts: int = 60,
) -> List[BoardPose]:
    """Board poses visible to *both* cameras, spanning depth and tilt.

    Stereo calibration only learns from frames where both views see the board,
    so poses are rejection-sampled against both frusta. The board is also
    oriented to face the bisector of the two view directions rather than either
    camera: a board square-on to camera 0 is edge-on to camera 1 when the pair
    converges at 30 degrees, which is what starved the first version of this
    generator (it produced usable pairs in fewer than half its frames).
    """
    rng = np.random.default_rng(seed)
    mid_centre = 0.5 * (cameras.cam0.center + cameras.cam1.center)
    forward = cameras.cam0.R[2] / np.linalg.norm(cameras.cam0.R[2]) + \
        cameras.cam1.R[2] / np.linalg.norm(cameras.cam1.R[2])
    forward /= max(np.linalg.norm(forward), 1e-9)

    poses: List[BoardPose] = []
    for f in range(n_frames):
        u = f / max(n_frames - 1, 1)
        for attempt in range(max_attempts):
            # Sweep depth, then jitter more widely with each failed attempt.
            spread = 1.0 + attempt * 0.12
            depth = 1.5 + 1.1 * (0.5 - 0.5 * np.cos(2 * np.pi * u * 1.7))
            centre = (
                mid_centre + forward * depth
                + rng.normal(0.0, 0.10 * spread, size=3)
            )
            # Face the bisector, then tilt: the tilt is what makes the
            # distortion and pose observable.
            base = _look_at_rotation(-forward)
            tilt = _rotation(
                float(rng.normal(0.0, 0.32)),
                float(rng.normal(0.0, 0.32)),
                float(rng.uniform(-np.pi / 3, np.pi / 3)),
            )
            pose = BoardPose.centred_at(base @ tilt, centre, board)
            if board_visible(cameras.cam0, board, pose, size) and \
                    board_visible(cameras.cam1, board, pose, size):
                poses.append(pose)
                break
        else:
            # Fall back to the previous pose rather than dropping a frame, so
            # the two clips stay frame-aligned.
            poses.append(poses[-1] if poses else BoardPose(np.eye(3), mid_centre))
    return poses


def render_stereo_videos(
    path0,
    path1,
    cameras: CameraPair,
    board: BoardSpec,
    size: Tuple[int, int],
    n_frames: int = 240,
    *,
    fps: float = 30.0,
    seed: int = 0,
) -> Tuple[int, int]:
    """Render a synchronised stereo pair for extrinsic calibration."""
    from .video import write_video

    poses = stereo_board_poses(cameras, board, size, n_frames, seed=seed)
    rng0 = np.random.default_rng(seed)
    rng1 = np.random.default_rng(seed + 1)

    def frames(camera: Camera, rng: np.random.Generator) -> Iterator[np.ndarray]:
        blank = np.full((size[1], size[0], 3), 110, np.uint8)
        for pose in poses:
            img = render_frame(camera, board, pose, size, rng=rng)
            yield blank if img is None else img

    n0 = write_video(path0, frames(cameras.cam0, rng0), fps, size=size)
    n1 = write_video(path1, frames(cameras.cam1, rng1), fps, size=size)
    return n0, n1
