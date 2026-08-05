"""Video IO, flash synchronisation, and calibration.

The calibration tests render a checkerboard through a camera whose parameters
are known and check that calibration recovers them. That is the check a
reprojection RMS cannot make: a calibration with a systematically wrong focal
length still fits its own detections perfectly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from pose3d.calibrate import (
    calibrate_extrinsics,
    calibrate_intrinsics,
    compare_cameras,
    detect_boards,
    enforce_time_spacing,
    farthest_point_selection,
)
from pose3d.camera import Camera, load_camera_pair
from pose3d.sync import (
    detect_flashes,
    match_flash_sequences,
)
from pose3d.synth_board import BoardSpec, render_mono_video, render_stereo_videos
from pose3d.video import (
    _chunk_ranges,
    default_workers,
    map_frames,
    probe,
    read_frames,
    write_video,
)

ROOT = Path(__file__).resolve().parent.parent
CAMERA_DIR = ROOT / "project" / "task30" / "camera_parameters"

# Small frames keep these tests quick; the geometry under test is unaffected.
TEST_SIZE = (540, 960)


# --------------------------------------------------------------------------- #
# Video IO
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def counting_video(tmp_path_factory):
    """A short video whose frames encode their own index in the top-left pixel."""
    path = tmp_path_factory.mktemp("video") / "counter.mp4"
    frames = []
    for i in range(90):
        frame = np.full((120, 160, 3), 30, np.uint8)
        cv2.putText(frame, str(i), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        frames.append(frame)
    write_video(path, frames, 30.0)
    return path


def test_probe_reads_metadata(counting_video):
    info = probe(counting_video)
    assert info.n_frames == 90
    assert info.fps == pytest.approx(30.0, abs=0.1)
    assert (info.width, info.height) == (160, 120)


def test_probe_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        probe(tmp_path / "nope.mp4")


def test_read_frames_returns_the_requested_indices(counting_video):
    wanted = [0, 5, 6, 40, 89]
    got = [idx for idx, _ in read_frames(counting_video, indices=wanted)]
    assert got == wanted


def test_read_frames_with_step(counting_video):
    got = [idx for idx, _ in read_frames(counting_video, start=10, stop=30, step=5)]
    assert got == [10, 15, 20, 25]


def test_read_frames_rejects_unsorted_indices(counting_video):
    with pytest.raises(ValueError, match="sorted"):
        list(read_frames(counting_video, indices=[5, 1]))


def test_chunk_ranges_are_contiguous_and_complete():
    indices = list(range(100))
    chunks = _chunk_ranges(indices, 7)
    assert sum(len(c) for c in chunks) == 100
    assert [i for c in chunks for i in c] == indices
    for chunk in chunks:
        assert chunk == list(range(chunk[0], chunk[-1] + 1))


def _mean_brightness(index, frame):
    return float(frame.mean())


def test_map_frames_agrees_serial_and_parallel(counting_video):
    indices = list(range(90))
    serial = map_frames(counting_video, _mean_brightness, indices=indices, workers=1)
    parallel = map_frames(counting_video, _mean_brightness, indices=indices, workers=4)
    assert [i for i, _ in serial] == [i for i, _ in parallel]
    assert np.allclose([v for _, v in serial], [v for _, v in parallel])


def test_default_workers_is_sane():
    n = default_workers()
    assert 1 <= n <= 12


# --------------------------------------------------------------------------- #
# Flash synchronisation
# --------------------------------------------------------------------------- #


def make_flash_series(frames: int, flash_frames, amplitude=8000, noise=40, seed=0):
    rng = np.random.default_rng(seed)
    series = rng.integers(0, noise, frames).astype(np.int64)
    for f in flash_frames:
        if 0 <= f < frames:
            series[f: f + 3] += amplitude
    return series


def test_flash_detection_finds_the_planted_flashes():
    flashes = [40, 120, 260, 400]
    series = make_flash_series(500, flashes)
    events = detect_flashes(series, 30.0)
    found = [e.frame for e in events]
    assert len(found) == len(flashes)
    for planted, detected in zip(flashes, found):
        assert abs(detected - planted) <= 1


def test_flash_detection_is_robust_to_a_bright_outlier():
    """A single huge spike must not raise the threshold past the real flashes."""
    series = make_flash_series(500, [40, 120, 260, 400])
    series[300] += 200000                      # a car headlight
    events = detect_flashes(series, 30.0)
    assert len(events) >= 4, "robust thresholding should survive one big outlier"


def test_flash_matching_recovers_a_known_offset():
    offset = 37
    a = make_flash_series(600, [50, 150, 300, 450], seed=1)
    b = make_flash_series(600, [50 + offset, 150 + offset, 300 + offset, 450 + offset], seed=2)
    result = match_flash_sequences(detect_flashes(a, 30.0), detect_flashes(b, 30.0), 30.0)
    assert abs(result.offset_frames - offset) <= 1
    assert result.matched >= 4
    assert result.confidence > 0.9


def test_flash_matching_survives_a_spurious_event():
    """One false positive used to truncate the whole match."""
    offset = 25
    a = make_flash_series(600, [50, 150, 300, 450], seed=3)
    b = make_flash_series(600, [50 + offset, 150 + offset, 210, 300 + offset, 450 + offset],
                          seed=4)
    result = match_flash_sequences(detect_flashes(a, 30.0), detect_flashes(b, 30.0), 30.0)
    assert abs(result.offset_frames - offset) <= 1
    assert result.matched >= 4


def test_flash_matching_reports_failure_rather_than_guessing():
    a = make_flash_series(300, [50], seed=5)
    result = match_flash_sequences(detect_flashes(a, 30.0), [], 30.0)
    assert result.matched == 0 and result.offset_frames == 0


# --------------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def truth_cameras():
    """The real task30 calibration, rescaled to the reduced test resolution.

    Rendering and evaluation must use the *same* camera model. Distortion
    coefficients live in normalised coordinates and so are resolution
    independent; only K scales.
    """
    from pose3d.camera import CameraPair

    full = load_camera_pair(CAMERA_DIR)
    scale = TEST_SIZE[0] / 1080.0

    def shrink(cam: Camera) -> Camera:
        K = cam.K.copy()
        K[:2] *= scale
        return Camera(K=K, dist=cam.dist, R=cam.R, t=cam.t, name=cam.name)

    return CameraPair(shrink(full.cam0), shrink(full.cam1), full.frame_offset)


@pytest.fixture(scope="module")
def small_board():
    # Larger squares so the board is well resolved at the reduced test size.
    return BoardSpec(rows=5, cols=8, square_size_m=0.060)


@pytest.fixture(scope="module")
def mono_clip(tmp_path_factory, truth_cameras, small_board):
    path = tmp_path_factory.mktemp("calib") / "mono0.mp4"
    render_mono_video(path, truth_cameras.cam0, small_board, TEST_SIZE,
                      n_frames=90, seed=11, distance_range=(1.1, 2.4))
    return path


def test_board_detection_finds_most_frames(mono_clip, small_board):
    detections = detect_boards(str(mono_clip), small_board, sample_fps=30.0,
                               sharpness_threshold=0.0, workers=1)
    assert len(detections) > 70, f"only {len(detections)}/90 detected"
    for d in detections[:5]:
        assert d.corners.shape == (small_board.n_corners, 2)


def test_farthest_point_selection_spreads_over_the_feature_space():
    class Fake:
        def __init__(self, feat, sharp):
            self.features = np.array(feat, float)
            self.sharpness = sharp

    candidates = [Fake([x, 0, 0, 0, 0, 0], 1.0) for x in np.linspace(0, 1, 40)]
    picked = farthest_point_selection(candidates, 5)
    xs = sorted(c.features[0] for c in picked)
    assert xs[0] < 0.05 and xs[-1] > 0.95, "the extremes must be represented"
    assert len(picked) == 5


def test_time_spacing_keeps_the_sharpest_per_bucket():
    class Fake:
        def __init__(self, frame, sharp):
            self.frame, self.sharpness = frame, sharp

    kept = enforce_time_spacing([Fake(0, 1.0), Fake(1, 5.0), Fake(20, 2.0)], 10)
    assert sorted(d.frame for d in kept) == [1, 20]


def test_intrinsics_recover_the_true_focal_length(mono_clip, small_board, truth_cameras):
    """The check a reprojection RMS cannot make."""
    detections = detect_boards(str(mono_clip), small_board, sample_fps=30.0,
                               sharpness_threshold=0.0, workers=1)
    result = calibrate_intrinsics(detections, small_board, TEST_SIZE, max_views=60)
    expected = truth_cameras.cam0.K

    fx_error = abs(result.K[0, 0] - expected[0, 0]) / expected[0, 0] * 100
    fy_error = abs(result.K[1, 1] - expected[1, 1]) / expected[1, 1] * 100
    assert fx_error < 5.0, f"fx off by {fx_error:.2f}%"
    assert fy_error < 5.0, f"fy off by {fy_error:.2f}%"
    assert result.rms_px < 1.0


def test_intrinsics_refuse_too_few_views(small_board):
    with pytest.raises(ValueError, match="at least"):
        calibrate_intrinsics([], small_board, TEST_SIZE)


def test_extrinsics_recover_the_true_baseline(tmp_path_factory, truth_cameras, small_board):
    """Guards the undistort-first fix.

    OpenCV 5.0's ``stereoCalibrate`` ignores the distortion coefficients it is
    handed when ``CALIB_FIX_INTRINSIC`` is set. Leaving distortion in put the
    baseline 12% out; undistorting the corners first brings it to 0.14%.
    """
    work = tmp_path_factory.mktemp("stereo")
    p0, p1 = work / "s0.mp4", work / "s1.mp4"
    render_stereo_videos(p0, p1, truth_cameras, small_board, TEST_SIZE,
                         n_frames=90, seed=13)

    det0 = detect_boards(str(p0), small_board, sample_fps=30.0,
                         sharpness_threshold=0.0, workers=1)
    det1 = detect_boards(str(p1), small_board, sample_fps=30.0,
                         sharpness_threshold=0.0, workers=1)
    if min(len(det0), len(det1)) < 25:
        pytest.skip(f"too few detections at reduced resolution ({len(det0)}/{len(det1)})")

    result = calibrate_extrinsics(det0, det1, small_board,
                                  truth_cameras.cam0.K, truth_cameras.cam0.dist,
                                  truth_cameras.cam1.K, truth_cameras.cam1.dist,
                                  TEST_SIZE, offsets=(0,), max_pairs=50)
    estimated = Camera(K=truth_cameras.cam1.K, dist=truth_cameras.cam1.dist,
                       R=result.R, t=result.t)
    metrics = compare_cameras(estimated, truth_cameras.cam1)
    true_baseline_mm = float(np.linalg.norm(truth_cameras.cam1.t)) * 1000
    baseline_error_percent = metrics["baseline_error_mm"] / true_baseline_mm * 100

    assert metrics["rotation_error_deg"] < 2.0, metrics
    assert baseline_error_percent < 5.0, f"baseline off by {baseline_error_percent:.2f}%"


def test_extrinsics_pick_the_right_frame_offset(tmp_path_factory, truth_cameras, small_board):
    work = tmp_path_factory.mktemp("stereo_offset")
    p0, p1 = work / "s0.mp4", work / "s1.mp4"
    render_stereo_videos(p0, p1, truth_cameras, small_board, TEST_SIZE,
                         n_frames=90, seed=17)
    det0 = detect_boards(str(p0), small_board, sample_fps=30.0,
                         sharpness_threshold=0.0, workers=1)
    det1 = detect_boards(str(p1), small_board, sample_fps=30.0,
                         sharpness_threshold=0.0, workers=1)
    if min(len(det0), len(det1)) < 25:
        pytest.skip("too few detections")

    result = calibrate_extrinsics(det0, det1, small_board,
                                  truth_cameras.cam0.K, truth_cameras.cam0.dist,
                                  truth_cameras.cam1.K, truth_cameras.cam1.dist,
                                  TEST_SIZE, offsets=(-1, 0, 1), max_pairs=50)
    assert result.frame_offset == 0
    assert result.per_offset_rms[0] < result.per_offset_rms[-1]
    assert result.per_offset_rms[0] < result.per_offset_rms[1]


def test_extrinsics_report_failure_when_nothing_pairs(small_board, truth_cameras):
    with pytest.raises(RuntimeError, match="failed for every candidate offset"):
        calibrate_extrinsics([], [], small_board,
                             truth_cameras.cam0.K, truth_cameras.cam0.dist,
                             truth_cameras.cam1.K, truth_cameras.cam1.dist,
                             TEST_SIZE, offsets=(0,))
