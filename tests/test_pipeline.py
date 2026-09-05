"""The reconstruction pipeline, end to end on synthetic motion.

These are the tests the old code could not have: everything lived inside
``if __name__ == '__main__'``, so no part of it was importable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pose3d import BODY25B, ReconstructionConfig, load_camera_pair, reconstruct
from pose3d.metrics import bone_length_cv_percent, jerk_rms, mpjpe_mm, pa_mpjpe_mm
from pose3d.pipeline import (
    _debias_residual,
    _fill_short_gaps,
    bootstrap_single_view,
    estimate_frame_offset,
    find_t_pose_frames,
    measure_bone_lengths,
    motion_mask,
    shift_series,
    synthesise_midhip,
    up_direction,
)
from pose3d.skeleton import scaled_bone_lengths
from pose3d.synth import (
    DetectorNoise,
    make_desynchronised_pair,
    make_sequence,
    perfect_observations,
    project_sequence,
)

CAMERA_DIR = Path(__file__).resolve().parent.parent / "project" / "task30" / "camera_parameters"


@pytest.fixture(scope="module")
def cameras():
    return load_camera_pair(CAMERA_DIR)


@pytest.fixture(scope="module")
def sequence(cameras):
    return make_sequence(cameras=cameras, subject_height_m=1.72)


@pytest.fixture(scope="module")
def observations(sequence, cameras):
    return project_sequence(sequence, cameras, seed=0)


# --------------------------------------------------------------------------- #
# Synthetic data itself
# --------------------------------------------------------------------------- #


def test_synthetic_bone_lengths_are_exact(sequence):
    """Ground truth must be rigid, or bone-length metrics measure the generator."""
    for p, c, name in BODY25B.bone_pairs:
        d = np.linalg.norm(sequence.points3d[:, p] - sequence.points3d[:, c], axis=-1)
        assert np.allclose(d, sequence.bone_lengths[name], atol=1e-9), name


def test_synthetic_subject_is_visible_to_both_cameras(sequence, cameras):
    for cam in (cameras.cam0, cameras.cam1):
        uv = cam.project(sequence.points3d)
        inside = (
            np.isfinite(uv[..., 0])
            & (uv[..., 0] >= 0) & (uv[..., 0] < 1080)
            & (uv[..., 1] >= 0) & (uv[..., 1] < 1920)
        )
        assert inside.mean() > 0.95, f"{cam.name} sees only {inside.mean():.1%}"


def test_detector_noise_leaves_a_realistic_mix(observations):
    both = np.isfinite(observations["kpts0"][..., 0]) & np.isfinite(observations["kpts1"][..., 0])
    single = np.isfinite(observations["kpts0"][..., 0]) ^ np.isfinite(observations["kpts1"][..., 0])
    assert 0.7 < both.mean() < 0.95, "two-view coverage should be high but not perfect"
    assert single.mean() > 0.03, "there should be single-view joints to bootstrap"


# --------------------------------------------------------------------------- #
# Stage behavior
# --------------------------------------------------------------------------- #


def test_shift_series_semantics():
    x = np.arange(6, dtype=float).reshape(6, 1, 1) * np.ones((6, 1, 2))
    c = np.ones((6, 1))
    shifted, _ = shift_series(x, c, 1.0)
    assert shifted[0, 0, 0] == pytest.approx(1.0)
    assert not np.isfinite(shifted[-1, 0, 0]), "past the end must be NaN, not clamped"
    half, _ = shift_series(x, c, 0.5)
    assert half[0, 0, 0] == pytest.approx(0.5)


def test_midhip_is_the_confidence_weighted_hip_midpoint():
    kpts = np.full((3, BODY25B.n_joints, 2), np.nan)
    conf = np.full((3, BODY25B.n_joints), np.nan)
    l, r, m = BODY25B.indices("LHip", "RHip", "MidHip")

    kpts[0, l], kpts[0, r] = (0.0, 0.0), (10.0, 0.0)
    conf[0, l], conf[0, r] = 1.0, 1.0
    kpts[1, l], kpts[1, r] = (0.0, 0.0), (10.0, 0.0)
    conf[1, l], conf[1, r] = 3.0, 1.0
    # frame 2: nothing at all

    out_k, out_c = synthesise_midhip(kpts, conf, BODY25B)
    assert out_k[0, m] == pytest.approx([5.0, 0.0])
    assert out_k[1, m] == pytest.approx([2.5, 0.0])
    assert not np.isfinite(out_k[2, m]).any()
    assert not np.isfinite(out_c[2, m])


def test_midhip_synthesis_emits_no_warnings(recwarn):
    """An all-missing frame used to trigger 'All-NaN slice encountered'."""
    kpts = np.full((2, BODY25B.n_joints, 2), np.nan)
    conf = np.full((2, BODY25B.n_joints), np.nan)
    synthesise_midhip(kpts, conf, BODY25B)
    assert [w for w in recwarn if issubclass(w.category, RuntimeWarning)] == []


def test_fill_short_gaps_respects_the_limit():
    values = np.arange(20, dtype=float).reshape(20, 1, 1)
    values[5] = np.nan                    # 1-frame gap
    values[10:16] = np.nan                # 6-frame gap
    filled = _fill_short_gaps(values, max_gap=3)
    assert filled[5, 0, 0] == pytest.approx(5.0)
    assert not np.isfinite(filled[12, 0, 0]), "a gap past the limit must not be invented"


def test_fill_short_gaps_never_extrapolates():
    values = np.full((10, 1, 1), np.nan)
    values[4:7] = 1.0
    filled = _fill_short_gaps(values, max_gap=5)
    assert not np.isfinite(filled[0, 0, 0])
    assert not np.isfinite(filled[-1, 0, 0])


def test_up_direction_is_measured_not_assumed(sequence):
    up = up_direction(sequence.points3d, BODY25B)
    assert np.linalg.norm(up) == pytest.approx(1.0)
    # The synthetic world puts "up" at -Y, matching the camera convention.
    assert up[1] < -0.9, f"expected up to be about -Y, got {up}"


def test_t_pose_detection_lands_in_the_t_pose_segment(sequence, cameras, observations):
    cfg = ReconstructionConfig(verbose=False, subject_height_m=1.72)
    k0, c0 = synthesise_midhip(observations["kpts0"].copy(), observations["conf0"].copy(), BODY25B)
    k1, c1 = synthesise_midhip(observations["kpts1"].copy(), observations["conf1"].copy(), BODY25B)
    for k, c in ((k0, c0), (k1, c1)):
        k[np.nan_to_num(c, nan=-1.0) < cfg.min_confidence] = np.nan

    from pose3d.geometry import triangulate_frames

    points = triangulate_frames(cameras.cam0.undistort(k0), cameras.cam1.undistort(k1),
                                c0, c1, cameras, min_confidence=cfg.min_confidence,
                                undistort=False)
    frames = find_t_pose_frames(points, BODY25B, cfg)
    assert frames, "no T-pose found in a clip that opens with one"

    t_pose_span = next(s for s in sequence.segments if s[0] == "t_pose")
    assert all(t_pose_span[1] <= f < t_pose_span[2] for f in frames), \
        f"T-pose frames {frames} fall outside {t_pose_span}"


def test_measured_bone_lengths_reject_implausible_values(cameras):
    """A bad measurement must not replace the prior, but a good one must."""
    cfg = ReconstructionConfig(verbose=False)
    # A subject who is not the reference height, so a correct measurement is
    # visibly different from the prior.
    subject = make_sequence(cameras=cameras, subject_height_m=1.90)
    prior = scaled_bone_lengths(BODY25B, 1.72)

    broken = subject.points3d.copy()
    broken[:, BODY25B.index("LElbow")] += 5.0        # elbow five meters off

    lengths = measure_bone_lengths(broken, BODY25B, list(range(20)), prior, cfg)
    assert lengths["humerus_l"] == pytest.approx(prior["humerus_l"]), \
        "an impossible humerus should fall back to the prior"
    assert lengths["femur_l"] == pytest.approx(subject.bone_lengths["femur_l"], rel=1e-6), \
        "an untouched bone should be personalised to the subject"
    assert lengths["femur_l"] != pytest.approx(prior["femur_l"], rel=1e-3)


def test_bootstrapping_adds_coverage(observations, cameras):
    cfg = ReconstructionConfig(verbose=False, subject_height_m=1.72)
    k0, c0 = synthesise_midhip(observations["kpts0"].copy(), observations["conf0"].copy(), BODY25B)
    k1, c1 = synthesise_midhip(observations["kpts1"].copy(), observations["conf1"].copy(), BODY25B)
    for k, c in ((k0, c0), (k1, c1)):
        k[np.nan_to_num(c, nan=-1.0) < cfg.min_confidence] = np.nan
    u0, u1 = cameras.cam0.undistort(k0), cameras.cam1.undistort(k1)

    from pose3d.geometry import triangulate_frames

    before = triangulate_frames(u0, u1, c0, c1, cameras,
                                min_confidence=cfg.min_confidence, undistort=False)
    after, added = bootstrap_single_view(before, (u0, u1), (c0, c1), cameras, BODY25B,
                                         scaled_bone_lengths(BODY25B, 1.72), cfg)
    assert added > 0
    assert np.isfinite(after[..., 0]).mean() > np.isfinite(before[..., 0]).mean()
    # Bootstrapping must never destroy a two-view point.
    had = np.isfinite(before[..., 0])
    assert np.isfinite(after[..., 0])[had].all()


# --------------------------------------------------------------------------- #
# Synchronization
# --------------------------------------------------------------------------- #


def test_motion_mask_selects_moving_samples():
    kpts = np.zeros((50, 3, 2))
    kpts[:, 1, 0] = np.arange(50) * 5.0        # joint 1 moves fast
    mask = motion_mask(kpts, percentile=80.0)
    assert mask[:, 1].sum() > mask[:, 0].sum()


def test_debias_is_a_no_op_on_integers_and_without_noise():
    assert _debias_residual(2.0, 1.0, 3.0) == pytest.approx(2.0)
    assert _debias_residual(2.0, 0.5, 0.0) == pytest.approx(2.0)
    # With noise present, a half-frame offset is penalized.
    assert _debias_residual(2.0, 0.5, 3.0) > 2.0


@pytest.mark.parametrize("true_offset", [-2.0, -1.0, 0.0, 1.0, 2.0])
def test_whole_frame_offsets_are_recovered(cameras, true_offset):
    """Whole-frame sync is what matters: a one-frame error costs ~3 mm MPJPE."""
    s0, s1 = make_desynchronised_pair(offset_frames=true_offset, cameras=cameras)
    o0 = perfect_observations(s0, cameras)
    o1 = perfect_observations(s1, cameras)
    cfg = ReconstructionConfig(verbose=False)
    estimate, log = estimate_frame_offset(o0["kpts0"], o0["conf0"],
                                          o1["kpts1"], o1["conf1"],
                                          cameras, BODY25B, cfg)
    assert round(estimate) == pytest.approx(true_offset), \
        f"estimated {estimate:+.2f}, expected {true_offset:+.2f}"


def test_offset_search_reports_a_margin(cameras):
    s0, s1 = make_desynchronised_pair(offset_frames=1.0, cameras=cameras)
    o0 = perfect_observations(s0, cameras)
    o1 = perfect_observations(s1, cameras)
    _, log = estimate_frame_offset(o0["kpts0"], o0["conf0"], o1["kpts1"], o1["conf1"],
                                   cameras, BODY25B, ReconstructionConfig(verbose=False))
    best = min(log, key=lambda k: log[k].get("score", np.inf))
    assert log[best]["margin"] > 0.05, "a clean signal should win decisively"


# --------------------------------------------------------------------------- #
# End to end
# --------------------------------------------------------------------------- #


def test_reconstruction_meets_its_accuracy_budget(sequence, observations, cameras):
    """Guards the headline numbers in the README against regression."""
    cfg = ReconstructionConfig(verbose=False, subject_height_m=1.72,
                               sigma_bone_m=0.012, sigma_accel_m=0.005, refine_passes=2)
    result = reconstruct(observations["kpts0"], observations["conf0"],
                         observations["kpts1"], observations["conf1"],
                         cameras, BODY25B, cfg, ground_truth=sequence.points3d)

    assert result.metrics["MPJPE_mm"] < 40.0, result.metrics
    assert result.metrics["PA_MPJPE_mm"] < 35.0, result.metrics
    assert result.metrics["PCK3D_150mm"] > 95.0, result.metrics
    assert result.metrics["BoneLengthCV_percent"] < 8.0, result.metrics
    assert result.metrics["ValidFraction_percent"] > 95.0, result.metrics


def test_refinement_beats_raw_triangulation(sequence, observations, cameras):
    """The refinement must earn its runtime on accuracy, rigidity and smoothness."""
    cfg = ReconstructionConfig(verbose=False, subject_height_m=1.72,
                               sigma_bone_m=0.012, sigma_accel_m=0.005, refine_passes=2)
    result = reconstruct(observations["kpts0"], observations["conf0"],
                         observations["kpts1"], observations["conf1"],
                         cameras, BODY25B, cfg, ground_truth=sequence.points3d)
    gt = sequence.points3d
    triangulated = result.stages["step1_triangulated"]

    assert mpjpe_mm(result.points3d, gt) < mpjpe_mm(triangulated, gt)
    assert bone_length_cv_percent(result.points3d, BODY25B) < \
        bone_length_cv_percent(triangulated, BODY25B)
    assert jerk_rms(result.points3d, 30.0) < jerk_rms(triangulated, 30.0)
    assert np.isfinite(result.points3d[..., 0]).mean() > \
        np.isfinite(triangulated[..., 0]).mean()


def test_reconstruction_never_invents_joints_outside_the_data(sequence, cameras):
    """A joint no camera ever sees must stay NaN, not get hallucinated."""
    obs = project_sequence(sequence, cameras, seed=3)
    dead = BODY25B.index("LBigToe")
    for key in ("kpts0", "kpts1"):
        obs[key][:, dead] = np.nan
    for key in ("conf0", "conf1"):
        obs[key][:, dead] = np.nan

    cfg = ReconstructionConfig(verbose=False, subject_height_m=1.72, refine_passes=1)
    result = reconstruct(obs["kpts0"], obs["conf0"], obs["kpts1"], obs["conf1"],
                         cameras, BODY25B, cfg)
    assert not np.isfinite(result.points3d[:, dead, 0]).any()


def test_pipeline_survives_an_empty_clip(cameras):
    empty_k = np.full((0, BODY25B.n_joints, 2), np.nan)
    empty_c = np.full((0, BODY25B.n_joints), np.nan)
    cfg = ReconstructionConfig(verbose=False)
    result = reconstruct(empty_k, empty_c, empty_k, empty_c, cameras, BODY25B, cfg)
    assert result.n_frames == 0


def test_config_validation_rejects_bad_settings():
    with pytest.raises(ValueError, match="window_overlap"):
        ReconstructionConfig(window_frames=10, window_overlap=10).validate()
    with pytest.raises(ValueError, match="odd"):
        ReconstructionConfig(savgol_window=20).validate()
    with pytest.raises(ValueError, match="sigma_bone_m"):
        ReconstructionConfig(sigma_bone_m=0.0).validate()
    with pytest.raises(ValueError, match="unknown config keys"):
        ReconstructionConfig.from_dict({"not_a_setting": 1})
