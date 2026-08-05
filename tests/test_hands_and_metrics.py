"""Hand kinematics, metrics, and the MMPose adapter.

The hand-frame tests exist because the old implementation raised ``IndexError``
on every single call -- it was handed a 5-row array and indexed row 17 -- and a
bare ``except Exception`` at the call site turned that into ``None``. The result
was a ``*_wrist_kinematics.json`` full of empty dicts on every run, with nothing
in the output to say so.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from pose3d.adapters import (
    coco_wholebody_hands,
    coco_wholebody_to_body25b,
    load_mmpose_as_body25b,
)
from pose3d.config import ReconstructionConfig
from pose3d.hands import (
    compute_wrist_kinematics,
    forearm_frame,
    hand_frame,
    interpolate_rotations,
    wrist_angles,
)
from pose3d.metrics import (
    bone_length_cv_percent,
    composite_score,
    dtw_distance,
    foot_slide_rate_percent,
    jerk_rms,
    mpjpe_mm,
    pa_mpjpe_mm,
    pck3d,
    sequence_distance_mm,
)
from pose3d.skeleton import (
    BODY25B,
    HAND_INDEX,
    HAND_N_JOINTS,
    HANDS_N_JOINTS,
    hand_joint_index,
)


def flat_hand(spread: float = 0.09) -> np.ndarray:
    """A synthetic open palm lying in the XY plane, wrist at the origin."""
    points = np.zeros((HAND_N_JOINTS, 3))
    fingers = {"THUMB": -0.035, "INDEX": -0.012, "MIDDLE": 0.004, "RING": 0.020, "PINKY": 0.034}
    for finger, x in fingers.items():
        for level, joint in enumerate(("MCP", "PIP", "DIP", "TIP")):
            name = f"{finger}_{'CMC' if finger == 'THUMB' and joint == 'MCP' else joint}"
            if name not in HAND_INDEX:
                continue
            points[HAND_INDEX[name]] = (x, spread * (0.45 + 0.18 * level), 0.0)
    points[HAND_INDEX["WRIST"]] = (0.0, 0.0, 0.0)
    return points


# --------------------------------------------------------------------------- #
# Hand frames
# --------------------------------------------------------------------------- #


def test_hand_frame_accepts_the_full_joint_set():
    points = flat_hand()
    conf = np.ones(HAND_N_JOINTS)
    R = hand_frame(points, conf, np.array([0.0, 1.0, 0.0]), "left")
    assert R is not None, "a clean open palm must produce a frame"
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-6)


def test_hand_frame_rejects_a_partial_array_loudly():
    """The old call site passed 5 rows; that must fail visibly, not silently."""
    with pytest.raises(ValueError, match="21 joints"):
        hand_frame(np.zeros((5, 3)), np.ones(5), np.array([0.0, 1.0, 0.0]), "left")


def test_hand_frame_returns_none_when_the_palm_is_missing():
    points = flat_hand()
    points[HAND_INDEX["WRIST"]] = np.nan
    assert hand_frame(points, np.ones(HAND_N_JOINTS),
                      np.array([0.0, 1.0, 0.0]), "left") is None


def test_hand_frame_x_axis_follows_the_fingers():
    points = flat_hand()
    R = hand_frame(points, np.ones(HAND_N_JOINTS), np.array([0.0, 1.0, 0.0]), "left")
    assert R is not None
    # Wrist -> MCP runs along +Y here, and that is the frame's first axis.
    assert abs(float(R[:, 0] @ np.array([0.0, 1.0, 0.0]))) > 0.9


def test_forearm_frame_needs_two_finite_points():
    assert forearm_frame(np.array([np.nan] * 3), np.zeros(3), "left") is None
    assert forearm_frame(np.zeros(3), np.zeros(3), "left") is None
    assert forearm_frame(np.zeros(3), np.array([0.0, 0.3, 0.0]), "left") is not None


def test_wrist_angles_are_zero_for_an_aligned_hand():
    identity = np.eye(3)
    angles = wrist_angles(identity, identity, "left")
    assert all(abs(v) < 1e-6 for v in angles.values())


def test_wrist_angles_are_mirrored_between_hands():
    """A positive number must mean the same anatomical direction on both sides."""
    forearm = np.eye(3)
    rotated = Rotation.from_euler("y", 20, degrees=True).as_matrix()
    left = wrist_angles(forearm, rotated, "left")
    right = wrist_angles(forearm, rotated, "right")
    assert left["RU"] == pytest.approx(-right["RU"], abs=1e-6)


def test_wrist_angles_handle_missing_frames():
    assert all(np.isnan(v) for v in wrist_angles(None, np.eye(3), "left").values())


def test_slerp_fills_short_gaps_only():
    rotations = [Rotation.identity(), None, None,
                 Rotation.from_euler("z", 30, degrees=True), None, None, None, None, None,
                 Rotation.from_euler("z", 60, degrees=True)]
    filled = interpolate_rotations(rotations, max_gap=3)
    assert filled[1] is not None and filled[2] is not None
    assert filled[5] is None, "a gap past the limit must stay empty"


def test_wrist_kinematics_produce_real_angles():
    """The end-to-end path that used to emit nothing at all."""
    n = 40
    hands = np.full((n, HANDS_N_JOINTS, 3), np.nan)
    body = np.full((n, BODY25B.n_joints, 3), np.nan)
    palm = flat_hand()

    for f in range(n):
        angle = np.deg2rad(20.0 * np.sin(f / 6.0))
        R = Rotation.from_euler("x", angle).as_matrix()
        for hand in ("left", "right"):
            base = hand_joint_index(hand, "WRIST")
            hands[f, base:base + HAND_N_JOINTS] = palm @ R.T + np.array([0.3, 1.0, 2.5])
        body[f, BODY25B.index("LElbow")] = (0.3, 1.0 - 0.26, 2.5)
        body[f, BODY25B.index("RElbow")] = (0.3, 1.0 - 0.26, 2.5)

    cfg = ReconstructionConfig(verbose=False)
    result = compute_wrist_kinematics(hands, np.ones((n, HANDS_N_JOINTS)), body, BODY25B, cfg)

    assert result.valid_left > n * 0.8, f"only {result.valid_left}/{n} left frames resolved"
    assert result.valid_right > n * 0.8
    assert len(result.angles) == n
    resolved = [a for a in result.angles if "left" in a]
    assert resolved, "wrist_kinematics.json must not be a list of empty dicts"
    assert all(np.isfinite(list(a["left"].values())).all() for a in resolved)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_mpjpe_is_zero_for_a_perfect_match():
    points = np.random.default_rng(0).normal(size=(10, 18, 3))
    assert mpjpe_mm(points, points) == pytest.approx(0.0)
    assert pck3d(points, points, 1.0) == pytest.approx(100.0)


def test_pa_mpjpe_ignores_a_rigid_offset():
    rng = np.random.default_rng(1)
    gt = rng.normal(size=(6, 18, 3))
    moved = gt * 1.3 + np.array([2.0, -1.0, 0.5])
    assert mpjpe_mm(moved, gt) > 100.0
    assert pa_mpjpe_mm(moved, gt) == pytest.approx(0.0, abs=1e-6)


def test_bone_cv_measures_rigidity_per_bone():
    """The old metric pooled every bone into one distribution, so it mostly
    measured 'femurs are longer than clavicles' and sat near 50% regardless."""
    rng = np.random.default_rng(2)
    points = np.zeros((50, BODY25B.n_joints, 3))
    for j in range(BODY25B.n_joints):
        points[:, j] = rng.normal(size=3) * 0.5
    rigid = np.repeat(points[:1], 50, axis=0)
    assert bone_length_cv_percent(rigid, BODY25B) == pytest.approx(0.0, abs=1e-6)

    jittered = rigid + rng.normal(0, 0.02, rigid.shape)
    assert 0.0 < bone_length_cv_percent(jittered, BODY25B) < 40.0


def test_jerk_rms_is_zero_for_constant_velocity():
    t = np.arange(30).reshape(30, 1, 1)
    points = np.tile(t * 0.01, (1, 4, 3))
    assert jerk_rms(points, 30.0) == pytest.approx(0.0, abs=1e-6)


def test_foot_slide_handles_axis_selection():
    """``pred[:, feet, [0, 2]]`` broadcasts to (F, 2) and the old code then
    called norm(axis=2) on it, which raised."""
    points = np.zeros((20, BODY25B.n_joints, 3))
    points[:, :, 1] = 1.0
    for name in ("LAnkle", "RAnkle"):
        points[:, BODY25B.index(name), 1] = 0.0
    value = foot_slide_rate_percent(points, BODY25B)
    assert np.isfinite(value) and value == pytest.approx(0.0)

    points[:, BODY25B.index("LAnkle"), 0] = np.arange(20) * 0.1     # sliding
    assert foot_slide_rate_percent(points, BODY25B) > 20.0


def test_dtw_matches_identical_sequences():
    rng = np.random.default_rng(3)
    seq = rng.normal(size=(40, 6))
    distance, length = dtw_distance(seq, seq, radius=5)
    assert distance == pytest.approx(0.0, abs=1e-9)
    assert length == 40


def test_dtw_absorbs_a_time_warp():
    base = np.linspace(0, 1, 60).reshape(-1, 1) * np.ones((1, 3))
    stretched = np.linspace(0, 1, 90).reshape(-1, 1) * np.ones((1, 3))
    distance, length = dtw_distance(base, stretched, radius=20)
    assert np.isfinite(distance)
    assert distance / max(length, 1) < 0.05


def test_sequence_distance_is_translation_invariant():
    rng = np.random.default_rng(4)
    a = rng.normal(size=(30, 18, 3))
    b = a + np.array([5.0, -3.0, 2.0])
    assert sequence_distance_mm(a, b) == pytest.approx(0.0, abs=1e-6)


def test_composite_score_ignores_missing_metrics():
    assert np.isnan(composite_score({}))
    perfect = composite_score({"MPJPE_mm": 0.0, "ReprojectionError_px": 0.0})
    assert perfect == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# MMPose adapter
# --------------------------------------------------------------------------- #


def make_mmpose_file(tmp_path, n_frames=5):
    import json

    rng = np.random.default_rng(5)
    frames = []
    for _ in range(n_frames):
        kp = (rng.uniform(100, 900, (133, 2))).tolist()
        frames.append({"instances": [{"keypoints": kp,
                                      "keypoint_scores": np.full(133, 0.9).tolist()}]})
    path = tmp_path / "results_cam0.json"
    path.write_text(json.dumps({"instance_info": frames}), encoding="utf-8")
    return path


def test_mmpose_adapter_maps_into_the_project_schema(tmp_path):
    path = make_mmpose_file(tmp_path)
    body, conf, hands, hand_conf = load_mmpose_as_body25b(path)
    assert body.shape == (5, len(BODY25B.raw_names), 2)
    assert hands.shape == (5, HANDS_N_JOINTS, 2)
    assert np.isfinite(body[:, BODY25B.raw_index("Nose")]).all()
    assert np.isfinite(hands).all()


def test_mmpose_adapter_synthesises_neck_from_shoulders():
    kpts = np.full((1, 133, 2), np.nan)
    conf = np.full((1, 133), np.nan)
    kpts[0, 5], conf[0, 5] = (0.0, 0.0), 0.8      # left shoulder
    kpts[0, 6], conf[0, 6] = (10.0, 0.0), 0.4     # right shoulder
    body, body_conf = coco_wholebody_to_body25b(kpts, conf)

    neck = BODY25B.raw_index("Neck")
    assert body[0, neck] == pytest.approx([5.0, 0.0])
    assert body_conf[0, neck] == pytest.approx(0.4), \
        "a derived joint must not claim more confidence than its inputs"


def test_mmpose_hand_ranges_land_on_the_right_hand():
    kpts = np.zeros((1, 133, 2))
    conf = np.ones((1, 133))
    kpts[0, 91] = (1.0, 1.0)       # left hand root
    kpts[0, 112] = (2.0, 2.0)      # right hand root
    hands, _ = coco_wholebody_hands(kpts, conf)
    assert hands[0, hand_joint_index("left", "WRIST")] == pytest.approx([1.0, 1.0])
    assert hands[0, hand_joint_index("right", "WRIST")] == pytest.approx([2.0, 2.0])
