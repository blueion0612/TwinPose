"""The keypoint schema and its index remapping.

These tests exist because the remapping used to be re-derived by hand in four
separate files, and one of them got it wrong: ``compare.py`` indexed 18-joint
reconstructions with the 26-joint map, so every joint lookup was silently off.
"""

from __future__ import annotations

import numpy as np
import pytest

from pose3d.skeleton import (
    BODY25B,
    BODY25B_DROPPED,
    BODY25B_RAW_NAMES,
    HAND_JOINT_NAMES,
    HAND_N_JOINTS,
    HANDS_N_JOINTS,
    Skeleton,
    hand_joint_index,
    hand_slice,
    height_from_bone_lengths,
    scaled_bone_lengths,
)


def test_drops_exactly_the_intended_joints():
    assert BODY25B.n_joints == len(BODY25B_RAW_NAMES) - len(BODY25B_DROPPED) == 18
    for name in BODY25B_DROPPED:
        assert name not in BODY25B
    for name in BODY25B_RAW_NAMES:
        if name not in BODY25B_DROPPED:
            assert name in BODY25B


def test_indices_are_contiguous_and_ordered():
    assert sorted(BODY25B.index(n) for n in BODY25B.names) == list(range(18))
    # Order must follow the raw layout, so a slice of raw data lines up.
    raw_order = [BODY25B.raw_index(n) for n in BODY25B.names]
    assert raw_order == sorted(raw_order)


def test_select_slices_raw_arrays_consistently():
    raw = np.arange(len(BODY25B_RAW_NAMES), dtype=float)
    frames = np.tile(raw, (5, 1))
    picked = BODY25B.select(frames)
    assert picked.shape == (5, 18)
    for name in BODY25B.names:
        assert picked[0, BODY25B.index(name)] == BODY25B.raw_index(name)


def test_select_rejects_wrong_width():
    with pytest.raises(ValueError, match="expected"):
        BODY25B.select(np.zeros((3, 7)))


def test_unknown_joint_raises_with_a_useful_message():
    with pytest.raises(KeyError, match="LEye"):
        BODY25B.index("LEye")


def test_bones_only_reference_surviving_joints():
    for bone in BODY25B.bones:
        assert bone.parent in BODY25B
        assert bone.child in BODY25B
    for a, b, c in BODY25B.chains:
        assert 0 <= a < 18 and 0 <= b < 18 and 0 <= c < 18


def test_topological_order_puts_parents_first():
    order = BODY25B.topological_order()
    assert len(order) == 18 and set(order) == set(range(18))
    seen = set()
    for i in order:
        parent = BODY25B.parents[i]
        assert parent is None or parent in seen
        seen.add(i)


def test_midhip_to_hip_prior_is_anatomical():
    """The old default was 0.52 m -- a torso length, not half a pelvis.

    That value fed single-view bootstrapping and the first refinement pass, so
    every bootstrapped hip landed about five times too far from the pelvis.
    """
    lengths = BODY25B.default_lengths
    for name in ("midhip_to_lhip", "midhip_to_rhip"):
        assert 0.06 <= lengths[name] <= 0.15, f"{name} = {lengths[name]}"


def test_height_scaling_round_trips():
    for height in (1.55, 1.72, 1.95):
        lengths = scaled_bone_lengths(BODY25B, height)
        assert height_from_bone_lengths(lengths) == pytest.approx(height, rel=1e-6)


def test_height_scaling_is_proportional():
    base = scaled_bone_lengths(BODY25B, 1.72)
    tall = scaled_bone_lengths(BODY25B, 1.72 * 1.1)
    for name in base:
        assert tall[name] == pytest.approx(base[name] * 1.1, rel=1e-9)


def test_scaled_bone_lengths_rejects_nonsense():
    with pytest.raises(ValueError):
        scaled_bone_lengths(BODY25B, -1.0)


def test_hand_layout_is_21_per_hand():
    assert HAND_N_JOINTS == 21
    assert HANDS_N_JOINTS == 42
    assert len(HAND_JOINT_NAMES) == 21
    assert hand_slice("left") == slice(0, 21)
    assert hand_slice("right") == slice(21, 42)
    assert hand_joint_index("left", "WRIST") == 0
    assert hand_joint_index("right", "WRIST") == 21
    # The old code laid hands out at stride 22 to keep the model's background
    # channel, which put "right wrist" on the left hand's background map.
    assert hand_joint_index("right", "WRIST") != 22


def test_hand_lookup_errors_are_specific():
    with pytest.raises(ValueError, match="left.*right"):
        hand_slice("middle")
    with pytest.raises(KeyError, match="ELBOW"):
        hand_joint_index("left", "ELBOW")


def test_custom_skeleton_can_drop_everything_touching_a_bone():
    from pose3d.skeleton import Bone

    skeleton = Skeleton.build(
        raw_names=("A", "B", "C"),
        dropped=("B",),
        bones=(Bone("A", "B", "ab", 0.1), Bone("A", "C", "ac", 0.2)),
        chains=(("A", "B", "C"),),
        parents={"A": None, "B": "A", "C": "A"},
    )
    assert skeleton.names == ("A", "C")
    assert [b.name for b in skeleton.bones] == ["ac"]
    assert skeleton.chains == ()


def test_building_with_an_unknown_dropped_joint_raises():
    from pose3d.skeleton import Bone

    with pytest.raises(ValueError, match="unknown joints"):
        Skeleton.build(("A",), ("Z",), (Bone("A", "A", "x", 1.0),), (), {"A": None})
