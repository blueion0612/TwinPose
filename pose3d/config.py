"""Tunable parameters for the reconstruction pipeline.

Collected in one dataclass so a run is fully described by a single object that
can be logged, diffed and round-tripped through JSON. The previous version kept
these as twenty-odd module-level globals in ``3D_estimation.py``, several of
which were assigned twice with different values (``DEFAULT_BONE_LENGTHS`` and
``KPTS_TO_EXCLUDE_FROM_3D`` were each defined three times).

Refinement weights are expressed as **uncertainties**, not gains. A residual
enters the cost as ``residual / sigma``, so ``sigma_bone_m = 0.012`` reads as
"a one-centimetre bone-length error is about as bad as a one-sigma reprojection
error" and a single Huber ``f_scale`` is meaningful across every block.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class ReconstructionConfig:
    """Everything the 3D stage can be tuned by."""

    # -- input conditioning ------------------------------------------------ #
    #: Observations below this confidence are dropped before triangulation.
    min_confidence: float = 0.30
    #: Looser gate used when scoring candidate frame offsets, where recall
    #: matters more than precision.
    sync_min_confidence: float = 0.10
    fps: float = 30.0

    # -- inter-camera synchronisation -------------------------------------- #
    #: Widest offset, in frames, the search will consider.
    sync_max_offset: float = 3.0
    #: Integer-grid step for the coarse pass.
    sync_coarse_step: float = 1.0
    #: Sub-frame step for the refinement pass around the coarse winner.
    #: Set to 0 to search whole frames only.
    sync_fine_step: float = 0.05
    #: Relative score improvement a sub-frame offset must show before it is
    #: preferred over the winning whole-frame offset. A half-frame sync error
    #: costs ~0.5 mm MPJPE on the synthetic benchmark while a whole-frame error
    #: costs ~3 mm, so the sub-frame search is only worth taking when it wins
    #: clearly.
    sync_fine_min_gain: float = 0.03
    #: Frames sampled when scoring an offset. The old code scored every frame
    #: for all 121 candidate offsets, which dominated the stage's runtime.
    sync_sample_frames: int = 400
    #: Skip the search and use the offset calibration stored instead.
    sync_use_calibration_offset: bool = False

    # -- subject model ----------------------------------------------------- #
    #: Standing height in metres. Anchors the bone prior before any T-pose
    #: measurement exists; ``None`` uses the 1.72 m population default.
    subject_height_m: Optional[float] = None
    #: Seconds from the start of the clip searched for T-pose frames.
    t_pose_search_seconds: float = 60.0
    #: Keep T-pose candidates scoring within this factor of the best frame.
    t_pose_score_tolerance: float = 1.15
    #: Trim this fraction off each tail before averaging measured bone lengths.
    t_pose_trim_fraction: float = 0.10
    #: Reject a T-pose bone measurement that disagrees with the height-scaled
    #: prior by more than this factor, in either direction.
    t_pose_max_deviation: float = 1.75

    # -- refinement --------------------------------------------------------- #
    #: Frames per sliding-window solve. The temporal prior is local, so
    #: windowing keeps each least-squares problem small without materially
    #: changing the optimum.
    window_frames: int = 120
    #: Overlap between consecutive windows, cross-faded with a cosine ramp.
    window_overlap: int = 30
    #: Number of sweeps over the whole sequence.
    refine_passes: int = 2
    #: Maximum solver evaluations per window.
    max_nfev: int = 60

    #: Expected 2D detector noise, in pixels. The data term's sigma.
    sigma_reproj_px: float = 3.0
    #: Expected bone-length error, in metres. Loose enough to absorb real soft
    #: tissue movement, tight enough to stop limbs stretching along the ray.
    sigma_bone_m: float = 0.012
    #: Expected frame-to-frame acceleration, in metres. This is the smoothness
    #: prior; it does the work the Savitzky-Golay pass and the "subtract 25% of
    #: the acceleration" step used to do, without their phase distortion.
    sigma_accel_m: float = 0.010
    #: Slack on the joint-limit hinge, in cosine units.
    sigma_limit: float = 0.05
    #: Huber transition point, in sigmas. Residuals beyond this are down-weighted
    #: linearly instead of quadratically, so a mis-detected keypoint cannot drag
    #: the whole window.
    huber_f_scale: float = 2.5

    # -- anatomical limits --------------------------------------------------- #
    #: Largest chain cosine allowed before the joint-limit hinge activates.
    #:
    #: For a chain (proximal, middle, distal), the cosine is taken between
    #: (proximal - middle) and (distal - middle). A straight limb gives -1; a
    #: folded one approaches +1. So the *upper* bound is the anatomical
    #: constraint: a knee cannot fold past roughly 150 degrees of flexion, which
    #: leaves about 30 degrees between the segments, i.e. cos ~ 0.87.
    #:
    #: The original code -- and the first version of this rewrite -- tested
    #: `cos < 0.15` instead, which is true for a *straight* limb and false for a
    #: bent one. As a soft penalty that actively fought straight legs during the
    #: walking stance phase. The synthetic ablation caught it: disabling the term
    #: entirely improved MPJPE from 25.4 mm to 24.2 mm.
    max_chain_cosine: float = 0.87

    # -- post-processing ----------------------------------------------------- #
    #: Longest run of missing frames that gets filled before refinement. Gaps
    #: longer than this stay NaN rather than being invented.
    max_interpolation_gap: int = 12
    #: Optional light Savitzky-Golay pass after refinement. Off by default: the
    #: acceleration prior already handles smoothness, and stacking both
    #: over-smooths fast motion.
    savgol_after_refine: bool = False
    savgol_window: int = 21
    savgol_poly: int = 3

    # -- hands ---------------------------------------------------------------- #
    hand_min_confidence: float = 0.10
    #: Longest gap SLERP will bridge in the hand orientation track.
    hand_max_rotation_gap: int = 8
    #: Smoothing window for the final wrist angle traces, in frames.
    hand_angle_smooth_window: int = 11

    # -- bookkeeping ---------------------------------------------------------- #
    #: Emit the per-step JSON files the plotting scripts read.
    save_intermediate: bool = True
    verbose: bool = True
    #: Seed for any stochastic step, so runs reproduce exactly.
    random_seed: int = 0

    # -- (de)serialisation ---------------------------------------------------- #
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ReconstructionConfig":
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def replace(self, **changes: Any) -> "ReconstructionConfig":
        data = self.to_dict()
        data.update(changes)
        return ReconstructionConfig.from_dict(data)

    def validate(self) -> "ReconstructionConfig":
        if self.window_overlap >= self.window_frames:
            raise ValueError(
                f"window_overlap ({self.window_overlap}) must be smaller than "
                f"window_frames ({self.window_frames})"
            )
        if self.savgol_window % 2 == 0:
            raise ValueError(f"savgol_window must be odd, got {self.savgol_window}")
        if self.savgol_poly >= self.savgol_window:
            raise ValueError("savgol_poly must be smaller than savgol_window")
        if self.sync_fine_step <= 0 or self.sync_coarse_step <= 0:
            raise ValueError("sync steps must be positive")
        if not 0.0 <= self.t_pose_trim_fraction < 0.5:
            raise ValueError("t_pose_trim_fraction must be in [0, 0.5)")
        for name in ("sigma_reproj_px", "sigma_bone_m", "sigma_accel_m", "sigma_limit"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.fps <= 0:
            raise ValueError("fps must be positive")
        return self
