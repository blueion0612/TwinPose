"""Video synchronization from a green flashlight, and checkerboard onset search.

Two cameras recording independently start at arbitrary times. Flashing a green
light in view of both puts a shared, unambiguous event in each stream; matching
the *intervals* between flashes then aligns the two clips to within a frame
without any synchronization hardware.

Speed
-----
Flash detection reads every frame of a two-minute clip. Doing that at full
1080x1920 resolution to count green pixels is wasted work -- a flash covers
thousands of pixels, so it survives an 8x downscale intact, and the frames can
be processed in parallel over contiguous ranges (see :mod:`pose3d.video`).
Together that is roughly a 20x reduction in this stage's cost.

Robustness
----------
The original matcher scored candidate alignments by counting how many
consecutive inter-flash gaps agreed, and stopped extending a match at the first
disagreement. One spurious detection between two real flashes -- a reflection, a
passing headlight -- merges two gaps into one and truncates the match there. The
matcher here scores every offset over the whole event set, so isolated false
positives cost one event instead of everything after them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .video import default_workers, map_frames, probe

#: HSV window for a green LED. Wide enough for the hue shift a saturated
#: highlight shows on phone sensors.
GREEN_LOWER = (40, 80, 120)
GREEN_UPPER = (90, 255, 255)


@dataclass
class FlashEvent:
    """One detected flash."""

    frame: int
    time_ms: float
    intensity: float


@dataclass
class SyncResult:
    """How two clips line up."""

    offset_frames: int
    events0: List[FlashEvent] = field(default_factory=list)
    events1: List[FlashEvent] = field(default_factory=list)
    matched: int = 0
    residual_ms: float = float("nan")
    confidence: float = 0.0

    def summary(self) -> str:
        return (
            f"offset {self.offset_frames:+d} frames from {self.matched} matched flashes "
            f"(residual {self.residual_ms:.1f} ms, confidence {self.confidence:.2f})"
        )


class _GreenCounter:
    """Picklable frame callback counting green pixels at reduced resolution."""

    def __init__(self, scale: float = 0.125):
        self.scale = float(scale)

    def __call__(self, index: int, frame: np.ndarray):
        import cv2

        if self.scale != 1.0:
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                               interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(GREEN_LOWER, np.uint8),
                           np.array(GREEN_UPPER, np.uint8))
        return int(cv2.countNonZero(mask))


def green_area_series(
    video: str,
    *,
    scale: float = 0.125,
    workers: Optional[int] = None,
    progress: Optional[object] = None,
    desc: str = "flash scan",
) -> Tuple[np.ndarray, float]:
    """Green-pixel count per frame, plus the clip's fps."""
    info = probe(video)
    results = map_frames(
        video, _GreenCounter(scale),
        workers=workers if workers is not None else default_workers(),
        progress=progress, desc=desc,
    )
    series = np.zeros(info.n_frames, dtype=np.int64)
    for idx, value in results:
        if 0 <= idx < info.n_frames:
            series[idx] = value
    return series, info.fps


def detect_flashes(
    green_area: np.ndarray,
    fps: float,
    *,
    min_sigma: float = 4.0,
    min_separation_frames: int = 4,
) -> List[FlashEvent]:
    """Find flash onsets as sharp positive jumps in the green-pixel count.

    Uses a robust (median/MAD) threshold instead of mean and standard deviation:
    the flashes themselves are large outliers, so they inflate a standard
    deviation enough to hide the smaller ones.
    """
    if green_area.size < 3:
        return []
    diffs = np.diff(green_area.astype(np.float64))
    median = float(np.median(diffs))
    mad = float(np.median(np.abs(diffs - median)))
    scale = 1.4826 * mad if mad > 0 else float(np.std(diffs)) or 1.0
    threshold = median + min_sigma * scale

    candidates = np.flatnonzero(diffs > threshold)
    if candidates.size == 0:
        return []

    # Group consecutive rising frames into one event, keeping the strongest.
    events: List[FlashEvent] = []
    group_start = candidates[0]
    previous = candidates[0]
    for idx in np.append(candidates[1:], -1):
        if idx == -1 or idx > previous + min_separation_frames:
            span = diffs[group_start:previous + 1]
            peak = int(group_start + int(np.argmax(span)))
            events.append(
                FlashEvent(frame=peak + 1, time_ms=(peak + 1) / fps * 1000.0,
                           intensity=float(span.max()))
            )
            if idx == -1:
                break
            group_start = idx
        previous = idx
    return events


def match_flash_sequences(
    events0: Sequence[FlashEvent],
    events1: Sequence[FlashEvent],
    fps: float,
    *,
    tolerance_ms: float = 40.0,
    max_offset_s: float = 60.0,
) -> SyncResult:
    """Find the constant time shift that aligns the two flash sequences.

    Every pairing of one event from each clip proposes a shift; the shift with
    the most events agreeing within ``tolerance_ms`` wins. This is a small
    RANSAC-style vote rather than a longest-run match, so a spurious detection
    costs one event rather than truncating the match.
    """
    if len(events0) < 2 or len(events1) < 2:
        return SyncResult(offset_frames=0, events0=list(events0), events1=list(events1))

    t0 = np.array([e.time_ms for e in events0])
    t1 = np.array([e.time_ms for e in events1])
    limit_ms = max_offset_s * 1000.0

    best = (0, -1, float("inf"))     # (shift_ms, votes, residual)
    for a in t0:
        for b in t1:
            shift = b - a
            if abs(shift) > limit_ms:
                continue
            deltas = np.abs(t1[None, :] - (t0[:, None] + shift))
            nearest = deltas.min(axis=1)
            hits = nearest <= tolerance_ms
            votes = int(hits.sum())
            residual = float(nearest[hits].mean()) if votes else float("inf")
            if votes > best[1] or (votes == best[1] and residual < best[2]):
                best = (shift, votes, residual)

    shift_ms, votes, residual = best
    if votes < 2:
        return SyncResult(offset_frames=0, events0=list(events0), events1=list(events1))

    # Refine: least-squares shift over the matched pairs only.
    deltas = np.abs(t1[None, :] - (t0[:, None] + shift_ms))
    nearest_idx = deltas.argmin(axis=1)
    nearest = deltas.min(axis=1)
    keep = nearest <= tolerance_ms
    if keep.any():
        shift_ms = float(np.mean(t1[nearest_idx[keep]] - t0[keep]))
        residual = float(np.mean(np.abs(t1[nearest_idx[keep]] - t0[keep] - shift_ms)))

    confidence = votes / min(len(events0), len(events1))
    return SyncResult(
        offset_frames=int(round(shift_ms / 1000.0 * fps)),
        events0=list(events0),
        events1=list(events1),
        matched=votes,
        residual_ms=residual,
        confidence=float(min(confidence, 1.0)),
    )


class _BoardPresent:
    """Picklable callback reporting whether a checkerboard is visible."""

    def __init__(self, pattern: Tuple[int, int], downscale: float = 0.5):
        self.pattern = pattern
        self.downscale = float(downscale)

    def __call__(self, index: int, frame: np.ndarray):
        import cv2

        if self.downscale != 1.0:
            frame = cv2.resize(frame, None, fx=self.downscale, fy=self.downscale,
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, _ = cv2.findChessboardCorners(
            gray, self.pattern,
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK,
        )
        return True if found else None


def find_first_board_frame(
    video: str,
    pattern: Tuple[int, int],
    *,
    start: int = 0,
    max_frames: int = 3600,
    step: int = 10,
    workers: Optional[int] = None,
    progress: Optional[object] = None,
) -> Optional[int]:
    """First frame at or after ``start`` where the checkerboard appears.

    Scans coarsely in parallel and then walks backwards from the first hit to
    pin down the onset, rather than testing every frame in order.
    """
    info = probe(video)
    stop = min(info.n_frames, start + max_frames)
    coarse = list(range(start, stop, max(1, step)))
    if not coarse:
        return None

    hits = map_frames(
        video, _BoardPresent(pattern), indices=coarse,
        workers=workers if workers is not None else default_workers(),
        progress=progress, desc="checkerboard search",
    )
    if not hits:
        return None
    first = hits[0][0]

    fine = list(range(max(start, first - step), first + 1))
    refined = map_frames(video, _BoardPresent(pattern), indices=fine, workers=1)
    return refined[0][0] if refined else first


def synchronise(
    video0: str,
    video1: str,
    *,
    scale: float = 0.125,
    workers: Optional[int] = None,
    progress: Optional[object] = None,
) -> SyncResult:
    """Full flash-based synchronization of two clips."""
    series0, fps0 = green_area_series(video0, scale=scale, workers=workers,
                                      progress=progress, desc="cam0 flash scan")
    series1, fps1 = green_area_series(video1, scale=scale, workers=workers,
                                      progress=progress, desc="cam1 flash scan")
    fps = fps0 or fps1 or 30.0
    events0 = detect_flashes(series0, fps0 or fps)
    events1 = detect_flashes(series1, fps1 or fps)
    return match_flash_sequences(events0, events1, fps)
