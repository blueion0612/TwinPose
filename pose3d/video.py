"""Fast video reading.

The single biggest avoidable cost in the old pipeline was the access pattern

.. code-block:: python

    for fid in range(0, total, step):
        cap.set(cv.CAP_PROP_POS_FRAMES, fid)
        ok, frame = cap.read()

Every ``set`` on a compressed stream forces the decoder to jump back to the
preceding keyframe and re-decode forward to the target. On phone H.264 with a
2-second GOP that is ~60 hidden frame decodes for each frame actually used, so
scanning a two-minute clip decodes it dozens of times over. ``calibration.py``
did this in three separate places and ``validate_calibration.py`` in a fourth,
where it seeked *twice per frame* to read consecutive frames.

Reading sequentially and skipping with ``grab()`` -- which demuxes and decodes
without converting to a NumPy array -- removes that entirely. Splitting the
frame range into contiguous chunks then lets every core work at once, with each
worker paying a single seek.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

PathLike = Union[str, Path]


@dataclass(frozen=True)
class VideoInfo:
    """Container metadata, read once."""

    path: str
    n_frames: int
    fps: float
    width: int
    height: int

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps if self.fps > 0 else float("nan")

    @property
    def is_portrait(self) -> bool:
        return self.height >= self.width


def probe(path: PathLike) -> VideoInfo:
    """Read container metadata without decoding any frames."""
    import cv2

    p = str(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"video not found: {p}")
    cap = cv2.VideoCapture(p)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {p}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        return VideoInfo(
            path=p,
            n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=fps,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def read_frames(
    path: PathLike,
    *,
    start: int = 0,
    stop: Optional[int] = None,
    step: int = 1,
    indices: Optional[Sequence[int]] = None,
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(frame_index, frame)`` by decoding forward, never seeking back.

    ``indices`` must be sorted ascending when given; frames between wanted ones
    are skipped with ``grab()``, which is roughly an order of magnitude cheaper
    than a full ``read()``.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    try:
        if indices is not None:
            wanted = list(indices)
            if not wanted:
                return
            if any(b < a for a, b in zip(wanted, wanted[1:])):
                raise ValueError("indices must be sorted ascending")
            cursor = wanted[0]
            if cursor > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, cursor)   # exactly one seek
            for target in wanted:
                while cursor < target:
                    if not cap.grab():
                        return
                    cursor += 1
                ok, frame = cap.read()
                if not ok:
                    return
                yield cursor, frame
                cursor += 1
            return

        if start > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)        # exactly one seek
        cursor = start
        while stop is None or cursor < stop:
            ok, frame = cap.read()
            if not ok:
                return
            yield cursor, frame
            cursor += 1
            for _ in range(step - 1):
                if not cap.grab():
                    return
                cursor += 1
    finally:
        cap.release()


def _chunk_ranges(indices: Sequence[int], n_chunks: int) -> List[List[int]]:
    """Split sorted indices into contiguous, roughly equal groups."""
    if n_chunks <= 1 or len(indices) <= 1:
        return [list(indices)]
    n_chunks = min(n_chunks, len(indices))
    bounds = np.linspace(0, len(indices), n_chunks + 1).astype(int)
    return [list(indices[a:b]) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def _worker(args: Tuple[str, List[int], Callable[[int, np.ndarray], Any]]) -> List[Tuple[int, Any]]:
    import cv2

    # OpenCV defaults to one thread per core. With N worker processes that is
    # N*cores threads fighting over the same cores, which is slower than either
    # strategy alone. One thread per process keeps the split clean.
    cv2.setNumThreads(1)

    path, indices, func = args
    out: List[Tuple[int, Any]] = []
    for idx, frame in read_frames(path, indices=indices):
        result = func(idx, frame)
        if result is not None:
            out.append((idx, result))
    return out


def map_frames(
    path: PathLike,
    func: Callable[[int, np.ndarray], Any],
    *,
    indices: Optional[Sequence[int]] = None,
    step: int = 1,
    workers: Optional[int] = None,
    progress: Optional[Callable] = None,
    desc: str = "",
) -> List[Tuple[int, Any]]:
    """Apply ``func(index, frame)`` across a video, in parallel over frame ranges.

    Each worker opens its own capture, seeks once to the head of its chunk and
    then decodes forward, so parallelism costs one seek per core rather than one
    per frame. Results are returned sorted by frame index; ``func`` returning
    ``None`` drops that frame from the output.

    ``func`` must be picklable -- a module-level function, or an instance of a
    class defined at module level. Set ``workers=1`` to run in-process, which is
    what the tests use and what small jobs fall back to.
    """
    info = probe(path)
    if indices is None:
        indices = list(range(0, info.n_frames, max(1, step)))
    indices = [int(i) for i in indices]
    if not indices:
        return []

    if workers is None:
        workers = default_workers()

    def run_serial() -> List[Tuple[int, Any]]:
        import cv2

        cv2.setNumThreads(0)         # 0 restores OpenCV's own default threading
        results: List[Tuple[int, Any]] = []
        iterator: Iterable[Tuple[int, np.ndarray]] = read_frames(path, indices=indices)
        if progress is not None:
            iterator = progress(iterator, total=len(indices), desc=desc)
        for idx, frame in iterator:
            value = func(idx, frame)
            if value is not None:
                results.append((idx, value))
        return results

    # Process startup on Windows costs ~0.5 s per worker; below this the pool
    # costs more than it saves.
    if workers <= 1 or len(indices) < 64:
        return run_serial()

    import concurrent.futures as cf

    chunks = _chunk_ranges(indices, workers)
    payload = [(str(path), chunk, func) for chunk in chunks]
    collected: List[Tuple[int, Any]] = []
    try:
        with cf.ProcessPoolExecutor(max_workers=len(chunks)) as pool:
            futures = [pool.submit(_worker, item) for item in payload]
            stream: Iterable[Any] = cf.as_completed(futures)
            if progress is not None:
                stream = progress(stream, total=len(futures), desc=desc)
            for future in stream:
                collected.extend(future.result())
    except (cf.process.BrokenProcessPool, RuntimeError, OSError, AttributeError) as exc:
        # Windows spawns workers by re-importing the caller's __main__, so a
        # script that calls this at module level (no `if __name__ == "__main__"`)
        # recurses and the pool dies. Falling back keeps the job correct rather
        # than dumping a multiprocessing traceback on the user.
        import warnings

        warnings.warn(
            f"parallel frame processing unavailable ({type(exc).__name__}: {exc}); "
            "falling back to a single process. If you are calling this from a "
            "script, guard the entry point with `if __name__ == \"__main__\":`.",
            RuntimeWarning,
            stacklevel=2,
        )
        return run_serial()
    collected.sort(key=lambda item: item[0])
    return collected


def default_workers(reserve: int = 2) -> int:
    """A worker count that is fast without oversubscribing.

    Frame processing mixes video decode (memory-bandwidth bound) with detection
    (CPU bound), so throughput peaks well below the logical core count and then
    falls off. Measured on a 12-core/24-thread Ryzen 9 5900X, scanning 1080x1920
    frames for a checkerboard:

    ======== =========  =======
    workers  fps        speedup
    ======== =========  =======
    1 (SMP)  5.8        1.00x
    4        17.9       3.11x
    8        24.1       4.18x
    12       23.3       4.04x
    22       20.1       3.49x
    ======== =========  =======

    Half the logical cores lands near the plateau on this machine and degrades
    gracefully on smaller ones.
    """
    total = os.cpu_count() or 1
    if total <= 4:
        return max(1, total - 1)
    return max(2, min(total // 2, 12))


def write_video(
    path: PathLike,
    frames: Iterable[np.ndarray],
    fps: float,
    *,
    size: Optional[Tuple[int, int]] = None,
    fourcc: str = "mp4v",
) -> int:
    """Write frames to a video file, returning how many were written."""
    import cv2

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    count = 0
    try:
        for frame in frames:
            if writer is None:
                h, w = frame.shape[:2]
                target = size or (w, h)
                writer = cv2.VideoWriter(
                    str(p), cv2.VideoWriter_fourcc(*fourcc), float(fps), target
                )
                if not writer.isOpened():
                    raise RuntimeError(f"could not open video writer for {p}")
            if size is not None and (frame.shape[1], frame.shape[0]) != size:
                frame = cv2.resize(frame, size)
            writer.write(frame)
            count += 1
    finally:
        if writer is not None:
            writer.release()
    return count
