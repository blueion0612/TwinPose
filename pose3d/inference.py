"""2D pose inference: device selection, batching, and heatmap decoding.

This is the pipeline's dominant cost -- five scales, two cameras, two hands, for
every frame of a 30-second clip -- so it is where optimization pays off most.

Three problems with the original ``estimation/Openpose.py``:

**The GPU was never used.** It sets ``DNN_BACKEND_CUDA`` and
``DNN_TARGET_CUDA_FP16``, but the ``opencv-contrib-python`` wheel on PyPI is
built without CUDA. OpenCV then falls back to CPU *silently*, so a run on a
machine with an idle RTX 3090 Ti looks exactly like a run on a machine with no
GPU at all. :func:`describe_backend` reports what is actually going to execute,
and :func:`select_backend` says so out loud instead of pretending.

**Everything ran one frame at a time.** Five ``net.forward()`` calls per frame
per camera, each on a batch of one. Batching frames into a single forward pass
keeps the BLAS threads busy between layers and is a large win on CPU as well as
on GPU.

**The multi-scale loop resized to a fixed network resolution.** Each scale
resized the image by ``s`` and then ``blobFromImage`` resized *again* to a fixed
``(W, H)``, so all five "scales" fed the network almost identical input -- the
supposed multi-scale averaging was mostly averaging the same thing five times at
five times the cost. Scales are applied to the network input size here, which is
what actually varies the receptive field.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class BackendInfo:
    """What inference will actually run on."""

    name: str
    device: str
    fp16: bool
    available_devices: Tuple[str, ...] = ()
    note: str = ""

    def __str__(self) -> str:
        precision = "FP16" if self.fp16 else "FP32"
        base = f"{self.name} on {self.device} ({precision})"
        return f"{base} -- {self.note}" if self.note else base


def describe_backend(prefer_gpu: bool = True) -> BackendInfo:
    """Report the inference device honestly, without loading a model.

    The important case is ``prefer_gpu=True`` on a machine that has a CUDA GPU
    but an OpenCV built without CUDA: that combination silently runs on CPU, and
    the note here is what tells the user why their run is taking an hour.
    """
    import cv2

    available: List[str] = ["cpu"]
    cuda_devices = 0
    try:
        cuda_devices = int(cv2.cuda.getCudaEnabledDeviceCount())
    except Exception:
        cuda_devices = 0
    if cuda_devices > 0:
        available.append("cuda")

    if not prefer_gpu:
        return BackendInfo("opencv-dnn", "cpu", False, tuple(available))

    if cuda_devices > 0:
        return BackendInfo("opencv-dnn", "cuda", True, tuple(available))

    note = (
        "OpenCV was built without CUDA, so the GPU cannot be used. "
        "The pip `opencv-contrib-python` wheel never includes CUDA; a CUDA-enabled "
        "OpenCV has to be built from source. Expect CPU inference to be roughly "
        "an order of magnitude slower."
    )
    if _has_nvidia_gpu():
        note = "An NVIDIA GPU is present but " + note[0].lower() + note[1:]
    return BackendInfo("opencv-dnn", "cpu", False, tuple(available), note)


def _has_nvidia_gpu() -> bool:
    """Cheap check for an NVIDIA GPU, so the CPU-fallback warning can be specific."""
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return False
    try:
        return subprocess.run(
            [exe, "-L"], capture_output=True, timeout=5
        ).returncode == 0
    except Exception:
        return False


@dataclass
class InferenceConfig:
    """Network input geometry and multi-scale settings."""

    #: Network input as (width, height). ``-1`` derives that axis from the
    #: image's aspect ratio.
    net_resolution: Tuple[int, int] = (-1, 736)
    #: Multi-scale factors applied to the *network input size*.
    scales: Tuple[float, ...] = (1.0, 0.75)
    #: Network output stride; input dimensions are padded to a multiple of it.
    stride: int = 8
    pad_value: int = 128
    #: Peak confidence below which a joint counts as undetected.
    min_confidence: float = 0.15
    #: Frames per forward pass. Larger is faster until VRAM or RAM runs out.
    batch_size: int = 8
    prefer_gpu: bool = True

    def resolve_size(self, width: int, height: int) -> Tuple[int, int]:
        """Network input size for an image, rounded up to the stride."""
        w, h = self.net_resolution
        if w == -1 and h == -1:
            raise ValueError("net_resolution cannot have both axes set to -1")
        if w == -1:
            w = int(np.ceil(width / height * h / self.stride) * self.stride)
        if h == -1:
            h = int(np.ceil(height / width * w / self.stride) * self.stride)
        w = int(np.ceil(w / self.stride) * self.stride)
        h = int(np.ceil(h / self.stride) * self.stride)
        return w, h


def select_backend(config: InferenceConfig, verbose: bool = True) -> BackendInfo:
    """Choose and announce the inference device."""
    info = describe_backend(config.prefer_gpu)
    if verbose:
        print(f"[inference] {info}")
        if info.note:
            warnings.warn(info.note, RuntimeWarning, stacklevel=2)
    return info


def pad_to_stride(
    image: np.ndarray, stride: int, pad_value: int = 128
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Pad right and bottom so both dimensions divide by ``stride``.

    Returns the padded image and ``(pad_bottom, pad_right)`` so the heatmaps can
    be cropped back afterwards.
    """
    h, w = image.shape[:2]
    pad_h = (-h) % stride
    pad_w = (-w) % stride
    if pad_h == 0 and pad_w == 0:
        return image, (0, 0)
    padded = np.pad(
        image,
        ((0, pad_h), (0, pad_w)) + ((0, 0),) * (image.ndim - 2),
        mode="constant",
        constant_values=pad_value,
    )
    return padded, (pad_h, pad_w)


def heatmaps_to_keypoints(
    heatmaps: np.ndarray, min_confidence: float = 0.15, refine: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Decode ``(J, H, W)`` heatmaps into keypoints and confidences.

    Parameters
    ----------
    refine
        Fit a parabola to each peak's immediate neighbors to recover sub-pixel
        position. The original took ``argmax`` alone, quantising every keypoint
        to the heatmap grid -- and since the heatmap is upsampled from a
        stride-8 network, that grid is coarse enough to matter: it puts a floor
        of a few pixels on the 2D error, which then propagates into every 3D
        point.

    Returns
    -------
    (keypoints, confidence)
        ``(J, 2)`` in heatmap pixel coordinates (NaN where undetected) and
        ``(J,)`` peak values.
    """
    if heatmaps.ndim != 3:
        raise ValueError(f"expected (J, H, W) heatmaps, got shape {heatmaps.shape}")
    n_joints, height, width = heatmaps.shape

    flat = heatmaps.reshape(n_joints, -1)
    peak = np.argmax(flat, axis=1)
    conf = flat[np.arange(n_joints), peak]
    ys, xs = np.divmod(peak, width)

    points = np.stack([xs, ys], axis=1).astype(float)

    if refine:
        for j in range(n_joints):
            x, y = int(xs[j]), int(ys[j])
            if 0 < x < width - 1:
                l, c, r = heatmaps[j, y, x - 1], heatmaps[j, y, x], heatmaps[j, y, x + 1]
                denom = l - 2.0 * c + r
                if abs(denom) > 1e-9:
                    points[j, 0] = x + np.clip(0.5 * (l - r) / denom, -0.5, 0.5)
            if 0 < y < height - 1:
                u, c, d = heatmaps[j, y - 1, x], heatmaps[j, y, x], heatmaps[j, y + 1, x]
                denom = u - 2.0 * c + d
                if abs(denom) > 1e-9:
                    points[j, 1] = y + np.clip(0.5 * (u - d) / denom, -0.5, 0.5)

    points[conf < min_confidence] = np.nan
    return points, conf.astype(float)


def hand_boxes_from_body(
    body_keypoints: np.ndarray,
    body_confidence: np.ndarray,
    image_shape: Tuple[int, int],
    *,
    index_map: Optional[Dict[str, int]] = None,
    min_confidence: float = 0.15,
    wrist_extension: float = 0.40,
    box_scale: float = 2.2,
    min_size: int = 40,
) -> List[Tuple[int, int, int, str]]:
    """Derive square hand crops from the wrist/elbow/shoulder chain.

    Returns ``(x, y, size, hand)`` boxes clipped to the image. The box is
    centered slightly beyond the wrist along the forearm, which is where the
    hand actually is.
    """
    from .skeleton import BODY25B

    idx = index_map or {name: BODY25B.index(name) for name in
                        ("LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist")}
    height, width = image_shape[:2]
    boxes: List[Tuple[int, int, int, str]] = []

    for hand, (s_key, e_key, w_key) in (
        ("left", ("LShoulder", "LElbow", "LWrist")),
        ("right", ("RShoulder", "RElbow", "RWrist")),
    ):
        try:
            s, e, w = idx[s_key], idx[e_key], idx[w_key]
        except KeyError:
            continue
        if min(body_confidence[s], body_confidence[e], body_confidence[w]) < min_confidence:
            continue
        shoulder, elbow, wrist = body_keypoints[s], body_keypoints[e], body_keypoints[w]
        if not np.isfinite([shoulder, elbow, wrist]).all():
            continue

        centre = wrist + wrist_extension * (wrist - elbow)
        forearm = float(np.linalg.norm(wrist - elbow))
        upper = float(np.linalg.norm(elbow - shoulder))
        size = box_scale * max(forearm, 0.9 * upper)

        x = int(round(centre[0] - size / 2.0))
        y = int(round(centre[1] - size / 2.0))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        size = int(min(size, width - x, height - y))
        if size >= min_size:
            boxes.append((x, y, size, hand))
    return boxes


class CaffePoseNet:
    """Batched multi-scale inference over an OpenPose Caffe model.

    The model files are not part of this repository; see
    ``estimation/model/README.md``. This class raises a clear error rather than
    a Caffe parse failure when they are missing.
    """

    def __init__(
        self,
        prototxt: os.PathLike,
        weights: os.PathLike,
        config: Optional[InferenceConfig] = None,
        *,
        n_outputs: Optional[int] = None,
        verbose: bool = True,
    ) -> None:
        import cv2

        self.config = config or InferenceConfig()
        prototxt, weights = Path(prototxt), Path(weights)
        missing = [str(p) for p in (prototxt, weights) if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                "model files not found: " + ", ".join(missing)
                + "\nThey are excluded from git for size and licence reasons; "
                  "see estimation/model/README.md for download instructions."
            )

        self.backend = select_backend(self.config, verbose=verbose)
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(weights))
        if self.backend.device == "cuda":
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(
                cv2.dnn.DNN_TARGET_CUDA_FP16 if self.backend.fp16 else cv2.dnn.DNN_TARGET_CUDA
            )
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            cv2.setNumThreads(0)     # let OpenCV use every core for the GEMMs
        self.n_outputs = n_outputs

    def infer_batch(self, frames: Sequence[np.ndarray]) -> np.ndarray:
        """Run multi-scale inference on a batch, returning ``(N, J, H, W)``.

        Heatmaps are averaged across scales at the frames' own resolution.
        """
        import cv2

        if not frames:
            return np.empty((0, 0, 0, 0), dtype=np.float32)
        height, width = frames[0].shape[:2]
        if any(f.shape[:2] != (height, width) for f in frames):
            raise ValueError("all frames in a batch must share the same size")

        base_w, base_h = self.config.resolve_size(width, height)
        accumulator: Optional[np.ndarray] = None
        used = 0

        for scale in self.config.scales:
            sw = int(np.ceil(base_w * scale / self.config.stride) * self.config.stride)
            sh = int(np.ceil(base_h * scale / self.config.stride) * self.config.stride)
            if sw < self.config.stride or sh < self.config.stride:
                continue

            blob = cv2.dnn.blobFromImages(
                list(frames), 1.0 / 255.0, (sw, sh), (0, 0, 0), swapRB=False, crop=False
            )
            self.net.setInput(blob)
            out = self.net.forward()                       # (N, C, h, w)
            if self.n_outputs is not None:
                out = out[:, : self.n_outputs]

            resized = np.empty((out.shape[0], out.shape[1], height, width), np.float32)
            for n in range(out.shape[0]):
                maps = np.transpose(out[n], (1, 2, 0))
                maps = cv2.resize(maps, (width, height), interpolation=cv2.INTER_CUBIC)
                resized[n] = np.transpose(maps, (2, 0, 1))

            accumulator = resized if accumulator is None else accumulator + resized
            used += 1

        if accumulator is None or used == 0:
            raise RuntimeError("no usable scales; check InferenceConfig.scales")
        return accumulator / used

    def keypoints_batch(
        self, frames: Sequence[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Keypoints and confidences for a batch: ``(N, J, 2)`` and ``(N, J)``."""
        heatmaps = self.infer_batch(frames)
        points, confs = [], []
        for n in range(heatmaps.shape[0]):
            p, c = heatmaps_to_keypoints(heatmaps[n], self.config.min_confidence)
            points.append(p)
            confs.append(c)
        return np.stack(points), np.stack(confs)
