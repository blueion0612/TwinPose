"""2D body and hand keypoint detection CLI.

Runs the OpenPose BODY_25B model over both estimation clips and the OpenPose
hand model over crops derived from the detected wrists, writing the JSON files
the 3D stage consumes.

    python estimation/Openpose.py --task_number 30 --trial_number 1
    python estimation/Openpose.py --task_number 30 --trial_number 1 --batch 16 --preview

The model weights are not in this repository -- see ``estimation/model/README.md``.

A note on speed
---------------
This stage dominates the pipeline's runtime. Before assuming it is using your
GPU, read what the tool prints on startup: the ``opencv-contrib-python`` wheel
from PyPI is built **without CUDA**, so ``DNN_BACKEND_CUDA`` silently runs on
the CPU. :mod:`pose3d.inference` detects and reports this rather than leaving
you to wonder why a machine with a fast GPU takes an hour.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from pose3d.inference import (
    CaffePoseNet,
    InferenceConfig,
    describe_backend,
    hand_boxes_from_body,
    heatmaps_to_keypoints,
)
from pose3d.kpio import save_body_keypoints, save_hand_keypoints
from pose3d.skeleton import BODY25B_RAW_NAMES, HAND_N_JOINTS, HANDS_N_JOINTS, hand_slice
from pose3d.video import probe, read_frames, write_video

MODEL_DIR = Path(__file__).resolve().parent / "model"
BODY_PROTO = MODEL_DIR / "BODY_25B" / "pose_deploy.prototxt"
BODY_WEIGHTS = MODEL_DIR / "BODY_25B" / "pose_iter_636000.caffemodel"
HAND_PROTO = MODEL_DIR / "hand" / "pose_deploy.prototxt"
HAND_WEIGHTS = MODEL_DIR / "hand" / "pose_iter_120000.caffemodel"

N_BODY_JOINTS = 25          # BODY_25B, before MidHip is synthesised
N_HAND_OUTPUTS = 22         # 21 joints plus a background channel

#: Skeleton used only for the preview overlay.
PREVIEW_EDGES = [
    (17, 5), (17, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4), (17, 18), (11, 17), (12, 17),
    (5, 11), (6, 12),
]


def draw_overlay(frame, kpts, confs, hands=None):
    """Draw the detected skeleton on a copy of the frame."""
    import cv2

    out = frame.copy()
    for a, b in PREVIEW_EDGES:
        if a < len(kpts) and b < len(kpts) and np.isfinite(kpts[a]).all() and np.isfinite(kpts[b]).all():
            cv2.line(out, tuple(kpts[a].astype(int)), tuple(kpts[b].astype(int)), (0, 255, 0), 2)
    for i, pt in enumerate(kpts):
        if np.isfinite(pt).all():
            cv2.circle(out, tuple(pt.astype(int)), 3, (0, 200, 255), -1)
            cv2.putText(out, f"{confs[i]:.2f}", tuple((pt + 5).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)
    if hands is not None:
        for pt in hands:
            if np.isfinite(pt).all():
                cv2.circle(out, tuple(pt.astype(int)), 3, (255, 100, 0), -1)
    return out


def run_camera(
    video: Path,
    body_net: CaffePoseNet,
    hand_net: CaffePoseNet | None,
    batch: int,
    preview_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect body and hand keypoints across one clip."""
    import cv2

    info = probe(video)
    all_kpts = np.full((info.n_frames, N_BODY_JOINTS, 2), np.nan)
    all_conf = np.full((info.n_frames, N_BODY_JOINTS), np.nan)
    hand_kpts = np.full((info.n_frames, HANDS_N_JOINTS, 2), np.nan)
    hand_conf = np.full((info.n_frames, HANDS_N_JOINTS), np.nan)

    writer_frames = [] if preview_path else None
    buffer: list[tuple[int, np.ndarray]] = []

    def flush() -> None:
        if not buffer:
            return
        indices = [i for i, _ in buffer]
        frames = [f for _, f in buffer]
        points, confs = body_net.keypoints_batch(frames)
        for k, idx in enumerate(indices):
            all_kpts[idx] = points[k][:N_BODY_JOINTS]
            all_conf[idx] = confs[k][:N_BODY_JOINTS]
            if hand_net is not None:
                _detect_hands(hand_net, frames[k], all_kpts[idx], all_conf[idx],
                              hand_kpts, hand_conf, idx)
            if writer_frames is not None:
                writer_frames.append(
                    draw_overlay(frames[k], all_kpts[idx], all_conf[idx],
                                 hand_kpts[idx] if hand_net is not None else None)
                )
        buffer.clear()

    for index, frame in tqdm(read_frames(video), total=info.n_frames,
                             desc=video.stem, unit="f"):
        buffer.append((index, frame))
        if len(buffer) >= batch:
            flush()
    flush()

    if preview_path and writer_frames:
        write_video(preview_path, writer_frames, info.fps)
    return all_kpts, all_conf, hand_kpts, hand_conf


def _detect_hands(hand_net, frame, body_kpts, body_conf, hand_kpts, hand_conf, index):
    """Crop each detected hand and run the hand model over the crops."""
    import cv2

    boxes = hand_boxes_from_body(
        body_kpts, np.nan_to_num(body_conf, nan=0.0), frame.shape,
        index_map={name: BODY25B_RAW_NAMES.index(name) for name in
                   ("LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist")},
        min_confidence=hand_net.config.min_confidence,
    )
    if not boxes:
        return
    crops, metas = [], []
    for x, y, size, hand in boxes:
        crop = frame[y:y + size, x:x + size]
        if crop.size == 0:
            continue
        crops.append(cv2.resize(crop, (368, 368)))
        metas.append((x, y, size, hand))
    if not crops:
        return

    heatmaps = hand_net.infer_batch(crops)
    for k, (x, y, size, hand) in enumerate(metas):
        points, confs = heatmaps_to_keypoints(
            heatmaps[k][:HAND_N_JOINTS], hand_net.config.min_confidence
        )
        scale = size / 368.0
        points = points * scale + np.array([x, y])
        sl = hand_slice(hand)
        hand_kpts[index, sl] = points
        hand_conf[index, sl] = confs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task_number", type=int, required=True)
    ap.add_argument("--trial_number", type=int, required=True)
    ap.add_argument("--batch", type=int, default=8,
                    help="frames per forward pass (raise it if you have VRAM/RAM)")
    ap.add_argument("--net_height", type=int, default=736,
                    help="network input height; the width follows the aspect ratio")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.75],
                    help="multi-scale factors applied to the network input size")
    ap.add_argument("--min_confidence", type=float, default=0.15)
    ap.add_argument("--cpu", action="store_true", help="force CPU inference")
    ap.add_argument("--no_hands", action="store_true", help="skip hand detection")
    ap.add_argument("--preview", action="store_true",
                    help="also write overlay videos next to the keypoints")
    ap.add_argument("--check_only", action="store_true",
                    help="report the inference device and exit")
    args = ap.parse_args()

    if args.check_only:
        print(describe_backend(prefer_gpu=not args.cpu))
        return 0

    trial_dir = ROOT / "project" / f"task{args.task_number}" / f"trial{args.trial_number}"
    out_dir = trial_dir / "2D"
    out_dir.mkdir(parents=True, exist_ok=True)
    videos = [trial_dir / "Estimation" / f"cam{i}.mp4" for i in range(2)]

    for path in videos:
        if not path.is_file():
            print(f"error: {path} not found. Run the synchronisation stage first.",
                  file=sys.stderr)
            return 1

    config = InferenceConfig(
        net_resolution=(-1, args.net_height),
        scales=tuple(args.scales),
        min_confidence=args.min_confidence,
        batch_size=args.batch,
        prefer_gpu=not args.cpu,
    )
    try:
        body_net = CaffePoseNet(BODY_PROTO, BODY_WEIGHTS, config, n_outputs=N_BODY_JOINTS)
        hand_net = (
            None if args.no_hands
            else CaffePoseNet(HAND_PROTO, HAND_WEIGHTS, config, n_outputs=N_HAND_OUTPUTS,
                              verbose=False)
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for i, video in enumerate(videos):
        print(f"\n[camera{i}] {video.name}")
        t0 = time.perf_counter()
        preview = out_dir / f"cam{i}_overlay.mp4" if args.preview else None
        kpts, conf, hkpts, hconf = run_camera(video, body_net, hand_net, args.batch, preview)
        elapsed = time.perf_counter() - t0

        save_body_keypoints(out_dir / f"kpts_cam{i}_all.json", kpts, conf)
        save_hand_keypoints(out_dir / f"hands_cam{i}_all.json", hkpts, hconf)
        n = kpts.shape[0]
        print(f"  {n} frames in {elapsed:.1f}s ({n / max(elapsed, 1e-9):.2f} fps)")
        print(f"  body keypoints detected on {np.isfinite(kpts[..., 0]).mean() * 100:.1f}% "
              f"of joint slots")
        if hand_net is not None:
            print(f"  hand keypoints detected on "
                  f"{np.isfinite(hkpts[..., 0]).mean() * 100:.1f}% of joint slots")

    print(f"\nWritten to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
