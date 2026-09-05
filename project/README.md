# project/ — data layout

Recorded videos and generated results live here. Large files (videos, keypoints,
reconstructions) are excluded from git; only the small calibration parameters
and evaluation metrics from `task30` are kept, as format samples and as the
camera model the synthetic benchmark projects through.

A **task** is one camera placement — `task30` means the two phones were about
30° apart. A **trial** is one recording at that placement. Intrinsics belong to
the task (they depend on the phones, not on where they stand); extrinsics belong
to a trial (they change whenever a camera moves).

```
project/
└── task<N>/                      # one camera placement, e.g. task30 = 30 degrees
    ├── mono0.mp4                 # [in]  intrinsics clip, camera 0 (~1 min)
    ├── mono1.mp4                 # [in]  intrinsics clip, camera 1 (~1 min)
    ├── camera_parameters/        # [out] calibration
    │   ├── camera0_intrinsics.json     K and distortion coefficients
    │   ├── camera1_intrinsics.json
    │   ├── camera0_extrinsics.json     identity: camera 0 defines the world
    │   └── camera1_extrinsics.json     R, t (cm) and best_offset
    ├── evaluation_metrics.json   # [out] one entry per trial
    └── trial<M>/
        ├── stereo0.mp4           # [in]  stereo clip with green-flash sync, camera 0
        ├── stereo1.mp4           # [in]  stereo clip with green-flash sync, camera 1
        ├── synchronized/         # [out] aligned calibration clips + sync_report.json
        ├── Estimation/           # [out] cam0.mp4 / cam1.mp4, aligned, for pose
        ├── 2D/                   # [out] kpts_cam*_all.json, hands_cam*_all.json
        └── 3D/                   # [out] per-stage reconstructions, final result,
                                  #       wrist kinematics, run report
```

## Units

Translations in `camera*_extrinsics.json` are in **centimetres**, because
`checkerboard_box_size_scale` in `calibration/calibration_settings.yaml` is given
in centimetres. `pose3d.camera.load_camera_pair` converts to meters on load, and
that is the only place the conversion happens — every 3D coordinate in the
pipeline is in meters.

## `best_offset`

`camera1_extrinsics.json` carries the inter-camera frame offset the calibration
offset search chose. Read it with `pose3d.camera.extract_frame_offset`, which
also accepts the older `frame_offset` and `offset` spellings.

## Recording protocol

Per task, once:

- `mono0.mp4` / `mono1.mp4` — about a minute each, moving the checkerboard slowly
  through the **whole frame**, including the corners, at a range of distances and
  tilts. Corner coverage is what determines the principal point; a board that
  stays near the center leaves it almost unconstrained, and the reprojection RMS
  will not tell you.

Per trial:

- `stereo0.mp4` / `stereo1.mp4` — start both recordings, flash a green light
  three or four times where both cameras can see it, then present the
  checkerboard to **both** cameras for the number of seconds you will pass as
  `--skip_start`, then step back and perform the motion. Open with a 5-second
  T-pose: the pipeline measures your bone lengths from it.

Run `python tools/inspect_footage.py --video project/task30/mono0.mp4` before
calibrating; it reports how often the board is actually detectable and how the
sharpness threshold would filter the clip.
