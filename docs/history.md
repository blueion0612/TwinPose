# Change history


This repository was dormant and unmaintained for some time. The reconstruction
stage had stopped working in ways nothing in its output revealed:

| Defect | Effect |
| --- | --- |
| f-string with nested same quotes | `3D_estimation.py` did not parse at all on the pinned Python 3.10 |
| Bundle adjustment results never written back | Both passes computed, printed an RMS, and were discarded |
| Hand frame given 5 rows, indexed row 17 | Every call raised `IndexError`; a bare `except` turned it into empty output, so `wrist_kinematics.json` was a list of empty dicts on every run |
| Reflected basis for the right hand | `Rotation.from_matrix` rejected it; every right-hand angle silently empty |
| `best_offset` written, `frame_offset` read | The MMPose pipeline always ran unaligned |
| `midhip_to_?hip` prior of 0.52 m | A torso length used as half a pelvis width; bootstrapped hips landed ~5× too far out |
| 26-joint index map applied to 18-joint files | `compare.py` read `LWrist` whenever it asked for `LShoulder` |
| Inter-trial metrics overwritten with NaN | Computed, then discarded two loops later |
| `pred[:, feet, [0, 2]]` | Broadcasts to `(F, 2)`; the following `norm(axis=2)` raised |
| Sync scored by a term ~100× the signal | Bone-length CV decided the offset; the search returned essentially arbitrary answers |
| Joint-limit test with the sign reversed | Penalised straight limbs instead of impossible folds; cost 2.3 mm of MPJPE |
| `cap.set` before every frame read | Video scanned dozens of times over |
| `DNN_BACKEND_CUDA` on a CUDA-less build | Silent CPU fallback with no indication |
| Distortion passed to `stereoCalibrate` | Ignored by OpenCV 5.0; baseline 12% wrong |

The keypoint schema had been re-derived by hand in four files; it now lives in
one. The MMPose variant was a 1,895-line near-copy of the main pipeline and is
now a 130-line adapter. Every defect above has a test.
