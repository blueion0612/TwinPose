# Recording guide

How to capture footage this pipeline can use.


Per **task** (one camera placement), once:

| Clip | Length | What to do |
| --- | --- | --- |
| `mono0.mp4`, `mono1.mp4` | ~1 min each | Move the checkerboard slowly through the **whole frame, corners included**, at a range of distances and tilts |

Per **trial**:

| Clip | What to do |
| --- | --- |
| `stereo0.mp4`, `stereo1.mp4` | Start both phones, flash the green torch 3–4 times in view of both, present the board to **both** cameras, then step back and move |

Open the motion with a 5-second T-pose, which is what the bone measurement uses.
Keep the subject between 1.5 m and 3.5 m. Put the files where
[`project/README.md`](project/README.md) says.

Before calibrating, check the footage is usable:

```bash
python tools/inspect_footage.py --video project/task30/mono0.mp4 --compare project/task30/mono1.mp4
```

It reports how often the board is actually detectable and how the sharpness
threshold would filter the clip, which is much cheaper than finding out after a full
calibration run.
