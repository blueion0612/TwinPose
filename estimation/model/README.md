# Model files (not included in this repository)

`estimation/Openpose.py` loads the following files from this folder.
They are excluded from git because of their size and license — download them
yourself and place them exactly as below:

```
estimation/model/
├── BODY_25B/
│   ├── pose_deploy.prototxt
│   └── pose_iter_636000.caffemodel
└── hand/
    ├── pose_deploy.prototxt
    └── pose_iter_120000.caffemodel   # OpenPose hand model (a pose_iter_102000.caffemodel also works — rename it)
```

Where to get them:

- **BODY_25B (experimental OpenPose body model)**
  https://github.com/CMU-Perceptual-Computing-Lab/openpose_train/tree/master/experimental_models
  (folder `body_25b` — contains `pose_deploy.prototxt` and the download link for the caffemodel)

- **Hand model**
  Prototxt: `models/hand/pose_deploy.prototxt` in https://github.com/CMU-Perceptual-Computing-Lab/openpose
  Weights: run OpenPose's `models/getModels.sh` (or `getModels.bat`) and copy `hand/pose_iter_102000.caffemodel`.

Note: the OpenPose models are released under CMU's license
(non-commercial research use). Check the OpenPose repository for details.

## Which device will actually run them

```bash
python main.py check
```

Read what that prints before assuming the GPU is involved. The
`opencv-contrib-python` wheel on PyPI is built **without CUDA**, so requesting
`DNN_BACKEND_CUDA` on it does not fail — OpenCV falls back to the CPU silently.
The only symptom is that the stage takes roughly an order of magnitude longer,
with nothing in the output to explain why. Getting the GPU involved means
building OpenCV from source with CUDA enabled.

Everything in this project except the 2D stage — including the accuracy
benchmark, `python main.py benchmark` — runs without these weights.
