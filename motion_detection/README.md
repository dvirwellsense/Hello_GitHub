# Patient motion detection – rigidity check prototype

Offline prototype used on phantom scan `arc22_office_20260830_121542` (static chest phantom).

Setup: `pip install opencv-python-headless numpy scipy`, then
`export SCAN_DIR=<folder with frames.csv and images/>  CALIB_JSON=<CalibrationResult_arc22.json>`.

| script | what it does |
|---|---|
| `calib.py` | Calibration frame mapping: raw 1280x720 -> resize 640x360 -> pad x by `m_iXoffset` -> 894x360, rational 8-coef model |
| `pairwise.py` | Adjacent pairs: 2D similarity residual (current method) vs epipolar residual on phantom points |
| `tracks.py FIRST LAST out.npy` | Long KLT tracks (forward-backward checked), labelled phantom / specular / background |
| `gantry.py`, `run_ba.py` | Gantry model: camera rigidly rotating about a fixed axis by the encoder angle; fits extrinsics + sync factor |
| `free_ba.py` | Bundle adjustment with a free pose per frame (image-based ego-motion) |
| `window_check.py` | Model fit quality on 8-frame sliding windows (encoder poses vs image poses) |
| `inject_epi.py` | Sensitivity: synthetic "limb" shifts injected into phantom tracks, scored with encoder-pose epipolar check |

Run order: `tracks.py 40 110 tracks_sweep.npy` -> `run_ba.py` -> `window_check.py` / `inject_epi.py`.

## Volunteer scan A28_1 (motion labelled 63–85)
`seq_eval.py SCAN_DIR person|phantom out.csv` scores every adjacent pair with: current 2D-alignment residual,
image-based epipolar distance, and encoder-pose epipolar distance (gantry model from the phantom fit, `gantry_fit.npy`).
`body.py` is a simple HSV person mask (shirt + skin) used to label body points. Result plot: `A28_1_scores.png`.
