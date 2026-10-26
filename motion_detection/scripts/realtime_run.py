"""End-to-end real-time run, frame by frame: decode the image, body mask (computed every N frames, reused in between),
MotionMonitor (motion/realtime.py). Writes one row per frame with the outputs and the time of each step.

Times: decode = JPEG decoding from memory (the file is read before the clock starts), mask = segmentation + erosion on
the frames where it is computed (0 otherwise), monitor = MotionMonitor.push; total = their sum.
Body mask: YOLOv8-seg person, union of all detections with conf >= 0.1, image rotated 90 deg clockwise, eroded by 21 px
(same as evaluate_scan.py --seg-select union --seg-conf 0.1). It is computed for frame f when f % N == 0, or when frame
f - f % N does not exist in the scan; otherwise the mask of frame f - f % N is reused (same rule as
results/mask_reuse_2026_10). --body-masks uses saved masks instead (no segmentation timed).
The exposure flag of each pair comes from --csv (an evaluate_scan.py run of the same scan); in the real system it would
come from the X-ray controller.

  python scripts/realtime_run.py --scan <dir> --csv <run.csv> --seg-weights yolov8n-seg.pt --seg-imgsz 640 --mask-every 4 --out <csv>
"""
import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

import _path  # noqa: F401
from motion.calib import Calibration
from motion.gantry import GantryModel
from motion.realtime import MotionMonitor
from motion.roi import DetectorROI
from motion.scan import Scan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--detector-mask', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--seg-weights', default='yolov8n-seg.pt')
    ap.add_argument('--seg-imgsz', type=int, default=640)
    ap.add_argument('--seg-conf', type=float, default=0.1)
    ap.add_argument('--mask-every', type=int, default=4)
    ap.add_argument('--erode', type=int, default=21)
    ap.add_argument('--body-masks', help='use saved masks (body_<frame>.png) instead of segmentation')
    ap.add_argument('--save-masks', help='folder: save the masks used')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    scan, calib = Scan(a.scan), Calibration(a.calib)
    mon = MotionMonitor(calib, GantryModel.load(a.gantry), DetectorROI(a.detector_mask, calib))
    ref = {int(r['frame']): r['exposure'] == 'True' for r in csv.DictReader(open(a.csv))}
    model = None
    if not a.body_masks:
        from ultralytics import YOLO
        model = YOLO(a.seg_weights)
        model.predict(np.zeros((1280, 720, 3), np.uint8), classes=[0], conf=a.seg_conf, imgsz=a.seg_imgsz, verbose=False)  # warm-up
    have = set(int(x) for x in scan.index)
    if a.save_masks:
        Path(a.save_masks).mkdir(parents=True, exist_ok=True)
    kern = np.ones((a.erode, a.erode), np.uint8)
    rows, mask, mask_from = [], None, None
    for i in range(len(scan)):
        fr = int(scan.index[i])
        data = np.frombuffer(Path(scan.files[i]).read_bytes(), np.uint8)  # the camera delivers the JPEG in memory: file read not timed
        t0 = time.perf_counter()
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        t1 = time.perf_counter()
        g = fr - fr % a.mask_every
        src = g if g in have else fr
        computed = False
        if a.body_masks:
            mask = cv2.imread(str(Path(a.body_masks) / f'body_{fr:06d}.png'), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros(img.shape[:2], np.uint8)
            mask_from = fr
        elif src != mask_from:
            rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            r = model.predict(rot, classes=[0], conf=a.seg_conf, imgsz=a.seg_imgsz, retina_masks=True, verbose=False)[0]
            m = np.zeros(rot.shape[:2], np.uint8)
            if r.masks is not None and len(r.masks):
                m = (r.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8) * 255
                m = cv2.resize(m, (rot.shape[1], rot.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = cv2.erode(cv2.rotate(m, cv2.ROTATE_90_COUNTERCLOCKWISE), kern)
            mask_from, computed = src, True
        t2 = time.perf_counter()
        out = mon.push(img, mask, float(scan.angle(i, a.sync)), ref.get(fr, False))
        t3 = time.perf_counter()
        if a.save_masks:
            cv2.imwrite(str(Path(a.save_masks) / f'body_{fr:06d}.png'), mask)
        row = dict(frame=fr, mask_from_frame=mask_from, mask_computed=int(computed), t_decode_ms=1000 * (t1 - t0),
                   t_mask_ms=1000 * (t2 - t1), t_monitor_ms=1000 * (t3 - t2), t_total_ms=1000 * (t3 - t0))
        if out is not None:
            row.update({k: v for k, v in out.items()})
        rows.append(row)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, restval='')
        w.writeheader()
        w.writerows(rows)
    try:
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    env = dict(command=' '.join(os.sys.argv), git_commit=commit, python=platform.python_version(), opencv=cv2.__version__,
               numpy=np.__version__, cv2_threads=cv2.getNumThreads(), cpu_count=os.cpu_count(),
               OMP_NUM_THREADS=os.environ.get('OMP_NUM_THREADS'))
    if model is not None:
        import torch
        import ultralytics
        env.update(torch=torch.__version__, torch_threads=torch.get_num_threads(), ultralytics=ultralytics.__version__)
    Path(a.out).with_suffix('.settings.json').write_text(json.dumps(env, indent=2))
    t = np.array([r['t_total_ms'] for r in rows[1:]])
    c = np.array([r['mask_computed'] for r in rows[1:]], bool)
    print(f'{Path(a.out).stem}: {len(rows)} frames; total ms mean {t.mean():.1f} p95 {np.percentile(t, 95):.1f} max {t.max():.1f}; '
          f'mask frames {c.sum()} (mean {t[c].mean() if c.any() else float("nan"):.1f}), reuse frames {(~c).sum()} (mean {t[~c].mean():.1f})')


if __name__ == '__main__':
    main()
