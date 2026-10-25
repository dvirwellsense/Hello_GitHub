"""Time per frame of each step of the pipeline, to compare with the camera interval (~0.25 s).

Runs the steps of evaluate_scan.py (defaults of October 2026: E3, --roi-fixed, k=16, chained second check L=4) one by
one on consecutive frames of a scan and reports the median time per frame of each step, single process. Also times
segmentation variants (YOLOv8 m / n, image size). The machine is the cloud container (4 cores, Xeon 2.1 GHz); the
target computer will differ, so the ratios matter more than the absolute numbers.

  python scripts/profile_realtime.py --scan <dir> --body-masks <dir> --first 50 --frames 30
"""
import argparse
import time
from pathlib import Path

import cv2
import numpy as np

import _path  # noqa: F401
import motion.masks as masks
from motion.calib import Calibration
from motion.gantry import GantryModel
from motion.rigidity import chained_score, cluster_score, epipolar_distance
from motion.roi import DetectorROI
from motion.scan import Scan
from motion.tracking import detect, track


def timed(store, key, fn, *args, **kw):
    t = time.perf_counter()
    out = fn(*args, **kw)
    store.setdefault(key, []).append(time.perf_counter() - t)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--body-masks', required=True)
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--detector-mask', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--first', type=int, default=50)
    ap.add_argument('--frames', type=int, default=30)
    ap.add_argument('--seg', default='yolov8m-seg.pt:960,yolov8m-seg.pt:640,yolov8n-seg.pt:960,yolov8n-seg.pt:640',
                    help='weights:imgsz list to time ("" = none)')
    a = ap.parse_args()
    scan, calib, gantry = Scan(a.scan), Calibration(a.calib), GantryModel.load(a.gantry)
    roi = DetectorROI(a.detector_mask, calib)
    region = roi.mask.astype(np.uint8) * 255
    f = calib.focal_px
    Kn = np.diag([1 / f, 1 / f, 1])
    T = {}
    rng = range(a.first, a.first + a.frames)
    grays = {}
    for i in rng:
        img = timed(T, '1 read image', scan.image, i)
        grays[i] = timed(T, '2 grey', cv2.cvtColor, img, cv2.COLOR_BGR2GRAY)
        if i - 1 not in grays:
            continue
        g0, g1 = grays[i - 1], grays[i]
        body = cv2.imread(str(Path(a.body_masks) / f'body_{int(scan.index[i - 1]):06d}.png'), cv2.IMREAD_GRAYSCALE)
        # background points (as in evaluate_pair): needed only for the bed guard, the ROI homography and dwell pairs
        p0 = timed(T, '3 background: detect (3000 corners, whole image)', detect, g0)
        timed(T, '4 background: track forward+backward', track, g0, g1, p0, return_fb=True)
        # body points above the detector (E3)
        m = ((body > 0) & (region > 0)).astype(np.uint8) * 255
        q0 = timed(T, '5 body: detect (masked_abs)', detect, g0, mask=m, absolute_quality=True)
        q1, ok, fb = timed(T, '6 body: track forward+backward', track, g0, g1, q0, return_fb=True)
        ok &= fb < 0.1

        def score():
            n0, n1 = calib.undistort_points(q0[ok]) * f, calib.undistort_points(q1[ok]) * f
            Fe = Kn.T @ gantry.essential(scan.angle(i - 1), scan.angle(i)) @ Kn
            return cluster_score(n0, epipolar_distance(Fe, n0, n1), 16)
        timed(T, '7 body: undistort + epipolar + k=16 score', score)
        if i - 4 in grays:
            b4 = cv2.imread(str(Path(a.body_masks) / f'body_{int(scan.index[i - 4]):06d}.png'), cv2.IMREAD_GRAYSCALE)
            timed(T, '8 second check: chained i-4 -> i', chained_score, [grays[k] for k in range(i - 4, i + 1)], b4,
                  np.zeros_like(b4), region, calib, gantry, scan.angle(i - 4), scan.angle(i))
    print(f'Scan {Path(a.scan).name}, frames {a.first}-{a.first + a.frames - 1}; median time per frame (ms)')
    tot = 0
    for k in sorted(T):
        v = 1000 * np.median(T[k])
        tot += v
        print(f'  {k:55s} {v:7.1f}')
    core = sum(1000 * np.median(T[k]) for k in T if k[0] in '125678')
    print(f'  {"all of the above":55s} {tot:7.1f}')
    print(f'  {"without background points (steps 3-4)":55s} {core:7.1f}')
    for item in [s for s in a.seg.split(',') if s]:
        w, sz = item.split(':')
        if not Path(w).exists():
            print(f'  segmentation {w} missing, skipped')
            continue
        from ultralytics import YOLO
        model = YOLO(w)
        ts = []
        for n, i in enumerate(rng):
            img = scan.image(i)
            rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            t = time.perf_counter()
            r = model.predict(rot, classes=[0], conf=0.1, imgsz=int(sz), retina_masks=True, verbose=False)[0]
            if r.masks is not None and len(r.masks):
                mm = (r.masks.data.cpu().numpy() > 0.5).any(axis=0).astype(np.uint8) * 255
                mm = cv2.resize(mm, (rot.shape[1], rot.shape[0]), interpolation=cv2.INTER_NEAREST)
                cv2.rotate(mm, cv2.ROTATE_90_COUNTERCLOCKWISE)
            if n >= 2:  # skip warm-up
                ts.append(time.perf_counter() - t)
        print(f'  segmentation {w} imgsz {sz} (union, conf 0.1): {1000 * np.median(ts):7.1f} ms per frame')


if __name__ == '__main__':
    main()
