"""Check motion/realtime.py against evaluate_scan.py and time it per frame.

Feeds a scan frame by frame to MotionMonitor (saved body masks, exposure flags and angles as in the evaluate_scan.py
run) and compares the scores and alarms with that run's CSV; prints the processing time per frame (image decoding
and the body mask not included: the mask is timed separately in profile_realtime.py).

  python scripts/realtime_check.py --scan <dir> --body-masks <dir> --csv results/k16_chain_2026_10/A44.csv
"""
import argparse
import csv
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


def fl(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--body-masks', required=True)
    ap.add_argument('--csv', required=True)
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--detector-mask', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--sync', type=float, default=0.9)
    a = ap.parse_args()
    scan, calib = Scan(a.scan), Calibration(a.calib)
    mon = MotionMonitor(calib, GantryModel.load(a.gantry), DetectorROI(a.detector_mask, calib))
    ref = {int(r['frame']): r for r in csv.DictReader(open(a.csv))}
    worst = dict(score=0.0, chain=0.0)
    mism, times = [], []
    for i in range(len(scan)):
        fr = int(scan.index[i])
        img = scan.image(i)
        body = cv2.imread(str(Path(a.body_masks) / f'body_{fr:06d}.png'), cv2.IMREAD_GRAYSCALE)
        if body is None:  # last frame: no pair starts there, its mask is never used
            body = np.zeros(img.shape[:2], np.uint8)
        exp = ref[fr]['exposure'] == 'True' if fr in ref else False
        t = time.perf_counter()
        out = mon.push(img, body, float(scan.angle(i, a.sync)), exp)
        times.append(time.perf_counter() - t)
        if out is None:
            continue
        r = ref[fr]
        col = 'static_cl' if out['same_angle'] else 'epi_enc_cl16'
        for key, mine, theirs in (('score', out.get(col, np.nan), fl(r[col])), ('chain', out.get('chain_cl16', np.nan), fl(r['chain_cl16']))):
            if np.isfinite(mine) != np.isfinite(theirs):
                mism.append((fr, key, mine, theirs))
            elif np.isfinite(mine):
                worst[key] = max(worst[key], abs(mine - theirs))
        for key in ('alarm_adjacent', 'alarm_chain', 'alarm'):
            if bool(out[key]) != (r[key] == 'True'):
                mism.append((fr, key, out[key], r[key]))
    t = 1000 * np.array(times[1:])
    print(f'{Path(a.csv).stem}: {len(t)} pairs; max |diff| adjacent score {worst["score"]:.2e} px, chained {worst["chain"]:.2e} px; '
          f'mismatches {len(mism)} {mism[:5]}')
    print(f'   time per frame (ms): median {np.median(t):.1f}, 95th pct {np.percentile(t, 95):.1f}, max {t.max():.1f}')


if __name__ == '__main__':
    main()
