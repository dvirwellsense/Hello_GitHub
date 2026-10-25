"""Diagnostic for slow drift: pair i-L -> i scored with points tracked DIRECTLY (as evaluate_scan.py --baseline L) or
CHAINED through every frame in between (small appearance change per step).

Body points are detected in frame i-L as in E3 (masked_abs inside body & detector region, union masks), tracked to
frame i, and scored with the encoder epipolar geometry of the two end frames (k=16 cluster score, px). Direct uses one
KLT step with the forward-backward limit of body points (0.1 px); chained uses that limit at every step.
If chained tracking lowers the noise of long baselines, their noise is tracking error; if not, it is geometry or real
slow motion. 'bed' scores bed-surface points within 150 raw px of the detector region (body dilated by 81 px removed),
chained, forward-backward 0.5 px as for background points: the bed does not breathe, so noise there is geometry.
Nothing here changes the score or any threshold.

  python scripts/chained_baseline_study.py --scan <scan_dir> --body-masks <masks_dir> --csv results/k16_2026_10/A07.csv \
      --out results/baseline_2026_10/chained/A07.csv
"""
import argparse
import csv
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

import _path  # noqa: F401
from motion.calib import Calibration
from motion.gantry import GantryModel
from motion.rigidity import cluster_score, epipolar_distance
from motion.roi import DetectorROI
from motion.scan import Scan
from motion.tracking import detect, track

G = {}


def init(a):
    G['a'] = a
    G['scan'] = Scan(a.scan)
    G['calib'] = Calibration(a.calib)
    G['gantry'] = GantryModel.load(a.gantry)
    G['roi'] = DetectorROI(a.detector_mask, G['calib'])
    G['roi'].near_raster = cv2.dilate(G['roi'].mask.astype(np.uint8), np.ones((301, 301), np.uint8)) > 0


def gray(i):
    return cv2.cvtColor(G['scan'].image(i), cv2.COLOR_BGR2GRAY)


def residual_score(p0, p1, j, i):
    a, scan, calib = G['a'], G['scan'], G['calib']
    if len(p0) < 16:
        return np.nan, len(p0)
    f = calib.focal_px
    n0, n1 = calib.undistort_points(p0) * f, calib.undistort_points(p1) * f
    Kn = np.diag([1 / f, 1 / f, 1])
    Fe = Kn.T @ G['gantry'].essential(scan.angle(j, a.sync), scan.angle(i, a.sync)) @ Kn
    return cluster_score(n0, epipolar_distance(Fe, n0, n1), 16), len(p0)


def run(i):
    a, scan, roi = G['a'], G['scan'], G['roi']
    out = dict(frame=int(scan.index[i]))
    imgs = {}
    for L in a.L:
        j = i - L
        if j < 0:
            continue
        body = cv2.imread(str(Path(a.body_masks) / f'body_{int(scan.index[j]):06d}.png'), cv2.IMREAD_GRAYSCALE)
        m = ((body > 0) & roi.mask).astype(np.uint8) * 255
        for k in range(j, i + 1):
            if k not in imgs:
                imgs[k] = gray(k)
        p0 = detect(imgs[j], mask=m, absolute_quality=True)
        # direct
        p1, ok, fb = track(imgs[j], imgs[i], p0, return_fb=True)
        ok &= fb < a.body_fb_max
        out[f'direct_L{L}'], out[f'n_direct_L{L}'] = residual_score(p0[ok], p1[ok], j, i)
        # chained
        cur, keep = p0.copy(), np.ones(len(p0), bool)
        for k in range(j, i):
            nxt, ok, fb = track(imgs[k], imgs[k + 1], cur, return_fb=True)
            keep &= ok & (fb < a.body_fb_max)
            cur = nxt
        out[f'chained_L{L}'], out[f'n_chained_L{L}'] = residual_score(p0[keep], cur[keep], j, i)
        # bed surface near the detector (rigid, no breathing), chained: same geometry, different object
        bm = (roi.near_raster & ~(cv2.dilate(body, np.ones((81, 81), np.uint8)) > 0)).astype(np.uint8) * 255
        q0 = detect(imgs[j], mask=bm, absolute_quality=True)
        cur, keep = q0.copy(), np.ones(len(q0), bool)
        for k in range(j, i):
            nxt, ok, fb = track(imgs[k], imgs[k + 1], cur, return_fb=True)
            keep &= ok & (fb < 0.5)
            cur = nxt
        out[f'bed_L{L}'], out[f'n_bed_L{L}'] = residual_score(q0[keep], cur[keep], j, i)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--body-masks', required=True)
    ap.add_argument('--csv', required=True, help='CSV of the real L=1 run: exposure window and same-angle pairs')
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--detector-mask', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--body-fb-max', type=float, default=0.1)
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--L', type=lambda v: tuple(int(x) for x in v.split(',')), default=(1, 2, 4, 8))
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    real = {int(r['frame']): r for r in csv.DictReader(open(a.csv))}
    scan = Scan(a.scan)
    Lmax = max(a.L)
    jobs = []
    for i in range(Lmax, len(scan)):
        span = [real.get(int(scan.index[k])) for k in range(i - Lmax + 1, i + 1)]
        if all(r is not None and r['exposure'] == 'True' and r['same_angle'] == 'False' for r in span):
            jobs.append(i)
    with Pool(a.workers, init, (a,)) as p:
        rows = p.map(run, jobs, chunksize=1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for L in a.L:
        for kind in ('direct', 'chained', 'bed'):
            v = np.array([r[f'{kind}_L{L}'] for r in rows], float)
            n = np.array([r[f'n_{kind}_L{L}'] for r in rows], float)
            fin = v[np.isfinite(v)]
            print(f'L={L} {kind:7s}: {len(fin)}/{len(v)} scored, median {np.median(fin):.3f}, max {fin.max():.3f} px, '
                  f'median points {np.median(n):.0f}')


if __name__ == '__main__':
    main()
