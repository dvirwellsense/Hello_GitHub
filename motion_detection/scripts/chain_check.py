"""Second check for slow drift and step-and-hold motion: pair i-L -> i (L=4) with points tracked in a chain.

The adjacent-pair score (evaluate_scan.py) sees about 1.1-1.5 mm per frame interval. A drift below that is never seen,
and a motion that ends within one frame interval gives one suspect pair (no alarm with persistence 2). Here the body
points of frame i-L (E3 detection inside body & detector region, union masks) are tracked through every frame in
between (KLT, forward-backward 0.1 px per step) and scored with the encoder epipolar geometry of the two end frames
(k=16 cluster score, px), as in chained_baseline_study.py.

  score   every pair i-L -> i whose frames are all inside the exposure window with the gantry moving
          -> CSV: frame (= i), chain_cl16, n_chain
  inject  static scans: synthetic motion in every second such pair (frame a and its mask untouched), then scored
            drift  frames i-L+k shifted by k*v (k = 1..L), v = --rates mm per frame
            step   frames i-L+2 .. i shifted by s (a step in the middle of the chain, then held), s = --steps mm
          region r80 (disc of 80 raw px around a random body point above the detector) or body; direction 0 or 60 deg
          to the epipolar line of the end frames; sizes in mm at the bed-plane scale, as in inject_motion.py.

  python scripts/chain_check.py score --scan <dir> --body-masks <dir> --csv results/k16_2026_10/A07.csv --out <csv>
  python scripts/chain_check.py inject --scan <dir> --body-masks <dir> --csv results/k16_2026_10/A07.csv --out <csv>
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

FEATHER_PX = 4
G = {}


def init(a):
    G['a'] = a
    G['scan'] = Scan(a.scan)
    G['calib'] = Calibration(a.calib)
    G['gantry'] = GantryModel.load(a.gantry)
    G['roi'] = DetectorROI(a.detector_mask, G['calib'])


def fe(j, i):
    a, f = G['a'], G['calib'].focal_px
    Kn = np.diag([1 / f, 1 / f, 1])
    return Kn.T @ G['gantry'].essential(G['scan'].angle(j, a.sync), G['scan'].angle(i, a.sync)) @ Kn


def chain_score(imgs, j, i, body):
    """imgs: dict frame -> gray image (possibly modified)."""
    a, calib = G['a'], G['calib']
    m = ((body > 0) & G['roi'].mask).astype(np.uint8) * 255
    p0 = detect(imgs[j], mask=m, absolute_quality=True)
    cur, keep = p0.copy(), np.ones(len(p0), bool)
    for k in range(j, i):
        nxt, ok, fb = track(imgs[k], imgs[k + 1], cur, return_fb=True)
        keep &= ok & (fb < a.body_fb_max)
        cur = nxt
    if keep.sum() < 16:
        return np.nan, int(keep.sum())
    f = calib.focal_px
    n0, n1 = calib.undistort_points(p0[keep]) * f, calib.undistort_points(cur[keep]) * f
    return cluster_score(n0, epipolar_distance(fe(j, i), n0, n1), 16), int(keep.sum())


def load(j, i):
    scan = G['scan']
    imgs = {k: cv2.cvtColor(scan.image(k), cv2.COLOR_BGR2GRAY) for k in range(j, i + 1)}
    body = cv2.imread(str(Path(G['a'].body_masks) / f'body_{int(scan.index[j]):06d}.png'), cv2.IMREAD_GRAYSCALE)
    return imgs, body


def run_score(i):
    j = i - G['a'].L
    imgs, body = load(j, i)
    s, n = chain_score(imgs, j, i, body)
    return dict(frame=int(G['scan'].index[i]), chain_cl16=s, n_chain=n)


def shifter(img, weight, pix, u, d_px):
    """Translate the weighted region of img by d_px undistorted px along unit vector u (see inject_motion.warp)."""
    calib = G['calib']
    f = calib.focal_px
    ys, xs, n = pix
    src = calib.distort_points((n - d_px * u * weight[ys, xs, None]) / f)
    mx, my = np.meshgrid(np.arange(img.shape[1], dtype=np.float32), np.arange(img.shape[0], dtype=np.float32))
    mx[ys, xs], my[ys, xs] = src[:, 0], src[:, 1]
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def run_inject(job):
    i, seed = job
    a, scan, calib = G['a'], G['scan'], G['calib']
    L = a.L
    j = i - L
    imgs, body = load(j, i)
    base = dict(frame=int(scan.index[i]))
    s, n = chain_score(imgs, j, i, body)
    rows = [dict(base, kind='none', region='none', angle_deg=0, size_mm=0.0, chain_cl16=s, n_chain=n)]
    full = cv2.dilate(body, np.ones((21, 21), np.uint8)) > 0
    ys, xs = np.nonzero((body > 0) & G['roi'].mask)
    if len(xs) == 0:
        return rows
    k = np.random.default_rng(seed).integers(len(xs))
    cx, cy = int(xs[k]), int(ys[k])
    f = calib.focal_px
    for region in a.regions:
        if region == 'body':
            m = full.astype(np.float32)
            centre = (float(np.mean(xs)), float(np.mean(ys)))
        else:
            disc = np.zeros(body.shape, np.uint8)
            cv2.circle(disc, (cx, cy), int(region[1:]), 1, -1)
            m = (disc.astype(bool) & full).astype(np.float32)
            centre = (cx, cy)
        w = cv2.GaussianBlur(m, (0, 0), FEATHER_PX / 2) * m.max()
        w[w < 0.01] = 0
        py, px = np.nonzero(w > 0)
        pix = (py, px, calib.undistort_points(np.c_[px, py].astype(np.float64)) * f)
        nc = calib.undistort_points(np.array([centre]))[0] * f
        line = np.r_[nc, 1] @ fe(j, i).T
        normal = line[:2] / np.hypot(*line[:2])
        tangent = np.array([-normal[1], normal[0]])
        for ang in a.angles:
            t = np.radians(ang)
            u = np.cos(t) * normal + np.sin(t) * tangent
            cases = [('drift', v, [kk * v for kk in range(1, L + 1)]) for v in a.rates]
            cases += [('step', st, [0.0 if kk < 2 else st for kk in range(1, L + 1)]) for st in a.steps]
            for kind, size, disp in cases:
                mod = dict(imgs)
                for kk, d in zip(range(1, L + 1), disp):
                    if d:
                        mod[j + kk] = shifter(imgs[j + kk], w, pix, u, d * calib.pixels_per_mm)
                s, n = chain_score(mod, j, i, body)
                rows.append(dict(base, kind=kind, region=region, angle_deg=ang, size_mm=size, chain_cl16=s, n_chain=n))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mode', choices=('score', 'inject'))
    ap.add_argument('--scan', required=True)
    ap.add_argument('--body-masks', required=True)
    ap.add_argument('--csv', required=True, help='CSV of the real adjacent-pair run: exposure window and same-angle pairs')
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--detector-mask', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--body-fb-max', type=float, default=0.1)
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--L', type=int, default=4)
    ap.add_argument('--rates', type=lambda v: tuple(float(x) for x in v.split(',')), default=(0.25, 0.5, 0.75, 1.0))
    ap.add_argument('--steps', type=lambda v: tuple(float(x) for x in v.split(',')), default=(1.0, 2.0, 3.0, 4.0, 5.0))
    ap.add_argument('--angles', type=lambda v: tuple(int(x) for x in v.split(',')), default=(0, 60))
    ap.add_argument('--regions', type=lambda v: tuple(v.split(',')), default=('r80', 'body'))
    ap.add_argument('--every', type=int, default=2)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    real = {int(r['frame']): r for r in csv.DictReader(open(a.csv))}
    scan = Scan(a.scan)
    elig = []
    for i in range(a.L, len(scan)):
        span = [real.get(int(scan.index[k])) for k in range(i - a.L + 1, i + 1)]
        if all(r is not None and r['exposure'] == 'True' and r['same_angle'] == 'False' for r in span):
            elig.append(i)
    with Pool(a.workers, init, (a,)) as p:
        if a.mode == 'score':
            rows = p.map(run_score, elig, chunksize=1)
        else:
            rows = [r for rr in p.map(run_inject, [(i, int(scan.index[i])) for i in elig[::a.every]], chunksize=1) for r in rr]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, restval='')
        w.writeheader()
        w.writerows(rows)
    v = np.array([r['chain_cl16'] for r in rows if r.get('kind', 'none') == 'none'], float)
    print(f'{a.mode}: {len(rows)} rows; no-injection pairs {len(v)}, median {np.nanmedian(v):.3f}, max {np.nanmax(v):.3f} px')


if __name__ == '__main__':
    main()
