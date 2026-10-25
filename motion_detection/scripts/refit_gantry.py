"""Refit the gantry camera model on long baselines, with optional deviations from the ideal circle.

Why: the epipolar residual of things that cannot move (rigid phantom, bed, floor) grows with the arc angle between the
two frames (results/baseline_2026_10). Between adjacent frames (~0.5 deg) the error is negligible; at 4 deg it is
not. The original model (data/gantry_arc22.json) was fitted once on the phantom scan.

Data: static points tracked in a chain through every frame of the exposure window (KLT, forward-backward 0.5 px),
from every second frame j to j+L, L = 1, 2, 4, 8, 16. Static points: everything in the phantom scan; in the volunteer
static scans everything outside the body (union body mask dilated by 81 px, as for background points in the score).
Objective: signed epipolar distance (px, undistorted calibration frame) of these correspondences, robust loss.

Variants, each adding to the previous one (one change at a time):
  V0  original model, no fit
  V1  camera pose refitted (6 numbers, same gauge as fit_gantry.py), sync fixed at 0.9
  V2  + sync (where the image sits between the angle reads before and after the capture)
  V3  + encoder scale
  V4  + camera pose deviation linear in the angle (6 numbers: wobble / flex)
  V5  + the same, quadratic in the angle (6 numbers)
Two folds: fit on one group of scans, report on the other. Nothing here changes the default pipeline.

  python scripts/refit_gantry.py --scans scans.txt --cache <dir> --out results/gantry_refit_2026_10
scans.txt: name|scan_dir|body_masks_dir or '-' (all static)|csv of an L=1 run (exposure window)
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

import _path  # noqa: F401
from motion.calib import Calibration
from motion.gantry import GantryModel, skew
from motion.scan import Scan
from motion.tracking import detect, track

LS = (1, 2, 4, 8, 16)
VARIANTS = ('V0', 'V1', 'V2', 'V3', 'V4', 'V5')
FOLDS = {'A': (('phantom', 'A07', 'legs_static'), ('A11', 'A27')),
         'B': (('A11', 'A27'), ('phantom', 'A07', 'legs_static'))}


def collect(name, scan_dir, masks, csv_path, calib, cache, points=600, step=2):
    out = Path(cache) / f'{name}.npz'
    if out.exists():
        return dict(np.load(out))
    scan = Scan(scan_dir)
    win = {int(r['frame']) for r in csv.DictReader(open(csv_path)) if r['exposure'] == 'True' and r['same_angle'] == 'False'}
    idx = [k for k in range(len(scan)) if int(scan.index[k]) in win]
    first, last = idx[0] - 1, idx[-1]  # frames of the window
    gray = {}

    def g(k):
        if k not in gray:
            gray[k] = cv2.cvtColor(scan.image(k), cv2.COLOR_BGR2GRAY)
        return gray[k]

    rec = {k: [] for k in ('L', 'bj', 'aj', 'bi', 'ai', 'x0', 'x1')}
    for j in range(first, last, step):
        m = None
        if masks != '-':
            body = cv2.imread(str(Path(masks) / f'body_{int(scan.index[j]):06d}.png'), cv2.IMREAD_GRAYSCALE)
            m = (cv2.dilate(body, np.ones((81, 81), np.uint8)) == 0).astype(np.uint8) * 255
        p0 = detect(g(j), points, m)
        cur, keep = p0.copy(), np.ones(len(p0), bool)
        for k in range(j, min(j + max(LS), last) + 1 - 1):
            nxt, ok = track(g(k), g(k + 1), cur)
            keep &= ok
            cur = nxt
            L = k + 1 - j
            if L in LS and keep.any():
                n = int(keep.sum())
                rec['L'] += [L] * n
                rec['bj'] += [scan.angle_before[j]] * n
                rec['aj'] += [scan.angle_after[j]] * n
                rec['bi'] += [scan.angle_before[k + 1]] * n
                rec['ai'] += [scan.angle_after[k + 1]] * n
                rec['x0'].append(calib.undistort_points(p0[keep]))
                rec['x1'].append(calib.undistort_points(cur[keep]))
        for k in [k for k in gray if k < j + step]:
            del gray[k]
    d = {k: (np.vstack(v) if k in ('x0', 'x1') else np.array(v, float)) for k, v in rec.items()}
    np.savez(out, **d)
    return d


N_EXT = dict(V0=0, V1=0, V2=0, V3=1, V4=7, V5=13)


def unpack(p, variant):
    cam, sync = p[:6], (p[6] if variant not in ('V0', 'V1') else 0.9)
    e = p[7:]
    ext = {}
    n = N_EXT[variant]
    if n >= 1:
        ext['scale'] = e[0]
    if n >= 7:
        ext['omega1'], ext['tau1'] = e[1:4], e[4:7]
    if n >= 13:
        ext['omega2'], ext['tau2'] = e[7:10], e[10:13]
    return cam, sync, ext


def residuals(model, sync, d, f):
    """Signed epipolar distance (px) of every correspondence."""
    ang_j = d['bj'] + sync * (d['aj'] - d['bj'])
    ang_i = d['bi'] + sync * (d['ai'] - d['bi'])
    Pa, Pb = model.projections(ang_j), model.projections(ang_i)
    R = Pb[:, :, :3] @ np.transpose(Pa[:, :, :3], (0, 2, 1))
    t = Pb[:, :, 3] - np.einsum('nij,nj->ni', R, Pa[:, :, 3])
    x0 = np.c_[d['x0'], np.ones(len(d['x0']))]
    x1 = np.c_[d['x1'], np.ones(len(d['x1']))]
    Rx = np.einsum('nij,nj->ni', R, x0)
    line = np.cross(t, Rx)  # E x0 = [t]x R x0
    return np.sum(line * x1, 1) / np.hypot(line[:, 0], line[:, 1]) * f


def fit(variant, base, d, f, sign):
    if variant == 'V0':
        return base, 0.9
    p0 = np.r_[base.cam, 0.9, np.zeros(N_EXT[variant])]

    def res(p):
        cam, sync, ext = unpack(p, variant)
        m = GantryModel(cam, sign, ext)
        c0 = m.camera_center()
        return np.r_[residuals(m, sync, d, f), 1e3 * c0[1], 1e3 * (np.hypot(c0[0], c0[1]) - 1), 1e3 * c0[2]]
    free = np.r_[np.ones(6, bool), variant not in ('V0', 'V1'), np.ones(N_EXT[variant], bool)]
    q0 = p0[free]

    def res_free(q):
        p = p0.copy()
        p[free] = q
        return res(p)
    sol = least_squares(res_free, q0, loss='soft_l1', f_scale=0.3, x_scale='jac', max_nfev=200)
    p = p0.copy()
    p[free] = sol.x
    cam, sync, ext = unpack(p, variant)
    return GantryModel(cam, sign, ext), sync


def stats(model, sync, d, f):
    r = np.abs(residuals(model, sync, d, f))
    return {L: (float(np.median(r[d['L'] == L])), float(np.percentile(r[d['L'] == L], 90))) for L in LS if (d['L'] == L).any()}


def merge(ds, n_max, seed=0):
    d = {k: np.concatenate([x[k] for x in ds]) for k in ds[0]}
    if len(d['L']) > n_max:
        sel = np.random.default_rng(seed).choice(len(d['L']), n_max, replace=False)
        d = {k: v[sel] for k, v in d.items()}
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scans', required=True)
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--cache', required=True)
    ap.add_argument('--fit-points', type=int, default=60000)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    calib, base = Calibration(a.calib), GantryModel.load(a.gantry)
    f = calib.focal_px
    Path(a.cache).mkdir(parents=True, exist_ok=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for line in open(a.scans):
        name, sd, masks, cp = line.strip().split('|')
        data[name] = collect(name, sd, masks, cp, calib, a.cache)
        print(f'{name}: {len(data[name]["L"])} correspondences', flush=True)
    summary = {}
    for fold, (train, test) in FOLDS.items():
        dtr = merge([data[n] for n in train], a.fit_points)
        print(f'\n== fold {fold}: fit on {", ".join(train)}; report on {", ".join(test)}')
        print('   |residual| of static points, median / 90th pct (px), per L = ' + ', '.join(map(str, LS)))
        for v in VARIANTS:
            m, sync = fit(v, base, dtr, f, base.sign)
            m.save(out / f'gantry_{fold}_{v}.json', sync=float(sync), fold=fold, variant=v, fitted_on=list(train))
            line = []
            for n in test:
                s = stats(m, sync, data[n], f)
                summary[(fold, v, n)] = s
                line.append(f'{n}: ' + ' '.join(f'{s[L][0]:.2f}/{s[L][1]:.2f}' for L in LS if L in s))
            tr = stats(m, sync, dtr, f)
            print(f'   {v} sync {sync:.3f}  train: ' + ' '.join(f'{tr[L][0]:.2f}' for L in LS if L in tr) + '  |  ' + '   '.join(line), flush=True)
    json.dump({f'{k[0]}|{k[1]}|{k[2]}': v for k, v in summary.items()}, open(out / 'summary.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
