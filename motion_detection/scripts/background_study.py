"""Background validation for the bg guard (G3): what drives the background residual, and does the
LOCAL background residual predict the error on the body in static controls?

Per moving pair inside the exposure window:
  * body points: detected inside body & ROI (E3: masked_abs, body forward-backward <= 0.1 px)
  * background points: E0 global detection, outside the dilated body mask, split into
        bed       inside the bed outline (data/bed_home_arc22.png, carried by the bed-plane homography)
                  and consistent with the bed-plane homography (board texture)
        offplane  inside the bed outline, NOT on the bed plane (reflections in the glossy board, cables)
        floor     outside the bed outline
  * SIGNED epipolar residual of every point (encoder model), px of the undistorted half-res frame
  * predictions of the signed body residual from the background (static control => truth is 0 motion,
    so the body residual IS the error to be predicted):
        none        0
        all_const   median signed residual of all background points
        floor_const median signed residual of floor points
        bed_const   median signed residual of bed points within --near px of the ROI
        bed_plane   least-squares plane a + b*x + c*y of signed bed residual (within --near of ROI)
        bed_knn     median signed residual of the 30 nearest bed points of each body point
    For each: median |body - prediction| and the cluster score (cl30) after correction.
  * cells (60 px): body |residual| median vs local bed |residual| median (within 150 px), for the
    "divide by local background" question.
Pairs where the background is missing (fewer than --min-bed bed points near the ROI) are reported as
INSUFFICIENT_BACKGROUND, not as reliable geometry.
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import _path  # noqa: F401
from motion.bed import classify_background
from motion.calib import Calibration
from motion.gantry import GantryModel
from motion.masks import phantom_mask, specular_mask
from motion.rigidity import cluster_score
from motion.roi import DetectorROI
from motion.scan import Scan
from motion.tracking import detect, track


def signed_epi(F, a, b):
    lines = np.c_[a, np.ones(len(a))] @ F.T
    return np.sum(lines * np.c_[b, np.ones(len(b))], 1) / np.hypot(lines[:, 0], lines[:, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--name', required=True)
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--body-masks', help='body_<frame>.png folder; omit with --phantom')
    ap.add_argument('--phantom', action='store_true', help='body = orange phantom (masks computed), specular removed')
    ap.add_argument('--roi', default='data/roi_arc22_calibrated.png', help='detector ROI, FIXED in image coordinates')
    ap.add_argument('--bed', default='data/bed_home_arc22.png', help='bed outline on the HOME frame')
    ap.add_argument('--near', type=float, default=150, help='px (raw) around the ROI for "local" background')
    ap.add_argument('--min-bed', type=int, default=30)
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    scan, calib, gantry = Scan(a.scan), Calibration(a.calib), GantryModel.load(a.gantry)
    f = calib.focal_px
    Kn = np.diag([1 / f, 1 / f, 1])
    roi = (cv2.imread(a.roi, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8) * 255
    roi_dist = cv2.distanceTransform(255 - roi, cv2.DIST_L2, 5)        # raw px distance to the ROI
    near_roi = roi_dist <= a.near
    bed = DetectorROI(a.bed, calib)
    exposure = scan.exposure_pairs()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs, cells = [], []
    prev = scan.image(0)
    for i in range(1, len(scan)):
        cur = scan.image(i)
        ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (prev, cur))
        if a.phantom:
            body = phantom_mask(prev)
            spec = specular_mask(prev) & body
            body = cv2.erode(body, np.ones((21, 21), np.uint8))
        else:
            body = cv2.imread(str(Path(a.body_masks) / f'body_{int(scan.index[i - 1]):06d}.png'), cv2.IMREAD_GRAYSCALE)
            spec = np.zeros_like(body)
        angle_a, angle_b = scan.angle(i - 1, a.sync), scan.angle(i, a.sync)
        # background: E0 global detection
        p0 = detect(ga)
        p1, ok = track(ga, gb, p0)
        p0, p1 = p0[ok], p1[ok]
        xi, yi = p0[:, 0].astype(int), p0[:, 1].astype(int)
        on_bg = cv2.dilate(body, np.ones((81, 81), np.uint8))[yi, xi] == 0
        g0, g1 = p0[on_bg], p1[on_bg]
        u0, u1 = calib.undistort_points(g0), calib.undistort_points(g1)
        lab = classify_background(u0, u1, bed(u0), 1.0 / f)
        isbed = lab == 'bed'
        if isbed.sum() >= 8:
            H, _ = cv2.findHomography(u0[isbed], u1[isbed], cv2.RANSAC, 1.0 / f)
            bed.advance(H)
        moving = abs(angle_b - angle_a) >= 0.02
        if not (exposure[i] and moving):
            prev = cur
            continue
        Fe = Kn.T @ gantry.essential(angle_a, angle_b) @ Kn
        s_bg = signed_epi(Fe, u0 * f, u1 * f)
        near = near_roi[g0[:, 1].astype(int), g0[:, 0].astype(int)]
        # body: E3 selection inside body & ROI
        m = ((body > 0) & (spec == 0) & (roi > 0)).astype(np.uint8) * 255
        q0 = detect(ga, mask=m, absolute_quality=True)
        q1, qok, qfb = track(ga, gb, q0, return_fb=True)
        keep = qok & (qfb < 0.1)
        q0, q1 = q0[keep], q1[keep]
        row = dict(frame=int(scan.index[i]), angle=float(angle_b), dangle=float(angle_b - angle_a), n_body=len(q0))
        for k in ('bed', 'offplane', 'floor'):
            sel = lab == k
            row[f'n_{k}'] = int(sel.sum())
            row[f'abs_{k}'] = float(np.median(np.abs(s_bg[sel]))) if sel.any() else np.nan
            row[f'signed_{k}'] = float(np.median(s_bg[sel])) if sel.any() else np.nan
            seln = sel & near
            row[f'n_{k}_near'] = int(seln.sum())
            row[f'abs_{k}_near'] = float(np.median(np.abs(s_bg[seln]))) if seln.any() else np.nan
        row['abs_all'] = float(np.median(np.abs(s_bg))) if len(s_bg) else np.nan
        row['bg_status'] = 'OK' if row['n_bed_near'] >= a.min_bed else 'INSUFFICIENT_BACKGROUND'
        if len(q0) >= 30:
            n0, n1 = calib.undistort_points(q0) * f, calib.undistort_points(q1) * f
            sb = signed_epi(Fe, n0, n1)
            bn = (lab == 'bed') & near
            fl = lab == 'floor'
            preds = {'none': np.zeros(len(sb)),
                     'all_const': np.full(len(sb), np.median(s_bg) if len(s_bg) else np.nan),
                     'floor_const': np.full(len(sb), np.median(s_bg[fl]) if fl.any() else np.nan),
                     'bed_const': np.full(len(sb), np.median(s_bg[bn]) if bn.sum() >= 8 else np.nan)}
            if bn.sum() >= 8:
                X = np.c_[np.ones(bn.sum()), u0[bn] * f]
                coef, *_ = np.linalg.lstsq(X, s_bg[bn], rcond=None)
                preds['bed_plane'] = np.c_[np.ones(len(n0)), n0] @ coef
                bpts = u0[bn] * f
                D = np.linalg.norm(n0[:, None] - bpts[None], axis=2)
                nn = np.argsort(D, 1)[:, :min(30, bn.sum())]
                preds['bed_knn'] = np.median(s_bg[bn][nn], axis=1)
            else:
                preds['bed_plane'] = preds['bed_knn'] = np.full(len(sb), np.nan)
            for k, p in preds.items():
                r = np.abs(sb - p)
                row[f'err_{k}'] = float(np.median(r)) if np.isfinite(r).all() else np.nan
                row[f'cl30_{k}'] = cluster_score(n0, r, 30) if np.isfinite(r).all() else np.nan
            row['signed_body'] = float(np.median(sb))
            # cells for the "divide by local background" question
            bpts_all = u0[lab == 'bed'] * f
            babs = np.abs(s_bg[lab == 'bed'])
            cid = (q0 // 60).astype(int)
            for c in np.unique(cid, axis=0):
                selc = np.all(cid == c, axis=1)
                if selc.sum() < 8:
                    continue
                cen = n0[selc].mean(0)
                d = np.linalg.norm(bpts_all - cen, axis=1)
                loc = d < 150 * 0.5  # 150 raw px ~ 75 px in the half-res undistorted frame
                if loc.sum() < 10:
                    continue
                cells.append(dict(frame=row['frame'], cx=float(cen[0]), cy=float(cen[1]), n_body=int(selc.sum()),
                                  body_abs=float(np.median(np.abs(sb[selc]))), body_signed=float(np.median(sb[selc])),
                                  bed_abs=float(np.median(babs[loc])), bed_signed=float(np.median(s_bg[lab == 'bed'][loc])),
                                  n_bed=int(loc.sum())))
        else:
            row['body_status'] = 'INSUFFICIENT_BODY'
        row.setdefault('body_status', 'OK')
        pairs.append(row)
        prev = cur
    keys = sorted({k for r in pairs for k in r}, key=lambda k: (k not in ('frame', 'angle', 'dangle'), k))
    with open(out / f'{a.name}_pairs.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(pairs)
    if cells:
        with open(out / f'{a.name}_cells.csv', 'w', newline='') as fh:
            w = csv.DictWriter(fh, cells[0].keys())
            w.writeheader()
            w.writerows(cells)
    print(f'{a.name}: {len(pairs)} moving exposure pairs, {len(cells)} cells')


if __name__ == '__main__':
    main()
