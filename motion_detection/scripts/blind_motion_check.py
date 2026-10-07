"""Diagnostic: does the score miss motion because it is along the epipolar line?

For every pair in the exposure window it tracks the body points (same settings as evaluate_scan.py, E3 +
saved body masks) and splits each point's image displacement into
  perp  = component across the epipolar line  (what the score measures; epi_enc_*)
  along = component along the epipolar line   (what the score cannot see: it is also where parallax goes)
A rigid body gives an `along` displacement per degree of arc that changes slowly from pair to pair, so the
excess motion along the line is  along_med - dangle * (median along-per-degree of the unlabelled pairs nearby).
Nothing here changes the score or any threshold.

Example:
  python scripts/blind_motion_check.py --scan <scan_dir> --body-masks <masks_dir> --label 55-69 \
      --out results/blind_check_2026_10/A30.csv
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from motion.calib import Calibration  # noqa: E402
from motion.gantry import GantryModel  # noqa: E402
from motion.rigidity import evaluate_pair  # noqa: E402
from motion.roi import DetectorROI  # noqa: E402
from motion.scan import Scan  # noqa: E402


def parse_labels(items):
    out = set()
    for it in items:
        a, b = (int(v) for v in it.split('-'))
        out |= set(range(a, b + 1))
    return out


def knn_median(pts, v, k=30):
    D = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    nn = np.argsort(D, 1)[:, :k]
    return np.median(v[nn], axis=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--body-masks', required=True)
    ap.add_argument('--label', action='append', default=[])
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--detector-mask', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--bed-near-px', type=int, default=150)
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--baseline-halfwidth', type=int, default=10, help='pairs on each side used for the baseline')
    ap.add_argument('--arrows', help='folder: save a displacement picture for every labelled pair')
    ap.add_argument('--include-outside', action='store_true', help='also process pairs outside the exposure window')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    labels = parse_labels(a.label)
    calib, gantry, scan = Calibration(a.calib), GantryModel.load(a.gantry), Scan(a.scan)
    roi = DetectorROI(a.detector_mask, calib)
    region = roi.mask.astype(np.uint8) * 255
    exposure = scan.exposure_pairs()
    f = calib.focal_px
    Kn = np.diag([1 / f, 1 / f, 1])
    rows = []
    for i in range(1, len(scan)):
        if not exposure[i] and not a.include_outside:
            continue
        body = cv2.imread(str(Path(a.body_masks) / f'body_{int(scan.index[i - 1]):06d}.png'), cv2.IMREAD_GRAYSCALE)
        img0, img1 = scan.image(i - 1), scan.image(i)
        ang0, ang1 = scan.angle(i - 1, a.sync), scan.angle(i, a.sync)
        r, pts = evaluate_pair(img0, img1, ang0, ang1, body, np.zeros_like(body), calib, gantry, roi=roi,
                               detect_mode='masked_abs', roi_region=region, body_fb_max=0.1, return_points=True)
        row = dict(frame=int(scan.index[i]), label=int(int(scan.index[i]) in labels), dangle=ang1 - ang0,
                   n_body=r['n_body'], perp_cl30=r.get('epi_enc_cl30', np.nan))
        if pts is None:
            rows.append(dict(row, along_med=np.nan, perp_med=np.nan, _pts=None))
            continue
        n0, n1, Fe = pts['n0'], pts['n1'], pts['Fe']
        line = np.c_[n0, np.ones(len(n0))] @ Fe.T
        d = np.c_[line[:, 1], -line[:, 0]] / np.hypot(line[:, 0], line[:, 1])[:, None]
        d *= np.where(d[:, :1] < 0, -1, 1)  # fixed orientation: pointing to +x
        disp = n1 - n0
        along = np.sum(disp * d, 1)
        perp = np.sum(disp * np.c_[-d[:, 1], d[:, 0]], 1)
        dang = ang1 - ang0
        rows.append(dict(row, along_med=float(np.median(along)), perp_med=float(np.median(perp)), _pts=(n0, along, perp, dang, pts)))
    # baseline: median along-per-degree of the pairs in the window that are not labelled, nearby
    for k, r in enumerate(rows):
        near = [q for q in rows[max(0, k - a.baseline_halfwidth):k + a.baseline_halfwidth + 1]
                if not q['label'] and q['_pts'] is not None and abs(q['dangle']) > 1e-3]
        if r['_pts'] is None or len(near) < 3:
            r.update(along_dev=np.nan, along_cl30_dev=np.nan)
            continue
        base = float(np.median([q['along_med'] / q['dangle'] for q in near]))
        n0, along, perp, dang, _ = r['_pts']
        dev = along - base * dang
        r['along_dev'] = float(np.median(dev))
        r['along_cl30_dev'] = float(np.max(np.abs(knn_median(n0, dev))))
        r['perp_cl30_signed'] = float(np.max(np.abs(knn_median(n0, perp))))
    if a.arrows:
        Path(a.arrows).mkdir(parents=True, exist_ok=True)
        for r in rows:
            if r['_pts'] is None or not r['label']:
                continue
            n0, along, perp, dang, pts = r['_pts']
            im = calib.undistort_image(scan.image(int([q for q in range(len(scan)) if int(scan.index[q]) == r['frame']][0])))
            p0 = pts['p0']
            u0 = n0 / f
            for (x, y), al, pe in zip(u0 * f + np.array([calib.K[0, 2], calib.K[1, 2]]), along, perp):
                cv2.circle(im, (int(x), int(y)), 2, (0, 255, 0), -1)
                cv2.arrowedLine(im, (int(x), int(y)), (int(x + 5 * al), int(y + 5 * pe)), (0, 100, 255), 1, tipLength=0.3)
            cv2.imwrite(str(Path(a.arrows) / f'pair_{r["frame"]:04d}.jpg'), im)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    keys = ['frame', 'label', 'dangle', 'n_body', 'perp_cl30', 'perp_med', 'along_med', 'along_dev', 'along_cl30_dev']
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k)) for k in keys})
    print('wrote', a.out, len(rows), 'pairs')


if __name__ == '__main__':
    main()
