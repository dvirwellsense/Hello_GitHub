"""Score every adjacent frame pair of a scan (causal, real-time capable) and raise motion alarms.

Scores per pair (pixels of the undistorted half-resolution frame, ~2.2 mm/px at the bed):
  affine_*   current method: 2D similarity from background applied to the body
  epi_img_*  epipolar distance, geometry estimated from background points
  epi_enc_*  epipolar distance, geometry from the encoder + gantry model   <- recommended
  *_cl = spatial cluster score (8 neighbouring points), *_med = median over all body points

Example:
  python scripts/evaluate_scan.py --scan scans/A28_1.zip --calib data/CalibrationResult_arc22.json \
      --gantry data/gantry_arc22.json --out results/A28_1.csv --plot results/A28_1.png --label 63-85
"""
import argparse
import csv

import cv2
import numpy as np

import _path  # noqa: F401
from motion.calib import Calibration
from motion.gantry import GantryModel
from motion.masks import body_and_specular
from motion.rigidity import evaluate_pair
from motion.scan import Scan


def alarms(score, threshold, persistence):
    run, out = 0, []
    for s in score:
        run = run + 1 if s > threshold else 0
        out.append(run >= persistence)
    return np.array(out)


def plot(rows, path, threshold, labelled):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = np.array([r['frame'] for r in rows])
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    panels = (('affine_cl', 'Current method: residual after 2D alignment (cluster score, px)', None),
              ('epi_enc_cl', 'Rigidity check: encoder-pose epipolar distance (cluster score, px)', threshold))
    for a, (key, title, thr) in zip(ax, panels):
        for lo, hi in labelled:
            a.axvspan(lo, hi, color='#eda100', alpha=0.18, lw=0)
        a.plot(f, [r[key] for r in rows], color='#2a78d6', lw=2, marker='o', ms=3)
        if thr:
            a.axhline(thr, color='#555', lw=1, ls='--')
            a.text(f[0], thr * 1.1, f'threshold {thr:g} px', va='bottom', fontsize=8, color='#333')
        a.set_title(title, fontsize=10, loc='left', color='#222')
        a.grid(axis='y', color='#e5e5e5', lw=0.8)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
        a.set_ylabel('px', fontsize=8, color='#555')
    ax[1].set_xlabel('frame index', fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True, help='scan folder or ZIP')
    ap.add_argument('--calib', required=True)
    ap.add_argument('--gantry', required=True)
    ap.add_argument('--mask', choices=('seg', 'color', 'phantom'), default='seg')
    ap.add_argument('--erode', type=int, default=21, help='erode body mask (px) to drop silhouette-edge points')
    ap.add_argument('--threshold', type=float, default=2.0)
    ap.add_argument('--persistence', type=int, default=2)
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True, help='CSV with per-pair scores')
    ap.add_argument('--plot', help='optional PNG')
    ap.add_argument('--label', action='append', default=[], help='labelled motion range, e.g. 63-85 (plot only)')
    a = ap.parse_args()
    scan, calib, gantry = Scan(a.scan), Calibration(a.calib), GantryModel.load(a.gantry)
    rows = []
    prev = scan.image(0)
    for i in range(1, len(scan)):
        cur = scan.image(i)
        body, spec = body_and_specular(a.mask, prev)
        if a.erode:
            body = cv2.erode(body, np.ones((a.erode, a.erode), np.uint8))
        r = evaluate_pair(prev, cur, scan.angle(i - 1, a.sync), scan.angle(i, a.sync), body, spec, calib, gantry)
        rows.append(dict(frame=int(scan.index[i]), angle=float(scan.angle(i, a.sync)),
                         dangle=float(scan.angle(i, a.sync) - scan.angle(i - 1, a.sync)), **r))
        prev = cur
    al = alarms([r['epi_enc_cl'] for r in rows], a.threshold, a.persistence)
    for r, x in zip(rows, al):
        r['alarm'] = bool(x)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    frames = [r['frame'] for r in rows if r['alarm']]
    print(f'{len(rows)} pairs, alarm frames: {frames}')
    if a.plot:
        plot(rows, a.plot, a.threshold, [tuple(map(int, s.split('-'))) for s in a.label])


if __name__ == '__main__':
    main()
