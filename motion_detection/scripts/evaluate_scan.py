"""Score every adjacent frame pair of a scan (causal, real-time capable) and raise motion alarms.

Gantry moving between the two frames -> epipolar check (encoder + gantry model).
Gantry stopped (same angle)          -> direct comparison: any coherent displacement is motion.
Only body points above the detector count (--detector-mask), and alarms are raised only inside the
exposure window (last frame of the start dwell .. first frame of the end dwell); the approach from
HOME and the return are scored but never alarm.

Scores per pair, pixels of the undistorted half-resolution frame (~2.2 mm/px at the bed plane):
  affine_*   current method: 2D similarity from background applied to the body
  epi_img_*  epipolar distance, geometry estimated from background points
  epi_enc_*  epipolar distance, geometry from the encoder + gantry model
  static_*   same-angle pairs: displacement after removing the common background shift
  *_cl = spatial cluster score (8 points), *_cl30 = 30 points, *_med = over all body points
  motion_mm = the score used for the alarm (epi_enc_cl30 or static_cl, converted to mm at the bed plane)

Alarm rule: score above threshold for --persistence consecutive pairs inside the exposure window.
Pairs where the gantry speed changes by >30% (start/stop/reversal) are skipped (accel_guard): there the
encoder-to-image timing error dominates. Re-run only the decision + plot with --from-csv.

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
from motion.roi import DetectorROI
from motion.scan import Scan


def plot(rows, path, a, labelled):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = np.array([r['frame'] for r in rows])
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw=dict(height_ratios=(1, 2)))
    exp = np.array([r['exposure'] for r in rows])
    if exp.any():
        for x in ax:
            x.axvspan(f[exp].min() - 0.5, f[exp].max() + 0.5, color='#e5e5e5', alpha=0.6, lw=0)
    ax[0].plot(f, [r['angle'] for r in rows], color='#555', lw=1.5)
    ax[0].set_title('Gantry angle (deg); grey = exposure window', fontsize=10, loc='left', color='#222')
    for lo, hi in labelled:
        ax[1].axvspan(lo - 0.5, hi + 0.5, color='#eda100', alpha=0.25, lw=0)
    m = np.array([r['motion_mm'] for r in rows], float)
    st = np.array([r['same_angle'] for r in rows])
    ax[1].plot(f[~st], m[~st], 'o', ms=3.5, color='#2a78d6', label='gantry moving (epipolar check)')
    ax[1].plot(f[st], m[st], 's', ms=3.5, color='#eb6834', label='gantry stopped (direct comparison)')
    al = np.array([r['alarm'] for r in rows])
    ax[1].plot(f[al], m[al], 'x', ms=9, mew=2, color='#222', label='alarm')
    g = np.array([r.get('accel_guard', False) for r in rows], bool)
    ax[1].plot(f[g], m[g], 'o', ms=6, mfc='none', color='#999', label='skipped (gantry accelerating)')
    ax[1].axhline(a.threshold * a.mm_per_px, color='#2a78d6', lw=1, ls='--')
    ax[1].axhline(a.static_threshold_mm, color='#eb6834', lw=1, ls='--')
    ax[1].set_yscale('log')
    ax[1].set_ylabel('motion score (mm, log)', fontsize=8, color='#555')
    ax[1].set_title('Motion score above the detector; dashed = thresholds, orange band = labelled motion', fontsize=10, loc='left', color='#222')
    ax[1].legend(fontsize=8, frameon=True, framealpha=0.9, edgecolor='none', loc='upper right')
    for x in ax:
        x.grid(axis='y', color='#e5e5e5', lw=0.8)
        for sp in ('top', 'right'):
            x.spines[sp].set_visible(False)
    ax[1].set_xlabel('frame index', fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=130)


def decide(rows, a):
    """Apply acceleration guard, thresholds and persistence. Adds motion_mm, accel_guard, alarm."""
    d = np.array([r['dangle'] for r in rows], float)
    jump = np.abs(np.diff(d)) > 0.3 * np.maximum(np.abs(d[1:]), np.abs(d[:-1]))
    guard = np.r_[False, jump] | np.r_[jump, False]
    run = 0
    for r, g in zip(rows, guard):
        if r['same_angle']:
            score_mm = r['static_cl'] * a.mm_per_px
            candidate = score_mm > a.static_threshold_mm
            g = False
        else:
            score_mm = r['epi_enc_cl30'] * a.mm_per_px
            candidate = r['epi_enc_cl30'] > a.threshold and not g
        active = r['exposure'] or a.all_phases
        run = run + 1 if (candidate and active) else 0
        r.update(motion_mm=float(score_mm), accel_guard=bool(g), alarm=run >= a.persistence)
    return rows


def read_csv(path):
    rows = []
    for x in csv.DictReader(open(path)):
        r = {}
        for k, v in x.items():
            if v in ('True', 'False'):
                r[k] = v == 'True'
            else:
                try:
                    r[k] = float(v) if v != '' else np.nan
                except ValueError:
                    r[k] = v
        r['frame'] = int(r['frame'])
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', help='scan folder or ZIP')
    ap.add_argument('--from-csv', help='skip image processing: re-apply the decision rule to a CSV from an earlier run')
    ap.add_argument('--calib', required=True)
    ap.add_argument('--gantry', required=True)
    ap.add_argument('--mask', choices=('seg', 'color', 'phantom'), default='seg')
    ap.add_argument('--erode', type=int, default=21, help='erode body mask (px) to drop silhouette-edge points')
    ap.add_argument('--detector-mask', help='PNG on the HOME frame (1280x720), white = above the detector')
    ap.add_argument('--detector-rect', help='x,y,w,h on the HOME frame, instead of --detector-mask')
    ap.add_argument('--threshold', type=float, default=0.5, help='gantry moving: epi_enc_cl30, px (0.5 px ~ 1.1 mm)')
    ap.add_argument('--static-threshold-mm', type=float, default=0.5, help='gantry stopped: displacement, mm')
    ap.add_argument('--persistence', type=int, default=1)
    ap.add_argument('--all-phases', action='store_true', help='also alarm during approach / return')
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True, help='CSV with per-pair scores')
    ap.add_argument('--plot', help='optional PNG')
    ap.add_argument('--label', action='append', default=[], help='labelled motion range, e.g. 63-85 (plot only)')
    a = ap.parse_args()
    calib = Calibration(a.calib)
    a.mm_per_px = 1.0 / calib.pixels_per_mm
    if a.from_csv:
        rows = decide(read_csv(a.from_csv), a)
        finish(rows, a)
        return
    scan, gantry = Scan(a.scan), GantryModel.load(a.gantry)
    roi = None
    if a.detector_mask:
        roi = DetectorROI(a.detector_mask, calib)
    elif a.detector_rect:
        roi = DetectorROI.from_rect(tuple(int(v) for v in a.detector_rect.split(',')), calib)
    phase = scan.phases()
    exposure = scan.exposure_pairs()
    rows = []
    prev = scan.image(0)
    for i in range(1, len(scan)):
        cur = scan.image(i)
        body, spec = body_and_specular(a.mask, prev)
        if a.erode:
            body = cv2.erode(body, np.ones((a.erode, a.erode), np.uint8))
        r = evaluate_pair(prev, cur, scan.angle(i - 1, a.sync), scan.angle(i, a.sync), body, spec, calib, gantry, roi=roi)
        H = r.pop('_H')
        if roi is not None:
            roi.advance(H)
        rows.append(dict(frame=int(scan.index[i]), phase=phase[i], exposure=bool(exposure[i]),
                         angle=float(scan.angle(i, a.sync)), dangle=float(scan.angle(i, a.sync) - scan.angle(i - 1, a.sync)), **r))
        prev = cur
    finish(decide(rows, a), a)


def finish(rows, a):
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    win = [r['frame'] for r in rows if r['exposure']]
    print(f'{len(rows)} pairs, exposure window frames {min(win) if win else None}-{max(win) if win else None}')
    print('alarm frames:', [r['frame'] for r in rows if r['alarm']])
    if a.plot:
        plot(rows, a.plot, a, [tuple(map(int, s.split('-'))) for s in a.label])


if __name__ == '__main__':
    main()
