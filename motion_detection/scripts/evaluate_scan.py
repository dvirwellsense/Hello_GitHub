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
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

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
    ax[1].plot(f[g], m[g], 'o', ms=6, mfc='none', color='#999', label='angle unreliable')
    ins = np.array([r.get('motion_state') == 'NO_BODY_DATA' and r['exposure'] for r in rows], bool)
    if ins.any():
        y0 = np.nanmin(m[np.isfinite(m)]) * 0.7 if np.isfinite(m).any() else 0.01
        ax[1].plot(f[ins], np.full(ins.sum(), y0), '|', ms=10, mew=2, color='#c0392b', label='insufficient data (<30 points)')
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


def guard_flags(rows, a):
    """Reliability of each moving pair with respect to the gantry angle (True = unreliable).

    legacy: |change of angle step between neighbouring pairs| > 30% of the step. Uses angle steps only
            (no time) and also looks at the NEXT pair, so it is not causal.
    time:   causal. For each frame the gantry speed during its capture (angle_before -> angle_after over
            request -> saved time) is compared with the speed over the previous request interval. If the
            speed changes, the linear interpolation that assigns an angle to the image is wrong by about
            |speed difference| x capture duration. That angle uncertainty (both frames of the pair) times
            the measured sensitivity of the residual to the angle (epi_sens_px_per_deg) is the expected
            residual error in px; the pair is unreliable if it exceeds --guard-px.
    bg:     median epipolar residual of ALL background points (epi_enc_bg) > --guard-bg-px. Dominated by
            floor points and reflections; see results/background_study for why this is not recommended.
    bed:    median residual of bed-PLANE points near the ROI (epi_bed_near, needs --bed) > --guard-bed-px.
            Fewer than --min-bed such points -> geometry_state BG_MISSING (cannot verify),
            never 'reliable'.
    none:   every pair reliable."""
    n = len(rows)
    flag = np.zeros(n, bool)
    if a.guard == 'none':
        return flag
    if a.guard == 'bed':
        for k, r in enumerate(rows):
            b = r.get('epi_bed_near', np.nan)
            flag[k] = (not r['same_angle']) and np.isfinite(b) and b > a.guard_bed_px
        return flag
    if a.guard == 'bg':
        for k, r in enumerate(rows):
            b = r.get('epi_enc_bg', np.nan)
            flag[k] = (not r['same_angle']) and np.isfinite(b) and b > a.guard_bg_px
        return flag
    if a.guard == 'legacy':
        d = np.array([r['dangle'] for r in rows], float)
        jump = np.abs(np.diff(d)) > 0.3 * np.maximum(np.abs(d[1:]), np.abs(d[:-1]))
        return np.r_[False, jump] | np.r_[jump, False]

    def unc(t_req, t_sav, a_bef, a_aft, t_req_prev, a_bef_prev):
        T = t_sav - t_req
        w_cap = (a_aft - a_bef) / T if T > 0 else 0.0
        dt = t_req - t_req_prev
        if not np.isfinite(t_req_prev) or dt <= 0:
            return abs(a_aft - a_bef)  # no history: assume the full in-capture change is uncertain
        w_int = (a_bef - a_bef_prev) / dt
        return abs(w_cap - w_int) * T

    prev_req, prev_bef = np.nan, np.nan  # request time / angle_before of the frame before frame a
    for k, r in enumerate(rows):
        ua = unc(r['t_req_a'], r['t_sav_a'], r['ab_a'], r['aa_a'], prev_req, prev_bef)
        ub = unc(r['t_req_b'], r['t_sav_b'], r['ab_b'], r['aa_b'], r['t_req_a'], r['ab_a'])
        u = float(np.hypot(ua, ub))
        sens = r.get('epi_sens_px_per_deg', np.nan)
        expected = u * sens if np.isfinite(sens) else 0.0
        r.update(angle_unc_deg=u, expected_err_px=float(expected))
        flag[k] = (not r['same_angle']) and expected > a.guard_px
        prev_req, prev_bef = r['t_req_a'], r['ab_a']
    return flag


LABELS = {
    ('SUSPECT', 'VERIFIED'): 'motion suspected; geometry verified by background',
    ('SUSPECT', 'NOT_CHECKED'): 'motion suspected; geometry not checked',
    ('SUSPECT', 'UNRELIABLE'): 'motion suspected; geometry unreliable',
    ('SUSPECT', 'BG_MISSING'): 'motion suspected; background missing for verification',
    ('NO_MOTION', 'VERIFIED'): 'no motion; geometry verified by background',
    ('NO_MOTION', 'NOT_CHECKED'): 'no motion; geometry not checked',
    ('NO_MOTION', 'UNRELIABLE'): 'no motion; geometry unreliable',
    ('NO_MOTION', 'BG_MISSING'): 'no motion; background missing for verification',
}


def decide(rows, a):
    """Two independent judgements per pair, then alarms.

    motion_state   (from the body score only)
        SUSPECT       score above threshold
        NO_MOTION     score below threshold
        NO_BODY_DATA  no valid score (fewer than 30 body points): insufficient information about motion
    geometry_state (from the guard only; never changes motion_state)
        VERIFIED      guard ran and the geometry of the pair is reliable
        UNRELIABLE    guard flags the pair (angle step / background residual)
        BG_MISSING    --guard bed and fewer than --min-bed bed points near the ROI: cannot verify
        NOT_CHECKED   no guard (or gantry stopped)
    state = 'motion_state/geometry_state'; label = readable text.
    candidate_raw: score above threshold, before anything else.
    alarm_confirmed: persistence over SUSPECT & (VERIFIED or NOT_CHECKED), inside the exposure window.
    alarm_any:       persistence over SUSPECT whatever the geometry (unreliable and missing background included).
    alarm:           alarm_confirmed with --guard-action drop, alarm_any with mark."""
    unreliable = guard_flags(rows, a)
    run_c = run_a = 0
    for r, g in zip(rows, unreliable):
        if r['same_angle']:
            score = r['static_cl']
            score_mm = score * a.mm_per_px
            thr_hit = score_mm > a.static_threshold_mm
        else:
            score = r['epi_enc_cl30']
            score_mm = score * a.mm_per_px
            thr_hit = score > a.threshold
        motion = 'NO_BODY_DATA' if not np.isfinite(score) else ('SUSPECT' if thr_hit else 'NO_MOTION')
        if r['same_angle'] or a.guard == 'none':
            geom = 'NOT_CHECKED'
        elif a.guard == 'bed' and not (r.get('n_bed_near', 0) >= a.min_bed):
            geom = 'BG_MISSING'
        else:
            geom = 'UNRELIABLE' if g else 'VERIFIED'
        active = r['exposure'] or a.all_phases
        run_c = run_c + 1 if (active and motion == 'SUSPECT' and geom in ('VERIFIED', 'NOT_CHECKED')) else 0
        run_a = run_a + 1 if (active and motion == 'SUSPECT') else 0
        conf, anyy = run_c >= a.persistence, run_a >= a.persistence
        r.update(motion_mm=float(score_mm), candidate_raw=bool(motion == 'SUSPECT'), accel_guard=bool(geom == 'UNRELIABLE'),
                 motion_state=motion, geometry_state=geom, state=f'{motion}/{geom}',
                 label=LABELS.get((motion, geom), 'insufficient body data (fewer than 30 points)'),
                 alarm_confirmed=conf, alarm_any=anyy, alarm=conf if a.guard_action == 'drop' else anyy)
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
    ap.add_argument('--detect', choices=('global', 'masked', 'masked_abs'), default='global',
                    help='global: corners on the whole image then filtered; masked: body corners detected inside body & ROI '
                         '(quality relative to the body); masked_abs: same region, quality threshold of global')
    ap.add_argument('--body-fb-max', type=float, default=0.5, help='forward-backward limit (px) for body points')
    ap.add_argument('--body-masks', help='folder with body_<frame>.png from --save-masks: use these exact (already eroded) masks instead of segmentation')
    ap.add_argument('--guard', choices=('legacy', 'time', 'bg', 'bed', 'none'), default='legacy', help='angle-reliability test, see guard_flags()')
    ap.add_argument('--guard-bg-px', type=float, default=0.4, help='bg guard: background residual (px) above which a pair is unreliable')
    ap.add_argument('--guard-bed-px', type=float, default=0.3, help='bed guard: bed-plane residual near the ROI (px)')
    ap.add_argument('--min-bed', type=int, default=30, help='bed guard: minimum bed-plane points near the ROI')
    ap.add_argument('--bed', help='bed outline PNG on the HOME frame (e.g. data/bed_home_arc22.png), carried by the bed homography')
    ap.add_argument('--bed-near-px', type=float, default=150, help='"near the ROI" distance for the bed statistics (raw px)')
    ap.add_argument('--roi-fixed', action='store_true', help='keep the detector ROI fixed in image coordinates (no homography)')
    ap.add_argument('--guard-px', type=float, default=0.25, help='time guard: expected residual error (px) above which a pair is unreliable')
    ap.add_argument('--guard-action', choices=('drop', 'mark'), default='drop',
                    help='drop: unreliable pairs cannot alarm (original); mark: they still alarm, reported as unreliable')
    ap.add_argument('--all-phases', action='store_true', help='also alarm during approach / return')
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True, help='CSV with per-pair scores')
    ap.add_argument('--plot', help='optional PNG')
    ap.add_argument('--label', action='append', default=[], help='labelled motion range, e.g. 63-85 (plot only)')
    ap.add_argument('--save-masks', help='folder: save the exact body mask used for every frame (PNG) + a preview overlay (JPG)')
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
    bed = DetectorROI(a.bed, calib) if a.bed else None
    near_fixed = None
    if roi is not None and a.roi_fixed:
        k = int(2 * a.bed_near_px + 1)
        near_fixed = cv2.dilate(roi.mask.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
    phase = scan.phases()
    exposure = scan.exposure_pairs()
    rows = []
    prev = scan.image(0)
    for i in range(1, len(scan)):
        cur = scan.image(i)
        if a.body_masks:
            body = cv2.imread(str(Path(a.body_masks) / f'body_{int(scan.index[i - 1]):06d}.png'), cv2.IMREAD_GRAYSCALE)
            if body is None:
                raise FileNotFoundError(f'body mask for frame {int(scan.index[i - 1])} in {a.body_masks}')
            spec = np.zeros_like(body)
        else:
            body, spec = body_and_specular(a.mask, prev)
            if a.erode:
                body = cv2.erode(body, np.ones((a.erode, a.erode), np.uint8))
        region = None
        if roi is not None and a.roi_fixed:
            region = roi.mask.astype(np.uint8) * 255
        elif roi is not None and a.detect != 'global':
            region = np.zeros(body.shape, np.uint8)
            cv2.fillPoly(region, roi.outline(), 255)
        near = None
        if bed is not None and roi is not None:
            if near_fixed is not None:
                near = near_fixed
            else:
                reg = np.zeros(body.shape, np.uint8)
                cv2.fillPoly(reg, roi.outline(), 255)
                k = int(2 * a.bed_near_px + 1)
                near = cv2.dilate(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0
        if a.save_masks:
            save_mask(a.save_masks, int(scan.index[i - 1]), prev, body, spec, roi)
        r = evaluate_pair(prev, cur, scan.angle(i - 1, a.sync), scan.angle(i, a.sync), body, spec, calib, gantry, roi=roi,
                          detect_mode=a.detect, roi_region=region, body_fb_max=a.body_fb_max, bed=bed, bg_near=near)
        H = r.pop('_H')
        if roi is not None and not a.roi_fixed:
            roi.advance(H)
        if bed is not None:
            bed.advance(H)
        rows.append(dict(frame=int(scan.index[i]), phase=phase[i], exposure=bool(exposure[i]),
                         angle=float(scan.angle(i, a.sync)), dangle=float(scan.angle(i, a.sync) - scan.angle(i - 1, a.sync)),
                         t_req_a=float(scan.time_s[i - 1]), t_sav_a=float(scan.saved_s[i - 1]),
                         ab_a=float(scan.angle_before[i - 1]), aa_a=float(scan.angle_after[i - 1]),
                         t_req_b=float(scan.time_s[i]), t_sav_b=float(scan.saved_s[i]),
                         ab_b=float(scan.angle_before[i]), aa_b=float(scan.angle_after[i]), **r))
        prev = cur
    finish(decide(rows, a), a)


def save_mask(folder, frame, img, body, spec, roi):
    """Save what the scoring actually used for pairs starting at this frame.
    body_<frame>.png: body mask after erosion (white = body).
    used_<frame>.png: points counted = body, not specular, and inside the detector region.
    view_<frame>.jpg: half-size preview; red = counted region, cyan = detector region outline."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    used = (body > 0) & (spec == 0)
    polys = roi.outline() if roi is not None else []
    if roi is not None:
        region = np.zeros(body.shape, np.uint8)
        cv2.fillPoly(region, polys, 255)
        used &= region > 0
    cv2.imwrite(str(folder / f'body_{frame:06d}.png'), body)
    cv2.imwrite(str(folder / f'used_{frame:06d}.png'), used.astype(np.uint8) * 255)
    v = img.copy()
    v[used] = (0.45 * v[used] + np.array([0, 0, 140])).astype(np.uint8)
    cv2.polylines(v, polys, True, (255, 255, 0), 2)
    cv2.putText(v, f'frame {frame}', (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)
    cv2.imwrite(str(folder / f'view_{frame:06d}.jpg'), cv2.resize(v, (640, 360)), [cv2.IMWRITE_JPEG_QUALITY, 85])


def sha256(path):
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def run_settings(a):
    """Everything needed to reproduce this run."""
    import scipy
    root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], cwd=root, capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain', '--', '.'], cwd=root, capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = None, None
    versions = dict(python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__, scipy=scipy.__version__)
    weights = None
    if a.mask == 'seg':
        import torch
        import ultralytics
        versions.update(ultralytics=ultralytics.__version__, torch=torch.__version__)
        weights = 'yolov8m-seg.pt'
    args = {k: v for k, v in vars(a).items() if k != 'mm_per_px'}
    return dict(command=' '.join(sys.argv), args=args, git_commit=commit, git_dirty=dirty, versions=versions,
                segmentation=dict(weights=weights, weights_sha256=sha256(weights) if weights else None,
                                  conf=0.25, imgsz=960, rotate='90 clockwise') if weights else None,
                calibration_sha256=sha256(a.calib), gantry_model=json.load(open(a.gantry)),
                mm_per_px=a.mm_per_px)


def finish(rows, a):
    Path(a.out).with_suffix('.settings.json').write_text(json.dumps(run_settings(a), indent=2, ensure_ascii=False))
    keys = list(dict.fromkeys(k for r in rows for k in r))  # union, first-seen order
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, restval='')
        w.writeheader()
        w.writerows(rows)
    win = [r['frame'] for r in rows if r['exposure']]
    print(f'{len(rows)} pairs, exposure window frames {min(win) if win else None}-{max(win) if win else None}')
    print('alarm frames:', [r['frame'] for r in rows if r['alarm']])
    ex = [r for r in rows if r['exposure']]
    from collections import Counter
    print('exposure-window motion_state:', dict(Counter(r['motion_state'] for r in ex)))
    print('exposure-window geometry_state:', dict(Counter(r['geometry_state'] for r in ex)))
    for key, text in (('SUSPECT/UNRELIABLE', 'motion suspected - geometry unreliable'),
                      ('SUSPECT/BG_MISSING', 'motion suspected - background missing for verification')):
        fr = [r['frame'] for r in ex if r['state'] == key]
        if fr:
            print(f'{text}: {fr}')
    if a.plot:
        plot(rows, a.plot, a, [tuple(map(int, s.split('-'))) for s in a.label])


if __name__ == '__main__':
    main()
