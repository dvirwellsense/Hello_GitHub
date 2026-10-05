"""Inspect the point cluster that set epi_enc_cl30 for one frame pair, and test WHY its residual is high.

The cluster score is: for every body point, the median residual of its 30 nearest body points; the
maximum over points. This script finds that maximising point, its 30 neighbours and their matches, and
runs tests that do not rely on where the points are:

  T1 window test     re-track each point with LK windows 7/11/15/21/31 px. A real displacement of the
                     surface gives the same residual for every window; a window that straddles two
                     surfaces at different depths gives a residual that changes with the window size.
  T2 sub-patch test  track a 5x5 grid of sub-points (+-8 px) inside the 21x21 window with 7x7 windows.
                     Along-line spread = p90-p10 of the displacement component ALONG the epipolar line.
                     Along the line the static displacement is pure parallax (depends on depth only), so
                     a large spread inside one window means it contains more than one depth.
                     Perp spread = same for the residual (perpendicular) component.
  T3 photometric     NCC of the 21x21 patch at p0 (frame a) and at its match p1 (frame b), and the LK
                     error, both from the original 21x21 tracking.
  T4 static reference: residual of BACKGROUND (bed) points within 150 px of the cluster, and of the
                     cluster's own previous/next pairs. The bed cannot move: if bed points next to the
                     cluster show the same residual, the cause is the camera/angle model at that place in
                     the image, not the body.
The same tests are run on a control group: the 30 body points with the lowest residual.

Example:
  python scripts/inspect_cluster.py --scan <scans_dir>/arc22_office_20260830_130408_volunteer_chest_manual_breath_hold_post \
      --calib data/CalibrationResult_arc22.json --gantry data/gantry_arc22.json \
      --body-masks masks/VC01_breath_hold --detector-rect 280,150,620,375 --frame 108 --out inspect/VC01_108
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

import _path  # noqa: F401
from motion.calib import Calibration
from motion.gantry import GantryModel
from motion.rigidity import evaluate_pair, epipolar_distance
from motion.roi import DetectorROI
from motion.scan import Scan
from motion.tracking import LK

WINDOWS = (7, 11, 15, 21, 31)


def lk_track(ga, gb, pts, win, fb_max=0.5):
    p0 = np.float32(pts).reshape(-1, 1, 2)
    lk = dict(LK, winSize=(win, win))
    p1, s, err = cv2.calcOpticalFlowPyrLK(ga, gb, p0, None, **lk)
    pb, sb, _ = cv2.calcOpticalFlowPyrLK(gb, ga, p1, None, **lk)
    fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
    ok = (s.ravel() == 1) & (sb.ravel() == 1) & (fb < fb_max)
    return p1.reshape(-1, 2), ok, fb, err.ravel()


def ncc(ga, gb, p0, p1, size=21):
    a = cv2.getRectSubPix(ga, (size, size), tuple(map(float, p0))).astype(np.float64)
    b = cv2.getRectSubPix(gb, (size, size), tuple(map(float, p1))).astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else np.nan


def residual_parts(calib, F, raw0, raw1):
    """Perpendicular residual and along-line displacement (px, undistorted half-res) of matches."""
    f = calib.focal_px
    n0, n1 = calib.undistort_points(raw0) * f, calib.undistort_points(raw1) * f
    perp = epipolar_distance(F, n0, n1)
    lines = np.c_[n0, np.ones(len(n0))] @ F.T
    t = np.c_[lines[:, 1], -lines[:, 0]] / np.hypot(lines[:, 0], lines[:, 1])[:, None]  # line direction
    along = np.sum((n1 - n0) * t, 1)
    return perp, along


def point_tests(ga, gb, calib, F, raw0):
    """All tests for one point (raw pixel coordinates in frame a)."""
    out = {}
    for w in WINDOWS:
        p1, ok, fb, _ = lk_track(ga, gb, [raw0], w)
        if ok[0]:
            perp, _ = residual_parts(calib, F, np.array([raw0]), p1[:1])
            out[f'e_win{w}'] = float(perp[0])
        else:
            out[f'e_win{w}'] = np.nan
    vals = np.array([out[f'e_win{w}'] for w in WINDOWS], float)
    out['win_range'] = float(np.nanmax(vals) - np.nanmin(vals)) if np.isfinite(vals).sum() >= 2 else np.nan
    g = np.arange(-8, 9, 4)
    sub = np.array([[raw0[0] + dx, raw0[1] + dy] for dy in g for dx in g], np.float32)
    s1, ok, _, _ = lk_track(ga, gb, sub, 7)
    out['sub_valid'] = int(ok.sum())
    if ok.sum() >= 6:
        perp, along = residual_parts(calib, F, sub[ok], s1[ok])
        out['sub_along_spread'] = float(np.percentile(along, 90) - np.percentile(along, 10))
        out['sub_perp_spread'] = float(np.percentile(perp, 90) - np.percentile(perp, 10))
        out['sub_perp_median'] = float(np.median(perp))
    else:
        out.update(sub_along_spread=np.nan, sub_perp_spread=np.nan, sub_perp_median=np.nan)
    p1, ok, fb, err = lk_track(ga, gb, [raw0], 21)
    out['lk_err'] = float(err[0])
    out['fb'] = float(fb[0])
    out['ncc'] = ncc(ga, gb, raw0, p1[0]) if ok[0] else np.nan
    return out


def figure(path, A, B, calib, F, grp, ctrl, e_all, p0_all, p1_all, frame, score):
    """Crop around the cluster in frames a and b, plus per-point window curves."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    P0, P1 = p0_all[grp], p1_all[grp]
    cx, cy = P0.mean(0)
    half = max(60, int(np.abs(P0 - [cx, cy]).max() + 40))
    x0, y0 = int(max(0, cx - half)), int(max(0, cy - half))
    x1, y1 = int(min(A.shape[1], cx + half)), int(min(A.shape[0], cy + half))
    fig = plt.figure(figsize=(13, 8))
    for k, (img, pts, title) in enumerate(((A, P0, f'frame a ({frame - 1}): the 30 points'),
                                           (B, P1, f'frame b ({frame}): their matches; residual x20'))):
        ax = fig.add_subplot(2, 3, k + 1)
        ax.imshow(cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2RGB), extent=(x0, x1, y1, y0))
        sc = ax.scatter(pts[:, 0], pts[:, 1], c=e_all[grp], cmap='inferno', vmin=0, vmax=max(1.0, e_all[grp].max()), s=28,
                        edgecolors='white', linewidths=0.6)
        if k == 1:
            f = calib.focal_px
            n0, n1 = calib.undistort_points(P0) * f, calib.undistort_points(P1) * f
            lines = np.c_[n0, np.ones(len(n0))] @ F.T
            nrm = lines[:, :2] / np.hypot(lines[:, 0], lines[:, 1])[:, None]
            dist = (np.sum(lines * np.c_[n1, np.ones(len(n1))], 1) / np.hypot(lines[:, 0], lines[:, 1]))
            # draw the perpendicular residual direction (approximately the same in raw pixels at this scale)
            for (x, y), nv, dd in zip(P1, nrm, dist):
                ax.plot([x, x - nv[0] * dd * 20 * 2], [y, y - nv[1] * dd * 20 * 2], color='#2a78d6', lw=1.2)
            fig.colorbar(sc, ax=ax, fraction=0.046, label='residual (px)')
        ax.set_title(title, fontsize=9)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y1, y0)
    ax = fig.add_subplot(2, 3, 3)
    ax.imshow(cv2.cvtColor(A, cv2.COLOR_BGR2RGB))
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, color='#eb6834', lw=2))
    ax.scatter(p0_all[ctrl, 0], p0_all[ctrl, 1], s=8, color='#1baf7a', label='control group')
    ax.legend(fontsize=7, loc='lower right')
    ax.set_title(f'whole frame; cluster score {score:.3f} px', fontsize=9)
    ax.axis('off')
    return fig, (x0, y0, x1, y1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--gantry', required=True)
    ap.add_argument('--body-masks', required=True, help='body_<frame>.png folder from --save-masks')
    ap.add_argument('--detector-mask')
    ap.add_argument('--detector-rect')
    ap.add_argument('--detect', choices=('global', 'masked', 'masked_abs'), default='global')
    ap.add_argument('--frame', type=int, required=True, help='image_index of frame b of the pair')
    ap.add_argument('--k', type=int, default=30)
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True, help='output folder')
    a = ap.parse_args()
    scan, calib, gantry = Scan(a.scan), Calibration(a.calib), GantryModel.load(a.gantry)
    roi = DetectorROI(a.detector_mask, calib) if a.detector_mask else (
        DetectorROI.from_rect(tuple(int(v) for v in a.detector_rect.split(',')), calib) if a.detector_rect else None)
    target = int(np.where(scan.index == a.frame)[0][0])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def load_body(i):
        return cv2.imread(str(Path(a.body_masks) / f'body_{int(scan.index[i]):06d}.png'), cv2.IMREAD_GRAYSCALE)

    # Replay all earlier pairs so the detector ROI is carried exactly as in evaluate_scan.py.
    prev = scan.image(0)
    for i in range(1, target + 1):
        cur = scan.image(i)
        body = load_body(i - 1)
        spec = np.zeros_like(body)
        region = None
        if roi is not None and a.detect != 'global':
            region = np.zeros(body.shape, np.uint8)
            cv2.fillPoly(region, roi.outline(), 255)
        res = evaluate_pair(prev, cur, scan.angle(i - 1, a.sync), scan.angle(i, a.sync), body, spec, calib, gantry,
                            roi=roi, detect_mode=a.detect, roi_region=region, return_points=(i == target))
        if i == target:
            r, pts = res
            break
        if roi is not None:
            roi.advance(res.pop('_H'))
        prev = cur
    if pts is None:
        raise SystemExit('no valid score for this pair (gantry stopped or too few points)')
    A, B = prev, cur
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (A, B))
    e, n0 = pts['e'], pts['n0']
    D = np.linalg.norm(n0[:, None] - n0[None], axis=2)
    nn = np.argsort(D, 1)[:, :a.k]
    med = np.median(e[nn], axis=1)
    c = int(np.argmax(med))
    grp = nn[c]
    ctrl = np.argsort(e)[:a.k]
    print(f'pair {a.frame - 1}->{a.frame}: epi_enc_cl30 = {med[c]:.4f} px (CSV value {r["epi_enc_cl30"]:.4f}), '
          f'{len(e)} body points, cluster centre at raw ({pts["p0"][c][0]:.1f}, {pts["p0"][c][1]:.1f})')

    rows = []
    for name, idx in (('cluster', grp), ('control', ctrl)):
        for j in idx:
            t = point_tests(ga, gb, calib, pts['Fe'], pts['p0'][j])
            rows.append(dict(group=name, x_a=float(pts['p0'][j][0]), y_a=float(pts['p0'][j][1]),
                             x_b=float(pts['p1'][j][0]), y_b=float(pts['p1'][j][1]), residual=float(e[j]), **t))
    with open(out / 'points.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    def summary(name):
        g = [x for x in rows if x['group'] == name]
        col = lambda k: np.array([x[k] for x in g], float)
        return dict(n=len(g), residual_median=float(np.nanmedian(col('residual'))),
                    win_range_median=float(np.nanmedian(col('win_range'))),
                    e_win7_median=float(np.nanmedian(col('e_win7'))), e_win21_median=float(np.nanmedian(col('e_win21'))),
                    e_win31_median=float(np.nanmedian(col('e_win31'))),
                    sub_along_spread_median=float(np.nanmedian(col('sub_along_spread'))),
                    sub_perp_spread_median=float(np.nanmedian(col('sub_perp_spread'))),
                    ncc_median=float(np.nanmedian(col('ncc'))), lk_err_median=float(np.nanmedian(col('lk_err'))),
                    fb_median=float(np.nanmedian(col('fb'))))
    centre = pts['p0'][grp].mean(0)
    def bg_near(cen, radius=150):
        d = np.linalg.norm(pts['bg_p0'] - cen, axis=1)
        sel = d < radius
        return dict(n=int(sel.sum()), median=float(np.median(pts['bg_e'][sel])) if sel.any() else np.nan,
                    p90=float(np.percentile(pts['bg_e'][sel], 90)) if sel.any() else np.nan)
    summ = dict(frame=a.frame, score=float(med[c]), csv_score=float(r['epi_enc_cl30']), n_body=len(e),
                cluster=summary('cluster'), control=summary('control'),
                bed_near_cluster=bg_near(centre), bed_near_control=bg_near(pts['p0'][ctrl].mean(0)),
                bed_all=dict(n=len(pts['bg_e']), median=float(np.median(pts['bg_e']))))
    (out / 'summary.json').write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))

    fig, _ = figure(out / 'cluster.png', A, B, calib, pts['Fe'], grp, ctrl, e, pts['p0'], pts['p1'], a.frame, med[c])
    import matplotlib.pyplot as plt
    ax = fig.add_subplot(2, 3, 4)
    for x in rows:
        if x['group'] == 'cluster':
            ax.plot(WINDOWS, [x[f'e_win{w}'] for w in WINDOWS], color='#eb6834', alpha=0.6, lw=1)
        else:
            ax.plot(WINDOWS, [x[f'e_win{w}'] for w in WINDOWS], color='#1baf7a', alpha=0.35, lw=1)
    ax.set_xlabel('LK window (px)', fontsize=8)
    ax.set_ylabel('residual (px)', fontsize=8)
    ax.set_title('T1: residual vs window (orange = cluster, green = control)', fontsize=9)
    for k, (key, title) in enumerate((('sub_along_spread', 'T2: along-line spread inside window (depth mix)'),
                                      ('ncc', 'T3: NCC of matched 21x21 patches'))):
        ax = fig.add_subplot(2, 3, 5 + k)
        cl = [x[key] for x in rows if x['group'] == 'cluster' and np.isfinite(x[key])]
        co = [x[key] for x in rows if x['group'] == 'control' and np.isfinite(x[key])]
        ax.boxplot([cl, co])
        ax.set_xticks([1, 2], ['cluster', 'control'])
        ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out / 'cluster.png', dpi=120)
    print('saved', out)


if __name__ == '__main__':
    main()
