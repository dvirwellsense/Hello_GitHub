"""Sensitivity by synthetic motion: move part of the body by a known amount in a real static pair, then score.

For pairs of a scan WITHOUT patient motion (exposure window, gantry moving, measurable), frame b of the pair is
re-sampled so that one body region is translated by a known displacement; frame a and its body mask are not
touched. The pair is then scored exactly as evaluate_scan.py does (E3: --detect masked_abs --body-fb-max 0.1,
--roi-fixed, saved union masks). Pairs with d = 0 are the control and must reproduce the CSV of the real run.

Displacement:
  size      mm at the bed-plane scale of the calibration (m_fPixelsInMM; 0.448 undistorted px per mm), applied
            as a translation in undistorted-image pixels and mapped back to raw pixels pixel by pixel.
  direction angle to the epipolar line through the region centre: 0 = across the line (fully visible to the
            score), 90 = along the line (invisible by construction).
  baseline  --baseline L scores the pair i-L -> i, as evaluate_scan.py --baseline L (slow drift).
  region    disc of the given radius (raw px, ~1.1 mm per raw px at the bed plane) around a random body point
            above the detector, intersected with the body; or 'body' = the whole body (patient slides).
Region edges are feathered over a few pixels. Nothing here changes the score or any threshold.

Example:
  python scripts/inject_motion.py --scan <scan_dir> --body-masks <masks_dir> --csv results/k16_2026_10/A07.csv \
      --out results/injection_2026_10/A07.csv
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
from motion.rigidity import evaluate_pair
from motion.roi import DetectorROI
from motion.scan import Scan

SIZES_MM = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)
ANGLES_DEG = (0, 30, 60, 90)
REGIONS = ('r40', 'r80', 'body')
FEATHER_PX = 4

G = {}


def init(a):
    G['a'] = a
    G['scan'] = Scan(a.scan)
    G['calib'] = Calibration(a.calib)
    G['gantry'] = GantryModel.load(a.gantry)
    G['roi'] = DetectorROI(a.detector_mask, G['calib'])


def score(img_a, img_b, ang_a, ang_b, body):
    a, roi = G['a'], G['roi']
    r = evaluate_pair(img_a, img_b, ang_a, ang_b, body, np.zeros_like(body), G['calib'], G['gantry'], roi=roi,
                      detect_mode='masked_abs', roi_region=roi.mask.astype(np.uint8) * 255, body_fb_max=a.body_fb_max)
    return {k: r[k] for k in ('n_body', 'epi_enc_cl', 'epi_enc_cl16', 'epi_enc_cl30', 'epi_enc_med')}


def region_pixels(weight):
    """Pixels of the region and their undistorted coordinates (computed once per region)."""
    ys, xs = np.nonzero(weight > 0)
    return ys, xs, G['calib'].undistort_points(np.c_[xs, ys].astype(np.float64)) * G['calib'].focal_px


def warp(img, weight, pix, n_a_centre, d_px, ang_deg):
    """Translate the weighted region of img by d_px undistorted pixels at ang_deg to the epipolar line.
    The line is taken through the region centre's position in frame b (a few px from its position in a)."""
    calib, gantry = G['calib'], G['gantry']
    f = calib.focal_px
    Kn = np.diag([1 / f, 1 / f, 1])
    Fe = Kn.T @ gantry.essential(G['ang_a'], G['ang_b']) @ Kn
    line = np.r_[n_a_centre, 1] @ Fe.T
    normal = line[:2] / np.hypot(*line[:2])
    tangent = np.array([-normal[1], normal[0]])
    t = np.radians(ang_deg)
    u = np.cos(t) * normal + np.sin(t) * tangent  # unit vector, undistorted px
    ys, xs, n = pix
    src = calib.distort_points((n - d_px * u * weight[ys, xs, None]) / f)  # backward map: where each pixel comes from
    mx, my = np.meshgrid(np.arange(img.shape[1], dtype=np.float32), np.arange(img.shape[0], dtype=np.float32))
    mx[ys, xs], my[ys, xs] = src[:, 0], src[:, 1]
    return cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def run_pair(job):
    i, seed = job
    a, scan, calib = G['a'], G['scan'], G['calib']
    j = i - a.baseline  # frame a of the pair
    img_a, img_b = scan.image(j), scan.image(i)
    frame_a = int(scan.index[j])
    body = cv2.imread(str(Path(a.body_masks) / f'body_{frame_a:06d}.png'), cv2.IMREAD_GRAYSCALE)
    G['ang_a'], G['ang_b'] = float(scan.angle(j, a.sync)), float(scan.angle(i, a.sync))
    base = dict(frame=int(scan.index[i]), angle=G['ang_b'])
    rows = [dict(base, region='none', size_mm=0.0, angle_deg=0, **score(img_a, img_b, G['ang_a'], G['ang_b'], body))]
    # body extent before erosion (the saved masks are eroded by 21 px)
    full = cv2.dilate(body, np.ones((21, 21), np.uint8)) > 0
    ys, xs = np.nonzero((body > 0) & G['roi'].mask)
    if len(xs) == 0:
        return rows
    rng = np.random.default_rng(seed)
    k = rng.integers(len(xs))
    cx, cy = int(xs[k]), int(ys[k])
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
        pix = region_pixels(w)
        n_c = calib.undistort_points(np.array([centre])) * calib.focal_px
        for size in a.sizes:
            d_px = size * calib.pixels_per_mm
            for ang in a.angles:
                b2 = warp(img_b, w, pix, n_c[0], d_px, ang)
                rows.append(dict(base, region=region, size_mm=size, angle_deg=ang, cx=centre[0], cy=centre[1],
                                 **score(img_a, b2, G['ang_a'], G['ang_b'], body)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--body-masks', required=True)
    ap.add_argument('--csv', required=True, help='CSV of the real run: which pairs are in the window and measurable')
    ap.add_argument('--calib', default='data/CalibrationResult_arc22.json')
    ap.add_argument('--gantry', default='data/gantry_arc22.json')
    ap.add_argument('--detector-mask', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--body-fb-max', type=float, default=0.1)
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--baseline', type=int, default=1,
                    help='pair i-L -> i (use the CSV of an evaluate_scan.py run with the same --baseline). The injected '
                         'size is then the displacement accumulated over L frames: a drift of size/L mm per frame')
    ap.add_argument('--sizes', type=lambda v: tuple(float(x) for x in v.split(',')), default=SIZES_MM)
    ap.add_argument('--angles', type=lambda v: tuple(int(x) for x in v.split(',')), default=ANGLES_DEG)
    ap.add_argument('--regions', type=lambda v: tuple(v.split(',')), default=REGIONS)
    ap.add_argument('--every', type=int, default=2, help='use every n-th eligible pair')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    real = list(csv.DictReader(open(a.csv)))
    ok = [int(r['frame']) for r in real if r['exposure'] == 'True' and r['same_angle'] == 'False'
          and r['epi_enc_cl16'] not in ('', 'nan')][::a.every]
    scan = Scan(a.scan)
    pos = {int(f): i for i, f in enumerate(scan.index)}
    jobs = [(pos[f], f) for f in ok]
    with Pool(a.workers, init, (a,)) as p:
        rows = [r for rr in p.map(run_pair, jobs, chunksize=1) for r in rr]
    keys = list(dict.fromkeys(k for r in rows for k in r))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys, restval='')
        w.writeheader()
        w.writerows(rows)
    ref = {int(r['frame']): float(r['epi_enc_cl16']) for r in real if r['epi_enc_cl16'] not in ('', 'nan')}
    ctl = [(r['frame'], r['epi_enc_cl16'], ref.get(r['frame'])) for r in rows if r['region'] == 'none']
    dev = max(abs(s - t) for _, s, t in ctl if t is not None and np.isfinite(s))
    print(f'{len(jobs)} pairs, {len(rows)} scored; control vs real run max |diff| k16 = {dev:.4f} px')


if __name__ == '__main__':
    main()
