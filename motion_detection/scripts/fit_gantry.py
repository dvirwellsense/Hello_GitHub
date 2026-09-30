"""Fit the gantry camera model from a STATIC scan (phantom or empty bed).

Example:
  python scripts/fit_gantry.py --scan scans/phantom_static --calib data/CalibrationResult_arc22.json \
      --first 40 --last 110 --out data/gantry_arc22.json
"""
import argparse

import cv2
import numpy as np

import _path  # noqa: F401
from motion.calib import Calibration
from motion.gantry import fit_gantry, initial_camera
from motion.masks import phantom_mask, person_mask_seg
from motion.scan import Scan
from motion.tracking import build_tracks


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--first', type=int, required=True, help='first frame index (0-based) of the sweep')
    ap.add_argument('--last', type=int, required=True)
    ap.add_argument('--mask', choices=('phantom', 'seg', 'none'), default='phantom', help='object to exclude from fitting points')
    ap.add_argument('--points', type=int, default=600)
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    scan, calib = Scan(a.scan), Calibration(a.calib)

    def label(img, pts):
        m = {'phantom': phantom_mask, 'seg': person_mask_seg}.get(a.mask, lambda im: np.zeros(im.shape[:2], np.uint8))(img)
        m = cv2.dilate(m, np.ones((41, 41), np.uint8))
        return ['obj' if m[int(y), int(x)] else 'bg' for x, y in pts]

    obs, lab = build_tracks(scan, a.first, a.last, label)
    tid = obs[:, 0].astype(int)
    lens = np.bincount(tid)
    bg = [t for t, l in lab.items() if l == 'bg' and lens[t] >= 25]
    sel = np.random.default_rng(0).choice(bg, min(a.points, len(bg)), replace=False)
    o = obs[np.isin(tid, sel)]
    u, pid = np.unique(o[:, 0].astype(int), return_inverse=True)
    fr = o[:, 1].astype(int)
    uv = calib.undistort_points(o[:, 2:4])
    angles = scan.angle(fr, a.sync)
    best = None
    for sign in (1, -1):
        for tilt in (-20, 0, 20):
            model, _, r = fit_gantry(pid, uv, angles, len(u), calib.focal_px, sign, initial_camera(tilt))
            print(f'sign {sign:+d} tilt {tilt:+d}: median reprojection {np.median(r):.3f} px')
            if best is None or np.median(r) < best[0]:
                best = (np.median(r), model)
    best[1].save(a.out, sync=a.sync, median_reprojection_px=float(best[0]), source_scan=str(scan.dir.name))
    print('saved', a.out)


if __name__ == '__main__':
    main()
