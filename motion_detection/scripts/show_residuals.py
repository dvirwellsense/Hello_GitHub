"""Draw per-point encoder-epipolar residuals of one frame pair (green <1 px, orange 1-2 px, red >2 px).

Example:
  python scripts/show_residuals.py --scan scans/A28_1.zip --calib data/CalibrationResult_arc22.json \
      --gantry data/gantry_arc22.json --frame 70 --out results/A28_1_frame70.jpg
"""
import argparse

import cv2
import numpy as np

import _path  # noqa: F401
from motion.calib import Calibration
from motion.gantry import GantryModel
from motion.masks import person_mask_seg
from motion.rigidity import epipolar_distance
from motion.scan import Scan
from motion.tracking import detect, track


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--calib', required=True)
    ap.add_argument('--gantry', required=True)
    ap.add_argument('--frame', type=int, required=True, help='image_index of the second frame of the pair')
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    scan, calib, gantry = Scan(a.scan), Calibration(a.calib), GantryModel.load(a.gantry)
    i = int(np.where(scan.index == a.frame)[0][0])
    A, B = scan.image(i - 1), scan.image(i)
    ga, gb = (cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) for x in (A, B))
    m = person_mask_seg(A)
    p0 = detect(ga, 3000, m)
    p1, ok = track(ga, gb, p0)
    p0, p1 = p0[ok], p1[ok]
    E = gantry.essential(scan.angle(i - 1, a.sync), scan.angle(i, a.sync))
    e = epipolar_distance(E, calib.undistort_points(p0), calib.undistort_points(p1)) * calib.focal_px
    v = B.copy()
    cnt, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(v, cnt, -1, (255, 255, 0), 2)
    for (x, y), r in zip(p1, e):
        c = (0, 0, 255) if r > 2 else (0, 200, 255) if r > 1 else (0, 200, 0)
        cv2.circle(v, (int(x), int(y)), 4 if r > 1 else 2, c, -1)
    cv2.putText(v, f'frame {a.frame}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(a.out, v)
    print('saved', a.out)


if __name__ == '__main__':
    main()
