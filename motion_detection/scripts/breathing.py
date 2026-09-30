"""Experimental breathing signal: encoder-pose epipolar distance on torso points between frames i-k and i.

Result so far (ARC22, VC01 scans): no separation between breath-hold and natural breathing.
Example:
  python scripts/breathing.py --scan scans/VC01_natural.zip --calib data/CalibrationResult_arc22.json \
      --gantry data/gantry_arc22.json --out results/VC01_natural_breathing.csv
"""
import argparse
import csv

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
    ap.add_argument('--gaps', default='2,4,8', help='frame gaps k')
    ap.add_argument('--sync', type=float, default=0.9)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    scan, calib, gantry = Scan(a.scan), Calibration(a.calib), GantryModel.load(a.gantry)
    gaps = [int(k) for k in a.gaps.split(',')]
    cache = {}

    def get(i):
        if i not in cache:
            img = scan.image(i)
            # erode: silhouette edges are depth discontinuities and give unreliable tracks
            cache[i] = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.erode(person_mask_seg(img), np.ones((31, 31), np.uint8)))
        return cache[i]

    rows = []
    for i in range(max(gaps), len(scan)):
        row = dict(frame=int(scan.index[i]), t=float(scan.time_s[i]), angle=float(scan.angle(i, a.sync)))
        gb, _ = get(i)
        for k in gaps:
            ga, ma = get(i - k)
            p0 = detect(ga, 1500, ma)
            p1, ok = track(ga, gb, p0, max_level=5)
            n0, n1 = calib.undistort_points(p0[ok]), calib.undistort_points(p1[ok])
            E = gantry.essential(scan.angle(i - k, a.sync), scan.angle(i, a.sync))
            e = epipolar_distance(E, n0, n1) * calib.focal_px
            row[f'epi_k{k}'] = float(np.median(e)) if len(e) > 20 else np.nan
        rows.append(row)
        for key in [q for q in cache if q < i - max(gaps)]:
            del cache[key]
    with open(a.out, 'w', newline='') as fh:
        w = csv.DictWriter(fh, rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print('saved', a.out)


if __name__ == '__main__':
    main()
