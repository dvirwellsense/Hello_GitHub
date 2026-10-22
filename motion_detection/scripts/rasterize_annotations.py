"""Rasterise the polygons of tools/mask_annotator.html (user_annotations.json, schema arc-full-boundary-v1) into PNG masks.

Polygons are applied in order: 'add' paints 255, 'subtract' paints 0. Vertices are rounded to integer pixels (OpenCV fillPoly),
like the annotation pipeline of the other chat.  Output name: <scan>_<frame:06d>_mask.png (1280x720 uint8, 255 = body).

  python scripts/rasterize_annotations.py user_annotations.json --out masks_human/
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def rasterize(ann):
    m = np.zeros((ann['height'], ann['width']), np.uint8)
    for poly in ann['polygons']:
        pts = np.round(np.array(poly['points'])).astype(np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(m, [pts], 255 if poly['mode'] == 'add' else 0)
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('json')
    ap.add_argument('--out', default='masks_human')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for ann in json.load(open(a.json))['annotations']:
        p = out / f"{ann['scan'] or 'scan'}_{int(ann['frame']):06d}_mask.png"
        cv2.imwrite(str(p), rasterize(ann))
        print('wrote', p)


if __name__ == '__main__':
    main()
