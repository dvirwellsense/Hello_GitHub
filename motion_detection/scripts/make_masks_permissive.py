"""Write body masks (body_<frame>.png, eroded like in evaluate_scan.py) for every frame of a scan with a permissively licensed model,
so that `evaluate_scan.py --body-masks <dir>` can use them. Run in an environment with mediapipe (see benchmark_permissive_masks.py).

  python scripts/make_masks_permissive.py --scan <scan_folder> --model selfie|multiclass --models <dir> --out <dir>
"""
import argparse
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_permissive_masks import mp_segmenter  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scan', required=True)
    ap.add_argument('--model', choices=['selfie', 'multiclass'], required=True)
    ap.add_argument('--models', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--erode', type=int, default=21)
    a = ap.parse_args()
    d = Path(a.models)
    fn = mp_segmenter(d / ('selfie_segmenter.tflite' if a.model == 'selfie' else 'selfie_multiclass_256x256.tflite'), 'selfie' if a.model == 'selfie' else 'multi')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted((Path(a.scan) / 'images').glob('frame_*.jpg')):
        idx = int(re.match(r'frame_0*(\d+)_', f.name).group(1))
        im = cv2.imread(str(f))
        m = np.asarray(fn(cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE))).astype(np.uint8) * 255
        m = cv2.rotate(m, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if m.shape != im.shape[:2]:
            m = cv2.resize(m, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
        if a.erode:
            m = cv2.erode(m, np.ones((a.erode, a.erode), np.uint8))
        cv2.imwrite(str(out / f'body_{idx:06d}.png'), m)
        n += 1
    print(a.scan, n, 'masks')


if __name__ == '__main__':
    main()
