"""Benchmark person-segmentation models with permissive licences (Apache-2.0 / MIT) against hand-drawn body masks.

Models (MediaPipe image segmenters, Apache-2.0; U2-Net human segmentation, Apache-2.0):
  selfie_segmenter.tflite, deeplab_v3.tflite (class 15 = person), selfie_multiclass_256x256.tflite (everything except background),
  u2net_human_seg.onnx.
The image is rotated by 90 degrees (the patient lies along the image x axis), like for YOLO. Reports IoU / recall / precision inside the
detector ROI and on the whole image, and ms per frame (CPU). Needs: mediapipe, onnxruntime, opencv, numpy (run in a venv).

  python scripts/benchmark_permissive_masks.py --models <dir with the model files> --pair frame.jpg=mask.png ... --out report.txt
"""
import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import onnxruntime as ort
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision


def mp_segmenter(path, kind):
    seg = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
        base_options=mpt.BaseOptions(model_asset_path=str(path)), output_category_mask=True, output_confidence_masks=True))

    def run(rot_bgr):
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2RGB))
        r = seg.segment(img)
        cat = r.category_mask.numpy_view()
        if kind == 'selfie':
            conf = r.confidence_masks[0].numpy_view()   # selfie: index 0 = person confidence in this model
            return conf > 0.5 if conf.mean() < 0.9 else cat == 0
        if kind == 'deeplab':
            return cat == 15
        return cat != 0                                  # multiclass: anything but background
    return run


def u2net(path):
    sess = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    name = sess.get_inputs()[0].name
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

    def run(rot_bgr):
        h, w = rot_bgr.shape[:2]
        x = cv2.resize(cv2.cvtColor(rot_bgr, cv2.COLOR_BGR2RGB), (320, 320)).astype(np.float32) / 255.0
        x = ((x - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)
        y = sess.run(None, {name: x})[0][0, 0]
        y = (y - y.min()) / (y.max() - y.min() + 1e-9)
        return cv2.resize(y, (w, h)) > 0.5
    return run


def st(p, g, reg=None):
    if reg is not None:
        p, g = p & reg, g & reg
    i, u = (p & g).sum(), (p | g).sum()
    return i / u, i / g.sum(), i / max(p.sum(), 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--models', required=True)
    ap.add_argument('--pair', action='append', required=True)
    ap.add_argument('--roi', default='data/roi_arc22_calibrated.png')
    ap.add_argument('--rotate', default='cw', choices=['cw', 'ccw', 'none'])
    ap.add_argument('--out')
    a = ap.parse_args()
    d = Path(a.models)
    roi = cv2.imread(a.roi, 0) > 0
    pairs = [(cv2.imread(p.split('=')[0]), cv2.imread(p.split('=')[1], 0) > 0, Path(p.split('=')[0]).stem) for p in a.pair]
    models = {'MediaPipe selfie_segmenter': mp_segmenter(d / 'selfie_segmenter.tflite', 'selfie'),
              'MediaPipe deeplab_v3 (person)': mp_segmenter(d / 'deeplab_v3.tflite', 'deeplab'),
              'MediaPipe selfie_multiclass_256': mp_segmenter(d / 'selfie_multiclass_256x256.tflite', 'multi'),
              'U2-Net human_seg': u2net(d / 'u2net_human_seg.onnx')}
    rot = {'cw': cv2.ROTATE_90_CLOCKWISE, 'ccw': cv2.ROTATE_90_COUNTERCLOCKWISE}.get(a.rotate)
    unrot = {'cw': cv2.ROTATE_90_COUNTERCLOCKWISE, 'ccw': cv2.ROTATE_90_CLOCKWISE}.get(a.rotate)
    lines = []
    for name, fn in models.items():
        res, ts = [], []
        for im, gt, fid in pairs:
            x = cv2.rotate(im, rot) if rot is not None else im
            t = time.time()
            m = np.asarray(fn(x)).astype(np.uint8)
            ts.append(time.time() - t)
            m = (cv2.rotate(m, unrot) if unrot is not None else m) > 0
            if m.shape != gt.shape:
                m = cv2.resize(m.astype(np.uint8), (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            res.append((fid, st(m, gt, roi), st(m, gt)))
        lines.append(f'{name:34s} ROI IoU mean {np.mean([r[1][0] for r in res]):.3f} min {min(r[1][0] for r in res):.3f} recall min {min(r[1][1] for r in res):.3f}'
                     f' | full IoU mean {np.mean([r[2][0] for r in res]):.3f} min {min(r[2][0] for r in res):.3f} | {1000 * np.mean(ts):.0f} ms/frame')
        lines.append('    ROI IoU per frame: ' + ', '.join(f'{r[0]} {r[1][0]:.3f}' for r in res))
    text = '\n'.join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
