"""Loading a recorded scan folder (frames.csv + images/)."""
import csv
import zipfile
from pathlib import Path

import cv2
import numpy as np


def resolve_scan_dir(path, extract_to=None):
    """Accept a scan folder or a ZIP; return the folder that holds frames.csv."""
    path = Path(path)
    if path.suffix.lower() == '.zip':
        extract_to = Path(extract_to or path.with_suffix(''))
        with zipfile.ZipFile(path) as zf:
            zf.extractall(extract_to)
        path = extract_to
    for d in [path, *path.rglob('*')]:
        if d.is_dir() and (d / 'frames.csv').exists():
            return d
    raise FileNotFoundError(f'frames.csv not found under {path}')


class Scan:
    def __init__(self, path):
        self.dir = resolve_scan_dir(path)
        rows = list(csv.DictReader(open(self.dir / 'frames.csv', encoding='utf-8-sig')))
        self.index = np.array([int(r['image_index']) for r in rows])
        self.files = [self.dir / r['image_file'].replace('\\', '/') for r in rows]
        self.angle_before = np.array([float(r['arm_angle_before_capture']) for r in rows])
        self.angle_after = np.array([float(r['arm_angle_after_capture']) for r in rows])
        t = np.array([np.datetime64(r['capture_request_utc'][:23]) for r in rows])
        self.time_s = (t - t[0]) / np.timedelta64(1, 'ms') / 1000.0

    def __len__(self):
        return len(self.files)

    def image(self, i):
        return cv2.imread(str(self.files[i]))

    def angle(self, i, sync=0.9):
        """Gantry angle at exposure. sync=0 -> reading before capture, 1 -> after.
        Fitted on ARC22 data: the image matches ~0.9 (close to the reading after capture)."""
        i = np.asarray(i)
        return self.angle_before[i] + sync * (self.angle_after[i] - self.angle_before[i])
