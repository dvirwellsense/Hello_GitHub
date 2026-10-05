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
        self.time_s = (t - t[0]) / np.timedelta64(1, 'ms') / 1000.0          # capture request (angle_before read)
        ts = np.array([np.datetime64(r['image_saved_utc'][:23]) for r in rows])
        self.saved_s = (ts - t[0]) / np.timedelta64(1, 'ms') / 1000.0        # image saved (angle_after read)

    def __len__(self):
        return len(self.files)

    def image(self, i):
        return cv2.imread(str(self.files[i]))

    def stationary(self, i, eps_deg=0.02):
        """Gantry did not move while frame i was captured."""
        return abs(self.angle_after[i] - self.angle_before[i]) < eps_deg

    def phases(self, eps_deg=0.02, min_away_deg=5.0):
        """Label every frame: 'approach' (home -> start angle), 'dwell_start', 'sweep', 'dwell_end', 'return'.
        Dwells are stationary runs far from the home angle; the sweep is between the first dwell and the
        next dwell on the opposite side. Without two dwells everything is labelled 'sweep'."""
        n = len(self)
        st = [self.stationary(i, eps_deg) for i in range(n)]
        home = self.angle(0)
        runs, i = [], 0
        while i < n:
            if st[i]:
                j = i
                while j + 1 < n and st[j + 1]:
                    j += 1
                if j > i and abs(self.angle(i) - home) > min_away_deg:
                    runs.append((i, j))
                i = j + 1
            else:
                i += 1
        lab = np.array(['sweep'] * n, dtype=object)
        if len(runs) < 2:
            return lab
        ds = runs[0]
        de = next((r for r in runs[1:] if np.sign(self.angle(r[0]) - home) != np.sign(self.angle(ds[0]) - home)), None)
        if de is None:
            return lab
        lab[:ds[0]] = 'approach'
        lab[ds[0]:ds[1] + 1] = 'dwell_start'
        lab[de[0]:de[1] + 1] = 'dwell_end'
        lab[de[1] + 1:] = 'return'
        return lab

    def exposure_pairs(self):
        """Boolean per frame i (pair i-1 -> i): both frames inside [last start-dwell frame, first end-dwell frame]."""
        lab = self.phases()
        idx = np.arange(len(self))
        if 'dwell_start' not in lab:
            return np.ones(len(self), bool)
        first = idx[lab == 'dwell_start'].max()
        last = idx[lab == 'dwell_end'].min()
        return (idx - 1 >= first) & (idx <= last)

    def angle(self, i, sync=0.9):
        """Gantry angle at exposure. sync=0 -> reading before capture, 1 -> after.
        Fitted on ARC22 data: the image matches ~0.9 (close to the reading after capture)."""
        i = np.asarray(i)
        return self.angle_before[i] + sync * (self.angle_after[i] - self.angle_before[i])
