import csv, glob, numpy as np, cv2
from calib import *
import os
SCAN = os.environ.get('SCAN_DIR') or glob.glob('scan/arc*')[0]
rows = list(csv.DictReader(open(SCAN + '/frames.csv', encoding='utf-8-sig')))
angle = np.array([float(r['arm_angle_effective']) for r in rows])
a_before = np.array([float(r['arm_angle_before_capture']) for r in rows])
a_after = np.array([float(r['arm_angle_after_capture']) for r in rows])
files = [SCAN + '/images/' + r['image_file'].split('\\')[-1] for r in rows]
def gray(i): return cv2.cvtColor(cv2.imread(files[i]), cv2.COLOR_BGR2GRAY)
def phantom_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (3, 60, 80), (25, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    k = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])
    return np.where(lab == k, 255, 0).astype(np.uint8)
def specular_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (0, 0, 215), (180, 90, 255))
    return cv2.dilate(m, np.ones((21, 21), np.uint8))
LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))
def track(g0, g1, p0, fb=0.5):
    p1, s, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **LK)
    pb, sb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **LK)
    ok = (s.ravel() == 1) & (sb.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < fb)
    return p1, ok
