"""Build long KLT tracks over a frame range. Output: obs list (track_id, frame, u, v raw px) + labels."""
from common import *
import sys
F0, F1 = int(sys.argv[1]), int(sys.argv[2])   # 0-based inclusive range
pos = {}   # track id -> list of (frame, x, y)
label = {} # track id -> 'ph' / 'bg' / 'spec'
nid = 0
g0 = gray(F0); im0 = cv2.imread(files[F0])
def new_points(g, im, existing):
    m = np.full(g.shape, 255, np.uint8)
    for p in existing: cv2.circle(m, (int(p[0]), int(p[1])), 8, 0, -1)
    p = cv2.goodFeaturesToTrack(g, 2500, 0.005, 8, mask=m)
    return np.zeros((0, 2), np.float32) if p is None else p.reshape(-1, 2)
def lab(im, pts):
    pm = phantom_mask(im); sp = specular_mask(im) & pm; bgm = cv2.dilate(pm, np.ones((41, 41), np.uint8))
    out = []
    for x, y in pts.astype(int):
        out.append('spec' if sp[y, x] else 'ph' if pm[y, x] else 'bg' if not bgm[y, x] else 'edge')
    return out
cur = new_points(g0, im0, []); ids = list(range(len(cur))); nid = len(cur)
for k, (i, l) in enumerate(zip(ids, lab(im0, cur))): pos[i] = [(F0, *cur[k])]; label[i] = l
for f in range(F0 + 1, F1 + 1):
    g1 = gray(f); im1 = cv2.imread(files[f])
    p1, ok = track(g0, g1, cur.reshape(-1, 1, 2).astype(np.float32))
    p1 = p1.reshape(-1, 2)
    ok &= (p1[:, 0] > 2) & (p1[:, 0] < 1277) & (p1[:, 1] > 2) & (p1[:, 1] < 717)
    cur = p1[ok]; ids = [i for i, o in zip(ids, ok) if o]
    for i, p in zip(ids, cur): pos[i].append((f, *p))
    new = new_points(g1, im1, cur)
    for p, l in zip(new, lab(im1, new)):
        pos[nid] = [(f, *p)]; label[nid] = l; ids.append(nid); nid += 1
    cur = np.vstack([cur, new]).astype(np.float32); g0 = g1
obs = [(t, f, x, y) for t, L in pos.items() if len(L) >= 8 for f, x, y in L]
lab_arr = {t: label[t] for t in pos if len(pos[t]) >= 8}
np.save(sys.argv[3], np.array(obs)); np.save(sys.argv[3].replace('.npy', '_lab.npy'), np.array(list(lab_arr.items()), dtype=object))
from collections import Counter
print(len(lab_arr), 'tracks', Counter(lab_arr.values()), 'obs', len(obs))
