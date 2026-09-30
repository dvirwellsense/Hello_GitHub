"""KLT point tracking with forward-backward check."""
import cv2
import numpy as np

LK = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))


def detect(gray, max_points=3000, mask=None, min_distance=7):
    p = cv2.goodFeaturesToTrack(gray, max_points, 0.005, min_distance, mask=mask)
    return np.zeros((0, 2), np.float32) if p is None else p.reshape(-1, 2)


def track(g0, g1, p0, fb_max_px=0.5, max_level=4):
    """Track points g0 -> g1. Returns (p1, ok) where ok passes status and forward-backward checks."""
    if len(p0) == 0:
        return p0.copy(), np.zeros(0, bool)
    lk = dict(LK, maxLevel=max_level)
    p0 = np.float32(p0).reshape(-1, 1, 2)
    p1, s, _ = cv2.calcOpticalFlowPyrLK(g0, g1, p0, None, **lk)
    pb, sb, _ = cv2.calcOpticalFlowPyrLK(g1, g0, p1, None, **lk)
    fb = np.linalg.norm((pb - p0).reshape(-1, 2), axis=1)
    ok = (s.ravel() == 1) & (sb.ravel() == 1) & (fb < fb_max_px)
    return p1.reshape(-1, 2), ok


def build_tracks(scan, first, last, label_fn, min_len=8):
    """Long tracks over frames [first, last] with re-detection in empty areas.
    label_fn(image, points) -> list of labels. Returns obs array (track, frame, x, y) and {track: label}."""
    pos, label = {}, {}

    def new_points(gray, img, existing):
        m = np.full(gray.shape, 255, np.uint8)
        for x, y in existing:
            cv2.circle(m, (int(x), int(y)), 8, 0, -1)
        p = detect(gray, 2500, m, 8)
        return p, label_fn(img, p)

    img = scan.image(first)
    g0 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cur, labs = new_points(g0, img, [])
    ids = list(range(len(cur)))
    for k, l in zip(ids, labs):
        pos[k] = [(first, *cur[k])]
        label[k] = l
    nid = len(cur)
    for f in range(first + 1, last + 1):
        img = scan.image(f)
        g1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        p1, ok = track(g0, g1, cur)
        ok &= (p1[:, 0] > 2) & (p1[:, 0] < g1.shape[1] - 3) & (p1[:, 1] > 2) & (p1[:, 1] < g1.shape[0] - 3)
        cur = p1[ok]
        ids = [i for i, o in zip(ids, ok) if o]
        for i, p in zip(ids, cur):
            pos[i].append((f, *p))
        new, labs = new_points(g1, img, cur)
        for p, l in zip(new, labs):
            pos[nid] = [(f, *p)]
            label[nid] = l
            ids.append(nid)
            nid += 1
        cur = np.vstack([cur, new]).astype(np.float32)
        g0 = g1
    obs = np.array([(t, f, x, y) for t, L in pos.items() if len(L) >= min_len for f, x, y in L])
    return obs, {t: label[t] for t in pos if len(pos[t]) >= min_len}
