"""Frame-by-frame (causal) motion monitor: the decision of evaluate_scan.py with only the work it needs.

Same scores and alarms as evaluate_scan.py with the defaults of October 2026 (E3: --detect masked_abs
--body-fb-max 0.1, --roi-fixed, k=16, threshold 0.5 px, persistence 2, second check --chain-L 4 --chain-threshold 1.2,
--guard-action mark), but:
  - no background points while the gantry moves (they only feed the guard, the ROI homography and diagnostics);
    they are tracked only for same-angle pairs, where the common image shift is removed;
  - no diagnostic scores (affine, image-estimated geometry, k=8/30, angle sensitivity);
  - the second check reuses the body points and the first tracking step of the adjacent pair, and advances the
    open chains by one step per frame, instead of detecting and tracking L steps again.
The caller supplies, per frame: the image, the body mask (eroded, as saved by --save-masks), the gantry angle at
exposure and whether the pair ending at this frame is inside the exposure window. See scripts/realtime_check.py.
"""
import cv2
import numpy as np

from .rigidity import cluster_score, epipolar_distance, static_cluster_score
from .tracking import detect, track


class MotionMonitor:
    def __init__(self, calib, gantry, roi, threshold=0.5, static_threshold_mm=0.5, persistence=2, chain_L=4,
                 chain_threshold=1.2, body_fb_max=0.1, k=16, same_angle_deg=0.02, bg_margin_px=81):
        self.calib, self.gantry, self.roi = calib, gantry, roi
        self.region = roi.mask.astype(np.uint8) * 255
        self.f = calib.focal_px
        self.Kn = np.diag([1 / self.f, 1 / self.f, 1])
        self.thr, self.sthr, self.persistence = threshold, static_threshold_mm, persistence
        self.CL, self.cthr, self.fbmax, self.k = chain_L, chain_threshold, body_fb_max, k
        self.same_deg, self.bg_margin = same_angle_deg, bg_margin_px
        self.mm_per_px = 1.0 / calib.pixels_per_mm
        self.prev = None          # (gray, body, angle) of the previous frame
        self.chains = []          # open chains: dict(start_angle, n_steps, p0, cur)
        self.hist = []            # per pair: (exposure, same_angle) of the last chain_L pairs
        self.run_adj = self.run_chain = 0

    def _und(self, p):
        return self.calib.undistort_points(p) if len(p) else np.zeros((0, 2))

    def _Fe(self, a0, a1):
        return self.Kn.T @ self.gantry.essential(a0, a1) @ self.Kn

    def push(self, img, body, angle, exposure):
        """Process the next frame. Returns None for the first frame, else a dict for the pair previous -> this frame."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.prev is None:
            self.prev = (gray, body, angle)
            return None
        g0, body0, a0 = self.prev
        same = abs(angle - a0) < self.same_deg
        calib, f = self.calib, self.f
        # body points of the previous frame above the detector (E3), tracked to this frame
        m = ((body0 > 0).astype(np.uint8) * 255) & self.region
        q0 = detect(g0, mask=m, absolute_quality=True)
        q1, ok, fb = track(g0, gray, q0, return_fb=True)
        first = ok & (fb < self.fbmax)                 # also the first step of the new chain
        u0 = self._und(q0[first])
        sel = self.roi(u0)                            # same lookup as evaluate_pair (identity homography: --roi-fixed)
        b0 = u0[sel] * f
        b1 = self._und(q1[first][sel]) * f
        out = dict(n_body=int(len(b0)), same_angle=bool(same))
        score = np.nan
        if same:
            p0 = detect(g0)
            p1, okb = track(g0, gray, p0)
            p0, p1 = p0[okb], p1[okb]
            xi, yi = p0[:, 0].astype(int), p0[:, 1].astype(int)
            on_bg = cv2.dilate(body0, np.ones((self.bg_margin, self.bg_margin), np.uint8))[yi, xi] == 0
            n0, n1 = calib.undistort_points(p0) * f, calib.undistort_points(p1) * f
            shift = np.median(n1[on_bg] - n0[on_bg], axis=0) if on_bg.sum() > 20 else np.zeros(2)
            score = static_cluster_score(b0, b1 - b0 - shift)
            hit = np.isfinite(score) and score * self.mm_per_px > self.sthr
            out['static_cl'] = score
        else:
            if len(b0) >= 8:
                score = cluster_score(b0, epipolar_distance(self._Fe(a0, angle), b0, b1), self.k)
            hit = np.isfinite(score) and score > self.thr
            out['epi_enc_cl16'] = score
        self.run_adj = self.run_adj + 1 if (exposure and hit) else 0
        out.update(suspect=bool(hit), alarm_adjacent=self.run_adj >= self.persistence,
                   motion_state='NO_BODY_DATA' if not np.isfinite(score) else ('SUSPECT' if hit else 'NO_MOTION'))
        # second check: advance the open chains by one step, open a new one, score the chain that reaches L steps
        chain_score = np.nan
        if self.CL:
            for c in self.chains:
                if len(c['cur']):
                    nxt, okc, fbc = track(g0, gray, c['cur'], return_fb=True)
                    keep = okc & (fbc < self.fbmax)
                    c['p0'], c['cur'] = c['p0'][keep], nxt[keep]
                c['n_steps'] += 1
            self.chains.append(dict(start_angle=a0, n_steps=1, p0=q0[first], cur=q1[first]))
            done = [c for c in self.chains if c['n_steps'] == self.CL]
            self.chains = [c for c in self.chains if c['n_steps'] < self.CL]
            if done:
                c = done[0]
                if len(c['p0']) >= self.k:
                    n0, n1 = calib.undistort_points(c['p0']) * f, calib.undistort_points(c['cur']) * f
                    chain_score = cluster_score(n0, epipolar_distance(self._Fe(c['start_angle'], angle), n0, n1), self.k)
            self.hist = (self.hist + [(bool(exposure), bool(same))])[-self.CL:]
            eligible = len(self.hist) == self.CL and all(e and not s for e, s in self.hist)
            csus = bool(eligible and np.isfinite(chain_score) and chain_score > self.cthr)
            self.run_chain = self.run_chain + 1 if csus else 0
            out.update(chain_cl16=chain_score, chain_suspect=csus, alarm_chain=self.run_chain >= self.persistence)
        out['alarm'] = bool(out['alarm_adjacent'] or out.get('alarm_chain', False))
        self.prev = (gray, body, angle)
        return out
