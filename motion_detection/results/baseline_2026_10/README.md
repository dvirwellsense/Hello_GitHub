# Slow drift: comparing frame i with frame i-L (October 2026)

Question: per adjacent pair the score sees about 1.1-1.5 mm (injection, results/injection_2026_10), so a slow drift
below that per frame interval is never seen, and a motion that is over within one frame is one suspect pair (no alarm
with persistence 2). Does a longer baseline (pair i-L -> i) help?

Runs
- L1, L2, L4, L8: `evaluate_scan.py --baseline L` (E3, union masks, --roi-fixed, --guard none), all 11 scans.
  L1 reproduces k16_2026_10 exactly. Report: report.txt (`python scripts/baseline_report.py results/baseline_2026_10`).
- chained/: `scripts/chained_baseline_study.py` on the static scans and the rigid chest phantom: the same pairs with
  points tracked directly (one KLT step i-L -> i) or chained through every frame in between, plus static bed/floor
  points near the detector (chained).
No threshold or setting of the default pipeline was changed.

Findings (k=16 score, px; ~0.45 px per mm across the line)
- Noise on static scans grows with L: median 0.10 / 0.16 / 0.31 / 0.52 px and max 0.31 / 1.06 / 2.5 / 7.2 px for
  L = 1 / 2 / 4 / 8 (direct). With the 0.5 px threshold, L=8 alarms on almost every pair of the static scans.
- Direct tracking over large angles loses body points (A07: ~1000 -> ~270 at L=8) and gives the outliers; chained
  tracking keeps the points and removes the outliers (A11 L8 max 7.2 -> 0.9 px), but the median still grows (and is
  higher than direct for legs_static: accumulated tracking error).
- The growth is also on things that cannot move: the rigid phantom (chained median 0.23 -> 1.23 px from L1 to L8) and
  static bed/floor points (A07 0.53 -> 2.6 px). So most of it is geometry (gantry model error ~ proportional to the arc
  angle between the frames; the model fit has a median reprojection error of 2.4 px over long tracks), not patient
  drift.
- What a longer baseline would buy as it is (threshold set above the static maximum, not applied): chained L4
  max 1.3 px -> about 0.8 mm per frame of drift, or a step of about 3 mm; L1 needs about 1.1-1.5 mm per frame.
  A modest gain, at the price of a second threshold.
