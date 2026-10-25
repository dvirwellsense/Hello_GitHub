# Synthetic motion injection (October 2026)

Static scans A07, A11, legs_static, A27: in every second measurable exposure-window pair, frame b was re-sampled so one
body region moves by a known amount (frame a and its mask untouched), then scored as in k16_2026_10 (E3, union masks,
k=16, threshold 0.5 px). Control pairs (no injection) reproduce the real run exactly (max diff 0.0000 px).
Sizes are mm at the bed-plane scale of the calibration (0.448 px per mm). Direction is the angle to the epipolar line
(0 = across, 90 = along). Regions: r40 / r80 = disc of 40 / 80 raw px radius (about 9 / 18 cm across) around a random
body point above the detector; body = the whole body. No threshold or setting was changed.

Files: <scan>.csv (one row per scored pair), report.txt (`python scripts/injection_report.py results/injection_2026_10/*.csv`).

Findings
- Score per mm of motion across the line: about 0.45 px. Noise floor of the control: median 0.10 px, max 0.29 px.
- With 0.5 px: 1.5 mm across the line is detected in 100% of pairs for r80 and the whole body; 1 mm in 37% (r80) and
  82% (whole body); 0.75 mm or less in none.
- Direction: what counts is the component across the line. 60 deg needs about 2-3 mm; along the line (90 deg) nothing
  is detected at any size up to 5 mm. This confirms the blind direction on real images.
- Small region on little texture: A27 (chest, 308 body points per pair) misses r40 motion in about half of the pairs
  even at 5 mm (k=8 0.76, k=16 0.53, k=30 0.26); scans with 500-1000 points detect all of them.
- Threshold trade-off on the real runs is in report.txt (0.3 px would detect 0.75 mm in 97% of r80 pairs, with 10
  alarm frames on negatives, most of them labelled pauses between motions in A44 and A12).
- Per pair only: with persistence 2 a motion that ends within one frame interval and is then held gives one suspect
  pair and no alarm.
