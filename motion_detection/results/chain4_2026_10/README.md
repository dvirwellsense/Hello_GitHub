# Second check: chained L=4 (October 2026)

`scripts/chain_check.py`: pair i-4 -> i, body points of frame i-4 (E3 detection, union masks) tracked through every frame
in between (forward-backward 0.1 px per step), k=16 cluster score with the original gantry model. score/ = all 11 scans
and the phantom, no injection; inject/ = static scans A07, A11, legs_static, A27, every second pair, synthetic drift
(frames shifted by v, 2v, 3v, 4v) or step (frames from the middle of the chain shifted by s). report.txt =
`python scripts/chain_check_report.py`. Thresholds are candidates; nothing is applied to the default pipeline.

Findings
- No injection: static scans + phantom median 0.33 px, 99th pct 1.13, max 1.32 (phantom; legs_static 1.18).
- Threshold 1.2 px (persistence 2, its own alarm, OR-ed with the adjacent-pair alarm):
  - no alarm on any negative span (0/552) and none on the phantom (one suspect pair, no alarm);
  - drift across the line: 0.75 mm per frame detected in 97% (r80) / 100% (whole body) of pairs; the adjacent-pair
    check detects 0% at that rate (it needs ~1-1.5 mm per frame). 0.5 mm per frame: 3% / 21%.
  - step and hold across the line: 3 mm detected in 97-100%; the adjacent-pair check sees it in one pair only (no alarm).
  - 60 deg to the line it is much weaker (drift 1 mm per frame: 2% r80 / 17% body; step 5 mm: 44% / 97%).
  - alarm frames it adds on the real scans all lie within 3 frames after labelled motion (the chain still spans it),
    e.g. A30 65-67 and 70-72, A44 77-79, A12 62-64, A08 88. So the alarm lasts up to 3 frames longer.
- 1.0 px: drift 0.5 mm/frame 40-95%, but 8 alarm frames on negative spans (A12 65-69, legs_static 88, 89, 93).
- 1.4 px: no false alarms either, drift 0.75 mm/frame drops to 59% (r80).
- The margin at 1.2 is thin: the largest static values are 1.18 (legs_static) and 1.32 (phantom, single pair).
