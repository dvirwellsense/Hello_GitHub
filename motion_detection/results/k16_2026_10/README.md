# Full re-run with cluster size 16 (the new default)

`evaluate_scan.py` now takes `--cluster-k` (8, 16, 30; default 16; `--cluster-k 30` reproduces the earlier behaviour).
E3 with the union masks, calibrated ROI, threshold 0.5 px, persistence 1. One CSV per scan (+ settings).

Exposure-window result, k=16 (moving = pairs the labeller confirmed):
| scan | moving pairs flagged | suspects outside |
|---|---|---|
| A30 | 12 of 12 | 0 |
| A44 | 20 of 20 (k=30: 17) | 1 (90, no motion: false alarm) |
| A08 | 14 of 14 | 0 |
| A12 | 5 of 5 | 1 (66, no motion: false alarm) |
| A26 | 4 of 9 (k=30: 3) | 0 |
| A28_1 | 24 of 24 labelled (pauses not checked) | not re-examined |
| A07, A11, legs, VC01 breath hold | no motion | 0 |

Persistence (post hoc, same CSVs): requiring 2 consecutive suspect pairs removes both false alarms (A44 90, A12 66) and
keeps every motion segment detected; it loses one isolated pair (A26 68) and delays an alarm by one pair. Not the default
(`--persistence` is still 1); 3 pairs gives the same on these scans. Tested on 10 scans only.
