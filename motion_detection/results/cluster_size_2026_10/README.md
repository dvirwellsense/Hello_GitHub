# Cluster size k of the score: one-factor study

`scripts/cluster_size_study.py` (per scan) and `scripts/cluster_size_report.py` (summary, `report.txt`).
Only k changes (median residual of the k nearest points, max over points). E3, union masks, calibrated ROI,
threshold 0.5 px for every k, no other change.

Negatives (about 500 pairs): static scans (legs, A07, A11, VC01 breath hold), the pauses confirmed by the labeller
(A30 65-67; A44 58, 62, 69, 73, 74, 77, 78, 81, 82) and unlabelled pairs more than 2 pairs from any label.
Moving pairs: the 32 pairs the labeller confirmed on A30 and A44.

| k | alarms at 0.5 on negatives | moving detected at 0.5 | A44 67 / 70 / 71 | new alarms vs k=30 |
|---|---|---|---|---|
| 8 | 13 of 512 | 32 of 32 | 6.5 / 2.1 / 9.0 | static legs 84, 85 (2.4, 4.7), A08 104, 105, 113, 116, 117, A44 73, ... |
| 12 | 4 of 510 | 32 of 32 | 6.4 / 2.1 / 5.2 | A30 67 (a pause, 0.51), A08 116 |
| 16 | 2 of 506 | 32 of 32 | 6.3 / 1.1 / 2.6 | none (same A44 90, A12 66 as k=30) |
| 20 | 2 of 505 | 31 of 32 | 5.6 / 1.7 / 0.3 | none |
| 30 (current) | 2 of 503 | 29 of 32 | 0.2 / 0.3 / 0.2 | - |
| 45 | 1 of 500 | 25 of 32 | 0.1 / 0.2 / 0.2 | - |

The two alarms that remain at k=16-30 (A44 pair 90, A12 pair 66) are borderline false alarms (0.52-0.55): the labeller confirmed no motion in either.
Caveat: k=16 was found on the same A30/A44 data it is judged on. A08, A12 and A26, once labelled precisely, are the
held-out test. The default in evaluate_scan.py is unchanged.

## Update: confirmed labels for A08, A12, A26 and the held-out result
Moving pairs now confirmed on five scans (60 pairs): A30 12, A44 20, A08 14 (74-87), A12 5 (57-61), A26 9 (59-62, 64-68).
Negatives: about 510 pairs. See `report.txt`.

| k | moving detected at 0.5 | false alarms at 0.5 |
|---|---|---|
| 16 | 55 of 60 | 2 of 512 |
| 30 (current) | 51 of 60 | 2 of 509 |

Held-out (A08, A12, A26: labelled after k=16 was chosen; 28 moving pairs): k=16 detects 23, k=30 detects 22; one false alarm
each (A12 pair 66). The gain there is a single pair (A26 pair 59, 0.47 -> 0.68). Most of the benefit is on A44 (development set).
Two false alarms in about 510 pairs means a 66-pair scan has roughly a one in four chance of one borderline spurious alarm,
with k=16 or k=30. The default in evaluate_scan.py is unchanged.
