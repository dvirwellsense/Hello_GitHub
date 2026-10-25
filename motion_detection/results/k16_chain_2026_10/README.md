# Final run with the default settings of October 2026

`evaluate_scan.py` with the defaults: k=16, threshold 0.5 px, persistence 2, and the chained second check
(--chain-L 4 --chain-threshold 1.2, results/chain4_2026_10). E3 (--detect masked_abs --body-fb-max 0.1), union masks,
--roi-fixed, --guard bed --guard-action mark, as in k16_2026_10. Per scan: <scan>.csv, .settings.json, .png
(x = alarm of the adjacent-pair check, purple + = alarm of the second check only).

- Adjacent-pair scores and alarms are identical to k16_p2_2026_10; the chained scores are identical to
  results/chain4_2026_10/score.
- Static scans (A07, A11, legs_static, VC01_bh, A27): no alarm.
- Alarm frames added by the second check, all within 3 frames after labelled motion (the chain still spans it):
  A30 65-67, 70-72; A44 58, 59, 62, 63, 77-79, 81-83, 86, 87; A08 88; A12 62-64; A26 62-64, 70; A28_1 88, 112.
