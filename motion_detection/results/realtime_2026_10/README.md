# Real time (October 2026)

Budget: the camera takes a frame every ~0.25 s (median request interval 0.25 s in A44, A07, A28_1).
Machine: the cloud container (4 cores, Xeon 2.1 GHz). The target computer will differ: ratios matter more.

Time per frame
- profile_A44.txt (`scripts/profile_realtime.py`, idle machine): segmentation union conf 0.1:
  YOLOv8m 960 px 479 ms, YOLOv8m 640 px 229 ms, YOLOv8n 960 px 89 ms, YOLOv8n 640 px 52 ms.
  The pipeline steps of evaluate_scan.py without segmentation: ~140 ms, of which 54 ms are background points
  that only feed diagnostics, and 48 ms the second check done from scratch.
- `motion/realtime.py` (MotionMonitor): causal, frame by frame, only the work the decision needs, the second check reuses
  the adjacent-pair tracking. monitor_check.txt (`scripts/realtime_check.py`, all 11 scans): scores and alarms identical
  to results/k16_chain_2026_10 (max difference 0, no mismatch); time per frame (saved masks, image decoding excluded)
  median 33-84 ms, 95th percentile 87-137 ms, max 163 ms. With the k-d tree for the k nearest neighbours
  (motion/rigidity.py knn; identical scores) the dense scans went from up to 344 ms to 163 ms max.

Effect on the results (alarms with the defaults, persistence 2 + second check), all 11 scans, vs k16_chain_2026_10
- mask computed every 2 / 4 / 8 frames and reused in between (../mask_reuse_2026_10): alarms identical in all three.
  Every 8: one more negative suspect pair (A07 94), no alarm.
- YOLOv8n 960, YOLOv8m 640, YOLOv8n 640 (seg_*): alarms identical (one more negative suspect pair, A07 94, no alarm).
- YOLOv8n 640 computed every 4 frames (v8n640_every4): alarms identical (same extra suspect, no alarm).

So YOLOv8n 640 every 4 frames costs ~13 ms per frame on average (52 ms on the frames where it runs) on top of the
monitor: max ~215 ms per frame on this machine, inside 0.25 s, with the same alarms. Note: YOLOv8 is AGPL-3.0 (n and m alike).
