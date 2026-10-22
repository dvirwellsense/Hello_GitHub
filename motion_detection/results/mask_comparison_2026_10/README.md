# YOLOv8m masks against the hand-drawn body masks (October 2026)

Human masks: `manual_mask_review_20261020_a27.zip` from the other chat (policy: skin, clothing and worn shoes; no bed or reflections).
Three frames from two people: A07 frame 53, A11 frame 80, A27 frame 92. **A27 is not available here, so only A07 and A11 were compared.**
The two other annotated frames mentioned in the other chat (A09, A28_1) were not in this zip. `scripts/compare_masks_with_human.py` reproduces everything.

Overlays (`*_overlay.jpg`): green = both, red = human only (missed by YOLO), blue = YOLO only (leakage), magenta = detector ROI.

| frame | mask | region | IoU | recall | precision |
|---|---|---|---|---|---|
| A07 53 | YOLO top 0.25 and union 0.1 (identical) | whole image | 0.700 | 0.859 | 0.790 |
| | | detector ROI | 0.675 | 1.000 | 0.675 |
| A11 80 | YOLO top 0.25 | whole image | 0.870 | 0.972 | 0.892 |
| | | detector ROI | 0.894 | 1.000 | 0.894 |
| A11 80 | YOLO union 0.1 | whole image | 0.822 | 0.979 | 0.837 |
| | | detector ROI | 0.894 | 1.000 | 0.894 |

Inside the detector ROI the masks cover everything the human drew (recall 1.00), but include extra area. In A07 the extra area is the bed between the legs
(34k pixels, about a third of the ROI), which the human policy excludes. Outside the ROI YOLO misses the head side/shoes at the image edge and the arms
(A07 recall 0.86) and leaks onto the bed/shadows (A11, union mask 47k pixels).

Effect on the score of the same static pair (`A07_53.txt`, `A11_80.txt`): the score does not change (k=16: 0.09 / 0.07 with either mask), while the number of body points
differs: A07 1053 (mine, as used) against 734 (human, eroded 21 px); A11 500 against 446. About 30% of my points in A07 lie on the bed between the legs.
Bed points are static, so they do not raise the score of a static pair; they could dilute a small moving region (to be tested on a moving scan with a hand-drawn mask).

Limits: two frames, two static pairs; no moving pair was evaluated; the human masks are an initial development annotation, not pixel-perfect.
