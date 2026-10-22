# YOLOv8m masks against the hand-drawn body masks (October 2026)

Human masks: `manual_mask_review_20261020_a27.zip` from the other chat (policy: skin, clothing and worn shoes; no bed or reflections).
Three frames from three volunteers' scans: A07 frame 53, A11 frame 80, A27 frame 92 (A27 added later; volunteer V03, chest, breath hold).
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
| A27 92 | YOLO top 0.25 | whole image | 0.879 | 0.946 | 0.925 |
| | | detector ROI | 0.987 | 1.000 | 0.987 |
| A27 92 | YOLO union 0.1 | whole image | 0.885 | 0.992 | 0.892 |
| | | detector ROI | 0.987 | 1.000 | 0.987 |

Inside the detector ROI the masks cover everything the human drew (recall 1.00), but include extra area. In A07 the extra area is the bed between the legs
(34k pixels, about a third of the ROI), which the human policy excludes. Outside the ROI YOLO misses the head side/shoes at the image edge and the arms
(A07 recall 0.86) and leaks onto the bed/shadows (A11, union mask 47k pixels).

Effect on the score of the same static pair (`A07_53.txt`, `A11_80.txt`): the score does not change (k=16: 0.09 / 0.07 with either mask), while the number of body points
differs: A07 1053 (mine, as used) against 734 (human, eroded 21 px); A11 500 against 446; A27 (92->93) 296 against 283, score 0.09 with every mask. About 30% of my points in A07 lie on the bed between the legs.
Bed points are static, so they do not raise the score of a static pair; they could dilute a small moving region (to be tested on a moving scan with a hand-drawn mask).

On the chest (A27) the automatic mask inside the detector ROI is almost identical to the hand-drawn one (IoU 0.987); the large leakage is on the bed between the legs (A07).
Limits: three frames, three static pairs; no moving pair was evaluated; the human masks are an initial development annotation, not pixel-perfect.

## A27 as a negative control (`results/a27_2026_10/`)
A27: volunteer V03, chest, breath hold maintained through the whole sweep (SCAN_LABEL: expected class NO MOTION). Run with the final settings (E3, union masks, k=16): exposure window pairs 59-125 (67), all 67 scored,
0 alarms, score median 0.11, maximum 0.28 (k=30: 0.09 / 0.23). Another clean static control on a chest.
