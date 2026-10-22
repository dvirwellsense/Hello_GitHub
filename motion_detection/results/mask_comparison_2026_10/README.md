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

## A08 frame 78 (moving pair 79, hand-drawn with `tools/mask_annotator.html`)
The annotation (`human_masks/user_annotations_A08_78.json`) follows the legs closely, includes the shoes and excludes the bed between the legs and the laptop.
YOLO (top and union identical): IoU 0.68 whole image, 0.65 inside the detector ROI (recall 1.00; the difference is again the bed between the legs).
Pair 78->79 (moving, a large motion): score k=8 / 16 / 30 = 5.08 / 5.02 / 4.94 with **both** masks; body points 874 (mine) against 660 (human, eroded 21 px) and 729 (raw).
So a large motion is not affected by the extra bed points. A small moving group (A44 pair 71, k=30 score 0.20 against 2.56 with k=16) is the more discriminating test.

## A44 frame 60 (moving pair 61, hand-drawn)
Legs crossed, knee raised: no bed gap inside the ROI. YOLO union against the hand-drawn mask: IoU 0.96 whole image, 0.98 inside the detector ROI (recall 0.997, precision 0.983).
The top mask (highest-confidence person) has IoU 0.74 on the whole image (recall 0.75, it misses a large part of the person) but 0.98 inside the ROI.
Pair 60->61: score k=8 / 16 / 30 = 3.88 / 3.57 / 3.02 with all three masks (mine 183 body points, human eroded 176, human raw 192).
Summary of the two moving pairs checked so far (A08 79, A44 61): the hand-drawn mask changes neither the score nor the detection. The decisive test remains A44 frame 70 (pair 71, small group, k=30 score 0.20).

## A44 frame 70 (pair 71: the small moving group missed with k=30) - the decisive test
The raw JSON (`user_annotations_A44_60_and_70_raw.json`) holds the frame-60 annotation again plus one annotation named `image-<number>.jpg` (the image was opened under another file name, so the tool could not read scan and frame);
it was relabelled A44 frame 70 (`user_annotations_A44_70.json`). The overlay fits the frame.
YOLO union against the hand-drawn mask: IoU 0.96 whole image, 0.985 inside the ROI. Pair 70->71, score k=8 / 16 / 30 = 8.98 / 2.56 / 0.20 with **all three masks** (mine 141 body points, human eroded 137, raw 148).
**Conclusion: the miss of this small motion with k=30 is not caused by the mask. It is the cluster size.** A hand-drawn mask does not change the score, so there is no reason to improve the automatic mask for the sake of detection.
