# Are the missed pairs motion along the epipolar line?

`scripts/blind_motion_check.py` splits the displacement of every tracked body point into the component
across the epipolar line (what the score measures) and along it (what the score cannot see), and compares the
along-line part with nearby unlabelled pairs. No score, mask or threshold is changed.

Settings: E3 with the union body masks, the calibrated ROI, baseline = median along-line shift per degree of the
unlabelled pairs within 10 pairs. Columns: `perp_cl30` (the existing score), `along_med` (median along-line shift,
px), `along_dev` (excess over the baseline), `along_cl30_dev` (cluster version, noisier).

Result (A30, A44, A08, A12):
- Missed labelled pairs (A30 65-67; A44 58, 62, 67, 69-71, 81, 82): the along-line excess is within the noise of
  the unlabelled pairs (A30 along_cl30_dev 0.35-0.40 vs median 0.37; A44 0.17-2.03 vs median 1.64, p95 2.23).
- The median along-line shift of every pair, detected or not, is 0.3 px or less.
- `A30_pair59_vs_pair66.jpg`: displacement arrows (x5), pair 59 (detected) vs pair 66 (missed). In 66 the tracked
  points barely move, so the label most likely includes a pause.
- Limit: the cluster version has a noise floor of about 0.4 (A30) to 2 (A44) px, so a small along-line motion
  could still hide. This is not proof.
- `arrows_*`: one picture per labelled pair.
