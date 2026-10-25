# Gantry model refit on long baselines (October 2026)

Question: the residual of static things grows with the arc angle between the frames (results/baseline_2026_10).
Can a better gantry model remove it, so that frame i can be compared with frame i-L (slow drift)?

Method (`scripts/refit_gantry.py`): static points (everything in the phantom scan; outside the body in A07, legs_static,
A11, A27) tracked in a chain from frame j to j+L, L = 1..16; the model is fitted on the signed epipolar distance.
Variants add one change at a time: V1 camera pose, V2 + sync, V3 + encoder scale, V4/V5 + pose deviation linear /
quadratic in the angle. Two folds (A: fit on phantom, A07, legs_static; B: fit on A11, A27), results on the other fold.
Files: gantry_<fold>_<variant>.json, fit_log.txt, heldout_residuals.txt, residual_by_region.txt; pipeline/ (all scans,
L = 1, 2, 4, 8, each scan with the V3 model of the fold it was not fitted on); chained/ (static scans, chained tracking).
No default setting was changed: data/gantry_arc22.json is still the model of the pipeline.

Findings
- V3 (encoder scale) is the only change that matters, and it is the same in both folds (scale 1.345 / 1.352, sync
  0.43 / 0.40). Held-out median residual of all static points at L=8: 1.1-1.5 px -> 0.31-0.34 px. V4/V5 add nothing.
- But the gain is at the image edges (floor, far background: 3-6 px -> 0.3-1 px). Near the detector region the original
  model was already good and V3 changes little (0.03-0.47 -> 0.05-0.35 px); on the phantom itself it does not help.
  So the scale is most likely compensating an error at the edges of the image (fisheye calibration or the floor), not
  a real encoder error: a 35% encoder error would also show near the centre, and it does not.
- What counts for the score does not change: body clusters of the static scans with chained tracking at L=8,
  original -> V3: A07 0.52 -> 0.50, A11 0.42 -> 0.43, A27 0.53 -> 0.50, legs_static 0.97 -> 1.12, phantom 1.23 -> 1.30 px.
  The adjacent-pair pipeline (L=1) gives the same results with either model (55/60 moving pairs, 2/623 negatives).
- So the growth of the body score with L is not the gantry model. Bed points near the detector stay low at L=8 (legs
  scan, centre: 0.03 px), while points on the body and on the rigid phantom (curved surfaces, above the bed) do not.
  Remaining candidates: tracking on curved surfaces whose appearance changes with the viewing angle, or an error that
  depends on height above the bed. The rigid phantom rules out patient motion as the main cause.
