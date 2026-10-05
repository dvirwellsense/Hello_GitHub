#!/usr/bin/env bash
# Experiments of 2026-10: point selection, angle-reliability guard, cluster inspection.
# Uses the SAVED body masks (identical masks in every experiment) and changes one component per step.
#
# Usage: bash scripts/run_experiments.sh <scans_dir> <masks_dir> [out_dir] [ROI option...]
#   ROI option default: --detector-rect 280,150,620,375   (pass e.g. --detector-mask our_roi.png instead)
set -euo pipefail
SCANS=${1:?scans dir}; MASKS=${2:?masks dir (from run_all.sh)}; OUT=${3:-experiments}; shift $(( $# >= 3 ? 3 : $# ))
ROI=("$@"); [ ${#ROI[@]} -eq 0 ] && ROI=(--detector-rect 280,150,620,375)
cd "$(dirname "$0")/.."
C=(--calib data/CalibrationResult_arc22.json --gantry data/gantry_arc22.json "${ROI[@]}")
declare -A S=([A28_1_chest_motion]=arc22_office_20260907_151502 [A26_hand_small_motion]=arc22_office_20260907_142049
  [legs_static]=arc22_office_20260830_142447_volunteer_both_legs_manual_normal_static_repea
  [VC01_breath_hold]=arc22_office_20260830_130408_volunteer_chest_manual_breath_hold_post
  [VC01_natural_breathing]=arc22_office_20260830_125412_volunteer_chest_manual_natural_breathing)

# Experiment 1: point selection (image processing; guard = legacy/drop so alarms are comparable)
#   E0 global      corners on the whole image, then filtered to body & ROI   (baseline)
#   E1 masked      body corners detected inside body & ROI, quality relative to the body      (E0 -> 1 change)
#   E2 masked_abs  same region, absolute quality threshold of E0                               (E1 -> 1 change)
#   E3 E2 + body forward-backward limit 0.1 px instead of 0.5                                  (E2 -> 1 change)
for n in "${!S[@]}"; do
  for e in "E0:--detect global" "E1:--detect masked" "E2:--detect masked_abs" "E3:--detect masked_abs --body-fb-max 0.1"; do
    E=${e%%:*}; opts=${e#*:}; mkdir -p "$OUT/$E"
    python3 scripts/evaluate_scan.py --scan "$SCANS/${S[$n]}" "${C[@]}" --body-masks "$MASKS/$n" $opts --out "$OUT/$E/$n.csv"
  done
done

# Experiment 2: angle reliability (decision only, from the E0 CSVs; no image processing)
#   G0 legacy/drop (baseline)  G1 legacy/mark (action changed)  G2 time/mark (criterion changed)
#   G3 bg/mark (criterion: background residual)                 E3G3: G3 applied to E3
mkdir -p "$OUT/G"
for n in "${!S[@]}"; do
  for g in "G0:--guard legacy --guard-action drop" "G1:--guard legacy --guard-action mark" \
           "G2:--guard time --guard-action mark" "G3:--guard bg --guard-action mark"; do
    G=${g%%:*}; opts=${g#*:}
    python3 scripts/evaluate_scan.py --from-csv "$OUT/E0/$n.csv" "${C[@]}" $opts --out "$OUT/G/${G}_$n.csv"
  done
  python3 scripts/evaluate_scan.py --from-csv "$OUT/E3/$n.csv" "${C[@]}" --guard bg --guard-action mark --out "$OUT/G/E3G3_$n.csv"
done

# Experiment 3: the 30 points behind the breath-hold score, frames 106-110
for f in 106 107 108 109 110; do
  python3 scripts/inspect_cluster.py --scan "$SCANS/${S[VC01_breath_hold]}" "${C[@]}" --body-masks "$MASKS/VC01_breath_hold" \
      --frame $f --out "$OUT/inspect_VC01_breath_hold_$f"
done
