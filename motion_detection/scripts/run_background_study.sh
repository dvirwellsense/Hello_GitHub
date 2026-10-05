#!/usr/bin/env bash
# Background validation (2026-10). Calibrated detector ROI, fixed in image coordinates.
# Usage: bash scripts/run_background_study.sh <scans_dir> <masks_dir> <phantom_scan_dir> [out_dir]
set -euo pipefail
SCANS=${1:?}; MASKS=${2:?}; PHANTOM=${3:?}; OUT=${4:-results/background_study_2026_10}
cd "$(dirname "$0")/.."
declare -A S=([A28_1_chest_motion]=arc22_office_20260907_151502 [A26_hand_small_motion]=arc22_office_20260907_142049
  [legs_static]=arc22_office_20260830_142447_volunteer_both_legs_manual_normal_static_repea
  [VC01_breath_hold]=arc22_office_20260830_130408_volunteer_chest_manual_breath_hold_post
  [VC01_natural_breathing]=arc22_office_20260830_125412_volunteer_chest_manual_natural_breathing)
declare -A SHORT=([A28_1_chest_motion]=A28_1_chest [A26_hand_small_motion]=A26_hand [legs_static]=legs_static
  [VC01_breath_hold]=VC01_breath_hold)
mkdir -p "$OUT/study" "$OUT/runs"
# 1+3: background classes and body-error prediction (static controls: phantom, legs, breath hold)
python3 scripts/background_study.py --scan "$PHANTOM" --name phantom --phantom --out "$OUT/study"
for n in "${!SHORT[@]}"; do
  python3 scripts/background_study.py --scan "$SCANS/${S[$n]}" --name "${SHORT[$n]}" --body-masks "$MASKS/$n" --out "$OUT/study"
done
# 2: E3 + calibrated fixed ROI + bed-plane guard (insufficient background -> INSUFFICIENT_DATA)
for n in "${!S[@]}"; do
  python3 scripts/evaluate_scan.py --scan "$SCANS/${S[$n]}" --calib data/CalibrationResult_arc22.json \
      --gantry data/gantry_arc22.json --detector-mask data/roi_arc22_calibrated.png --roi-fixed \
      --bed data/bed_home_arc22.png --body-masks "$MASKS/$n" --detect masked_abs --body-fb-max 0.1 \
      --guard bed --guard-action mark --out "$OUT/runs/$n.csv" --plot "$OUT/runs/$n.png"
done
