#!/usr/bin/env bash
# Reproduce everything in results/ (CSV, plot, settings) and save the body masks used.
#
# Usage:  bash scripts/run_all.sh <scans_dir> [out_dir] [masks_dir]
#   <scans_dir> must contain these scan folders (or .zip files with the same names):
#     arc22_office_20260907_151502                                   A28_1  chest motion, labelled 63-85
#     arc22_office_20260907_142049                                   A26    small hand motion
#     arc22_office_20260830_142447_volunteer_both_legs_manual_normal_static_repea   legs, static
#     arc22_office_20260830_130408_volunteer_chest_manual_breath_hold_post          VC01 breath hold
#     arc22_office_20260830_125412_volunteer_chest_manual_natural_breathing         VC01 natural breathing
set -euo pipefail
SCANS=${1:?scans dir}
OUT=${2:-results}
MASKS=${3:-masks}
cd "$(dirname "$0")/.."

# Settings used for every result (all other options are the defaults in evaluate_scan.py):
#   --mask seg (yolov8m-seg, conf 0.25, imgsz 960)  --erode 21
#   --threshold 0.5 px (epi_enc_cl30)  --static-threshold-mm 0.5  --persistence 1  --sync 0.9
#   detector region: PLACEHOLDER rectangle between the four white squares on the HOME frame
COMMON=(--calib data/CalibrationResult_arc22.json --gantry data/gantry_arc22.json
        --detector-rect 280,150,620,375)

find_scan() { for p in "$SCANS/$1" "$SCANS/$1.zip"; do [ -e "$p" ] && { echo "$p"; return; }; done; echo "missing scan $1" >&2; exit 1; }

run() {  # name scan_folder [extra args...]
  local name=$1 scan; scan=$(find_scan "$2"); shift 2
  echo "== $name"
  python3 scripts/evaluate_scan.py --scan "$scan" "${COMMON[@]}" \
      --out "$OUT/$name.csv" --plot "$OUT/$name.png" --save-masks "$MASKS/$name" "$@"
}

mkdir -p "$OUT" "$MASKS"
run A28_1_chest_motion       arc22_office_20260907_151502 --label 63-85
run A26_hand_small_motion    arc22_office_20260907_142049
run legs_static              arc22_office_20260830_142447_volunteer_both_legs_manual_normal_static_repea
run VC01_breath_hold         arc22_office_20260830_130408_volunteer_chest_manual_breath_hold_post
run VC01_natural_breathing   arc22_office_20260830_125412_volunteer_chest_manual_natural_breathing
