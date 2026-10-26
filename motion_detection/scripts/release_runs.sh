#!/bin/bash
# Release runs of October 2026 (motion_detection_release_2026_10_26): every number of the release README comes from here.
#
#   bash scripts/release_runs.sh <scans_dir> <masks_dir> <out_dir> [phase ...]
#
# <scans_dir>  folder with the 11 scan folders under their original names (see scripts/labels_2026_10.py, SCANS[..]['folder'])
# <masks_dir>  masks/ of the release: yolov8m_union/<scan>, mediapipe_selfie/<scan>, mediapipe_multiclass/<scan>,
#              yolov8n640_every4/<scan>  (body_<frame>.png, already eroded)
# <out_dir>    results; one folder per run
# phases (default all): eval injection realtime tables
#   eval       regular (--chain-L 0) and combined (defaults) on the YOLOv8m masks, and the defaults on the other masks
#   injection  synthetic motion on the static scans (adjacent pairs and chained L=4)
#   realtime   scripts/realtime_run.py: saved masks; live YOLOv8n 640 every 4 frames; live YOLOv8m 960 every frame
#              (run alone on an idle machine: these are timing runs; needs yolov8n-seg.pt and yolov8m-seg.pt)
#   tables     scripts/release_tables.py
# Run from motion_detection/. Python packages: requirements.txt (versions in the release env/ folder).
set -e
SCANS=$1; MASKS=$2; OUT=$3; shift 3
PHASES=${*:-eval injection realtime tables}
cd "$(dirname "$0")/.."
NAMES="A30 A44 A08 A12 A26 A28_1 A07 A11 legs_static VC01_bh A27"
folder() { python3 -c "import sys; sys.path.insert(0, 'scripts'); import labels_2026_10 as L; print(L.SCANS['$1']['folder'])"; }
COMMON="--calib data/CalibrationResult_arc22.json --gantry data/gantry_arc22.json --detector-mask data/roi_arc22_calibrated.png \
 --roi-fixed --bed data/bed_home_arc22.png --detect masked_abs --body-fb-max 0.1 --guard bed --guard-action mark"
P=${JOBS:-4}

if [[ " $PHASES " == *" eval "* ]]; then
  jobs=()
  for n in $NAMES; do
    f=$SCANS/$(folder $n)
    jobs+=("python3 scripts/evaluate_scan.py --scan $f $COMMON --body-masks $MASKS/yolov8m_union/$n --chain-L 0 --out $OUT/regular/$n.csv")
    jobs+=("python3 scripts/evaluate_scan.py --scan $f $COMMON --body-masks $MASKS/yolov8m_union/$n --out $OUT/combined/$n.csv --plot $OUT/combined/$n.png")
    for m in mediapipe_selfie mediapipe_multiclass yolov8n640_every4; do
      jobs+=("python3 scripts/evaluate_scan.py --scan $f $COMMON --body-masks $MASKS/$m/$n --out $OUT/$m/$n.csv")
    done
  done
  mkdir -p $OUT/regular $OUT/combined $OUT/mediapipe_selfie $OUT/mediapipe_multiclass $OUT/yolov8n640_every4 $OUT/logs
  printf '%s\n' "${jobs[@]}" > $OUT/logs/eval_commands.txt
  printf '%s\0' "${jobs[@]}" | xargs -0 -P $P -I{} sh -c '{} > /dev/null 2>&1 || echo "FAILED: {}"'
fi

if [[ " $PHASES " == *" injection "* ]]; then
  mkdir -p $OUT/injection $OUT/chain_injection
  for n in A07 A11 legs_static A27; do
    f=$SCANS/$(folder $n)
    python3 scripts/inject_motion.py --scan $f --body-masks $MASKS/yolov8m_union/$n --csv $OUT/combined/$n.csv --out $OUT/injection/$n.csv
    python3 scripts/chain_check.py inject --scan $f --body-masks $MASKS/yolov8m_union/$n --csv $OUT/combined/$n.csv --out $OUT/chain_injection/$n.csv
  done
  python3 scripts/injection_report.py $OUT/injection/*.csv --real $OUT/combined > $OUT/injection/report.txt
  python3 scripts/chain_check_report.py $OUT --score-dir $OUT/combined --inject-dir $OUT/chain_injection \
    --adjacent-injection $OUT/injection --real $OUT/combined > $OUT/chain_injection/report.txt
fi

if [[ " $PHASES " == *" realtime "* ]]; then
  for n in $NAMES; do
    f=$SCANS/$(folder $n)
    python3 scripts/realtime_run.py --scan $f --csv $OUT/combined/$n.csv --body-masks $MASKS/yolov8m_union/$n --out $OUT/realtime_saved_masks/$n.csv
    python3 scripts/realtime_run.py --scan $f --csv $OUT/combined/$n.csv --seg-weights ${YOLOV8N:-yolov8n-seg.pt} --seg-imgsz 640 --mask-every 4 \
      --save-masks $OUT/realtime_masks_v8n640/$n --out $OUT/realtime_v8n640_every4/$n.csv
    python3 scripts/realtime_run.py --scan $f --csv $OUT/combined/$n.csv --seg-weights ${YOLOV8M:-yolov8m-seg.pt} --seg-imgsz 960 --mask-every 1 \
      --out $OUT/realtime_v8m960_every1/$n.csv
  done
fi

if [[ " $PHASES " == *" tables "* ]]; then
  python3 scripts/release_tables.py --runs $OUT --out $OUT/tables
fi
