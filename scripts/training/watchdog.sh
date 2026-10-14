#!/bin/bash
# usage: watchdog.sh <fold> <trainer>
FOLD=$1; TR=$2
OUT=/data/nnUNet_results/Dataset101_BraTSMET/${TR}__nnUNetResEncUNetLPlans__3d_fullres/fold_${FOLD}
PAT="nnUNetv2_train 101 3d_fullres $FOLD -tr $TR "
gpu_busy() { [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; }
while true; do
  if [ -f "$OUT/checkpoint_final.pth" ] && ! pgrep -f "$PAT" >/dev/null; then echo "$(date '+%F %T') finished"; exit 0; fi
  if ! gpu_busy; then
    sleep 120
    if ! gpu_busy; then
      echo "$(date '+%F %T') training not on GPU -> restarting with --c"
      pkill -f "$PAT"; sleep 10; pkill -9 -f "$PAT"; sleep 5
      nohup ~/run_primus.sh $FOLD $TR --c >> ~/logs/${TR}_fold${FOLD}.log 2>&1 &
      sleep 1800   # allow time for startup and compilation
    fi
  fi
  sleep 300
done
