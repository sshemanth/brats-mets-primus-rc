#!/bin/bash
# Predict the challenge validation set with ResEnc-L fold 0 and Primus fold 0, ensemble, zip, upload.
# usage: nohup ~/predict_val.sh /data/val_images > ~/logs/predict_val.log 2>&1 &
# Safe to re-run after a crash: --c skips cases that are already predicted.
set -e
source ~/primus-env/bin/activate
export nnUNet_raw=/data/nnUNet_raw nnUNet_preprocessed=/data/nnUNet_preprocessed nnUNet_results=/data/nnUNet_results
export MALLOC_ARENA_MAX=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1

VAL=$1
R=/data/nnUNet_results/Dataset101_BraTSMET
OUT=/data/val_pred
mkdir -p $OUT/resenc $OUT/primus

echo "$(date '+%F %T') [1/4] ResEnc-L fold 0"
nnUNetv2_predict_from_modelfolder -i $VAL -o $OUT/resenc --c --save_probabilities -f 0 -chk checkpoint_final.pth \
  -m $R/nnUNetTrainerCoreCavityV3ResEncL__nnUNetResEncUNetLPlans__3d_fullres

echo "$(date '+%F %T') [2/4] Primus fold 0"
nnUNetv2_predict_from_modelfolder -i $VAL -o $OUT/primus --c --save_probabilities -f 0 -chk checkpoint_final.pth \
  -m $R/nnUNetTrainerPrimusV3S_RegionAwareRC__nnUNetResEncUNetLPlans__3d_fullres

echo "$(date '+%F %T') [3/4] Ensemble + final post-processing"
python ~/ensemble_submit.py --resenc $OUT/resenc --primus $OUT/primus --images $VAL \
  --out /data/submission --zip /data/submission.zip

echo "$(date '+%F %T') [4/4] Upload to bucket"
gcloud storage cp /data/submission.zip gs://brats_data_imp/primus_runs/submission/
gcloud storage rsync -r $OUT gs://brats_data_imp/primus_runs/val_pred

echo "$(date '+%F %T') ALL DONE"
