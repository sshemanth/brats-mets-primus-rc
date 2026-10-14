#!/bin/bash
source ~/primus-env/bin/activate
export nnUNet_raw=/data/nnUNet_raw
export nnUNet_preprocessed=/data/nnUNet_preprocessed
export nnUNet_results=/data/nnUNet_results
export nnUNet_n_proc_DA=6
export MALLOC_ARENA_MAX=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
nnUNetv2_train 101 3d_fullres "$1" -tr "$2" -p nnUNetResEncUNetLPlans --npz "${@:3}"
