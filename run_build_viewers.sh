#!/bin/zsh
set -e
cd "$(dirname "$0")"
PY=thesis_env/bin/python
echo "=== unfiltered viewer (AScanCNN-nofilt + BiLSTM-unfilt) ==="
ASCAN_CACHE2=ascan_cache_atlanta2 \
ASCAN_CNN_CKPT=results_v2_atlanta2/ascan_classifier/ascan_cnn_depth_nofilt.pt \
ASCAN_BILSTM_CKPT=runs/unet_bilstm/unet_bilstm_best.pt \
ASCAN_SLICES_DIR=slices_v2_atlanta2 \
ASCAN_VIEWER_OUT=runs/unet_bilstm/viewer_ascan_compare.html \
  $PY build_ascan_compare_viewer.py
echo "=== band-pass viewer (AScanCNN-bp + BiLSTM-bp) ==="
ASCAN_CACHE2=ascan_cache_atlanta2_bp \
ASCAN_CNN_CKPT=results_v2_atlanta2/ascan_classifier/ascan_cnn_depth_bp.pt \
ASCAN_BILSTM_CKPT=runs/unet_bilstm/unet_bilstm_bp_best.pt \
ASCAN_SLICES_DIR=slices_v2_atlanta2_bp \
ASCAN_VIEWER_OUT=runs/unet_bilstm/viewer_ascan_compare_bp.html \
  $PY build_ascan_compare_viewer.py
echo "DONE both viewers."
