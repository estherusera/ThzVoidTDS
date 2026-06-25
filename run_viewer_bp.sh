#!/bin/zsh
# Export band-pass-filtered 50-slice atlanta2 slices, train a 50-slice U-Net on
# both unfiltered (baseline) and filtered slices, and build both viewers so the
# band-pass effect can be seen slice-by-slice in viewer.html.
set -e
cd "$(dirname "$0")"
PY=thesis_env/bin/python

echo "===== 1/4  Export filtered 50-slice slices (0.1-2.5 THz) ====="
THZ_BANDPASS=1 THZ_OUT_DIR=slices_v2_atlanta2_bp \
  $PY export_slices.py atlanta2 50

echo "\n===== 2/4  Train baseline (unfiltered) 50-slice U-Net ====="
THZ_BANDPASS=0 THZ_SLICES_DIR=slices_v2_atlanta2 THZ_MODEL_NAME=unet_50_nofilt.pt \
  $PY thz_slice_pipelinev2.py train atlanta2

echo "\n===== 3/4  Train band-pass 50-slice U-Net ====="
THZ_BANDPASS=1 THZ_SLICES_DIR=slices_v2_atlanta2_bp THZ_MODEL_NAME=unet_50_bp.pt \
  $PY thz_slice_pipelinev2.py train atlanta2

echo "\n===== 4/4  Build both viewers ====="
$PY build_viewer.py atlanta2_50_nofilt
$PY build_viewer.py atlanta2_bp

echo "\nDONE."
echo "  baseline viewer: results_v2_atlanta2/viewer_50_nofilt.html"
echo "  band-pass viewer: results_v2_atlanta2/viewer_50_bp.html"
