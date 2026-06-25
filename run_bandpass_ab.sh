#!/bin/zsh
# A/B: band-pass-filtered vs unfiltered A-scan depth-CNN on atlanta2 (TRAIN/VAL).
# Both arms run under identical current code. Crossdomain (atlanta1) is skipped
# in both (ASCAN_CACHE1 -> empty dir) so the comparison is symmetric and fast.
#
# Usage:  ./run_bandpass_ab.sh [epochs]    (default 30; use a small number to smoke-test)
set -e
cd "$(dirname "$0")"
PY=thesis_env/bin/python
EP=${1:-30}
export ASCAN_EPOCHS=$EP
mkdir -p ascan_cache_empty   # forces eval to skip the crossdomain section

echo "############ EPOCHS=$EP ############"

echo "\n===== ARM A: UNFILTERED baseline (reuse existing cache) ====="
# `train` runs train() then evaluate() in one go.
ASCAN_VARIANT=_nofilt THZ_BANDPASS=0 \
  ASCAN_CACHE2=ascan_cache_atlanta2 ASCAN_CACHE1=ascan_cache_empty \
  $PY ascan_classifier.py train

echo "\n===== ARM B: BAND-PASS 0.1-2.5 THz (rebuild filtered cache) ====="
ASCAN_VARIANT=_bp THZ_BANDPASS=1 \
  ASCAN_CACHE2=ascan_cache_atlanta2_bp ASCAN_CACHE1=ascan_cache_empty \
  $PY ascan_classifier.py prepare-atlanta2
ASCAN_VARIANT=_bp THZ_BANDPASS=1 \
  ASCAN_CACHE2=ascan_cache_atlanta2_bp ASCAN_CACHE1=ascan_cache_empty \
  $PY ascan_classifier.py train

echo "\n===== A/B DONE — summaries ====="
echo "baseline: results_v2_atlanta2/ascan_classifier/summary_depth_nofilt.json"
echo "bandpass: results_v2_atlanta2/ascan_classifier/summary_depth_bp.json"
