#!/bin/zsh
# Band-pass-filtered A-scan comparison: build filtered atlanta1 cache, retrain
# the U-Net-BiLSTM on the filtered atlanta2 cache, then evaluate BOTH A-scan
# models (1D-CNN-bp already trained, + BiLSTM-bp) through the unified per-blob
# harness on the filtered caches (same-domain + cross-domain).
set -e
cd "$(dirname "$0")"
PY=thesis_env/bin/python
BP2=ascan_cache_atlanta2_bp
BP1=ascan_cache_atlanta1_bp

echo "===== 1/3  Build filtered atlanta1 cache (cross-domain) ====="
THZ_BANDPASS=1 ASCAN_CACHE1=$BP1 \
  $PY ascan_classifier.py prepare-atlanta1

echo "\n===== 2/3  Retrain U-Net-BiLSTM on filtered atlanta2 cache ====="
THZ_BANDPASS=1 ASCAN_CACHE2=$BP2 \
  $PY train_ascan_unet_bilstm.py --epochs 40 --tag _bp

echo "\n===== 3/3  Unified per-blob eval (filtered, both A-scan models) ====="
THZ_BANDPASS=1 ASCAN_CACHE2=$BP2 ASCAN_CACHE1=$BP1 \
  $PY eval_unified_bp_compare.py

echo "\nDONE. filtered comparison → runs/unet_bilstm/eval_bp_compare.json"
