# U-Net-BiLSTM vs existing models — unified per-blob F1

Zhang et al. (2024) U-Net-BiLSTM A-scan damage segmenter (1D U-Net with BiLSTM on
the two deepest skips), trained on the atlanta2 A-scan cache under the fixed
sample-level split (TRAIN={1-10,12,15}, VAL={11,13}, NOVOID={14}; atlanta1 held
out for cross-domain). 40 epochs, Dice+weighted-BCE, AdamW+cosine. Best val
per-point F1 = 0.399. Params ≈ 1.05M.

All F1 below use the SAME unified per-blob matcher (mesh.contains GT footprint,
8-symmetry rotation alignment, per-blob Hungarian matching with an 8 mm centroid
gate), so the three models are directly comparable.

## 3-way comparison (atlanta2, all void samples)

| Model | F1 | Precision | Recall |
|---|---|---|---|
| A-scan 1D-CNN (depth classifier) | **0.703** | 0.962 | 0.618 |
| C-scan U-Net (manual labels)     | 0.437 | 0.867 | 0.320 |
| **U-Net-BiLSTM (NEW)**           | 0.521 | 0.929 | 0.401 |

## U-Net-BiLSTM by domain

| Split | F1 | P | R | notes |
|---|---|---|---|---|
| atlanta2 all-void (TRAIN+VAL) | 0.521 | 0.929 | 0.401 | partly on training data, as is the 1D-CNN number |
| atlanta2 VAL only (11, 13)    | 0.295 | 1.000 | 0.190 | 2 hard samples (s11: 21 prism voids) |
| atlanta1 cross-domain         | 0.580 | 0.911 | 0.514 | fully held out — best of the three splits |

## Per-point vs per-blob metric gap (U-Net-BiLSTM, atlanta2 void samples)

| Metric | F1 | P | R |
|---|---|---|---|
| Per-time-point (2048 samples) | 0.650 | 0.507 | 0.907 |
| Per-blob (unified)            | 0.521 | 0.929 | 0.401 |

The two metrics measure different things and do NOT move together: per-point F1 is
micro-averaged over time samples (broad-in-time predictions inflate recall to 0.91
but precision is only 0.51); the per-blob stage then projects over depth, keeps
connected components ≥ MIN_AREA_PX, and matches one blob per void with an 8 mm
gate — which raises precision to 0.93 (spurious time points don't form coherent
2D blobs) but lowers recall to 0.40 (many voids never form a matchable blob).

## Verdict

BiLSTM-in-skips **beats the C-scan U-Net** (0.521 vs 0.437, +0.084) but **does NOT
beat the existing A-scan 1D-CNN depth classifier** (0.521 vs 0.703). It generalises
well across scan campaigns (cross-domain 0.580 > its own same-domain 0.521). All
three models share the same failure mode — high precision, recall-limited on
many-void samples (s7 spheres, s10/s11 stripes/prisms). Whether the BiLSTM itself
helps vs a plain 1D U-Net is a separate question; run `train_ascan_unet_bilstm.py
--no_bilstm` for that ablation.
