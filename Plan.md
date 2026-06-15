# THz Void Detection — Project Plan

> Roadmap & task tracker. Maintained by the assistant. Last updated: 2026-06-08.
> See `Wiki.md` for full project state and `References.md` for sources.

---

## Status legend
- [x] done
- [~] in progress / partial
- [ ] not started

---

## Phase 1 — Data foundation (DONE)
- [x] Load raw THz volumes from tprj (`thz_slice_pipeline.py`)
- [x] Surface-flatten + depth-slice + isotropic-zoom pipeline
- [x] Record ground-truth void specs for all 15 samples (see Wiki §3)
- [x] Abandon reference-subtraction approach (`void_detector.py`) — ineffective
- [x] Interactive ROI labeler; crop every sample to 20×20 mm = 100×100 px
- [x] Per-slice manual void labeler (`thz_slice_pipelinev2.py label`)

## Phase 2 — atlanta1 baseline (DONE)
- [x] Label atlanta1 voids (148 positive slices @ 50-slice)
- [x] Train base U-Net `unet_v2.pt` (atlanta1, 50-slice)
- [x] Prediction viewer + paper figure of selected slices

## Phase 3 — atlanta2 (clean data) (DONE)
- [x] Evaluate new atlanta2 scans — uniform 0.2 mm isotropic, all 30×30 mm, clean
- [x] Make all scripts dataset-aware (`atlanta1` / `atlanta2` arg)
- [x] ROI-label + slice-label atlanta2 (176 positive slices @ 50-slice)
- [x] A-scan analysis (signed waveforms, void vs background) on atlanta2

## Phase 4 — Self-training / pseudo-labeling (DONE)
- [x] Export atlanta2 @ 500 slices
- [x] Pseudo-label 500-slice with `unet_v2.pt` (offsets ±10)
- [x] QC pseudo vs manual (mean IoU 0.40 / Dice 0.49)
- [x] Retrain `unet_pseudo.pt` (500-slice, manual weighted 5×) → Dice 0.937
- [x] Compare old vs new model on atlanta2 → +0.089 IoU / +0.101 Dice
- [x] Export atlanta2 @ 2050 slices (full time resolution)
- [x] Pseudo-label 2050-slice with `unet_pseudo.pt` (offsets ±40)
- [x] Retrain `unet_pseudo_2050.pt` (2050-slice, ±40) → Dice 0.978
- [x] Viewers for 2050 (±40 and ±10 variants)

## Phase 5 — Cross-domain test (DONE)
- [x] Export atlanta1 @ 2050 slices
- [x] Eval `unet_pseudo_2050.pt` on atlanta1 manual labels → IoU 0.532 / Dice 0.653
- [x] Cross-domain viewers (±40 and ±10)
- [x] Eval `unet_pseudo_2050.pt` on atlanta2 manual labels → IoU 0.492 / Dice 0.597
      (same-domain honest check vs human labels)

## Phase 6 — Void characterization vs STL CAD (DONE)
- [x] Establish scan-sample → STL filename mapping (off-by-one above sample 7)
- [x] STL voxelization via per-component contains() for watertight (≤ 2k faces),
      bbox-only fallback for non-watertight / complex meshes
- [x] Coarse 200-point depth grid for GT voxelization (memory-safe vs full 2050)
- [x] 2D-projection + per-blob depth profile for prediction blob extraction
- [x] Threshold sweep + 8-symmetry Hungarian matching (gated 8 mm)
- [x] MIN_GT_VOLUME_MM3 = 0.05 filter for spurious mesh fragments
- [x] Per-sample + aggregate report (`void_characterize.py all`)
- [x] **Headline result**: mean F1 0.44, precision 0.87, centroid dist 1.34 mm,
      volume blow-up ~19× mean (~5× mode) — confirms THz physical inflation

## Phase 7 — A-scan 1D-CNN classifier (IN PROGRESS, Jun 2026)
- [x] Cache builder: surface-flattened ROI-cropped A-scans + per-pixel 50-bit
      labels for atlanta1 and atlanta2 (`ascan_cache_atlanta{1,2}/`)
- [x] STL mapping verified identical to `void_characterize.py` (+ extra `"5b"`)
- [x] Splits: TRAIN={1..10,12,15} VAL={11,13} NOVOID={14}; atlanta1 fully held out
- [x] Baseline 2D 1D-CNN (`ascan_cnn.pt`):
      val F1_gt 0.66, val F1_stl 0.44 (vs bbox projection)
- [x] Publication figures (`ascan_classifier_figures.py`):
      per-sample F1 bars, prob histograms, showcase grid, designed-vs-detectable
- [~] Depth-aware 50-dim head (`ascan_cnn_depth.pt`, AdaptiveAvgPool1d(50) +
      Conv1d(128→1, 1×1)) — trained 30 epochs; val per-pixel F1_3D plateaus
      at 0.20 but val F1_stl is excellent under the unified metric (see below)
- [x] Reconcile A-scan STL metric with `void_characterize.py`
      (`ascan_vs_stl.py`):
      reuses voxelize_gt() + extract_pred_blobs() + 8-symmetry Hungarian
- [x] **Headline**: A-scan F1_unified = **0.703** (P 0.96 R 0.62) vs the
      C-scan U-Net's 0.44 on the same 14 void samples; NoVoid sanity passes
      (0 blobs, was 3109 FP pixels under per-pixel)
- [ ] Improve recall on multi-void samples (s7=19 voids R=0.16, s10/s11=21
      voids R=0.14) — investigate whether MIN_AREA_PX=100 is dropping real
      small voids, or whether the model truly misses them
- [ ] Cross-domain atlanta1 evaluation of the A-scan classifier under unified
      metric
- [ ] Resolve the "F1_3D 0.20" vs "F1_unified 0.70" gap: the per-pixel × per-
      depth metric the training script reports is much harsher than the
      per-blob match; clarify which one is reported in the paper

## Phase 8 — STL-supervised U-Net (NEW, Jun 2026)
- [x] `stl_to_labels.py`: build soft per-slice labels from STL via
      `mesh.contains()` voxelisation + per-sample symmetry (IoU vs manual) +
      3D Gaussian blur (σ_xy=1 px, σ_z=1 slice)
- [x] s13 (Cilindros, no detectable scan voids) and s14 (NoVoid) → zero masks
- [x] `train_unet_stl.py`: 60-epoch training on STL labels →
      `unet_stl.pt` (~3 min on MPS, final Dice 0.59 on train)
- [x] `eval_unet_stl.py`: unified per-blob F1 → **F1=0.523, P=0.929, R=0.398**
      (vs manual U-Net 0.468 / 0.867 / 0.320 → +0.055 F1)
- [x] `compare_3way.py`: head-to-head figure + STL-U-Net showcase
- [ ] Improve recall: try σ=0 (hard labels), longer training, focal/over-sample
      for s7/s10/s11 (the many-small-voids samples)
- [ ] Re-pick sym for s2/s3/s6/s7 (low manual IoU); consider dropping them
      from training if alignment is genuinely ambiguous
- [ ] Combine STL + manual labels (weighted) — STL for shape, manual for
      "what THz can actually see"
- [ ] Honest leave-one-sample-out evaluation

---

## NEXT / OPEN (priority order)

### A. Honest evaluation (HIGH — needed for paper credibility)
- [ ] Leave-one-sample-out cross-validation on atlanta2: for each void sample, retrain
      without it, evaluate on its manual labels. Report mean ± std IoU/Dice.
- [ ] Report **void-conditional** metrics (exclude trivially-correct empty slices).
- [ ] Hold-out split alternative if LOO is too slow (~13×45 min).
- [ ] Unified STL evaluation between U-Net and A-scan classifier (same projection,
      same matching) so the two pipelines can share a headline F1 number.

### B. Paper figures (MEDIUM)
- [ ] Clean hero figure: best 4–6 cross-domain cases (input | GT | prediction).
- [ ] A-scan physics figure (sample 11 + 1–2 others), publication quality.
- [ ] Architecture / pipeline schematic.
- [ ] Metrics table: old vs new vs cross-domain.

### C. Model improvements (MEDIUM)
- [ ] Add an absolute-depth channel (normalised z plane) so the model knows slice depth.
- [ ] Try focal loss or hard-negative mining for the void/background imbalance.
- [ ] Consider polygon labelling or convex-hull post-processing to reduce rectangle bias.
- [ ] Iterate self-training once more (use `unet_pseudo_2050` to relabel, retrain).

### D. Housekeeping (LOW)
- [ ] Decide which large artifacts to keep vs regenerate (slice `.npy` dirs are big).
- [ ] Keep `Wiki.md` / `Plan.md` / `References.md` in sync after each work session.
- [ ] Commit code + small artifacts to GitHub (estherusera/ThzVoidTDS).

---

## Decisions log (why things are the way they are)
- **Per-slice 2D U-Net, not 3D**: too few labels for 3D; 2.5D context channels are
  the cheap compromise. (Wiki §7)
- **Offset step scales with slice count**: keep physical depth window ~±0.084 mm
  constant across 50/500/2050-slice models. User chose ±10 once and ±40 once — ±40
  matches the training window, ±10 is the tight experiment.
- **Manual weighted 5× over pseudo**: trust human labels more than model's own output
  while still using pseudo coverage. Sample 5 pseudo was low quality.
- **NO_VOID_SAMPLES = {"14"}**: forced zero masks; sample 10 also effectively no-void
  in atlanta2.
- **Train from scratch (not fine-tune)** for pseudo runs: atlanta2 set is ~50× larger
  than atlanta1's, so scratch avoids inheriting rectangle bias.
