# THz Void Detection — Project Plan

> Roadmap & task tracker. Maintained by the assistant. Last updated: 2026-06-24.
> See `Wiki.md` for full project state and `References.md` for sources.

## Status legend
- [x] done  ·  [~] partial  ·  [ ] not started

---

## Completed phases (see Wiki for detail)

**P1 Data foundation** — tprj loader, surface-flatten + depth-slice + isotropic-zoom
pipeline, ground-truth void specs (Wiki §3), 20×20 mm ROI crop, per-slice manual
labeler. Reference-subtraction detector abandoned.

**P2 atlanta1 baseline** — 148 positive slices, base U-Net `unet_v2.pt`.

**P3 atlanta2 (clean data)** — dataset-aware scripts, 176 positive slices, A-scan
physics analysis.

**P4 Self-training / pseudo-labeling** — atlanta2 @ 500 then 2050 slices,
`unet_pseudo.pt` / `unet_pseudo_2050.pt` (manual weighted 5×).

**P5 Cross-domain** — `unet_pseudo_2050` → atlanta1 IoU 0.532 / Dice 0.653
(beats atlanta1's native model); same-domain honest check IoU 0.492.

**P6 Void characterization vs STL CAD** — `void_characterize.py`: mesh-contains GT,
8-symmetry Hungarian match (8 mm gate). C-scan U-Net headline: mean F1 0.44, P 0.87,
volume blow-up ~19×.

**P7 A-scan 1D-CNN classifier** — cache builder + depth-aware CNN; unified per-blob
F1 **0.703** (P 0.96 R 0.62) via `ascan_vs_stl.py`. NoVoid sanity passes.

**P8 STL-supervised U-Net** — `unet_stl.pt`; unified F1 0.523 (+0.055 over manual).

**P9 Band-pass denoising (Jun 2026)** — `thz_slice_pipeline.bandpass_filter`
(0.1–2.5 THz Butterworth, env-toggled, default ON). Fixed two bugs: sosfiltfilt
energy blow-up (median-sub + zero-pad) and surface-alignment ring-snapping (detect
surface on raw envelope, slice on filtered). **Effect is architecture-dependent
(P10): helps C-scan U-Net (+0.054) & U-Net-BiLSTM (+0.088), regresses 1D-CNN
(−0.122) on the per-blob metric.** Harness `run_bandpass_ab.sh`; Wiki §9.

**P10 U-Net-BiLSTM A-scan segmenter (Jun 2026)** — Zhang et al. 2024 reimplemented:
`ascan_unet_bilstm.py` (1D U-Net + BiLSTM on the 2 deepest skips), per-time-point
output reduced to 50 bins for the unified harness. Trained + evaluated unfiltered
and band-pass, plus cross-domain. Comparison viewers built. Wiki §15.
- Unified per-blob (atlanta2 all-void): 1D-CNN **0.679** > C-scan 0.561(50-sl) >
  BiLSTM 0.521 (unfiltered). With band-pass the ranking flips: BiLSTM **0.610** >
  1D-CNN 0.556. BiLSTM-bp best on held-out VAL (0.525) and cross-domain (0.669).
- Per-point (2048) vs per-blob gap documented (0.650 vs 0.521 for unfiltered BiLSTM).

---

## NEXT / OPEN (priority order)

### A. Honest evaluation (HIGH — paper credibility)
- [ ] Leave-one-sample-out (or held-out) CV on atlanta2; report mean ± std. All
      headline F1 to date are computed partly on training samples.
- [ ] Multi-seed runs — current band-pass/BiLSTM deltas are single-seed on a
      2-sample VAL; confirm the ±0.05–0.12 effects are real, not noise.
- [x] Void-conditional metrics (done — unified per-blob is void-only).
- [x] Unified STL eval shared across U-Net / A-scan / BiLSTM (done — `void_characterize`
      matcher reused by all).
- [x] Cross-domain A-scan eval under the unified metric (done — 1D-CNN 0.650,
      BiLSTM 0.580→0.669 bp).

### B. Model improvements (MEDIUM)
- [ ] BiLSTM ablation: `--no_bilstm` vs full, to isolate the BiLSTM-in-skips
      contribution (vs plain 1D U-Net).
- [ ] Depth-stratified sampling/weighting to fix the depth-coverage starvation
      (no voids <1 mm or >3.8 mm in training — see `dataset_properties.py`).
- [ ] Improve recall on many-void samples (s7, s10, s11) for all models.
- [ ] Wavelet denoising (Dicky's params) as the next signal-processing lever;
      deconvolution (Erwan) after.
- [ ] Band-pass on the full 2050 pseudo C-scan pipeline (current C-scan bp is a
      50-slice proxy, not the 0.44 headline model).

### C. Paper figures (MEDIUM)
- [ ] Hero cross-domain figure (input | GT | prediction).
- [ ] Architecture/pipeline schematic; old vs new vs cross-domain metrics table.
- [x] Evaluation-metrics definitions doc (`evaluation_metrics.docx`).
- [x] Dataset-properties / data-repartition figure (`dataset_properties.py`).

### D. Housekeeping (LOW)
- [x] `.gitignore` excludes large regenerable data; first GitHub push done.
- [ ] Push the A-scan-BiLSTM + band-pass session work (committed-ready, not pushed).
- [x] Keep Wiki/Plan/References in sync (this refactor).

---

## Decisions log (why things are the way they are)
- **2.5D per-slice U-Net, not 3D**: too few labels for 3D; neighbour channels are the
  cheap compromise.
- **Offset step scales with slice count**: keeps the physical depth window ~constant
  across 50/500/2050-slice models.
- **Manual weighted 5× over pseudo**: trust human labels more than the model's own.
- **Train from scratch for pseudo runs**: atlanta2 set ~50× larger than atlanta1's.
- **Band-pass default ON but architecture-dependent**: kept ON because it helps the
  C-scan/slice pipeline and the BiLSTM; the 1D-CNN regresses on per-blob, so it is
  NOT a universal win. Env-toggle (`THZ_BANDPASS=0`) reproduces the unfiltered path.
- **Surface alignment on the RAW envelope** (not filtered): the band-pass rings the
  surface pulse, so aligning on the filtered envelope caused blocky depth artifacts.
- **Two A-scan model families share the unified per-blob metric**: 1D-CNN and
  U-Net-BiLSTM both operate per-pixel on the time-domain waveform; their per-time/
  per-depth outputs are reduced to a (50,NY,NX) volume only for scoring, so they are
  directly comparable to the C-scan U-Net.
