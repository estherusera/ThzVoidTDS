# THz Void Detection — References

> External references used in this project. Maintained by the assistant.
> Last updated: 2026-06-24. Add a source here whenever it informs a decision.

---

## Methods & background

### Terahertz time-domain spectroscopy (THz-TDS) NDT
- THz-TDS reflection imaging for NDT of composites: pulses reflect at
  refractive-index interfaces; a void (CFRP→air) produces an extra echo with
  **inverted polarity** relative to the front surface, and echo spacing encodes void
  thickness. (Justifies the A-scan analysis.)
- Depth conversion: **n ≈ 1.57** for CFRP/plastic; depth = (Δt·c)/(2·n),
  c = 0.29979 mm/ps (`process_to_slices`).

### Hilbert envelope / analytic signal
- `scipy.signal.hilbert` — analytic-signal envelope, for surface detection and
  per-pixel A-scan flattening.

### Band-pass denoising (Jun 2026)
- Zero-phase Butterworth band-pass (`scipy.signal.butter` + `sosfiltfilt`), default
  **0.1–2.5 THz**, applied per A-scan along the time axis before enveloping. Cutoff
  range is the standard useful THz-TDS band; sampling fs ≈ 45.9 THz (Nyquist 22.9).
- Edge handling: median-subtract + zero-pad before `sosfiltfilt` to avoid
  boundary-ring energy blow-up; surface alignment detected on the RAW envelope to
  avoid ring-snapping artifacts. (`thz_slice_pipeline.bandpass_filter`; Wiki §4/§9.)

### U-Net (segmentation architecture)
- Ronneberger, Fischer, Brox (2015), "U-Net: Convolutional Networks for Biomedical
  Image Segmentation", MICCAI. https://arxiv.org/abs/1505.04597 — basis for the
  encoder–decoder + skip-connection models (2D C-scan U-Net and the 1D A-scan U-Net).

### U-Net-BiLSTM for THz defect detection (primary new reference, Jun 2026)
- Zhang et al. (2024), "Quantitative Detection of Defects in Multi-Layer Lightweight
  Composite Structures Using THz-TDS Based on a U-Net-BiLSTM Network", *Materials*
  17(4):839. https://doi.org/10.3390/ma17040839 — reimplemented as
  `ascan_unet_bilstm.py`: a 1D U-Net with a BiLSTM on the deepest skip connections,
  per-time-point A-scan damage segmentation.
- BiLSTM / LSTM: Hochreiter & Schmidhuber (1997), "Long Short-Term Memory",
  *Neural Computation* 9(8); Schuster & Paliwal (1997), "Bidirectional Recurrent
  Neural Networks", IEEE TSP 45(11). (`nn.LSTM(bidirectional=True)` on skips.)

### Loss functions
- Dice loss: Milletari et al. (2016), "V-Net", https://arxiv.org/abs/1606.04797 —
  soft Dice for class-imbalanced segmentation. BCE+Dice is the robust default;
  focal loss (Lin et al. 2017, https://arxiv.org/abs/1708.02002) is the alternative
  for the ~1–5% positive fraction in the per-time A-scan labels.

### Self-training / pseudo-labeling
- Train on labelled → predict on unlabelled → add confident pseudo-labels → retrain;
  here manual labels are up-weighted 5× to limit error entrenchment.

### Evaluation metrics (segmentation)
See `evaluation_metrics.docx` for the full write-up (IoU, mIoU, Dice, precision,
recall, pixel accuracy) with formulas. Core sources:
- Jaccard (1912), *New Phytologist* 11(2) — IoU / Jaccard index.
- Dice (1945), *Ecology* 26(3); Sørensen (1948), *Biol. Skr.* 5 — Dice coefficient.
- van Rijsbergen (1979), *Information Retrieval* (2nd ed.) — precision/recall/F1.
- Everingham et al. (2010), PASCAL VOC, *IJCV* 88(2) — mIoU as the segmentation std.
- Taha & Hanbury (2015), *BMC Medical Imaging* 15:29 — metric survey; why accuracy
  fails under class imbalance.

---

## Software / tools
- PyTorch (models, training; MPS backend on Apple Silicon) — https://pytorch.org
- NumPy, SciPy (`ndimage`, `signal`) — arrays, zoom, Hilbert, Butterworth filtering.
- trimesh — STL mesh loading + `mesh.contains()` GT voxelization.
- Matplotlib, Pillow — figures and base64 PNGs for the HTML viewers.
- Plotly — interactive 3D volume viewer (`view_3d.py`).
- `gh` CLI — GitHub auth & repo management.

---

## Project resources
- GitHub: https://github.com/estherusera/ThzVoidTDS
- Raw data (local only, not in git): `3D_print_esther_atlanta.tprj` (atlanta1),
  `3D_print_esther_atlanta2.tprj` (atlanta2).
- Sample CAD: `AllSamples/*.step` and `AllSamples/stl/*.stl` (designed void geometry).

---

## To add later (placeholders)
- [ ] Specific THz-NDT-of-CFRP paper(s) for the literature review.
- [ ] THz scanner / acquisition software citation (tprj format origin).
- [ ] Dicky's wavelet-denoising parameters (when provided).
- [ ] Erwan's deconvolution method (when integrated).
- [ ] Dataset DOI if the scans are published.
