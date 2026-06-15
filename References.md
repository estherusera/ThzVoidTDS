# THz Void Detection — References

> External references used in this project. Maintained by the assistant.
> Last updated: 2026-05-28. Add a source here whenever it informs a decision.

---

## Methods & background

### Terahertz time-domain spectroscopy (THz-TDS) NDT
- THz-TDS reflection imaging for non-destructive testing of composites: pulses
  reflect at refractive-index interfaces; a void (CFRP→air) produces an extra echo
  with **inverted polarity** relative to the front surface, and echo spacing encodes
  void thickness. (Standard THz-NDT theory — used to justify the A-scan analysis.)
- Refractive index used for depth conversion: **n ≈ 1.57** for CFRP/plastic;
  depth = (Δt · c) / (2·n), c = 0.29979 mm/ps. (Constants in `process_to_slices`.)

### Hilbert envelope / analytic signal
- `scipy.signal.hilbert` — analytic-signal envelope, used for surface detection and
  per-pixel flattening of A-scans.

### U-Net (segmentation architecture)
- Ronneberger, Fischer, Brox (2015), "U-Net: Convolutional Networks for Biomedical
  Image Segmentation", MICCAI. https://arxiv.org/abs/1505.04597
  — basis for the encoder–decoder + skip-connection model here.

### Loss functions
- Dice loss: Milletari et al. (2016), "V-Net" — soft Dice for class-imbalanced
  segmentation. https://arxiv.org/abs/1606.04797
- Combined BCE+Dice is a common robust default for small/imbalanced masks.

### Self-training / pseudo-labeling
- General semi-supervised self-training: train on labelled data → predict on
  unlabelled → add confident predictions as pseudo-labels → retrain. Here adapted
  with manual labels up-weighted 5× to limit error entrenchment.

---

## Software / tools
- PyTorch (model, training; MPS backend on Apple Silicon) — https://pytorch.org
- NumPy, SciPy (`ndimage`, `signal`) — array ops, zoom, Hilbert, peak finding.
- Matplotlib — all figures.
- Pillow (PIL) — PNG encoding for the HTML viewers.
- `gh` CLI — GitHub auth & repo management.

---

## Project resources
- GitHub repo: https://github.com/estherusera/ThzVoidTDS
- Raw data: `3D_print_esther_atlanta.tprj` (atlanta1), `3D_print_esther_atlanta2.tprj`
  (atlanta2) — local only, not in git.

---

## To add later (placeholders — fill when used)
- [ ] Specific THz-NDT-of-CFRP paper(s) for the literature review.
- [ ] The THz scanner / acquisition software citation (tprj format origin).
- [ ] Any dataset DOI if the scans are published.
