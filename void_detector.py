"""
void_detector.py — Reference-subtraction void detection
=========================================================
Uses CV15-NoVoid (sample 15) as a reference. For each other sample,
computes |test - reference| per depth slice. Regions that differ
significantly from the reference are flagged as likely voids.

Outputs:
  void_detector_plots/{name}_void_diff.png   — visual grid per sample
  labels/void_detector_{name}_slice_masks.npy — binary masks (uint8)

Run:
    thesis_env/bin/python void_detector.py
"""

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage
from scipy.signal import hilbert

warnings.filterwarnings("ignore")

# ── constants ─────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

SAMPLES = {
    "s2":  dict(label="CV02",           dx=0.2,   dy=0.5, outdir="output_atlanta_s2"),
    "s3":  dict(label="CV03",           dx=0.2,   dy=0.5, outdir="output_atlanta_s3"),
    "s4":  dict(label="CV04",           dx=0.2,   dy=0.5, outdir="output_atlanta_s4"),
    "s5":  dict(label="CV05",           dx=0.2,   dy=0.5, outdir="output_atlanta_s5"),
    "s5b": dict(label="CV05-repeat",    dx=0.2,   dy=0.5, outdir="output_atlanta_s5b"),
    "s8":  dict(label="CV08",           dx=0.2,   dy=0.2, outdir="output_atlanta_s8"),
    "s10": dict(label="CV10-Diagonal",  dx=0.493, dy=0.2, outdir="output_atlanta_s10"),
    "s11": dict(label="CV11-Stripes",   dx=0.5,   dy=0.2, outdir="output_atlanta_s11"),
    "s12": dict(label="CV12-Prism",     dx=0.5,   dy=0.2, outdir="output_atlanta_s12"),
    "s13": dict(label="CV13-Random",    dx=0.5,   dy=0.2, outdir="output_atlanta_s13"),
    "s14": dict(label="CV14-Cilindros", dx=0.5,   dy=0.2, outdir="output_atlanta_s14"),
}
REFERENCE = dict(label="CV15-NoVoid", dx=0.5, dy=0.2, outdir="output_atlanta_s15")

N_SLICES   = 100
K_SIGMA    = 2.5      # flag pixels where diff > mean + k*std of that slice
N_SHOW     = 12       # rows in the visualisation grid
PLOTS_DIR  = BASE_DIR / "void_detector_plots"
LABELS_DIR = BASE_DIR / "labels"
# ──────────────────────────────────────────────────────────────────────────────


def load_and_process(outdir: str, dx_mm: float, dy_mm: float,
                     n_slices: int = N_SLICES):
    """Load thz_volume.npy from outdir, compute Hilbert envelope, flatten
    surface, extract n_slices depth slices, resample to isotropic pixels,
    and normalise each slice to [0, 1].

    The stored volumes are (ny, nx, nt) — time on axis 2.

    Returns
    -------
    slices    : float32  (n_slices, Ny, Nx)  normalised to [0, 1]
    depths_mm : float32  (n_slices,)          depth below surface in mm
    """
    d = BASE_DIR / outdir
    volume = np.load(d / "thz_volume.npy")           # (ny, nx, nt)
    dz_mm  = json.loads((d / "metadata.json").read_text())["dz_mm"]
    ny, nx, nt = volume.shape

    # 1. Hilbert envelope along time axis
    envelope = np.abs(hilbert(volume, axis=2)).astype(np.float32)

    # 2. Global surface index from mean A-scan
    mean_ascan  = envelope.mean(axis=(0, 1))          # (nt,)
    surface_idx = int(np.argmax(mean_ascan))

    # 3. Per-pixel local surface in ±20 % window
    margin = int(0.2 * nt)
    s0 = max(0, surface_idx - margin)
    s1 = min(nt, surface_idx + margin)
    local_surface = s0 + np.argmax(envelope[:, :, s0:s1], axis=2)  # (ny, nx)
    target_idx    = int(np.median(local_surface))

    # 4. Surface flattening — roll each A-scan so surface lands at target_idx
    flat_env = np.zeros_like(envelope)
    for iy in range(ny):
        for ix in range(nx):
            shift = target_idx - int(local_surface[iy, ix])
            flat_env[iy, ix, :] = np.roll(envelope[iy, ix, :], shift)

    # 5. Extract evenly-spaced depth slices
    slice_indices = np.linspace(target_idx, nt - 1, n_slices, dtype=int)
    slices = flat_env[:, :, slice_indices].transpose(2, 0, 1).copy()  # (n_s, ny, nx)
    depths_mm = (slice_indices - target_idx) * dz_mm

    # 6. Isotropic spatial resampling (upsample coarse axis to finer pitch)
    if abs(dx_mm - dy_mm) / max(dx_mm, dy_mm) > 0.05:
        target = min(dx_mm, dy_mm)
        zoom_y = dy_mm / target
        zoom_x = dx_mm / target
        slices = scipy.ndimage.zoom(slices, (1, zoom_y, zoom_x), order=1)

    # 7. Per-slice normalisation to [0, 1]
    for i in range(slices.shape[0]):
        vmax = float(np.percentile(slices[i], 99)) + 1e-8
        slices[i] = np.clip(slices[i] / vmax, 0.0, 1.0)

    return slices.astype(np.float32), depths_mm.astype(np.float32)


def align_to_reference(test_slices: np.ndarray,
                       ref_slices: np.ndarray) -> np.ndarray:
    """Resize test_slices spatial dims to match ref_slices if needed."""
    _, tNy, tNx = test_slices.shape
    _, rNy, rNx = ref_slices.shape
    if (tNy, tNx) == (rNy, rNx):
        return test_slices
    zoom_y = rNy / tNy
    zoom_x = rNx / tNx
    print(f"    [align] resizing ({tNy}×{tNx}) → ({rNy}×{rNx})  "
          f"zoom=({zoom_y:.3f}, {zoom_x:.3f})  "
          f"(note: test covers different physical area than reference)")
    return scipy.ndimage.zoom(test_slices, (1, zoom_y, zoom_x), order=1
                              ).astype(np.float32)


def compute_diff_masks(test_slices: np.ndarray,
                       ref_slices: np.ndarray,
                       k: float = K_SIGMA):
    """Per-slice absolute difference with adaptive k-sigma threshold.

    Returns
    -------
    diff : float32  (n_slices, Ny, Nx)  absolute difference
    mask : uint8    (n_slices, Ny, Nx)  1 where diff > mean + k*std
    """
    diff = np.abs(test_slices - ref_slices).astype(np.float32)
    mask = np.zeros_like(diff, dtype=np.uint8)
    for i in range(diff.shape[0]):
        mu    = diff[i].mean()
        sigma = diff[i].std()
        mask[i] = (diff[i] > mu + k * sigma).astype(np.uint8)
    return diff, mask


def visualize_and_save(name: str, label: str,
                       test_slices: np.ndarray,
                       ref_slices: np.ndarray,
                       diff: np.ndarray,
                       mask: np.ndarray,
                       depths_mm: np.ndarray,
                       plots_dir: Path,
                       labels_dir: Path) -> None:
    """Save a 3-column × N_SHOW-row figure and binary mask file."""
    n_s = test_slices.shape[0]
    show_idx = np.linspace(0, n_s - 1, N_SHOW, dtype=int)

    fig, axes = plt.subplots(N_SHOW, 3,
                             figsize=(9, N_SHOW * 2.2),
                             dpi=100)
    fig.suptitle(f"{name} ({label})  vs  CV15-NoVoid\n"
                 f"Threshold: diff > mean + {K_SIGMA}σ per slice",
                 fontsize=12, fontweight="bold", y=0.995)

    col_titles = ["THz slice", "|test − ref|  (hot)", "Binary mask"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=9, pad=4)

    for row, si in enumerate(show_idx):
        ax_thz  = axes[row, 0]
        ax_diff = axes[row, 1]
        ax_mask = axes[row, 2]

        ax_thz.imshow(test_slices[si], cmap="inferno", vmin=0, vmax=1,
                      aspect="equal")
        ax_diff.imshow(diff[si], cmap="hot",
                       vmin=0, vmax=float(np.percentile(diff, 99)),
                       aspect="equal")
        ax_mask.imshow(mask[si], cmap="gray", vmin=0, vmax=1,
                       aspect="equal")

        for ax in (ax_thz, ax_diff, ax_mask):
            ax.axis("off")
        ax_thz.set_ylabel(f"{depths_mm[si]:.2f} mm", fontsize=7,
                          rotation=0, labelpad=30, va="center")
        ax_thz.yaxis.set_label_position("left")

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plots_dir.mkdir(exist_ok=True)
    fig.savefig(plots_dir / f"{name}_void_diff.png",
                dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Save binary mask
    labels_dir.mkdir(exist_ok=True)
    np.save(labels_dir / f"void_detector_{name}_slice_masks.npy", mask)

    n_void_px     = int(mask.sum())
    n_void_slices = int((mask.sum(axis=(1, 2)) > 0).sum())
    print(f"    → {n_void_px:6d} void pixels total  |  "
          f"{n_void_slices}/{n_s} slices with detections")


def main():
    print("=" * 60)
    print("THz void detector — reference subtraction")
    print("=" * 60)

    # Process reference once
    print(f"\nProcessing reference: {REFERENCE['label']} ({REFERENCE['outdir']})")
    ref_slices, ref_depths = load_and_process(
        REFERENCE["outdir"], REFERENCE["dx"], REFERENCE["dy"])
    print(f"  Reference shape: {ref_slices.shape}  "
          f"depth range: 0 – {ref_depths[-1]:.2f} mm")

    # Sanity check: reference vs itself (should give near-zero diff)
    print("\n--- Sanity check: s15 vs s15 (expect ~0 detections) ---")
    diff_self, mask_self = compute_diff_masks(ref_slices, ref_slices)
    visualize_and_save("s15_self", "CV15-NoVoid (self-check)",
                       ref_slices, ref_slices,
                       diff_self, mask_self, ref_depths,
                       PLOTS_DIR, LABELS_DIR)

    # Process each test sample
    results = {}
    for name, meta in SAMPLES.items():
        print(f"\n--- {name} ({meta['label']}) ---")
        try:
            test_slices, depths = load_and_process(
                meta["outdir"], meta["dx"], meta["dy"])
            print(f"  shape={test_slices.shape}  "
                  f"depth range: 0 – {depths[-1]:.2f} mm")

            test_aligned = align_to_reference(test_slices, ref_slices)
            diff, mask   = compute_diff_masks(test_aligned, ref_slices)
            visualize_and_save(name, meta["label"],
                               test_aligned, ref_slices,
                               diff, mask, depths,
                               PLOTS_DIR, LABELS_DIR)
            results[name] = {
                "void_px":     int(mask.sum()),
                "void_slices": int((mask.sum(axis=(1, 2)) > 0).sum()),
                "n_slices":    mask.shape[0],
            }
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback; traceback.print_exc()
            results[name] = None

    # Summary table
    print("\n" + "=" * 60)
    print(f"{'Sample':<8}  {'Label':<20}  {'Void px':>8}  {'Slices w/ voids':>16}")
    print("-" * 60)
    for name, r in results.items():
        label = SAMPLES[name]["label"]
        if r is None:
            print(f"  {name:<8}  {label:<20}  {'FAILED':>8}")
        else:
            print(f"  {name:<8}  {label:<20}  {r['void_px']:>8}  "
                  f"  {r['void_slices']:>3}/{r['n_slices']}")
    print("=" * 60)
    print(f"\nPlots  → {PLOTS_DIR}/")
    print(f"Masks  → {LABELS_DIR}/void_detector_*_slice_masks.npy")


if __name__ == "__main__":
    main()
