"""
stl_diagnose.py — Diagnostic for STL label issues flagged on the 3-way viewer.

For each problem sample, show:
  • all 8 in-plane symmetries of the STL z-projection overlaid on the
    manual z-projection (so the right sym can be picked by eye)
  • the per-slice depth profile of the STL mask vs the per-slice depth
    profile of the actual input volume's reflection strength (so any
    z-offset between CAD and scan frames is visible)
  • where in depth s13's STL voids would sit if they were not zeroed

Run
    thesis_env/bin/python stl_diagnose.py 3 4 8 13
"""
import sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

_CLI = [a for a in sys.argv[1:] if not a.startswith("-")]
TARGETS = _CLI if _CLI else ["3", "4", "8", "13"]

sys.path.insert(0, str(Path(__file__).parent))
from void_characterize import SAMPLE_TO_STL, STL_DIR, PX_MM, NX, NY
from stl_to_labels import voxelize_gt_3d, apply_sym_mask
from scipy.ndimage import gaussian_filter

SLICES_DIR    = Path("slices_v2_atlanta2")
MANUAL_LABELS = Path("labels_v2_atlanta2")
OUT_DIR       = Path("results_v2_atlanta2/stl_labels/diagnose")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def all_syms_figure(name):
    """3×3 grid: panel 0 = manual, panels 1–8 = STL after each sym."""
    depths_path = SLICES_DIR / f"{name}_depths.npy"
    if not depths_path.exists():
        print(f"  {name}: no depths"); return
    depths = np.load(depths_path).astype(np.float32)
    stl_path = STL_DIR / SAMPLE_TO_STL[name]

    m3d = voxelize_gt_3d(stl_path, depths)              # CAD frame, sym=0
    m3d_blur = gaussian_filter(m3d, sigma=(1, 1, 1))
    m3d_blur = np.clip(m3d_blur, 0, 1)

    man_path = MANUAL_LABELS / f"{name}_slice_masks.npy"
    man_proj = np.zeros((NY, NX), dtype=np.float32)
    if man_path.exists():
        man = np.load(man_path).astype(np.float32)
        man_proj = (man.max(axis=0) > 0).astype(np.float32)

    fig, axes = plt.subplots(3, 3, figsize=(8.5, 8.5), dpi=140)
    fig.patch.set_facecolor("white")

    ax = axes.flat[0]
    rgb = np.zeros((NY, NX, 3), dtype=np.float32)
    rgb[..., 2] = man_proj
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title("Manual (blue) — reference", fontsize=9, fontweight="bold")
    ax.set_xticks([]); ax.set_yticks([])

    for sym_id in range(8):
        ax = axes.flat[sym_id + 1]
        m_s = apply_sym_mask(m3d_blur, sym_id)
        stl_proj = (m_s.max(axis=0) > 0.5).astype(np.float32)
        rgb = np.zeros((NY, NX, 3), dtype=np.float32)
        rgb[..., 0] = stl_proj
        rgb[..., 2] = man_proj
        inter = int((stl_proj.astype(bool) & man_proj.astype(bool)).sum())
        union = int((stl_proj.astype(bool) | man_proj.astype(bool)).sum())
        iou = inter / max(union, 1)
        ax.imshow(rgb, interpolation="nearest")
        title = f"sym={sym_id}  IoU={iou:.2f}"
        if sym_id == 0:
            title += "  (CAD frame)"
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle(f"Sample {name} — STL z-projection under each sym  vs  manual",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    out = OUT_DIR / f"all_syms_S{name}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")


def depth_profile_figure(name):
    """Plot STL mask mass per slice vs input slice intensity per slice."""
    depths_path  = SLICES_DIR / f"{name}_depths.npy"
    slices_path  = SLICES_DIR / f"{name}_slices.npy"
    if not depths_path.exists() or not slices_path.exists():
        return
    depths = np.load(depths_path).astype(np.float32)
    slices = np.load(slices_path).astype(np.float32)
    stl_path = STL_DIR / SAMPLE_TO_STL[name]

    m3d = voxelize_gt_3d(stl_path, depths)
    stl_per_slice = m3d.sum(axis=(1, 2))
    input_per_slice = slices.mean(axis=(1, 2))    # mean intensity (mostly background)
    # also the standard deviation per slice — high std = strong features (voids)
    input_std_per_slice = slices.std(axis=(1, 2))

    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=140)
    ax2 = ax.twinx()

    ax.plot(depths, stl_per_slice / max(stl_per_slice.max(), 1), "r-",
            label=f"STL mask (n={int((stl_per_slice > 0).sum())} sl)")
    ax2.plot(depths, input_std_per_slice / max(input_std_per_slice.max(), 1e-9),
             "b-", alpha=0.6, label="input slice std (normalised)")

    # Annotate any STL peak location and any input peak location
    if stl_per_slice.max() > 0:
        stl_peak = depths[int(np.argmax(stl_per_slice))]
        ax.axvline(stl_peak, color="r", linestyle=":", alpha=0.5)
        ax.text(stl_peak, 0.95, f"STL peak  z={stl_peak:.2f}",
                color="r", fontsize=8, rotation=90, va="top")
    input_peak = depths[int(np.argmax(input_std_per_slice))]
    ax2.axvline(input_peak, color="b", linestyle=":", alpha=0.5)
    ax.text(input_peak, 0.05, f"input peak  z={input_peak:.2f}",
            color="b", fontsize=8, rotation=90, va="bottom")

    ax.set_xlabel("Depth (mm)")
    ax.set_ylabel("STL mask sum (normalised)", color="r")
    ax2.set_ylabel("Input slice std (normalised)", color="b")
    ax.set_title(f"Sample {name} — STL depth profile vs input slice features",
                 fontsize=10, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / f"depth_profile_S{name}.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  saved: {out}")


def main():
    print(f"Diagnosing samples: {TARGETS}")
    for name in TARGETS:
        print(f"\n· sample {name}")
        all_syms_figure(name)
        depth_profile_figure(name)
    print(f"\nFigures in {OUT_DIR}/")


if __name__ == "__main__":
    main()
