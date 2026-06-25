"""
dataset_properties.py — characterise the void label distribution
=================================================================
Answers "is the model trained with enough voids at every depth / width, and how
much do voids overlap?" by labelling every void in the manual masks with its
depth, lateral width, axial thickness, and overlap (number of voids stacked
along the same A-scan), then plotting the data repartition.

Two views are produced:
  • Per-A-scan  — each (y, x) pixel column through the 50 depth bins. Used for
    the depth-coverage and overlap histograms (this is exactly what the A-scan
    classifier sees: one label vector of length 50 per pixel).
  • Per-blob    — 3D connected components (voids) for width / thickness / volume.

Source labels: manual masks `{name}_slice_masks.npy` (50, 100, 100) uint8 with
the matching `{name}_depths.npy` (50,) depth axis (mm). In-plane pixel = 0.2 mm.

Usage:
    thesis_env/bin/python dataset_properties.py            # atlanta2 (default)
    thesis_env/bin/python dataset_properties.py atlanta1
"""

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import label

DX_MM = 0.2  # in-plane pixel size (20 mm / 100 px)

DATASETS = {
    "atlanta1": dict(labels_dir="labels_v2",          depths_dir="slices_v2"),
    "atlanta2": dict(labels_dir="labels_v2_atlanta2", depths_dir="slices_v2_atlanta2"),
}
# Samples that are no-void references / have no detectable voids (see Wiki §3).
NO_VOID = {"14"}


def depth_segments(col):
    """Disjoint runs of 1s along a length-50 column. Returns list of (lo, hi)."""
    idx = np.where(col > 0)[0]
    if len(idx) == 0:
        return []
    breaks = np.where(np.diff(idx) > 1)[0] + 1
    return [(int(g[0]), int(g[-1])) for g in np.split(idx, breaks)]


def analyse_sample(name, labels_dir, depths_dir):
    mask = np.load(labels_dir / f"{name}_slice_masks.npy").astype(bool)  # (50,100,100)
    depths = np.load(depths_dir / f"{name}_depths.npy").astype(float)    # (50,) mm
    nz, ny, nx = mask.shape
    # local depth step per bin (mm), for single-bin axial thickness
    dz = np.gradient(depths)

    # ── per-A-scan view: columns through depth ──────────────────────────────
    cols = mask.reshape(nz, ny * nx).T          # (10000, 50)
    void_cols = cols[cols.any(axis=1)]          # only A-scans that hit a void
    n_void_ascans = int(len(void_cols))
    depth_hist = cols.sum(axis=0)               # void count per depth bin (len 50)
    overlap_counts = []                         # # stacked voids per void A-scan
    seg_depth_mm, seg_thick_mm = [], []         # per depth-segment
    for c in void_cols:
        segs = depth_segments(c)
        overlap_counts.append(len(segs))
        for lo, hi in segs:
            seg_depth_mm.append(float(depths[(lo + hi) // 2]))
            seg_thick_mm.append(float(depths[hi] - depths[lo] + dz[(lo + hi) // 2]))

    # ── per-blob view: 3D connected components ───────────────────────────────
    lab, n_blob = label(mask)                   # 6-connectivity default
    blob_width_mm, blob_thick_mm, blob_depth_mm, blob_vol_mm3 = [], [], [], []
    vox_mm3 = DX_MM * DX_MM * float(np.mean(dz))
    for b in range(1, n_blob + 1):
        zz, yy, xx = np.where(lab == b)
        # lateral footprint = union of pixels over depth → equivalent diameter
        foot = len(set(zip(yy.tolist(), xx.tolist())))
        blob_width_mm.append(2.0 * np.sqrt(foot / np.pi) * DX_MM)
        blob_thick_mm.append(float(depths[zz.max()] - depths[zz.min()] + dz[int(zz.mean())]))
        blob_depth_mm.append(float(depths[int(round(zz.mean()))]))
        blob_vol_mm3.append(len(zz) * vox_mm3)

    return dict(
        name=name, n_void_ascans=n_void_ascans, n_blobs=int(n_blob),
        depth_hist=depth_hist, overlap_counts=overlap_counts,
        seg_depth_mm=seg_depth_mm, seg_thick_mm=seg_thick_mm,
        blob_width_mm=blob_width_mm, blob_thick_mm=blob_thick_mm,
        blob_depth_mm=blob_depth_mm, blob_vol_mm3=blob_vol_mm3,
        depths=depths,
    )


def main():
    dataset = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in DATASETS else "atlanta2"
    cfg = DATASETS[dataset]
    labels_dir, depths_dir = Path(cfg["labels_dir"]), Path(cfg["depths_dir"])
    if not labels_dir.is_dir():
        sys.exit(f"no such dir: {labels_dir}")

    names = sorted((p.stem.replace("_slice_masks", "")
                    for p in labels_dir.glob("*_slice_masks.npy")),
                   key=lambda x: int(x.rstrip("b")))
    print(f"=== dataset properties: {dataset} ({len(names)} samples) ===\n")

    results = [analyse_sample(n, labels_dir, depths_dir) for n in names]
    depths_ref = results[0]["depths"]

    # ── per-sample table ─────────────────────────────────────────────────────
    print(f"{'samp':>4} {'voidAscans':>10} {'blobs':>5} {'medWidth':>9} "
          f"{'medThick':>9} {'medDepth':>9} {'%overlap':>9}")
    agg = dict(depth_hist=np.zeros(50), overlap=[], seg_depth=[], seg_thick=[],
               width=[], thick=[], depth=[], vol=[])
    rows = []
    for r in results:
        ov = r["overlap_counts"]
        pct_overlap = 100 * np.mean(np.array(ov) >= 2) if ov else 0.0
        mw = np.median(r["blob_width_mm"]) if r["blob_width_mm"] else float("nan")
        mt = np.median(r["blob_thick_mm"]) if r["blob_thick_mm"] else float("nan")
        md = np.median(r["blob_depth_mm"]) if r["blob_depth_mm"] else float("nan")
        print(f"{r['name']:>4} {r['n_void_ascans']:>10} {r['n_blobs']:>5} "
              f"{mw:>9.2f} {mt:>9.2f} {md:>9.2f} {pct_overlap:>8.1f}%")
        rows.append(dict(name=r["name"], n_void_ascans=r["n_void_ascans"],
                         n_blobs=r["n_blobs"], median_width_mm=mw,
                         median_thick_mm=mt, median_depth_mm=md,
                         pct_ascans_overlapping=pct_overlap))
        if r["name"] in NO_VOID:
            continue
        agg["depth_hist"] += r["depth_hist"]
        agg["overlap"] += r["overlap_counts"]
        agg["seg_depth"] += r["seg_depth_mm"]
        agg["seg_thick"] += r["seg_thick_mm"]
        agg["width"] += r["blob_width_mm"]
        agg["thick"] += r["blob_thick_mm"]
        agg["depth"] += r["blob_depth_mm"]
        agg["vol"] += r["blob_vol_mm3"]

    total_ascans = int(agg["depth_hist"].sum())
    n_overlap = int(np.sum(np.array(agg["overlap"]) >= 2))
    print(f"\nTotal void A-scan/depth labels: {total_ascans}")
    print(f"Void A-scans with >=2 stacked voids: {n_overlap} "
          f"({100*n_overlap/max(len(agg['overlap']),1):.1f}%)")
    print(f"Total voids (3D blobs): {len(agg['width'])}")

    # ── figure ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(2, 3, figsize=(16, 9), dpi=130)

    ax[0, 0].bar(depths_ref, agg["depth_hist"],
                 width=np.gradient(depths_ref) * 0.9, color="#3b6")
    ax[0, 0].set(title="Void A-scans per depth bin (coverage)",
                 xlabel="depth (mm)", ylabel="# void A-scans")

    ax[0, 1].hist(agg["seg_thick"], bins=30, color="#36b")
    ax[0, 1].set(title="Axial thickness per void segment",
                 xlabel="thickness (mm)", ylabel="count")

    ax[0, 2].hist(agg["width"], bins=30, color="#b63")
    ax[0, 2].set(title="Lateral width per void (blob)",
                 xlabel="equivalent diameter (mm)", ylabel="count")

    ov = np.array(agg["overlap"])
    bins = np.arange(0.5, (ov.max() if len(ov) else 1) + 1.5)
    ax[1, 0].hist(ov, bins=bins, color="#963", rwidth=0.85)
    ax[1, 0].set(title="Overlap: # voids stacked per void A-scan",
                 xlabel="stacked voids along depth", ylabel="# void A-scans")
    ax[1, 0].set_xticks(range(1, int(ov.max()) + 1) if len(ov) else [1])

    sc = ax[1, 1].scatter(agg["depth"], agg["width"],
                          c=agg["thick"], cmap="viridis", s=30)
    ax[1, 1].set(title="Void width vs depth (colour = thickness)",
                 xlabel="depth (mm)", ylabel="width (mm)")
    fig.colorbar(sc, ax=ax[1, 1], label="thickness (mm)")

    names_plt = [r["name"] for r in results]
    counts = [r["n_void_ascans"] for r in results]
    ax[1, 2].bar(range(len(names_plt)), counts, color="#69b")
    ax[1, 2].set(title="Void A-scans per sample (class balance)",
                 xlabel="sample", ylabel="# void A-scans")
    ax[1, 2].set_xticks(range(len(names_plt)))
    ax[1, 2].set_xticklabels(names_plt, rotation=45, ha="right")

    plt.suptitle(f"Void label distribution — {dataset} "
                 f"({len(agg['width'])} voids, {total_ascans} void A-scan labels)",
                 fontsize=14)
    plt.tight_layout()
    out_png = f"dataset_properties_{dataset}.png"
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_png.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"\nSaved figure: {out_png} (+ .pdf)")

    # ── machine-readable summary ──────────────────────────────────────────────
    summary = dict(
        dataset=dataset,
        n_samples=len(names),
        total_void_ascan_labels=total_ascans,
        total_voids=len(agg["width"]),
        pct_ascans_overlapping=float(100 * n_overlap / max(len(agg["overlap"]), 1)),
        depth_bins_mm=[float(d) for d in depths_ref],
        void_ascans_per_depth_bin=[int(v) for v in agg["depth_hist"]],
        per_sample=rows,
    )
    out_json = f"dataset_properties_{dataset}.json"
    with open(out_json, "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved summary: {out_json}")


if __name__ == "__main__":
    main()
