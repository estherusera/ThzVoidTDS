"""
ascan_vs_stl_figures.py — Comparison figures for the A-scan 1D-CNN classifier
under the unified STL metric (see ascan_vs_stl.py).

Produces (all into results_v2_atlanta2/ascan_classifier/vs_stl_unified/):
  1. head_to_head.png        — bar chart, A-scan F1_unified vs C-scan U-Net F1
                                per sample + aggregates.
  2. per_sample_showcase.png — 6 representative samples × 3 panels (STL z-proj,
                                A-scan max-proj, A-scan + matched centroids).
  3. old_vs_unified_f1.png   — already produced by ascan_vs_stl.py (kept here
                                for reference).

Run:
    thesis_env/bin/python ascan_vs_stl_figures.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, str(Path(__file__).parent))

# Re-use the heavy lifting from ascan_vs_stl.py.
from ascan_vs_stl import (
    DepthAwareAScanCNN, MODEL_CKPT, DEPTHS_DIR, OUT_DIR,
    predict_prob_volume, device,
)
from void_characterize import (
    voxelize_gt, extract_pred_blobs, best_alignment_and_match,
    SAMPLE_TO_STL, STL_DIR, PX_MM, NX, NY,
)
from ascan_classifier import CACHE_ATL2

UNIFIED_JSON = OUT_DIR / "summary_unified.json"
UNET_AGG     = Path("results_v2_atlanta2/void_char/aggregate.json")

# 6 representative samples for the showcase: 3 best, 3 hard.
SHOWCASE_SAMPLES = ["1", "4", "15", "8", "12", "10"]


# ── load saved results ───────────────────────────────────────────────────────
def load_unified():
    with open(UNIFIED_JSON) as fh:
        rows = json.load(fh)
    return {r["sample"]: r for r in rows}


def load_unet():
    with open(UNET_AGG) as fh:
        d = json.load(fh)
    return {r["sample"]: r for r in d["per_sample"]}, d


# ── (1) head-to-head bar chart ───────────────────────────────────────────────
def plot_head_to_head():
    ascan_by = load_unified()
    unet_by, unet_agg = load_unet()

    # All void samples present in BOTH
    samples = sorted([s for s in ascan_by
                      if ascan_by[s]["n_gt"] > 0 and s in unet_by],
                     key=lambda x: int(x.rstrip("b")))
    names  = samples
    ascan  = [ascan_by[s]["f1_unified"] for s in samples]
    unet   = [unet_by[s]["f1"]           for s in samples]

    x = np.arange(len(names)); w = 0.38

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 6.4), dpi=140,
        gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.patch.set_facecolor("white")

    ax_top.bar(x - w/2, unet,  w, label="C-scan U-Net (void_characterize)",
               color="#1f78b4")
    ax_top.bar(x + w/2, ascan, w, label="A-scan 1D-CNN (ascan_vs_stl)",
               color="#33a02c")
    ax_top.set_xticks(x); ax_top.set_xticklabels(names)
    ax_top.set_ylabel("F1 (per-blob, 8 mm gate)")
    ax_top.set_ylim(0, 1.05)
    ax_top.grid(axis="y", alpha=0.3)
    ax_top.set_title(
        "Same metric (mesh.contains + 8-symmetry + Hungarian + 8 mm gate)\n"
        "A-scan 1D-CNN vs C-scan U-Net on atlanta2 — per sample",
        fontsize=11, fontweight="bold")
    ax_top.legend(loc="upper right", fontsize=9)
    ax_top.set_xlabel("Sample")

    # Aggregate bar at the bottom
    mu_ascan = float(np.mean(ascan)); mu_unet = float(np.mean(unet))
    p_ascan  = float(np.mean([ascan_by[s]["p_unified"] for s in samples]))
    r_ascan  = float(np.mean([ascan_by[s]["r_unified"] for s in samples]))
    p_unet   = float(unet_agg["mean_precision"])
    r_unet   = float(unet_agg["mean_recall"])

    metrics = ["F1", "Precision", "Recall"]
    unet_v  = [mu_unet,  p_unet,  r_unet]
    asc_v   = [mu_ascan, p_ascan, r_ascan]
    xx = np.arange(len(metrics))
    ax_bot.bar(xx - w/2, unet_v, w, color="#1f78b4")
    ax_bot.bar(xx + w/2, asc_v,  w, color="#33a02c")
    for xi, (uv, av) in enumerate(zip(unet_v, asc_v)):
        ax_bot.text(xi - w/2, uv + 0.02, f"{uv:.3f}", ha="center", fontsize=8,
                    color="#1f78b4", fontweight="bold")
        ax_bot.text(xi + w/2, av + 0.02, f"{av:.3f}", ha="center", fontsize=8,
                    color="#33a02c", fontweight="bold")
    ax_bot.set_xticks(xx); ax_bot.set_xticklabels(metrics)
    ax_bot.set_ylim(0, 1.15)
    ax_bot.grid(axis="y", alpha=0.3)
    ax_bot.set_title(f"Aggregate over {len(samples)} void samples",
                     fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "head_to_head.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")
    return mu_ascan, mu_unet, samples


# ── (2) per-sample showcase ──────────────────────────────────────────────────
def plot_showcase(model, ascan_by):
    nrows = len(SHOWCASE_SAMPLES)
    fig, axes = plt.subplots(nrows, 3, figsize=(9.5, 2.4 * nrows), dpi=140)
    fig.patch.set_facecolor("white")

    for row, name in enumerate(SHOWCASE_SAMPLES):
        info = ascan_by[name]
        thr  = info["best_threshold"]
        sym  = info["best_symmetry"]

        cache_path  = CACHE_ATL2 / f"{name}.npz"
        depths_path = DEPTHS_DIR / f"{name}_depths.npy"
        stl_path    = STL_DIR / SAMPLE_TO_STL[name]
        depths = np.load(depths_path).astype(np.float32)

        # GT footprint (union of per-void z-projections)
        gt_masks, gt_meta = voxelize_gt(stl_path, depths)
        gt_proj = np.zeros((NY, NX), dtype=bool)
        for m in gt_masks:
            if m is not None:
                gt_proj |= m

        # Model output and 2D max-projection
        probs = predict_prob_volume(model, cache_path)
        pred_proj = (probs.max(axis=0) > thr)
        pred_blobs = extract_pred_blobs(probs, thr, depths)

        # Matching at best threshold (so we can mark which pred centroids hit)
        if gt_meta and pred_blobs:
            match = best_alignment_and_match(gt_meta, pred_blobs)
            G_aligned = match["G_aligned"]
            matched_pred_ids = {pi for _, pi, _ in match["matches"]}
        else:
            G_aligned = [m["centroid_mm"] for m in gt_meta]
            matched_pred_ids = set()

        # ── panel 0: STL z-projection (orange) ───────────────────────────
        ax = axes[row, 0]
        ax.imshow(gt_proj, cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"S{name}  STL ({SAMPLE_TO_STL[name].replace('.stl','')})\n"
                     f"n_gt={info['n_gt']}",
                     fontsize=8, fontweight="bold")

        # ── panel 1: A-scan max-projected prob > thr (green) ─────────────
        ax = axes[row, 1]
        ax.imshow(pred_proj, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"A-scan pred (thr={thr:.2f})\n"
                     f"n_pred={info['n_pred_blobs']}",
                     fontsize=8, fontweight="bold")

        # ── panel 2: centroid overlay (GT ○ green=matched red=missed,
        #            pred ×: green=hit, blue=spurious) ───────────────────
        ax = axes[row, 2]
        bg = np.full((NY, NX), 0.97)
        ax.imshow(bg, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        # GT centroids
        for gi, g in enumerate(G_aligned):
            hit = any(gi == m[0] for m in (
                (best_alignment_and_match(gt_meta, pred_blobs)["matches"]
                 if (gt_meta and pred_blobs) else [])))
            ax.plot(g[0]/PX_MM, g[1]/PX_MM, "o", ms=10, mfc="none",
                    mec=("#27ae60" if hit else "#c0392b"), lw=2)
        # Pred centroids
        for pi, m in enumerate(pred_blobs):
            cx, cy, _ = m["centroid_mm"]
            color = "#27ae60" if pi in matched_pred_ids else "#1f5fa8"
            ax.plot(cx/PX_MM, cy/PX_MM, "x", ms=8, mew=2, color=color)
        ax.set_xlim(0, NX); ax.set_ylim(NY, 0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"Centroids   F1={info['f1_unified']:.2f}  "
                     f"P={info['p_unified']:.2f}  R={info['r_unified']:.2f}\n"
                     f"sym={sym}",
                     fontsize=8, fontweight="bold")

    # Legend
    handles = [
        mpatches.Patch(facecolor="#ffcc99", edgecolor="black",
                       label="STL void footprint (mesh.contains z-proj)"),
        mpatches.Patch(facecolor="#aae0aa", edgecolor="black",
                       label="A-scan prediction (prob > thr)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="#27ae60", markersize=10, lw=2,
                   label="GT void (matched)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="none",
                   markeredgecolor="#c0392b", markersize=10, lw=2,
                   label="GT void (missed)"),
        plt.Line2D([0], [0], marker="x", color="#27ae60", lw=0,
                   markersize=10, mew=2, label="A-scan blob (hit)"),
        plt.Line2D([0], [0], marker="x", color="#1f5fa8", lw=0,
                   markersize=10, mew=2, label="A-scan blob (spurious)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle("A-scan 1D-CNN under unified STL metric — representative samples",
                 fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout(rect=[0, 0.03, 1, 0.99])
    out = OUT_DIR / "per_sample_showcase.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    if not UNIFIED_JSON.exists():
        sys.exit(f"missing {UNIFIED_JSON} — run ascan_vs_stl.py first")
    if not UNET_AGG.exists():
        sys.exit(f"missing {UNET_AGG} — run void_characterize.py all first")

    ascan_by = load_unified()
    mu_a, mu_u, samples = plot_head_to_head()
    print(f"  Aggregate over {len(samples)} samples: "
          f"A-scan F1={mu_a:.3f}  U-Net F1={mu_u:.3f}  Δ={mu_a-mu_u:+.3f}")

    model_ckpt = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    model = DepthAwareAScanCNN().to(device)
    model.load_state_dict(model_ckpt["model_state"])
    model.eval()
    plot_showcase(model, ascan_by)


if __name__ == "__main__":
    main()
