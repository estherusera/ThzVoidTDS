"""
ascan_classifier_figures.py — Publication-quality figures for the 1D A-scan
classifier results.

Produces:
  1. f1_per_sample.png         — bar chart of F1 vs GT and F1 vs STL, grouped
                                  by train/val/no-void split.
  2. designed_vs_detectable.png — quantifies the gap between STL design voids
                                  and manually-detectable voids, per sample.
  3. showcase.png              — best-train / best-val / no-void samples shown
                                  as input slice + GT + STL + CNN prob + binary.
  4. prob_distributions.png    — histograms of predicted void probability for
                                  GT-positive vs GT-negative pixels, per sample.

Run:
    thesis_env/bin/python ascan_classifier_figures.py
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.argv = [sys.argv[0], "predict", "atlanta2"]
from ascan_classifier import (
    AScanCNN, CACHE_DIR, MANUAL_DIR, NY, NX,
    SAMPLE_TO_STL, TRAIN_SAMPLES, VAL_SAMPLES, NOVOID_SANITY,
    stl_projection, device,
)

OUT_DIR    = Path("results_v2_atlanta2/ascan_classifier")
MODEL_CKPT = OUT_DIR / "ascan_cnn.pt"
SLICES_50  = Path("slices_v2_atlanta2")

with open(OUT_DIR / "summary.json") as fh:
    summary = json.load(fh)

# Order samples numerically
ALL_NAMES = sorted(set(TRAIN_SAMPLES + VAL_SAMPLES + NOVOID_SANITY),
                   key=lambda x: int(x.rstrip("b")))
SUM_BY_NAME = {r["sample"]: r for r in summary}


# ── load model + recompute probability maps (also gets non-binary probabilities) ──
print("Loading model + computing probability maps for all samples…")
ckpt  = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
model = AScanCNN().to(device)
model.load_state_dict(ckpt["model_state"]); model.eval()

prob_maps, gt_maps, stl_maps = {}, {}, {}
all_probs_pos, all_probs_neg = {}, {}    # for histogram
for name in ALL_NAMES:
    cache = np.load(CACHE_DIR / f"{name}.npz")
    ascans = cache["ascans"]; labels = cache["labels"]
    mean = ascans.mean(axis=1, keepdims=True)
    std  = ascans.std(axis=1, keepdims=True) + 1e-6
    a_norm = (ascans - mean) / std
    a_t = torch.from_numpy(a_norm).unsqueeze(1).to(device)
    probs = np.zeros(a_t.shape[0], dtype=np.float32)
    CHUNK = 1024
    with torch.no_grad():
        for k in range(0, a_t.shape[0], CHUNK):
            logits = model(a_t[k:k+CHUNK])
            probs[k:k+CHUNK] = torch.sigmoid(logits).cpu().numpy()
    prob_maps[name] = probs.reshape(NY, NX)
    gt_maps[name]   = labels.reshape(NY, NX).astype(np.uint8)
    stl_maps[name]  = stl_projection(name).astype(np.uint8)
    all_probs_pos[name] = probs[labels == 1]
    all_probs_neg[name] = probs[labels == 0]


def split_color(name):
    if name in VAL_SAMPLES:     return "#e67e22"  # orange
    if name in NOVOID_SANITY:   return "#7f8c8d"  # grey
    return "#2980b9"                              # blue (train)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — F1 per sample, GT and STL
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5), dpi=160)
fig.patch.set_facecolor("white")

xpos = np.arange(len(ALL_NAMES))
W = 0.4
f1_gt  = [SUM_BY_NAME[n]["f1_gt"]  for n in ALL_NAMES]
f1_stl = [SUM_BY_NAME[n]["f1_stl"] for n in ALL_NAMES]
colors_gt  = [split_color(n) for n in ALL_NAMES]

bars1 = ax.bar(xpos - W/2, f1_gt, W, color=colors_gt, edgecolor="black", lw=0.6,
               label="F1 vs manual GT")
bars2 = ax.bar(xpos + W/2, f1_stl, W, color=colors_gt, alpha=0.45,
               hatch="//", edgecolor="black", lw=0.6,
               label="F1 vs STL design")

# Mean lines
train_idx = [i for i, n in enumerate(ALL_NAMES) if n in TRAIN_SAMPLES]
val_idx   = [i for i, n in enumerate(ALL_NAMES) if n in VAL_SAMPLES]
train_mean_gt  = np.mean([f1_gt[i]  for i in train_idx])
val_mean_gt    = np.mean([f1_gt[i]  for i in val_idx])
train_mean_stl = np.mean([f1_stl[i] for i in train_idx])
val_mean_stl   = np.mean([f1_stl[i] for i in val_idx])

ax.axhline(train_mean_gt, color="#2980b9", ls="--", lw=1,
           label=f"train mean F1 (vs GT) = {train_mean_gt:.2f}")
ax.axhline(val_mean_gt,   color="#e67e22", ls="--", lw=1,
           label=f"val mean F1 (vs GT) = {val_mean_gt:.2f}")

ax.set_xticks(xpos); ax.set_xticklabels(ALL_NAMES, fontsize=10)
ax.set_xlabel("Sample", fontsize=11)
ax.set_ylabel("F1 score", fontsize=11)
ax.set_ylim(0, 1.0)
ax.set_title("A-scan classifier F1 per sample — manual GT vs STL design",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Custom legend reflecting train/val/no-void colour coding
legend_split = [
    mpatches.Patch(color="#2980b9", label="train sample"),
    mpatches.Patch(color="#e67e22", label="val sample (held out)"),
    mpatches.Patch(color="#7f8c8d", label="no-void sample (sanity)"),
]
leg1 = ax.legend(handles=legend_split, loc="upper left", fontsize=9,
                 title="Split", framealpha=0.95)
ax.add_artist(leg1)
ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

plt.tight_layout()
out = OUT_DIR / "f1_per_sample.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Designed vs Detectable: STL ∩ GT, STL \ GT, GT \ STL counts per sample
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5.5), dpi=160)
fig.patch.set_facecolor("white")

inter   = []
stl_only = []
gt_only  = []
for n in ALL_NAMES:
    g = gt_maps[n].astype(bool)
    s = stl_maps[n].astype(bool)
    inter.append(int((g & s).sum()))
    stl_only.append(int((s & ~g).sum()))
    gt_only.append(int((g & ~s).sum()))

ax.bar(xpos, inter,                    color="#27ae60", label="STL ∩ GT  (visible & designed)")
ax.bar(xpos, stl_only, bottom=inter,   color="#e74c3c",
       label="STL \\ GT  (designed but invisible)")
ax.bar(xpos, gt_only,  bottom=np.array(inter)+np.array(stl_only),
       color="#3498db", label="GT \\ STL  (visible but not designed)")

ax.set_xticks(xpos); ax.set_xticklabels(ALL_NAMES, fontsize=10)
ax.set_xlabel("Sample", fontsize=11)
ax.set_ylabel("Pixel count", fontsize=11)
ax.set_title("Designed (STL) vs detectable (manual GT) void area per sample",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9.5, loc="upper right", framealpha=0.95)
ax.grid(axis="y", alpha=0.3)

# Annotate the "invisible" fraction per sample
for i, n in enumerate(ALL_NAMES):
    total_stl = inter[i] + stl_only[i]
    if total_stl > 0:
        pct_invisible = stl_only[i] / total_stl * 100
        if pct_invisible >= 25:
            ax.text(i, inter[i] + stl_only[i] / 2,
                    f"{pct_invisible:.0f}%\nhidden", ha="center", va="center",
                    fontsize=7.5, color="white", fontweight="bold")

plt.tight_layout()
out = OUT_DIR / "designed_vs_detectable.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Showcase: input mid-slice + GT + STL + CNN prob + binary
#            for 6 chosen samples (best train, best val, weak case, no-void)
# ─────────────────────────────────────────────────────────────────────────────
SHOWCASE = ["4", "1", "13", "11", "2", "14"]   # mix of strong, val, weak, no-void
fig, axes = plt.subplots(len(SHOWCASE), 5, figsize=(12, 2.4 * len(SHOWCASE)), dpi=160)
fig.patch.set_facecolor("white")

col_titles = ["Input slice (mid-depth)", "Manual GT", "STL design (z-proj)",
              "CNN probability", "CNN > 0.5"]

for r_idx, name in enumerate(SHOWCASE):
    row = SUM_BY_NAME[name]
    slices = np.load(SLICES_50 / f"{name}_slices.npy")     # (50, 100, 100)
    mid = slices.shape[0] // 3                             # roughly 1.4 mm depth
    inp = slices[mid]

    axes[r_idx, 0].imshow(inp,                 cmap="Reds", vmin=0, vmax=1,
                          interpolation="nearest")
    axes[r_idx, 1].imshow(gt_maps[name],       cmap="Reds", vmin=0, vmax=1,
                          interpolation="nearest")
    axes[r_idx, 2].imshow(stl_maps[name],      cmap="Reds", vmin=0, vmax=1,
                          interpolation="nearest")
    axes[r_idx, 3].imshow(prob_maps[name],     cmap="Reds", vmin=0, vmax=1,
                          interpolation="nearest")
    axes[r_idx, 4].imshow((prob_maps[name] > 0.5).astype(np.uint8),
                          cmap="Reds", vmin=0, vmax=1, interpolation="nearest")

    split = row["split"]
    badge_colour = "#2980b9" if split == "TRAIN" else ("#e67e22" if split == "VAL"
                                                       else "#7f8c8d")
    label = (f"S{name}  [{split}]\nF1 vs GT = {row['f1_gt']:.2f}\n"
             f"F1 vs STL = {row['f1_stl']:.2f}")
    axes[r_idx, 0].set_ylabel(label, fontsize=8.5, rotation=0,
                              ha="right", va="center", labelpad=10,
                              color=badge_colour, fontweight="bold")

    for ax in axes[r_idx]:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#888"); sp.set_linewidth(0.5)
    if r_idx == 0:
        for c, t in enumerate(col_titles):
            axes[0, c].set_title(t, fontsize=10, fontweight="bold", pad=5)

plt.suptitle("A-scan classifier showcase — input · manual GT · STL design · "
             "CNN probability · binary prediction",
             fontsize=11.5, fontweight="bold", y=0.997)
plt.tight_layout()
out = OUT_DIR / "showcase.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Probability distribution: void vs non-void per sample
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(13, 11), dpi=160)
fig.patch.set_facecolor("white")
axes = axes.flatten()

bins = np.linspace(0, 1, 41)

for i, name in enumerate(ALL_NAMES):
    ax = axes[i]
    pos = all_probs_pos[name]
    neg = all_probs_neg[name]
    if len(pos):
        ax.hist(pos, bins=bins, color="#c0392b", alpha=0.70,
                label=f"GT void  (n={len(pos)})", edgecolor="black", lw=0.3)
    if len(neg):
        ax.hist(neg, bins=bins, color="#2980b9", alpha=0.55,
                label=f"GT bg    (n={len(neg)})", edgecolor="black", lw=0.3)
    ax.axvline(0.5, color="black", ls="--", lw=0.8)
    split = SUM_BY_NAME[name]["split"]
    badge_colour = "#2980b9" if split == "TRAIN" else ("#e67e22" if split == "VAL"
                                                       else "#7f8c8d")
    ax.set_title(f"S{name}  [{split}]  ·  F1 vs GT = {SUM_BY_NAME[name]['f1_gt']:.2f}",
                 fontsize=9.5, color=badge_colour, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_xlabel("P(void)", fontsize=8)
    ax.set_ylabel("# pixels", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc="upper center", framealpha=0.85)
    ax.grid(axis="y", alpha=0.3)

# Hide unused axes
for j in range(len(ALL_NAMES), len(axes)):
    axes[j].axis("off")

plt.suptitle("Predicted void probability — GT-positive vs GT-negative pixels",
             fontsize=12, fontweight="bold", y=0.997)
plt.tight_layout()
out = OUT_DIR / "prob_distributions.png"
plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
plt.savefig(out.with_suffix(".pdf"), dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {out}")

print(f"\nAll 4 figures saved to {OUT_DIR}/")
