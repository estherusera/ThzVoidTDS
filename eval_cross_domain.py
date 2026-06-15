"""
eval_cross_domain.py — Test the atlanta2-trained 2050-slice model on the
OLD atlanta1 dataset (different resolutions, noisier scans).

For every atlanta1 manually-labelled slice, run the model at the depth-matched
2050-slice index (offsets ±40) and compare to the manual mask.

Run:
    thesis_env/bin/python eval_cross_domain.py
"""

import sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from pathlib import Path

sys.argv = [sys.argv[0], "predict", "atlanta2"]
sys.path.insert(0, ".")
from thz_slice_pipelinev2 import UNet

# ── config ───────────────────────────────────────────────────────────────────
MODEL_CKPT  = Path("results_v2_atlanta2/unet_pseudo_2050.pt")
SLICES_2050 = Path("slices_v2_2050")           # atlanta1 @ 2050 slices
SLICES_50   = Path("slices_v2")                # atlanta1 @ 50 slices (manual depths)
MANUAL_DIR  = Path("labels_v2")                # atlanta1 manual masks (50-slice)
OUT_DIR     = Path("results_v2")
OFFSETS     = [-40, -20, 0, 20, 40]

device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")

ckpt  = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
cfg   = ckpt["config"]
model = UNet(in_channels=cfg["in_channels"], base_filters=cfg["base_filters"])
model.load_state_dict(ckpt["model_state"]); model.to(device).eval()
print(f"Model: {MODEL_CKPT}  (offsets {cfg.get('offsets')})\n")


def iou_dice(gt, pred):
    g, p = gt > 0, pred > 0
    u = (g | p).sum()
    if u == 0:
        return 1.0, 1.0
    i = (g & p).sum()
    return i / u, 2 * i / (g.sum() + p.sum() + 1e-8)


def run(slices, i, n_s):
    chans = [slices[int(np.clip(i + o, 0, n_s - 1))] for o in OFFSETS]
    x = torch.from_numpy(np.stack(chans)).unsqueeze(0).to(device).float()
    with torch.no_grad():
        return torch.sigmoid(model(x)).squeeze().cpu().numpy()


samples = sorted([p.stem.replace("_slice_masks", "")
                  for p in MANUAL_DIR.glob("*_slice_masks.npy")],
                 key=lambda x: (int(x.rstrip("b")), x))

rows, all_iou, all_dice = [], [], []
showcase = []   # (name, depth, input, gt, prob) for the figure

for name in samples:
    s2050_p = SLICES_2050 / f"{name}_slices.npy"
    if not s2050_p.exists():
        print(f"  {name}: no 2050 slices, skip"); continue
    slices2050 = np.load(s2050_p)
    depths2050 = np.load(SLICES_2050 / f"{name}_depths.npy")
    depths50   = np.load(SLICES_50   / f"{name}_depths.npy")
    manual     = np.load(MANUAL_DIR  / f"{name}_slice_masks.npy")
    n_s = slices2050.shape[0]

    lbl_idx = np.where(manual.sum(axis=(1, 2)) > 0)[0]
    if len(lbl_idx) == 0:
        rows.append((name, 0, None, None)); continue

    ious, dices = [], []
    best = (-1, None)   # (px, data) to pick busiest slice for showcase
    for i50 in lbl_idx:
        i2050 = int(np.argmin(np.abs(depths2050 - depths50[i50])))
        prob  = run(slices2050, i2050, n_s)
        a, b  = iou_dice(manual[i50], prob > 0.5)
        ious.append(a); dices.append(b)
        npx = int(manual[i50].sum())
        if npx > best[0]:
            best = (npx, (name, float(depths50[i50]),
                          slices2050[i2050], manual[i50], prob))

    rows.append((name, len(lbl_idx), float(np.mean(ious)), float(np.mean(dices))))
    all_iou += ious; all_dice += dices
    if best[1]:
        showcase.append(best[1])

# ── print table ──────────────────────────────────────────────────────────────
print(f"{'sample':>6s}  {'#lbl':>4s}  {'IoU':>6s}  {'Dice':>6s}")
print("─" * 34)
for r in rows:
    if r[1] == 0:
        print(f"{r[0]:>6s}  {r[1]:>4d}    (no manual)")
    else:
        print(f"{r[0]:>6s}  {r[1]:>4d}   {r[2]:.3f}  {r[3]:.3f}")
print("─" * 34)
print(f"OVERALL ({len(all_iou)} slices):  "
      f"IoU {np.mean(all_iou):.3f}   Dice {np.mean(all_dice):.3f}")

# ── showcase figure ──────────────────────────────────────────────────────────
N = len(showcase)
fig, axes = plt.subplots(N, 3, figsize=(7, 2.2 * N), dpi=150)
fig.patch.set_facecolor("white")
if N == 1:
    axes = axes[None, :]
for r, (name, depth, inp, gt, prob) in enumerate(showcase):
    axes[r, 0].imshow(inp, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    gt_rgb = np.ones((*gt.shape, 3)); gt_rgb[gt > 0] = [0.72, 0.11, 0.11]
    axes[r, 1].imshow(gt_rgb, interpolation="nearest")
    axes[r, 2].imshow(prob, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    for ax in axes[r]:
        ax.set_xticks([]); ax.set_yticks([])
    axes[r, 0].set_ylabel(f"S{name}\n$z$={depth:.2f}mm", fontsize=8,
                          rotation=0, ha="right", va="center", labelpad=4)
    if r == 0:
        axes[r, 0].set_title("Input (atlanta1)", fontsize=9, fontweight="bold")
        axes[r, 1].set_title("Manual GT", fontsize=9, fontweight="bold")
        axes[r, 2].set_title("Pred (atlanta2 model)", fontsize=9, fontweight="bold")

plt.suptitle("Cross-domain: atlanta2-trained 2050-slice model on atlanta1",
             fontsize=11, fontweight="bold", y=0.997)
plt.tight_layout()
out = OUT_DIR / "cross_domain_atlanta1.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {out}")
