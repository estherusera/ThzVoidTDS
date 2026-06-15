"""
retrain_pseudo.py — Retrain U-Net on the enriched atlanta2 dataset
===================================================================
Combines 500-slice pseudo-labels with the manual 50-slice ground-truth
labels (mapped to their depth-matched indices in the 500-slice array).
Manual-labeled slices get a 5× weight in the loss so the model trusts
them more than its own pseudo-predictions.

Inputs:
  slices_v2_atlanta2_500/{name}_slices.npy   (500, 100, 100) float32
  slices_v2_atlanta2_500/{name}_depths.npy   (500,)          float32
  slices_v2_atlanta2/{name}_depths.npy       (50,)           float32  (manual depths)
  labels_v2_atlanta2/{name}_slice_masks.npy  (50, 100, 100)  uint8    (manual masks)
  labels_v2_atlanta2_pseudo/{name}_pseudo_mask.npy   (500, 100, 100)  uint8

Output:
  results_v2_atlanta2/unet_pseudo.pt
  results_v2_atlanta2/training_curves_pseudo.png
  results_v2_atlanta2/{name}_predictions_pseudo.png  (per sample)

Run:
    thesis_env/bin/python retrain_pseudo.py
"""

import sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

# ── CLI args (read BEFORE sys.argv is rewritten for pipeline import) ─────────
# Usage:
#   retrain_pseudo.py                       → 500 slices, step 5  (default)
#   retrain_pseudo.py 2050 20 30            → 2050 slices, step 20 (±40), 30 epochs
N_SLICES_SAMPLE  = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OFFSET_STEP      = int(sys.argv[2]) if len(sys.argv) > 2 else 5
N_EPOCHS         = int(sys.argv[3]) if len(sys.argv) > 3 else 60

sys.path.insert(0, str(Path(__file__).parent))
# Force atlanta2 mode in pipelinev2 for path resolution / NO_VOID_SAMPLES
sys.argv = [sys.argv[0], "predict", "atlanta2"]
from thz_slice_pipelinev2 import UNet, DiceBCELoss, NO_VOID_SAMPLES

# ── paths (derived from N_SLICES_SAMPLE) ─────────────────────────────────────
SLICES_DIR   = Path(f"slices_v2_atlanta2_{N_SLICES_SAMPLE}")
MANUAL_DEPTH = Path("slices_v2_atlanta2")         # 50-slice depths (manual GT)
MANUAL_DIR   = Path("labels_v2_atlanta2")
# back-compat: 500-slice run uses the original short name
PSEUDO_DIR   = (Path("labels_v2_atlanta2_pseudo") if N_SLICES_SAMPLE == 500
                else Path(f"labels_v2_atlanta2_pseudo_{N_SLICES_SAMPLE}"))
OUT_DIR      = Path("results_v2_atlanta2")
OUT_DIR.mkdir(exist_ok=True)

CKPT_NAME    = ("unet_pseudo.pt" if N_SLICES_SAMPLE == 500
                else f"unet_pseudo_{N_SLICES_SAMPLE}.pt")
CURVES_NAME  = ("training_curves_pseudo.png" if N_SLICES_SAMPLE == 500
                else f"training_curves_pseudo_{N_SLICES_SAMPLE}.png")

# ── training config ──────────────────────────────────────────────────────────
BATCH_SIZE       = 16
LR               = 1e-3
MANUAL_WEIGHT    = 5.0      # how much more manual labels count vs pseudo
PSEUDO_WEIGHT    = 1.0
CONTEXT_N        = 2
OFFSETS          = [o * OFFSET_STEP for o in range(-CONTEXT_N, CONTEXT_N + 1)]


# ── dataset that yields (input, target, weight) per slice ────────────────────
class WeightedSliceDataset(Dataset):
    """Each item: (input C×H×W, mask 1×H×W, weight scalar)."""

    def __init__(self, samples, augment=False):
        self.items   = []
        self.augment = augment

        n_manual = n_pseudo = n_novoid = 0
        for sd in samples:
            slices    = sd["slices"]
            mask_vol  = sd["mask_vol"]      # (500, 100, 100) uint8
            weights   = sd["weight_vol"]    # (500,)
            n_s       = slices.shape[0]

            for i in range(n_s):
                channels = [slices[int(np.clip(i + off, 0, n_s - 1))]
                            for off in OFFSETS]
                x = np.stack(channels, axis=0).astype(np.float32)
                y = mask_vol[i].astype(np.float32)
                w = float(weights[i])
                self.items.append((x, y, w))

                if sd["name"] in NO_VOID_SAMPLES:
                    n_novoid += 1
                elif w >= MANUAL_WEIGHT:
                    n_manual += 1
                else:
                    n_pseudo += 1

        print(f"  Dataset: {len(self.items)} slice-mask-weight triples")
        print(f"    manual-label slices:   {n_manual}  (weight {MANUAL_WEIGHT})")
        print(f"    pseudo-label slices:   {n_pseudo}  (weight {PSEUDO_WEIGHT})")
        print(f"    no-void zero-masks:    {n_novoid}  (weight {PSEUDO_WEIGHT})")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y, w = self.items[idx]
        x = torch.from_numpy(x)
        y = torch.from_numpy(y).unsqueeze(0)
        if self.augment:
            if torch.rand(1) > 0.5:
                x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
            if torch.rand(1) > 0.5:
                x, y = torch.flip(x, [-2]), torch.flip(y, [-2])
            if torch.rand(1) > 0.5:
                x = torch.rot90(x, 2, [-2, -1])
                y = torch.rot90(y, 2, [-2, -1])
            x = torch.clamp(x + 0.02 * torch.randn_like(x), 0, 1)
        return x, y, torch.tensor(w, dtype=torch.float32)


# ── build combined labels for one sample ─────────────────────────────────────
def build_sample(name):
    """Load slices/depths and assemble (combined_mask, weight_vol)."""
    slices    = np.load(SLICES_DIR  / f"{name}_slices.npy")           # (500, H, W)
    depths500 = np.load(SLICES_DIR  / f"{name}_depths.npy")           # (500,)

    n_s, H, W = slices.shape
    mask_vol   = np.zeros((n_s, H, W), dtype=np.uint8)
    weight_vol = np.full((n_s,), PSEUDO_WEIGHT, dtype=np.float32)

    # No-void samples: keep mask all zeros, do NOT trust pseudo for them
    if name in NO_VOID_SAMPLES:
        print(f"    {name}: no-void → all-zero target, weight {PSEUDO_WEIGHT}")
        return dict(name=name, slices=slices, depths=depths500,
                    mask_vol=mask_vol, weight_vol=weight_vol)

    # 1) start with pseudo labels everywhere
    pseudo_path = PSEUDO_DIR / f"{name}_pseudo_mask.npy"
    if pseudo_path.exists():
        mask_vol = np.load(pseudo_path).astype(np.uint8)
        weight_vol[:] = PSEUDO_WEIGHT

    # 2) overwrite at the depth-matched indices where manual labels exist
    manual_path = MANUAL_DIR / f"{name}_slice_masks.npy"
    if manual_path.exists():
        manual    = np.load(manual_path)                              # (50, H, W)
        depths50  = np.load(MANUAL_DEPTH / f"{name}_depths.npy")      # (50,)
        labelled  = np.where(manual.sum(axis=(1, 2)) > 0)[0]
        n_overridden = 0
        # Also override the empty manual slices (negatives are useful too)
        for i50 in range(len(depths50)):
            target_d = depths50[i50]
            i500     = int(np.argmin(np.abs(depths500 - target_d)))
            mask_vol[i500]   = manual[i50]
            weight_vol[i500] = MANUAL_WEIGHT
            n_overridden    += 1
        print(f"    {name}: pseudo ({N_SLICES_SAMPLE}) + {n_overridden} manual @ "
              f"weight {MANUAL_WEIGHT}  "
              f"({len(labelled)} of manual are positive)")
    else:
        print(f"    {name}: pseudo only")

    return dict(name=name, slices=slices, depths=depths500,
                mask_vol=mask_vol, weight_vol=weight_vol)


# ── weighted loss ────────────────────────────────────────────────────────────
class WeightedDiceBCELoss(nn.Module):
    """DiceBCE per item, scaled by the per-item weight, averaged over batch."""

    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_pix     = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets, weights):
        # BCE per pixel, then mean over H,W per item
        bce = self.bce_pix(logits, targets).mean(dim=(1, 2, 3))   # (B,)

        # Dice per item
        p     = torch.sigmoid(logits)
        inter = (p * targets).sum(dim=(1, 2, 3))
        union = p.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
        dice  = (2 * inter + 1) / (union + 1)                     # (B,)

        per_item = (1 - self.dice_weight) * bce + self.dice_weight * (1 - dice)

        # apply weights then average
        return (per_item * weights).sum() / (weights.sum() + 1e-8)


# ── train loop ───────────────────────────────────────────────────────────────
def train(model, ds, n_epochs, lr, device):
    loader  = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    optim   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optim, n_epochs)
    crit    = WeightedDiceBCELoss(0.5)
    model.to(device)

    history = {"loss": [], "dice": [], "iou": []}
    n_p     = sum(p.numel() for p in model.parameters())
    print(f"\nTraining: {len(ds)} items, {n_epochs} epochs, "
          f"batch={BATCH_SIZE}, {n_p:,} params, device={device}")
    print("─" * 70)

    t0 = time.time()
    for epoch in range(n_epochs):
        model.train()
        ep_loss = ep_dice = ep_iou = nb = 0
        for x, y, w in loader:
            x, y, w = x.to(device), y.to(device), w.to(device)
            logits  = model(x)
            loss    = crit(logits, y, w)
            optim.zero_grad(); loss.backward(); optim.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                inter = (preds * y).sum(dim=(2, 3))
                denom = preds.sum(dim=(2, 3)) + y.sum(dim=(2, 3))
                dice  = ((2 * inter + 1) / (denom + 1)).mean()
                iou   = ((inter + 1) / (denom - inter + 1)).mean()

            ep_loss += loss.item(); ep_dice += dice.item()
            ep_iou  += iou.item();  nb += 1

        sched.step()
        history["loss"].append(ep_loss / nb)
        history["dice"].append(ep_dice / nb)
        history["iou"].append(ep_iou  / nb)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch+1:3d}/{n_epochs} | "
                  f"Loss {ep_loss/nb:.4f} | "
                  f"Dice {ep_dice/nb:.4f} | "
                  f"IoU {ep_iou/nb:.4f} | "
                  f"elapsed {elapsed/60:.1f} min")
    return history


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"\n=== retrain_pseudo (atlanta2, {N_SLICES_SAMPLE}-slice + weighted manual) ===\n")
    print(f"  slices dir:    {SLICES_DIR}/")
    print(f"  pseudo dir:    {PSEUDO_DIR}/")
    print(f"  manual weight: {MANUAL_WEIGHT}")
    print(f"  pseudo weight: {PSEUDO_WEIGHT}")
    print(f"  offsets:       {OFFSETS}")
    print(f"  epochs:        {N_EPOCHS}")
    print(f"  batch:         {BATCH_SIZE}\n")

    sample_names = sorted(
        [p.stem.replace("_slices", "") for p in SLICES_DIR.glob("*_slices.npy")],
        key=lambda x: int(x.rstrip("b"))
    )
    print(f"  samples: {sample_names}\n")
    print("Building combined label volumes:")
    samples = [build_sample(n) for n in sample_names]

    dataset = WeightedSliceDataset(samples, augment=True)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if hasattr(torch.backends, "mps") and
                            torch.backends.mps.is_available()
              else "cpu")
    model = UNet(in_channels=len(OFFSETS), base_filters=32)

    history = train(model, dataset, n_epochs=N_EPOCHS, lr=LR, device=device)

    # save model
    ckpt_path = OUT_DIR / CKPT_NAME
    torch.save({
        "model_state": model.state_dict(),
        "config": {"in_channels": len(OFFSETS), "base_filters": 32,
                   "offsets": OFFSETS,
                   "n_slices_per_sample": N_SLICES_SAMPLE,
                   "manual_weight": MANUAL_WEIGHT,
                   "pseudo_weight": PSEUDO_WEIGHT},
        "history": history,
    }, ckpt_path)
    print(f"\nSaved: {ckpt_path}")

    # training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    axes[0].plot(history["loss"]); axes[0].set_title("Loss"); axes[0].set_yscale("log")
    axes[1].plot(history["dice"]); axes[1].set_title("Dice"); axes[1].set_ylim(0, 1)
    axes[2].plot(history["iou"]);  axes[2].set_title("IoU");  axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
    plt.suptitle(f"Training (pseudo-labels + 5× manual, "
                 f"{N_SLICES_SAMPLE}-slice, offsets {OFFSETS})", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_DIR / CURVES_NAME, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_DIR}/{CURVES_NAME}")

    print(f"\nFinal — Loss {history['loss'][-1]:.4f} | "
          f"Dice {history['dice'][-1]:.4f} | "
          f"IoU {history['iou'][-1]:.4f}")


if __name__ == "__main__":
    main()
