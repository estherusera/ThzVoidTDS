"""
train_unet_stl.py — Train a 50-slice 2.5D U-Net on the STL-derived soft labels
produced by stl_to_labels.py (labels_stl_atlanta2/), as opposed to the
hand-drawn rectangular masks in labels_v2_atlanta2/.

Same architecture / hyperparams as thz_slice_pipelinev2.py train mode (UNet
in_channels=5, base_filters=32, Adam + cosine, DiceBCELoss), but:
  • Reads soft float labels (in [0,1]) instead of binary uint8
  • include_unlabeled=True so empty slices contribute as negatives
  • Saves to results_v2_atlanta2/unet_stl.pt

Run
    thesis_env/bin/python train_unet_stl.py            # 60 epochs default
    thesis_env/bin/python train_unet_stl.py 100        # custom epoch count
"""
import sys, time, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

# void_characterize / thz_slice_pipelinev2 rewrite sys.argv on import; capture
# our CLI first.
_CLI = [a for a in sys.argv[1:] if not a.startswith("-")]

sys.path.insert(0, str(Path(__file__).parent))
# thz_slice_pipelinev2 reads sys.argv[1:2] for mode+dataset. Set a safe dummy.
sys.argv = [sys.argv[0], "predict", "atlanta2"]
from thz_slice_pipelinev2 import UNet, SliceDataset, DiceBCELoss

SLICES_DIR  = Path("slices_v2_atlanta2")
LABELS_DIR  = Path("labels_stl_atlanta2")
RESULTS_DIR = Path("results_v2_atlanta2")
OUT_MODEL   = RESULTS_DIR / "unet_stl.pt"
OUT_CURVES  = RESULTS_DIR / "unet_stl_training_curves.png"
OUT_VIZ_DIR = RESULTS_DIR / "unet_stl_predictions"
OUT_VIZ_DIR.mkdir(parents=True, exist_ok=True)

N_EPOCHS = int(_CLI[0]) if _CLI else 60
LR       = 1e-3
CONTEXT  = 2
BATCH    = 8

device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def load_data():
    all_slices, all_masks, names = [], [], []
    for slices_p in sorted(SLICES_DIR.glob("*_slices.npy"),
                           key=lambda p: int(p.stem.split("_")[0].rstrip("b"))):
        name = slices_p.stem.replace("_slices", "")
        labels_p = LABELS_DIR / f"{name}_slice_masks.npy"
        if not labels_p.exists():
            print(f"  skip {name}: no STL label"); continue
        slices = np.load(slices_p).astype(np.float32)
        masks  = np.load(labels_p).astype(np.float32)
        if slices.shape[0] != masks.shape[0]:
            print(f"  skip {name}: shape mismatch {slices.shape} vs {masks.shape}")
            continue
        all_slices.append(slices); all_masks.append(masks); names.append(name)
        n_pos = int((masks.max(axis=(1, 2)) > 0.5).sum())
        print(f"  {name}: {n_pos}/{slices.shape[0]} slices have STL signal "
              f"(frac>{0.5}: {float((masks>0.5).sum())/masks.size*100:.2f}%)")
    return all_slices, all_masks, names


def visualize(model, all_slices, all_masks, names, depths_by):
    model.eval()
    for slices, masks, name in zip(all_slices, all_masks, names):
        n_s = slices.shape[0]
        pred = np.zeros((n_s, 100, 100), dtype=np.float32)
        with torch.no_grad():
            for i in range(n_s):
                ch = [slices[int(np.clip(i+o, 0, n_s-1))]
                      for o in range(-CONTEXT, CONTEXT+1)]
                x = torch.from_numpy(np.stack(ch)).unsqueeze(0).to(device)
                pred[i] = torch.sigmoid(model(x)).cpu().squeeze().numpy()

        # Pick 10 slices evenly spaced
        depths = depths_by[name]
        n_show = min(n_s, 10)
        idxs = np.linspace(0, n_s-1, n_show, dtype=int)
        fig, axes = plt.subplots(4, n_show, figsize=(2.5*n_show, 8), dpi=110)
        for j, si in enumerate(idxs):
            axes[0, j].imshow(slices[si], cmap="Reds", vmin=0, vmax=1); axes[0, j].axis("off")
            axes[0, j].set_title(f"d={depths[si]:.2f}mm", fontsize=8)
            axes[1, j].imshow(masks[si],  cmap="Oranges", vmin=0, vmax=1); axes[1, j].axis("off")
            axes[2, j].imshow(pred[si],   cmap="Greens",  vmin=0, vmax=1); axes[2, j].axis("off")
            axes[3, j].imshow((pred[si] > 0.5).astype(float),
                              cmap="Greens", vmin=0, vmax=1); axes[3, j].axis("off")
        for r, lbl in enumerate(("Input slice", "STL soft label",
                                 "Pred (sigmoid)", "Pred > 0.5")):
            axes[r, 0].set_ylabel(lbl, fontsize=10)
        plt.suptitle(f"Sample {name} — U-Net trained on STL labels",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUT_VIZ_DIR / f"{name}_predictions.png",
                    dpi=110, bbox_inches="tight")
        plt.close()
    print(f"Predictions saved to {OUT_VIZ_DIR}/")


def main():
    print(f"Device: {device}   epochs={N_EPOCHS}   batch={BATCH}")
    print(f"Loading STL labels from {LABELS_DIR}/")
    all_slices, all_masks, names = load_data()
    if not all_slices:
        sys.exit("No data")
    depths_by = {n: np.load(SLICES_DIR / f"{n}_depths.npy") for n in names}

    dataset = SliceDataset(all_slices, all_masks,
                           context_slices=CONTEXT, augment=True,
                           include_unlabeled=True)
    loader  = DataLoader(dataset, batch_size=BATCH, shuffle=True, drop_last=False)

    in_ch = CONTEXT*2 + 1
    model = UNet(in_channels=in_ch, base_filters=32).to(device)
    n_p   = sum(p.numel() for p in model.parameters())
    print(f"UNet: {n_p:,} params  in_channels={in_ch}\n")

    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPOCHS)
    crit  = DiceBCELoss(0.5)

    hist = {"loss": [], "dice": [], "iou": []}
    t0 = time.time()
    print("─" * 70)
    for ep in range(N_EPOCHS):
        model.train()
        ep_loss = ep_dice = ep_iou = 0; nb = 0
        for x, y in loader:
            x = x.to(device); y = y.to(device)
            logits = model(x); loss = crit(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            with torch.no_grad():
                p  = (torch.sigmoid(logits) > 0.5).float()
                yb = (y > 0.5).float()
                inter = (p * yb).sum(dim=(2, 3))
                denom = p.sum(dim=(2, 3)) + yb.sum(dim=(2, 3))
                dice  = ((2*inter + 1) / (denom + 1)).mean()
                iou   = ((inter + 1) / (denom - inter + 1)).mean()
            ep_loss += loss.item(); ep_dice += dice.item(); ep_iou += iou.item()
            nb += 1
        sched.step()
        hist["loss"].append(ep_loss/nb)
        hist["dice"].append(ep_dice/nb)
        hist["iou"].append(ep_iou/nb)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  ep {ep+1:3d}/{N_EPOCHS}  loss={ep_loss/nb:.4f}  "
                  f"dice={ep_dice/nb:.3f}  iou={ep_iou/nb:.3f}  "
                  f"({(time.time()-t0)/60:.1f} min)")

    torch.save({
        "model_state": model.state_dict(),
        "config": {"in_channels": in_ch, "base_filters": 32,
                   "context": CONTEXT, "n_slices": 50,
                   "labels_dir": str(LABELS_DIR)},
        "history": hist,
    }, OUT_MODEL)
    print(f"\nSaved: {OUT_MODEL}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=140)
    axes[0].plot(hist["loss"]); axes[0].set_title("Loss"); axes[0].set_yscale("log")
    axes[1].plot(hist["dice"]); axes[1].set_title("Dice"); axes[1].set_ylim(0, 1)
    axes[2].plot(hist["iou"]);  axes[2].set_title("IoU");  axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    plt.suptitle("U-Net on STL labels (atlanta2, 50-slice)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_CURVES, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_CURVES}")

    visualize(model, all_slices, all_masks, names, depths_by)


if __name__ == "__main__":
    main()
