"""
train_ascan_unet_bilstm.py — train the U-Net-BiLSTM A-scan damage segmenter
===========================================================================
Per-time-point void segmentation on the atlanta2 A-scan cache, under the fixed
sample-level split (no leakage): TRAIN={1..10,12,15}, VAL={11,13}, NOVOID={14}.

Loss      : soft Dice + weighted BCE (pos_weight from the void/background ratio);
            `--loss focal` swaps the BCE term for focal loss.
Optimiser : AdamW + cosine schedule, per-scan amplitude normalisation, and the
            depth model's augmentations (time roll + noise).
Logging   : every epoch appends {epoch, val per-point F1, val Dice, ...} to
            runs/unet_bilstm/metrics.jsonl and refreshes runs/unet_bilstm/curves.png.
            Best-by-VAL-F1 checkpoint → runs/unet_bilstm/unet_bilstm_best.pt.

Run (background):
    thesis_env/bin/python train_ascan_unet_bilstm.py [--epochs 40] [--loss dice_bce|focal]
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys
# Capture our real CLI args BEFORE importing ascan_classifier, which rewrites
# sys.argv to ["...","predict","atlanta2"] as an import side effect.
_ARGV = sys.argv[1:]
sys.path.insert(0, str(Path(__file__).parent))
from ascan_unet_bilstm import UNetBiLSTM1D, AScanSegDataset, SEQ_LEN
from ascan_classifier import CACHE_ATL2, TRAIN_SAMPLES, VAL_SAMPLES

RUN_DIR = Path("runs/unet_bilstm")
RUN_DIR.mkdir(parents=True, exist_ok=True)


# ── losses ───────────────────────────────────────────────────────────────────
def soft_dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=1)
    denom = p.sum(dim=1) + target.sum(dim=1)
    dice  = (2 * inter + eps) / (denom + eps)
    return (1 - dice).mean()


class DiceBCE(nn.Module):
    def __init__(self, pos_weight):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight))

    def forward(self, logits, target):
        return 0.5 * soft_dice_loss(logits, target) + 0.5 * self.bce(logits, target)


class DiceFocal(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma

    def focal(self, logits, target):
        p = torch.sigmoid(logits)
        ce = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        pt = p * target + (1 - p) * (1 - target)
        w  = self.alpha * target + (1 - self.alpha) * (1 - target)
        return (w * (1 - pt) ** self.gamma * ce).mean()

    def forward(self, logits, target):
        return 0.5 * soft_dice_loss(logits, target) + 0.5 * self.focal(logits, target)


# ── eval ─────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tp = fp = fn = 0
    dice_num = dice_den = 0.0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        p = torch.sigmoid(model(x))
        pred = (p > 0.5).float()
        tp += float((pred * y).sum())
        fp += float((pred * (1 - y)).sum())
        fn += float(((1 - pred) * y).sum())
        dice_num += float((2 * p * y).sum())
        dice_den += float((p + y).sum())
    prec = tp / max(tp + fp, 1.0)
    rec  = tp / max(tp + fn, 1.0)
    f1   = 2 * prec * rec / max(prec + rec, 1e-9)
    dice = dice_num / max(dice_den, 1e-9)
    return dict(f1=f1, precision=prec, recall=rec, dice=dice)


def plot_curves(history, path):
    if not history:
        return
    ep = [h["epoch"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), dpi=130)
    ax[0].plot(ep, [h["train_loss"] for h in history], label="train loss")
    ax[0].set(xlabel="epoch", ylabel="loss", title="Training loss"); ax[0].grid(alpha=0.3); ax[0].legend()
    ax[1].plot(ep, [h["val_f1"] for h in history],   label="val per-point F1")
    ax[1].plot(ep, [h["val_dice"] for h in history], label="val soft Dice")
    ax[1].plot(ep, [h["val_recall"] for h in history], ls=":", label="val recall")
    ax[1].plot(ep, [h["val_precision"] for h in history], ls=":", label="val precision")
    ax[1].set(xlabel="epoch", ylim=(0, 1), title="Validation (per-point)"); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss", choices=["dice_bce", "focal"], default="dice_bce")
    ap.add_argument("--pos_weight_cap", type=float, default=20.0)
    ap.add_argument("--no_bilstm", action="store_true", help="ablation: plain 1D U-Net")
    ap.add_argument("--tag", default="", help="suffix for outputs, e.g. _bp (avoids clobbering)")
    args = ap.parse_args(_ARGV)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={device}  split TRAIN={TRAIN_SAMPLES} VAL={VAL_SAMPLES}")

    train_ds = AScanSegDataset(TRAIN_SAMPLES, CACHE_ATL2, augment=True)
    val_ds   = AScanSegDataset(VAL_SAMPLES,   CACHE_ATL2, augment=False)
    print(f"train items={len(train_ds)}  val items={len(val_ds)}")

    f = train_ds.pos_fraction()
    pos_weight = min((1 - f) / max(f, 1e-6), args.pos_weight_cap)
    print(f"train void fraction={f:.4f}  → pos_weight={pos_weight:.1f} "
          f"(raw {(1-f)/max(f,1e-6):.1f}, capped at {args.pos_weight_cap})")

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)

    model = UNetBiLSTM1D(bilstm_deep_skips=not args.no_bilstm).to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"model params={n_p:,}  bilstm={'OFF' if args.no_bilstm else 'ON'}  loss={args.loss}")

    crit = (DiceFocal() if args.loss == "focal" else DiceBCE(pos_weight)).to(device)
    opt  = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    tag = args.tag
    metrics_path = RUN_DIR / f"metrics{tag}.jsonl"
    ckpt_path    = RUN_DIR / f"unet_bilstm{tag}_best.pt"
    curves_path  = RUN_DIR / f"curves{tag}.png"
    metrics_path.write_text("")  # fresh log
    history, best_f1 = [], -1.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train(); ep_loss = nb = 0
        for x, y in train_ld:
            x = x.to(device); y = y.to(device)
            loss = crit(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            ep_loss += float(loss); nb += 1
        sched.step()
        train_loss = ep_loss / max(nb, 1)
        val = evaluate(model, val_ld, device)

        row = dict(epoch=epoch, train_loss=train_loss,
                   val_f1=val["f1"], val_dice=val["dice"],
                   val_precision=val["precision"], val_recall=val["recall"],
                   lr=sched.get_last_lr()[0], elapsed_min=(time.time()-t0)/60)
        history.append(row)
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        plot_curves(history, curves_path)
        print(f"epoch {epoch:3d}/{args.epochs}  loss {train_loss:.4f}  "
              f"valF1 {val['f1']:.3f}  valDice {val['dice']:.3f}  "
              f"P {val['precision']:.3f} R {val['recall']:.3f}  "
              f"[{row['elapsed_min']:.1f} min]", flush=True)

        if val["f1"] > best_f1:
            best_f1 = val["f1"]
            torch.save({"model_state": model.state_dict(),
                        "config": {"base": 16, "bilstm": not args.no_bilstm,
                                   "seq_len": SEQ_LEN, "loss": args.loss,
                                   "pos_weight": pos_weight},
                        "epoch": epoch, "val": val},
                       ckpt_path)

    print(f"\nDone. best val per-point F1 = {best_f1:.3f}  → {ckpt_path}")
    print(f"metrics: {metrics_path}   curves: {curves_path}")


if __name__ == "__main__":
    main()
