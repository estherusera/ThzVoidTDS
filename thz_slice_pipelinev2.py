"""
thz_slice_pipelinev2.py — ROI-cropped per-slice labeling, training, prediction
===============================================================================
Same workflow as thz_slice_pipeline.py but each sample is cropped to its
20×20 mm ROI (from sample_rois.json) before labeling / training / inference.
All samples become 100×100 px after cropping.

Usage:
    thesis_env/bin/python thz_slice_pipelinev2.py label
    thesis_env/bin/python thz_slice_pipelinev2.py train
    thesis_env/bin/python thz_slice_pipelinev2.py predict
"""

import sys, json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RectangleSelector
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# ── reuse loading / processing from v1 ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from thz_slice_pipeline import load_all_volumes, process_to_slices

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# ── constants ─────────────────────────────────────────────────────────────────
TPRJ      = "3D_print_esther_atlanta.tprj"
ROI_JSON  = "sample_rois.json"
N_SLICES  = 50
CONTEXT   = 2       # ±2 neighbouring slices as extra input channels
N_EPOCHS  = 300
LR        = 1e-3

LABELS_DIR  = Path("labels_v2")
RESULTS_DIR = Path("results_v2")

PHYSICAL_NAMES = {
    "1":    "1",
    "2":    "2",
    "3":    "3",
    "4":    "4",
    "5":    "5",
    "5b":   "5b",
    "6-9":  "6",
    "7":    "7",
    "8":    "8",
    "9-6.": "9",
    "10":   "10",
    "11":   "11",
    "12":   "12",
    "13":   "13",
    "14":   "14",
    "15":   "15",
}

# Sample 1's tprj range metadata is doubled (±30 mm instead of ±15 mm),
# causing load_all_volumes to infer wrong spacing. Override to match sample 7.
SPACING_OVERRIDES = {
    "1": dict(dx_mm=0.2, dy_mm=0.5),
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_roi_samples(rois: dict) -> list:
    """Load tprj, process to depth slices, crop each sample to its 20×20 mm ROI."""
    print("Loading volumes from tprj…")
    raw_samples = load_all_volumes([TPRJ])

    print(f"\nProcessing {len(raw_samples)} samples to {N_SLICES} depth slices…")
    samples_data = []
    for s in raw_samples:
        tprj_name = s["name"].split("/")[-1]
        if tprj_name.lower().startswith("test"):
            continue

        phys = PHYSICAL_NAMES.get(tprj_name, tprj_name)

        if phys not in rois:
            print(f"  –  {phys}: no ROI in {ROI_JSON}, skipping")
            continue

        # Apply spacing override before processing
        if phys in SPACING_OVERRIDES:
            s = dict(s)
            s.update(SPACING_OVERRIDES[phys])

        try:
            slices, depths, _, profile = process_to_slices(s, n_slices=N_SLICES)

            # Crop to ROI
            roi = rois[phys]
            r0, r1, c0, c1 = roi["r0"], roi["r1"], roi["c0"], roi["c1"]
            slices = slices[:, r0:r1, c0:c1].copy()   # (N_SLICES, 100, 100)

            samples_data.append({
                "name":          phys,
                "slices":        slices,
                "depths_mm":     depths,
                "depth_profile": profile,
            })
            print(f"  ✓  {phys}  {slices.shape}")
        except Exception as e:
            print(f"  ✗  {phys}: {e}")

    return samples_data


# ============================================================
# INTERACTIVE PER-SLICE LABELER  (adapted from v1)
# ============================================================

class SliceLabeler:
    """
    Label voids in individual depth slices of ROI-cropped samples.

    Controls:
      Left-drag   — ADD void rectangle
      Right-drag  — REMOVE rectangle
      c           — clear current slice mask
      s           — save current sample masks
      n / p       — next / previous sample
      q           — quit (auto-saves all)
    """

    def __init__(self, samples_data, output_dir=LABELS_DIR):
        self.samples    = samples_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.current_sample = 0
        self.masks = {}

        for sd in samples_data:
            name = sd["name"]
            mask_path = self.output_dir / f"{name}_slice_masks.npy"
            n_s, Ny, Nx = sd["slices"].shape
            if mask_path.exists():
                loaded = np.load(mask_path)
                if loaded.shape == (n_s, Ny, Nx):
                    self.masks[name] = loaded
                    n_lbl = (loaded.sum(axis=(1, 2)) > 0).sum()
                    print(f"  Loaded: {name}  ({n_lbl} slices labeled)")
                else:
                    print(f"  Shape mismatch for {name} — reinitializing")
                    self.masks[name] = np.zeros((n_s, Ny, Nx), dtype=np.uint8)
            else:
                self.masks[name] = np.zeros((n_s, Ny, Nx), dtype=np.uint8)

    def run(self):
        self.fig = plt.figure(figsize=(18, 7))
        gs = self.fig.add_gridspec(2, 5, height_ratios=[1, 0.06],
                                   hspace=0.3, wspace=0.3)

        self.ax_slice   = self.fig.add_subplot(gs[0, 0:2])
        self.ax_mask    = self.fig.add_subplot(gs[0, 2])
        self.ax_overlay = self.fig.add_subplot(gs[0, 3])
        self.ax_summary = self.fig.add_subplot(gs[0, 4])

        ax_slider = self.fig.add_subplot(gs[1, 0:3])
        n_slices  = self.samples[0]["slices"].shape[0]
        self.slider = Slider(ax_slider, "Slice", 0, n_slices - 1,
                             valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)

        ax_clear = self.fig.add_axes([0.65, 0.02, 0.07, 0.04])
        ax_save  = self.fig.add_axes([0.73, 0.02, 0.07, 0.04])
        ax_next  = self.fig.add_axes([0.81, 0.02, 0.07, 0.04])
        ax_prev  = self.fig.add_axes([0.89, 0.02, 0.07, 0.04])

        self.btn_clear = Button(ax_clear, "Clear(c)")
        self.btn_save  = Button(ax_save,  "Save(s)")
        self.btn_next  = Button(ax_next,  "Next(n)")
        self.btn_prev  = Button(ax_prev,  "Prev(p)")

        self.btn_clear.on_clicked(lambda e: self._clear_slice())
        self.btn_save.on_clicked( lambda e: self._save_current())
        self.btn_next.on_clicked( lambda e: self._change_sample(1))
        self.btn_prev.on_clicked( lambda e: self._change_sample(-1))

        self.rect_add = RectangleSelector(
            self.ax_slice, self._on_add, useblit=True, button=[1],
            props=dict(facecolor="lime", alpha=0.3), interactive=False)
        self.rect_remove = RectangleSelector(
            self.ax_slice, self._on_remove, useblit=True, button=[3],
            props=dict(facecolor="red", alpha=0.3), interactive=False)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        self._display()
        plt.show()
        self._save_all()

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_current(self):
        sd       = self.samples[self.current_sample]
        si       = int(self.slider.val)
        mask_vol = self.masks[sd["name"]]
        return sd, si, mask_vol

    def _display(self):
        sd, si, mask_vol = self._get_current()
        slices  = sd["slices"]
        depths  = sd["depths_mm"]
        n_slices = slices.shape[0]

        for ax in (self.ax_slice, self.ax_mask, self.ax_overlay, self.ax_summary):
            ax.clear()

        self.ax_slice.imshow(slices[si], cmap="Reds", vmin=0, vmax=1)
        self.ax_slice.set_title(
            f"Slice {si}/{n_slices-1}  depth={depths[si]:.2f} mm\n"
            f"Left-drag=ADD   Right-drag=REMOVE", fontsize=10)
        self.ax_slice.axis("off")

        self.ax_mask.imshow(mask_vol[si], cmap="gray", vmin=0, vmax=1)
        self.ax_mask.set_title(f"Mask ({mask_vol[si].sum()} px)", fontsize=10)
        self.ax_mask.axis("off")

        overlay = np.zeros((*slices[si].shape, 3))
        overlay[..., 0] = slices[si]
        overlay[..., 1] = slices[si] * 0.3 + mask_vol[si] * 0.7
        overlay[..., 2] = slices[si] * 0.3
        self.ax_overlay.imshow(np.clip(overlay, 0, 1))
        self.ax_overlay.set_title("Overlay", fontsize=10)
        self.ax_overlay.axis("off")

        # Depth profile panel
        from scipy.signal import find_peaks as _fp
        labeled_per_slice = mask_vol.sum(axis=(1, 2))
        profile = sd.get("depth_profile", np.zeros(n_slices))
        p_min, p_max = profile.min(), profile.max()
        p_norm = (profile - p_min) / (p_max - p_min + 1e-8)
        peaks, _ = _fp(p_norm, prominence=0.10, distance=3)

        ax = self.ax_summary
        ax.fill_betweenx(range(n_slices), 0, p_norm, alpha=0.35, color="steelblue")
        ax.plot(p_norm, range(n_slices), color="steelblue", lw=1)
        for pk in peaks:
            ax.plot(p_norm[pk], pk, "o", color="orange", ms=6, zorder=5)
            ax.text(p_norm[pk] + 0.03, pk, f"{depths[pk]:.2f}mm",
                    va="center", fontsize=6, color="orange")
        for i, lps in enumerate(labeled_per_slice):
            if lps > 0:
                ax.axhspan(i - 0.4, i + 0.4, xmin=0.92, xmax=1.0,
                           color="limegreen", alpha=0.7)
        ax.axhline(si, color="red", lw=1.5, alpha=0.8, linestyle="--")
        ax.text(0.01, si - 0.8, f"← {depths[si]:.2f}mm",
                fontsize=6, color="red", va="top")
        tick_step = max(1, n_slices // 20)
        ax.set_yticks(range(0, n_slices, tick_step))
        ax.set_yticklabels([f"{depths[i]:.1f}" for i in range(0, n_slices, tick_step)],
                           fontsize=6)
        ax.set_xlim(0, 1.35)
        ax.set_xlabel("Norm. amplitude", fontsize=7)
        ax.set_ylabel("Depth (mm)", fontsize=7)
        ax.set_title(f"Depth profile  ({len(peaks)} peaks)", fontsize=9)
        ax.invert_yaxis()

        n_labeled = (labeled_per_slice > 0).sum()
        self.fig.suptitle(
            f"Sample {self.current_sample+1}/{len(self.samples)}: {sd['name']}  |  "
            f"{n_labeled}/{n_slices} slices labeled  |  "
            f"Keys: c=clear  s=save  n/p=next/prev  q=quit",
            fontsize=11, fontweight="bold")
        self.fig.canvas.draw_idle()

    # ── event callbacks ───────────────────────────────────────────────────────

    def _on_slider(self, val):
        self._display()

    def _on_add(self, eclick, erelease):
        sd, si, mask_vol = self._get_current()
        x0, x1 = sorted([int(eclick.xdata), int(erelease.xdata)])
        y0, y1 = sorted([int(eclick.ydata), int(erelease.ydata)])
        Ny, Nx = mask_vol.shape[1], mask_vol.shape[2]
        mask_vol[si, max(0,y0):min(Ny,y1+1), max(0,x0):min(Nx,x1+1)] = 1
        self._display()

    def _on_remove(self, eclick, erelease):
        sd, si, mask_vol = self._get_current()
        x0, x1 = sorted([int(eclick.xdata), int(erelease.xdata)])
        y0, y1 = sorted([int(eclick.ydata), int(erelease.ydata)])
        Ny, Nx = mask_vol.shape[1], mask_vol.shape[2]
        mask_vol[si, max(0,y0):min(Ny,y1+1), max(0,x0):min(Nx,x1+1)] = 0
        self._display()

    def _clear_slice(self):
        sd, si, mask_vol = self._get_current()
        mask_vol[si] = 0
        self._display()

    def _on_key(self, event):
        if event.key == "c":
            self._clear_slice()
        elif event.key == "s":
            self._save_current()
        elif event.key == "n":
            self._change_sample(1)
        elif event.key == "p":
            self._change_sample(-1)
        elif event.key == "q":
            self._save_all()
            plt.close(self.fig)

    def _change_sample(self, delta):
        self._save_current()
        self.current_sample = (self.current_sample + delta) % len(self.samples)
        n_slices = self.samples[self.current_sample]["slices"].shape[0]
        self.slider.valmax = n_slices - 1
        self.slider.set_val(0)
        self._display()

    def _save_current(self):
        sd = self.samples[self.current_sample]
        name = sd["name"]
        np.save(self.output_dir / f"{name}_slice_masks.npy", self.masks[name])
        n_lbl = (self.masks[name].sum(axis=(1, 2)) > 0).sum()
        print(f"  Saved {name}: {n_lbl} slices with labels")

    def _save_all(self):
        for sd in self.samples:
            np.save(self.output_dir / f"{sd['name']}_slice_masks.npy",
                    self.masks[sd["name"]])
        print(f"\nAll masks saved to {self.output_dir}/")


# ============================================================
# U-NET
# ============================================================

if HAS_TORCH:
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        """(B, C, H, W) → (B, 1, H, W) — works at any spatial size ≥8."""
        def __init__(self, in_channels=5, base_filters=32):
            super().__init__()
            f = base_filters
            self.enc1      = DoubleConv(in_channels, f)
            self.enc2      = DoubleConv(f,   f*2)
            self.enc3      = DoubleConv(f*2, f*4)
            self.pool      = nn.MaxPool2d(2)
            self.bottleneck= DoubleConv(f*4, f*8)
            self.up3       = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
            self.dec3      = DoubleConv(f*8, f*4)
            self.up2       = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
            self.dec2      = DoubleConv(f*4, f*2)
            self.up1       = nn.ConvTranspose2d(f*2, f,   2, stride=2)
            self.dec1      = DoubleConv(f*2, f)
            self.out_conv  = nn.Conv2d(f, 1, 1)

        def forward(self, x):
            _, _, H, W = x.shape
            ph = (8 - H % 8) % 8
            pw = (8 - W % 8) % 8
            if ph or pw:
                x = F.pad(x, [0, pw, 0, ph], mode="reflect")

            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b  = self.bottleneck(self.pool(e3))

            d3 = self._cat(self.up3(b),  e3); d3 = self.dec3(d3)
            d2 = self._cat(self.up2(d3), e2); d2 = self.dec2(d2)
            d1 = self._cat(self.up1(d2), e1); d1 = self.dec1(d1)
            return self.out_conv(d1)[:, :, :H, :W]

        def _cat(self, up, skip):
            dh = skip.shape[2] - up.shape[2]
            dw = skip.shape[3] - up.shape[3]
            up = F.pad(up, [0, dw, 0, dh])
            return torch.cat([up, skip], dim=1)

    class SliceDataset(Dataset):
        """
        Each item: (C, 100, 100) input tensor + (1, 100, 100) mask.
        C = 2*context+1 neighbouring depth slices.
        No resize — all ROI crops are already 100×100.
        """
        def __init__(self, all_slices, all_masks, context_slices=2,
                     augment=False, include_unlabeled=False):
            self.items   = []
            self.context = context_slices
            self.augment = augment

            for slices, masks in zip(all_slices, all_masks):
                n_s = slices.shape[0]
                for i in range(n_s):
                    has_label = masks[i].sum() > 0
                    if not has_label and not include_unlabeled:
                        continue
                    channels = []
                    for offset in range(-context_slices, context_slices + 1):
                        j = int(np.clip(i + offset, 0, n_s - 1))
                        channels.append(slices[j])
                    inp = np.stack(channels, axis=0).astype(np.float32)
                    self.items.append((inp, masks[i].astype(np.float32)))

            print(f"  Dataset: {len(self.items)} slice-mask pairs "
                  f"({context_slices*2+1} channels each, 100×100 px)")

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            x, y = self.items[idx]
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(y).unsqueeze(0)

            if self.augment:
                if torch.rand(1) > 0.5:
                    x, y = torch.flip(x, [-1]), torch.flip(y, [-1])
                if torch.rand(1) > 0.5:
                    x, y = torch.flip(x, [-2]), torch.flip(y, [-2])
                if torch.rand(1) > 0.5:
                    x, y = torch.rot90(x, 2, [-2,-1]), torch.rot90(y, 2, [-2,-1])
                x = torch.clamp(x + 0.02 * torch.randn_like(x), 0, 1)

            return x, y

    class DiceBCELoss(nn.Module):
        def __init__(self, dice_weight=0.5):
            super().__init__()
            self.dice_weight = dice_weight
            self.bce = nn.BCEWithLogitsLoss()

        def forward(self, logits, targets):
            bce  = self.bce(logits, targets)
            p    = torch.sigmoid(logits)
            inter = (p * targets).sum(dim=(2, 3))
            union = p.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
            dice  = (2 * inter + 1) / (union + 1)
            return (1 - self.dice_weight) * bce + self.dice_weight * (1 - dice.mean())


# ============================================================
# TRAINING
# ============================================================

def train(model, dataset, n_epochs=N_EPOCHS, lr=LR, device="cpu"):
    loader    = DataLoader(dataset, batch_size=min(len(dataset), 8),
                           shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = DiceBCELoss(0.5)
    model.to(device)

    history = {"loss": [], "dice": [], "iou": []}
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining: {len(dataset)} pairs, {n_epochs} epochs, "
          f"{n_params:,} params, device={device}")
    print("─" * 60)

    for epoch in range(n_epochs):
        model.train()
        ep_loss = ep_dice = ep_iou = nb = 0

        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                inter = (preds * y).sum(dim=(2, 3))
                denom = preds.sum(dim=(2, 3)) + y.sum(dim=(2, 3))
                dice  = ((2*inter+1) / (denom+1)).mean()
                iou   = ((inter+1)   / (denom-inter+1)).mean()

            ep_loss += loss.item(); ep_dice += dice.item()
            ep_iou  += iou.item();  nb += 1

        scheduler.step()
        history["loss"].append(ep_loss/nb)
        history["dice"].append(ep_dice/nb)
        history["iou"].append(ep_iou/nb)

        if (epoch+1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d} | Loss {ep_loss/nb:.4f} | "
                  f"Dice {ep_dice/nb:.4f} | IoU {ep_iou/nb:.4f}")

    return history


# ============================================================
# VISUALISATION
# ============================================================

def visualize_predictions(model, samples_data, device="cpu"):
    model.eval()
    RESULTS_DIR.mkdir(exist_ok=True)

    for sd in samples_data:
        slices = sd["slices"]
        masks  = sd.get("masks", np.zeros_like(slices))
        depths = sd["depths_mm"]
        n_s, Ny, Nx = slices.shape
        name = sd["name"]

        pred_masks = np.zeros((n_s, Ny, Nx))
        with torch.no_grad():
            for i in range(n_s):
                channels = [slices[int(np.clip(i+o, 0, n_s-1))]
                            for o in range(-CONTEXT, CONTEXT+1)]
                inp = torch.FloatTensor(np.stack(channels)).unsqueeze(0).to(device)
                pred_masks[i] = torch.sigmoid(model(inp)).cpu().squeeze().numpy()

        pred_binary = (pred_masks > 0.5).astype(float)
        n_show  = min(n_s, 10)
        indices = np.linspace(0, n_s-1, n_show, dtype=int)

        fig, axes = plt.subplots(3, n_show, figsize=(2.5*n_show, 7), dpi=100)
        for j, si in enumerate(indices):
            axes[0, j].imshow(slices[si], cmap="Reds", vmin=0, vmax=1)
            axes[0, j].set_title(f"d={depths[si]:.1f}mm", fontsize=8)
            axes[0, j].axis("off")
            axes[1, j].imshow(masks[si], cmap="gray", vmin=0, vmax=1)
            axes[1, j].axis("off")
            overlay = np.zeros((Ny, Nx, 3))
            overlay[..., 0] = slices[si]
            overlay[..., 1] = pred_binary[si] * 0.7
            axes[2, j].imshow(np.clip(overlay, 0, 1))
            axes[2, j].axis("off")

        axes[0, 0].set_ylabel("Input",      fontsize=10)
        axes[1, 0].set_ylabel("GT Mask",    fontsize=10)
        axes[2, 0].set_ylabel("Prediction", fontsize=10)
        plt.suptitle(f"Sample {name} — Per-Slice Predictions",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"{name}_predictions.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Results saved to {RESULTS_DIR}/")


def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    axes[0].plot(history["loss"]); axes[0].set_title("Loss"); axes[0].set_yscale("log")
    axes[1].plot(history["dice"]); axes[1].set_title("Dice"); axes[1].set_ylim(0, 1)
    axes[2].plot(history["iou"]);  axes[2].set_title("IoU");  axes[2].set_ylim(0, 1)
    for ax in axes:
        ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
    plt.suptitle("Training Curves", fontsize=14)
    plt.tight_layout()
    RESULTS_DIR.mkdir(exist_ok=True)
    plt.savefig(RESULTS_DIR / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python thz_slice_pipelinev2.py label")
        print("  python thz_slice_pipelinev2.py train")
        print("  python thz_slice_pipelinev2.py predict")
        sys.exit(1)

    mode = sys.argv[1]

    DEVICE = "cpu"
    if HAS_TORCH:
        DEVICE = ("cuda"  if torch.cuda.is_available() else
                  "mps"   if hasattr(torch.backends, "mps") and
                             torch.backends.mps.is_available() else "cpu")

    # Load ROIs
    with open(ROI_JSON) as fh:
        rois = json.load(fh)

    # Load and crop samples
    samples_data = load_roi_samples(rois)
    if not samples_data:
        print("No samples loaded. Check tprj path and sample_rois.json.")
        sys.exit(1)

    # ── LABEL ──────────────────────────────────────────────────────────────────
    if mode == "label":
        matplotlib.use("TkAgg")
        print(f"\nLaunching labeler for {len(samples_data)} samples…")
        labeler = SliceLabeler(samples_data)
        labeler.run()

    # ── TRAIN ──────────────────────────────────────────────────────────────────
    elif mode == "train":
        assert HAS_TORCH, "PyTorch required for training"
        matplotlib.use("Agg")

        all_slices, all_masks = [], []
        for sd in samples_data:
            mask_path = LABELS_DIR / f"{sd['name']}_slice_masks.npy"
            if not mask_path.exists():
                print(f"  {sd['name']}: no mask file, skipping")
                continue
            masks  = np.load(mask_path)
            n_lbl  = (masks.sum(axis=(1, 2)) > 0).sum()
            if n_lbl == 0:
                print(f"  {sd['name']}: no labeled slices, skipping")
                continue
            sd["masks"] = masks
            all_slices.append(sd["slices"])
            all_masks.append(masks)
            print(f"  {sd['name']}: {n_lbl}/{N_SLICES} slices labeled")

        if not all_slices:
            print("No labeled data found. Run 'label' mode first.")
            sys.exit(1)

        total_lbl   = sum((m.sum(axis=(1,2)) > 0).sum() for m in all_masks)
        total_empty = sum((m.sum(axis=(1,2)) == 0).sum() for m in all_masks)
        print(f"\nTotal: {total_lbl} void slices, {total_empty} empty slices")

        dataset = SliceDataset(all_slices, all_masks,
                               context_slices=CONTEXT, augment=True,
                               include_unlabeled=True)

        in_ch = CONTEXT * 2 + 1
        model = UNet(in_channels=in_ch, base_filters=32)
        history = train(model, dataset, n_epochs=N_EPOCHS, lr=LR, device=DEVICE)
        plot_history(history)

        RESULTS_DIR.mkdir(exist_ok=True)
        torch.save({
            "model_state": model.state_dict(),
            "config": {"in_channels": in_ch, "base_filters": 32,
                       "context": CONTEXT, "n_slices": N_SLICES},
            "history": history,
        }, RESULTS_DIR / "unet_v2.pt")
        print(f"Model saved: {RESULTS_DIR}/unet_v2.pt")

        visualize_predictions(model,
                              [sd for sd in samples_data if "masks" in sd],
                              device=DEVICE)
        print(f"\n{'='*60}")
        print(f"Final — Loss {history['loss'][-1]:.4f} | "
              f"Dice {history['dice'][-1]:.4f} | IoU {history['iou'][-1]:.4f}")

    # ── PREDICT ────────────────────────────────────────────────────────────────
    elif mode == "predict":
        assert HAS_TORCH, "PyTorch required"
        matplotlib.use("Agg")

        ckpt  = torch.load(RESULTS_DIR / "unet_v2.pt",
                           map_location=DEVICE, weights_only=False)
        cfg   = ckpt["config"]
        model = UNet(in_channels=cfg["in_channels"],
                     base_filters=cfg["base_filters"])
        model.load_state_dict(ckpt["model_state"])
        model.to(DEVICE)
        visualize_predictions(model, samples_data, device=DEVICE)

    else:
        print(f"Unknown mode: {mode}. Use: label, train, predict")
