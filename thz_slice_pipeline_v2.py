"""
THz Void Detection v2 — Domain Transfer Learning
==================================================
Three experiments comparing:
  1. Baseline: THz-only training
  2. Domain Transfer: train on Mendeley CFRP thermal defect data, test on THz
  3. Pretrain+Finetune: Mendeley pretrain → THz finetune

Run:  python thz_slice_pipeline_v2.py
"""

import h5py
import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from pathlib import Path
import warnings
import os
import zipfile
import io
import requests
from PIL import Image
from collections import Counter

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ============================================================
# CONFIG
# ============================================================
IMG_SIZE = 64
N_SLICES = 20
CONTEXT = 2  # ±2 neighboring slices → 5 channels
DEVICE = ('cuda' if torch.cuda.is_available() else
          'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

RESULTS_DIR = Path('results_v2')
MENDELEY_DIR = Path('mendeley_cfrp')
TPRJ_FILES = ['3D_print_esther_atlanta']
LABELS_DIR = Path('labels')


# ============================================================
# DATA LOADING & PROCESSING (reused from v1)
# ============================================================

def load_all_volumes(tprj_paths):
    """Load all samples from .tprj files."""
    all_samples = []
    for tprj_path in tprj_paths:
        with h5py.File(tprj_path, 'r') as f:
            data_paths = []
            def find_data(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith('raw data/sample/data'):
                    data_paths.append(name)
            f.visititems(find_data)

            for dpath in data_paths:
                parts = dpath.split('/')
                dv_idx = parts.index('DataViews')
                sample_name = parts[dv_idx + 1]
                parent = dpath.replace('/data', '')
                line_path = parent + '/line'

                try:
                    raw = f[dpath][:]
                    line_data = np.array(f[line_path][:]).flatten()
                    attrs = dict(f[parent].attrs)
                    N_time, N_flat = raw.shape
                    Ny = len(line_data)
                    Nx = N_flat // Ny
                    if Ny * Nx > N_flat:
                        continue

                    volume = raw[:, :Ny * Nx].reshape(N_time, Ny, Nx)
                    x_offset = float(np.array(attrs.get('X_Offset', [[0]])).flatten()[0])
                    x_spacing = float(np.array(attrs.get('X_Spacing', [[0.02]])).flatten()[0])
                    time_axis = x_offset + np.arange(N_time) * x_spacing

                    all_samples.append({
                        'name': f"{Path(tprj_path).stem}/{sample_name}",
                        'volume': volume, 'time_axis': time_axis,
                        'Ny': Ny, 'Nx': Nx,
                    })
                    print(f"  ✓ {sample_name}: ({N_time}, {Ny}, {Nx})")
                except Exception as e:
                    print(f"  ✗ {sample_name}: {e}")
    return all_samples


def process_to_slices(sample, n_slices=20, n_pla=1.57, c=0.29979):
    """Process sample → normalized depth slices."""
    volume = sample['volume']
    N_time, Ny, Nx = volume.shape
    time_axis = sample['time_axis']

    envelope = np.abs(hilbert(volume, axis=0))

    mean_ascan = np.mean(envelope, axis=(1, 2))
    surface_idx = np.argmax(mean_ascan)

    margin = int(0.2 * N_time)
    s0, s1 = max(0, surface_idx - margin), min(N_time, surface_idx + margin)
    local_surface = s0 + np.argmax(envelope[s0:s1], axis=0)
    target_idx = int(np.median(local_surface))

    flat_env = np.zeros_like(envelope)
    for iy in range(Ny):
        for ix in range(Nx):
            shift = target_idx - local_surface[iy, ix]
            flat_env[:, iy, ix] = np.roll(envelope[:, iy, ix], shift)

    margin_bot = int(0.05 * N_time)
    slice_indices = np.linspace(target_idx, N_time - margin_bot, n_slices, dtype=int)
    slices = flat_env[slice_indices].copy()

    surface_time = time_axis[target_idx]
    slice_depths_mm = ((time_axis[slice_indices] - surface_time) * c) / (2 * n_pla)

    for i in range(n_slices):
        vmax = np.percentile(slices[i], 99) + 1e-8
        slices[i] = np.clip(slices[i] / vmax, 0, 1).astype(np.float32)

    return slices, slice_depths_mm, target_idx


# ============================================================
# U-NET (reused from v1)
# ============================================================

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_filters=32):
        super().__init__()
        f = base_filters
        self.enc1 = DoubleConv(in_channels, f)
        self.enc2 = DoubleConv(f, f * 2)
        self.enc3 = DoubleConv(f * 2, f * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(f * 4, f * 8)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, 2, stride=2)
        self.dec3 = DoubleConv(f * 8, f * 4)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, 2, stride=2)
        self.dec2 = DoubleConv(f * 4, f * 2)
        self.up1 = nn.ConvTranspose2d(f * 2, f, 2, stride=2)
        self.dec1 = DoubleConv(f * 2, f)
        self.out_conv = nn.Conv2d(f, 1, 1)

    def forward(self, x):
        _, _, H, W = x.shape
        pad_h = (8 - H % 8) % 8 #porque la imagen del sample no tiene un tamaño par
        pad_w = (8 - W % 8) % 8
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self._cat(d3, e3); d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = self._cat(d2, e2); d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = self._cat(d1, e1); d1 = self.dec1(d1)

        return self.out_conv(d1)[:, :, :H, :W]

    def _cat(self, up, skip):
        dh = skip.shape[2] - up.shape[2]
        dw = skip.shape[3] - up.shape[3]
        up = F.pad(up, [0, dw, 0, dh])
        return torch.cat([up, skip], dim=1)


class DiceBCELoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        inter = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * inter + 1) / (union + 1)
        return (1 - self.dice_weight) * bce + self.dice_weight * (1 - dice.mean())


# ============================================================
# MENDELEY CFRP DATASET
# ============================================================

def download_mendeley_dataset():
    """Download Mendeley CFRP thermal defect dataset if not already present."""
    img_dir = MENDELEY_DIR / 'images'
    mask_dir = MENDELEY_DIR / 'masks'

    if img_dir.exists() and len(list(img_dir.glob('*.png'))) > 100:
        print(f"  Mendeley dataset already downloaded ({MENDELEY_DIR})")
        return

    MENDELEY_DIR.mkdir(exist_ok=True)
    img_dir.mkdir(exist_ok=True)
    mask_dir.mkdir(exist_ok=True)

    print("  Downloading Mendeley CFRP dataset...")

    # Get file download URLs from the public API
    api_url = "https://data.mendeley.com/public-api/datasets/jrsb4b9yy5"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    dataset_info = resp.json()

    # Build map: filename → download_url
    file_map = {}
    for f in dataset_info.get('files', []):
        fname = f.get('filename', '')
        dl_url = f.get('content_details', {}).get('download_url', '')
        if fname and dl_url:
            file_map[fname] = dl_url

    # Download originalData.zip → images/
    if 'originalData.zip' in file_map:
        print("  Downloading originalData.zip (images)...")
        resp = requests.get(file_map['originalData.zip'], timeout=300)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            n = 0
            for member in zf.namelist():
                if member.startswith('__MACOSX') or Path(member).name.startswith('._'):
                    continue
                if member.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    data = zf.read(member)
                    base = Path(member).name
                    with open(img_dir / base, 'wb') as f:
                        f.write(data)
                    n += 1
            print(f"    Extracted {n} images")

    # Download annotatedData.zip → masks/
    if 'annotatedData.zip' in file_map:
        print("  Downloading annotatedData.zip (masks)...")
        resp = requests.get(file_map['annotatedData.zip'], timeout=120)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            n = 0
            for member in zf.namelist():
                if member.startswith('__MACOSX') or Path(member).name.startswith('._'):
                    continue
                if member.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                    data = zf.read(member)
                    base = Path(member).name
                    with open(mask_dir / base, 'wb') as f:
                        f.write(data)
                    n += 1
            print(f"    Extracted {n} masks")

    n_img = len(list(img_dir.glob('*')))
    n_mask = len(list(mask_dir.glob('*')))
    print(f"  Final count: {n_img} images, {n_mask} masks")


def augment_pair(x, y):
    """Shared augmentation for image-mask pairs.
    x: (C, H, W), y: (1, H, W). Returns augmented copies.
    """
    # Random horizontal flip
    if torch.rand(1) > 0.5:
        x = torch.flip(x, [-1])
        y = torch.flip(y, [-1])
    # Random vertical flip
    if torch.rand(1) > 0.5:
        x = torch.flip(x, [-2])
        y = torch.flip(y, [-2])
    # Random 90° rotations (0, 90, 180, 270)
    k = torch.randint(0, 4, (1,)).item()
    if k > 0:
        x = torch.rot90(x, k, [-2, -1])
        y = torch.rot90(y, k, [-2, -1])
    # Brightness jitter ±15%
    brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.3
    x = x * brightness
    # Contrast jitter ±15%
    mean_val = x.mean()
    contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.3
    x = (x - mean_val) * contrast + mean_val
    # Gaussian noise
    x = x + 0.03 * torch.randn_like(x)
    x = torch.clamp(x, 0, 1)
    return x, y


class MendeleyDataset(Dataset):
    """Mendeley CFRP thermal defect dataset — 1 channel input, resized to IMG_SIZE."""

    def __init__(self, image_paths, mask_paths, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('L')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        x = np.array(img, dtype=np.float32) / 255.0
        x = torch.FloatTensor(x).unsqueeze(0)

        mask = Image.open(self.mask_paths[idx]).convert('L')
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)
        y = np.array(mask, dtype=np.float32)
        y = (y > 127).astype(np.float32)
        y = torch.FloatTensor(y).unsqueeze(0)

        if self.augment:
            x, y = augment_pair(x, y)

        return x, y


def load_mendeley_splits():
    """Load Mendeley dataset and return train/test splits."""
    img_dir = MENDELEY_DIR / 'images'
    mask_dir = MENDELEY_DIR / 'masks'

    img_files = sorted([f for f in img_dir.iterdir()
                        if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                        and not f.name.startswith('._')])
    mask_files = sorted([f for f in mask_dir.iterdir()
                         if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                         and not f.name.startswith('._')])

    # Match images to masks by filename stem
    img_stems = {f.stem: f for f in img_files}
    mask_stems = {f.stem: f for f in mask_files}

    # Try direct matching first
    paired_imgs, paired_masks = [], []
    for stem, img_path in img_stems.items():
        if stem in mask_stems:
            paired_imgs.append(img_path)
            paired_masks.append(mask_stems[stem])
            continue
        # Try common mask naming patterns
        for suffix in ['_mask', '_label', '_gt', '_GT', '_Mask', '_Label']:
            mask_stem = stem + suffix
            if mask_stem in mask_stems:
                paired_imgs.append(img_path)
                paired_masks.append(mask_stems[mask_stem])
                break
        else:
            # Try removing suffix from mask stems to match
            for mstem, mpath in mask_stems.items():
                for suffix in ['_mask', '_label', '_gt', '_GT', '_Mask', '_Label']:
                    if mstem.endswith(suffix) and mstem[:-len(suffix)] == stem:
                        paired_imgs.append(img_path)
                        paired_masks.append(mpath)
                        break

    # Deduplicate
    seen = set()
    unique_imgs, unique_masks = [], []
    for img, mask in zip(paired_imgs, paired_masks):
        key = (str(img), str(mask))
        if key not in seen:
            seen.add(key)
            unique_imgs.append(img)
            unique_masks.append(mask)
    paired_imgs, paired_masks = unique_imgs, unique_masks

    n = len(paired_imgs)
    print(f"  Mendeley: {n} matched image-mask pairs")

    if n == 0:
        # Fallback: if counts match, pair by sorted order
        if len(img_files) == len(mask_files) and len(img_files) > 0:
            paired_imgs = img_files
            paired_masks = mask_files
            n = len(paired_imgs)
            print(f"  Fallback: paired {n} by sorted order")
        else:
            raise RuntimeError(
                f"Could not match images ({len(img_files)}) to masks ({len(mask_files)}). "
                f"Check {MENDELEY_DIR} structure.")

    # 85/15 split
    np.random.seed(42)
    indices = np.random.permutation(n)
    split = int(0.85 * n)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_imgs = [paired_imgs[i] for i in train_idx]
    train_masks = [paired_masks[i] for i in train_idx]
    test_imgs = [paired_imgs[i] for i in test_idx]
    test_masks = [paired_masks[i] for i in test_idx]

    print(f"  Split: {len(train_imgs)} train, {len(test_imgs)} test")
    return train_imgs, train_masks, test_imgs, test_masks


# ============================================================
# THz DATASET (64x64, 5-channel context)
# ============================================================

class THzSliceDataset64(Dataset):
    """THz depth slice dataset resized to 64x64, with ±context neighboring slices."""

    def __init__(self, all_slices, all_masks, context_slices=2, augment=False):
        self.items = []
        self.context = context_slices

        for slices, masks in zip(all_slices, all_masks):
            n_s, Ny, Nx = slices.shape
            for i in range(n_s):
                has_label = masks[i].sum() > 0
                if not has_label:
                    continue

                channels = []
                for offset in range(-context_slices, context_slices + 1):
                    j = np.clip(i + offset, 0, n_s - 1)
                    # Resize to IMG_SIZE x IMG_SIZE
                    ch = Image.fromarray(slices[j]).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                    channels.append(np.array(ch, dtype=np.float32))

                inp = np.stack(channels, axis=0)  # (2*context+1, IMG_SIZE, IMG_SIZE)

                # Resize mask
                m = Image.fromarray(masks[i].astype(np.float32)).resize(
                    (IMG_SIZE, IMG_SIZE), Image.NEAREST)
                mask = np.array(m, dtype=np.float32)
                mask = (mask > 0.5).astype(np.float32)

                self.items.append((inp, mask))

        print(f"  THzSliceDataset64: {len(self.items)} labeled pairs "
              f"({context_slices * 2 + 1} ch, {IMG_SIZE}x{IMG_SIZE})")
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        x, y = self.items[idx]
        x = torch.FloatTensor(x)
        y = torch.FloatTensor(y).unsqueeze(0)

        if self.augment:
            x, y = augment_pair(x, y)

        return x, y


# ============================================================
# CHANNEL ADAPTATION
# ============================================================

def adapt_first_conv(pretrained_state_dict, new_in_channels=5, center_channel=2):
    """
    Adapt 1-channel pretrained weights to multi-channel input.
    Copy pretrained weights into center channel, zero-initialize others.
    """
    adapted = {}
    for key, val in pretrained_state_dict.items():
        if key == 'enc1.conv.0.weight':
            # Original shape: (out_ch, 1, kH, kW)
            out_ch, _, kH, kW = val.shape
            new_weight = torch.zeros(out_ch, new_in_channels, kH, kW)
            new_weight[:, center_channel:center_channel + 1, :, :] = val
            adapted[key] = new_weight
        else:
            adapted[key] = val
    return adapted


def freeze_encoder(model):
    """Freeze encoder + bottleneck weights (enc1, enc2, enc3, bottleneck, pool)."""
    frozen = 0
    for name, param in model.named_parameters():
        if any(name.startswith(p) for p in ['enc1', 'enc2', 'enc3', 'bottleneck', 'pool']):
            param.requires_grad = False
            frozen += param.numel()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"    Frozen: {frozen:,} params | Trainable: {trainable:,} / {total:,}")


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_model(model, train_loader, n_epochs, lr, device, label=""):
    """Train model, return history dict."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = DiceBCELoss(0.5)
    model.to(device)

    history = {'loss': [], 'dice': [], 'iou': []}

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Training [{label}]: {n_epochs} epochs, {n_params:,} params, device={device}")

    for epoch in range(n_epochs):
        model.train()
        ep_loss, ep_dice, ep_iou, nb = 0, 0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                inter = (preds * y).sum(dim=(2, 3))
                dice = ((2 * inter + 1) / (preds.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) + 1)).mean()
                iou = ((inter + 1) / (preds.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) - inter + 1)).mean()

            ep_loss += loss.item()
            ep_dice += dice.item()
            ep_iou += iou.item()
            nb += 1

        scheduler.step()
        history['loss'].append(ep_loss / nb)
        history['dice'].append(ep_dice / nb)
        history['iou'].append(ep_iou / nb)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:4d} | Loss: {ep_loss / nb:.4f} | "
                  f"Dice: {ep_dice / nb:.4f} | IoU: {ep_iou / nb:.4f}")

    return history


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set. Returns dict with dice, iou, predictions."""
    model.eval()
    model.to(device)

    all_preds = []
    all_targets = []
    total_dice, total_iou, nb = 0, 0, 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = (torch.sigmoid(logits) > 0.5).float()

            inter = (preds * y).sum(dim=(2, 3))
            dice = ((2 * inter + 1) / (preds.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) + 1)).mean()
            iou = ((inter + 1) / (preds.sum(dim=(2, 3)) + y.sum(dim=(2, 3)) - inter + 1)).mean()

            total_dice += dice.item()
            total_iou += iou.item()
            nb += 1

            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    return {
        'dice': total_dice / nb,
        'iou': total_iou / nb,
        'predictions': torch.cat(all_preds, dim=0),
        'targets': torch.cat(all_targets, dim=0),
    }


# ============================================================
# VISUALIZATION
# ============================================================

def plot_comparison(results, histories, save_path):
    """Bar charts for Dice/IoU + training curves."""
    exp_names = list(results.keys())
    dices = [results[n]['dice'] for n in exp_names]
    ious = [results[n]['iou'] for n in exp_names]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Bar chart — Dice
    colors = ['#2196F3', '#FF9800', '#4CAF50']
    axes[0, 0].bar(exp_names, dices, color=colors[:len(exp_names)])
    axes[0, 0].set_ylabel('Dice Score')
    axes[0, 0].set_title('Test Dice Score')
    axes[0, 0].set_ylim(0, 1)
    for i, v in enumerate(dices):
        axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # Bar chart — IoU
    axes[0, 1].bar(exp_names, ious, color=colors[:len(exp_names)])
    axes[0, 1].set_ylabel('IoU Score')
    axes[0, 1].set_title('Test IoU Score')
    axes[0, 1].set_ylim(0, 1)
    for i, v in enumerate(ious):
        axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontweight='bold')

    # Training curves — Loss
    for name in exp_names:
        if name in histories:
            axes[1, 0].plot(histories[name]['loss'], label=name)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Training Loss')
    axes[1, 0].set_yscale('log')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Training curves — Dice
    for name in exp_names:
        if name in histories:
            axes[1, 1].plot(histories[name]['dice'], label=name)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice')
    axes[1, 1].set_title('Training Dice')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Domain Transfer Learning — Experiment Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_prediction_comparison(results, test_inputs, save_path, n_show=5):
    """Side-by-side predictions from all experiments on test slices."""
    exp_names = list(results.keys())
    n_exp = len(exp_names)

    # Get test inputs/targets from first experiment
    first_exp = exp_names[0]
    targets = results[first_exp]['targets']
    n_total = targets.shape[0]
    n_show = min(n_show, n_total)

    # Pick slices with most void pixels for visibility
    void_counts = targets.sum(dim=(1, 2, 3))
    indices = torch.argsort(void_counts, descending=True)[:n_show]

    fig, axes = plt.subplots(n_exp + 2, n_show, figsize=(3 * n_show, 3 * (n_exp + 2)))
    if n_show == 1:
        axes = axes.reshape(-1, 1)

    for j, idx in enumerate(indices):
        # Input (center channel or single channel)
        inp = test_inputs[idx]
        if inp.shape[0] > 1:
            center = inp.shape[0] // 2
            axes[0, j].imshow(inp[center].numpy(), cmap='Reds', vmin=0, vmax=1)
        else:
            axes[0, j].imshow(inp[0].numpy(), cmap='Reds', vmin=0, vmax=1)
        axes[0, j].set_title(f'Slice {idx.item()}', fontsize=9)
        axes[0, j].axis('off')

        # Ground truth
        axes[1, j].imshow(targets[idx, 0].numpy(), cmap='gray', vmin=0, vmax=1)
        axes[1, j].axis('off')

        # Predictions from each experiment
        for k, name in enumerate(exp_names):
            pred = results[name]['predictions'][idx, 0].numpy()
            axes[k + 2, j].imshow(pred, cmap='gray', vmin=0, vmax=1)
            axes[k + 2, j].axis('off')

    # Row labels
    axes[0, 0].set_ylabel('Input', fontsize=10)
    axes[1, 0].set_ylabel('Ground Truth', fontsize=10)
    for k, name in enumerate(exp_names):
        axes[k + 2, 0].set_ylabel(name, fontsize=9)

    plt.suptitle('Prediction Comparison on THz Test Slices', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def print_results_table(results):
    """Print comparison table."""
    print(f"\n{'=' * 60}")
    print(f"{'Experiment':<25} {'Dice':>10} {'IoU':>10}")
    print(f"{'-' * 60}")
    for name, res in results.items():
        print(f"{name:<25} {res['dice']:>10.4f} {res['iou']:>10.4f}")
    print(f"{'=' * 60}")

    # Highlight best
    best_dice_name = max(results, key=lambda n: results[n]['dice'])
    best_iou_name = max(results, key=lambda n: results[n]['iou'])
    print(f"  Best Dice: {best_dice_name} ({results[best_dice_name]['dice']:.4f})")
    print(f"  Best IoU:  {best_iou_name} ({results[best_iou_name]['iou']:.4f})")


# ============================================================
# MAIN
# ============================================================

class CenterChannelLoader:
    """Wraps a multi-channel loader to yield only the center channel."""
    def __init__(self, loader, center=2):
        self.loader = loader
        self.center = center
    def __iter__(self):
        for x, y in self.loader:
            yield x[:, self.center:self.center + 1, :, :], y
    def __len__(self):
        return len(self.loader)


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    # ----------------------------------------------------------
    # 1. Load THz data from all .tprj files
    # ----------------------------------------------------------
    print("=" * 60)
    print("STEP 1: Loading THz data")
    print("=" * 60)

    all_samples = load_all_volumes(TPRJ_FILES)

    print("\nProcessing to depth slices...")
    all_slices_list = []  # list of (n_slices, Ny, Nx) arrays
    all_masks_list = []
    sample_names = []
    for sample in all_samples:
        try:
            slices, depths, _ = process_to_slices(sample, n_slices=N_SLICES)
            name = sample['name']
            name_safe = name.replace('/', '_')
            mask_path = LABELS_DIR / f'{name_safe}_slice_masks.npy'

            if not mask_path.exists():
                print(f"  {name}: no mask file")
                continue

            masks = np.load(mask_path)
            n_labeled = int((masks.sum(axis=(1, 2)) > 0).sum())
            print(f"  {name}: {n_labeled}/{N_SLICES} slices labeled")

            if n_labeled == 0:
                continue

            all_slices_list.append(slices)
            all_masks_list.append(masks)
            sample_names.append(name)
        except Exception as e:
            print(f"  SKIP {sample['name']}: {e}")

    # Build all labeled slice items (before splitting)
    all_items = []  # list of (input_array, mask_array, sample_name, slice_idx)
    for slices, masks, name in zip(all_slices_list, all_masks_list, sample_names):
        n_s = slices.shape[0]
        for i in range(n_s):
            if masks[i].sum() == 0:
                continue
            # Build context channels, resize to 64x64
            channels = []
            for offset in range(-CONTEXT, CONTEXT + 1):
                j = np.clip(i + offset, 0, n_s - 1)
                ch = Image.fromarray(slices[j]).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
                channels.append(np.array(ch, dtype=np.float32))
            inp = np.stack(channels, axis=0)

            m = Image.fromarray(masks[i].astype(np.float32)).resize(
                (IMG_SIZE, IMG_SIZE), Image.NEAREST)
            mask = (np.array(m, dtype=np.float32) > 0.5).astype(np.float32)

            all_items.append((inp, mask, name, i))

    total = len(all_items)
    print(f"\n  Total labeled slices: {total}")
    for name in sample_names:
        count = sum(1 for _, _, n, _ in all_items if n == name)
        print(f"    {name}: {count}")

    if total < 5:
        print("\nERROR: Not enough labeled slices.")
        return

    # Random 80/20 split
    np.random.seed(42)
    indices = np.random.permutation(total)
    split = int(0.8 * total)
    train_idx = indices[:split]
    test_idx = indices[split:]

    train_items = [all_items[i] for i in train_idx]
    test_items = [all_items[i] for i in test_idx]

    # Show split composition
    train_samples = Counter(name for _, _, name, _ in train_items)
    test_samples = Counter(name for _, _, name, _ in test_items)
    print(f"\n  Train ({len(train_items)} slices):")
    for name, count in sorted(train_samples.items()):
        print(f"    {name}: {count}")
    print(f"  Test ({len(test_items)} slices):")
    for name, count in sorted(test_samples.items()):
        print(f"    {name}: {count}")

    # Build datasets from pre-computed items
    class THzPrecomputedDataset(Dataset):
        def __init__(self, items, augment=False):
            self.items = items
            self.augment = augment
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            inp, mask, _, _ = self.items[idx]
            x = torch.FloatTensor(inp)
            y = torch.FloatTensor(mask).unsqueeze(0)
            if self.augment:
                x, y = augment_pair(x, y)
            return x, y

    thz_train_ds = THzPrecomputedDataset(train_items, augment=True)
    thz_test_ds = THzPrecomputedDataset(test_items, augment=False)

    print(f"\n  Train dataset: {len(thz_train_ds)} pairs ({CONTEXT * 2 + 1} ch, {IMG_SIZE}x{IMG_SIZE})")
    print(f"  Test dataset:  {len(thz_test_ds)} pairs ({CONTEXT * 2 + 1} ch, {IMG_SIZE}x{IMG_SIZE})")

    thz_train_loader = DataLoader(thz_train_ds, batch_size=8, shuffle=True, drop_last=False)
    thz_test_loader = DataLoader(thz_test_ds, batch_size=8, shuffle=False)

    test_inputs = torch.stack([thz_test_ds[i][0] for i in range(len(thz_test_ds))])

    # ----------------------------------------------------------
    # 2. Download & load Mendeley data
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 2: Mendeley CFRP dataset")
    print("=" * 60)

    download_mendeley_dataset()
    mend_train_imgs, mend_train_masks, _, _ = load_mendeley_splits()
    mend_train_ds = MendeleyDataset(mend_train_imgs, mend_train_masks, augment=True)
    mend_train_loader = DataLoader(mend_train_ds, batch_size=32, shuffle=True, drop_last=False)

    # ----------------------------------------------------------
    # 3. Run experiments
    # ----------------------------------------------------------
    all_results = {}
    all_histories = {}

    # ---- Experiment 1: Baseline (THz only) ----
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Baseline (THz only)")
    print("=" * 60)

    model_exp1 = UNet(in_channels=CONTEXT * 2 + 1, base_filters=32)
    hist1 = train_model(model_exp1, thz_train_loader, n_epochs=500, lr=1e-3,
                        device=DEVICE, label="Exp1-Baseline")
    res1 = evaluate_model(model_exp1, thz_test_loader, DEVICE)
    print(f"  Test — Dice: {res1['dice']:.4f}, IoU: {res1['iou']:.4f}")

    torch.save(model_exp1.state_dict(), RESULTS_DIR / 'exp1_baseline.pt')
    all_results['Exp1: Baseline'] = res1
    all_histories['Exp1: Baseline'] = hist1

    # ---- Shared: Pretrain on Mendeley (used by Exp 2 and Exp 3) ----
    print("\n" + "=" * 60)
    print("PRETRAINING on Mendeley (shared by Exp 2 & 3)")
    print("=" * 60)

    model_pretrained = UNet(in_channels=1, base_filters=32)
    hist_pre = train_model(model_pretrained, mend_train_loader, n_epochs=100, lr=1e-3,
                           device=DEVICE, label="Pretrain-Mendeley")
    pretrained_sd = model_pretrained.state_dict()
    adapted_sd = adapt_first_conv(pretrained_sd, new_in_channels=CONTEXT * 2 + 1,
                                   center_channel=CONTEXT)

    # ---- Experiment 2: Pretrain + Finetune (full, all layers unfrozen) ----
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Pretrain + Finetune (all layers)")
    print("=" * 60)

    model_exp2 = UNet(in_channels=CONTEXT * 2 + 1, base_filters=32)
    model_exp2.load_state_dict(adapted_sd)

    hist2 = train_model(model_exp2, thz_train_loader, n_epochs=500, lr=1e-4,
                        device=DEVICE, label="Exp2-FT-Full")
    res2 = evaluate_model(model_exp2, thz_test_loader, DEVICE)
    print(f"  Test — Dice: {res2['dice']:.4f}, IoU: {res2['iou']:.4f}")

    torch.save(model_exp2.state_dict(), RESULTS_DIR / 'exp2_finetune_full.pt')
    all_results['Exp2: PT+FT (full)'] = res2
    all_histories['Exp2: PT+FT (full)'] = {
        'loss': hist_pre['loss'] + hist2['loss'],
        'dice': hist_pre['dice'] + hist2['dice'],
        'iou': hist_pre['iou'] + hist2['iou'],
    }

    # ---- Experiment 3: Pretrain + Finetune (frozen encoder) ----
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Pretrain + Finetune (frozen encoder)")
    print("=" * 60)

    model_exp3 = UNet(in_channels=CONTEXT * 2 + 1, base_filters=32)
    model_exp3.load_state_dict(adapted_sd)
    freeze_encoder(model_exp3)

    hist3 = train_model(model_exp3, thz_train_loader, n_epochs=500, lr=1e-4,
                        device=DEVICE, label="Exp3-FT-FrozenEnc")
    res3 = evaluate_model(model_exp3, thz_test_loader, DEVICE)
    print(f"  Test — Dice: {res3['dice']:.4f}, IoU: {res3['iou']:.4f}")

    torch.save(model_exp3.state_dict(), RESULTS_DIR / 'exp3_finetune_frozen.pt')
    all_results['Exp3: PT+FT (frozen)'] = res3
    all_histories['Exp3: PT+FT (frozen)'] = {
        'loss': hist_pre['loss'] + hist3['loss'],
        'dice': hist_pre['dice'] + hist3['dice'],
        'iou': hist_pre['iou'] + hist3['iou'],
    }

    # ----------------------------------------------------------
    # 4. Results
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print_results_table(all_results)

    plot_comparison(all_results, all_histories, RESULTS_DIR / 'experiment_comparison.png')
    plot_prediction_comparison(all_results, test_inputs,
                               RESULTS_DIR / 'prediction_comparison.png')

    # Save JSON results
    json_results = {
        'split': f'{len(train_items)} train / {len(test_items)} test (random 80/20, seed=42)',
        'train_composition': dict(train_samples),
        'test_composition': dict(test_samples),
    }
    for name, res in all_results.items():
        json_results[name] = {'dice': res['dice'], 'iou': res['iou']}
    with open(RESULTS_DIR / 'comparison_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n  Saved: {RESULTS_DIR / 'comparison_results.json'}")

    print(f"\nAll outputs saved to {RESULTS_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
