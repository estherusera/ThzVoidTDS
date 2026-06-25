"""
ascan_unet_bilstm.py — 1D U-Net + BiLSTM A-scan damage-segmentation model
=========================================================================
Implements the U-Net-BiLSTM of Zhang et al. (2024), "Quantitative Detection of
Defects in Multi-Layer Lightweight Composite Structures Using THz-TDS Based on a
U-Net-BiLSTM Network", Materials 17(4):839 — adapted to this repo's A-scan cache.

Per-time-point binary segmentation: input one surface-flattened A-scan
(B, 1, 2048), output a void probability at every time sample (B, 2048). The
paper's contribution is a BiLSTM inserted on the deepest skip connections so the
decoder sees temporally-contextualised features; we keep it off the two
full-length skips for speed.

This module also holds the shared data helpers (label building in the
apparent-depth frame, dataset, and the 2048→50-bin reduction used to plug the
predictions into the existing per-blob unified evaluation harness).

Label provenance: the per-time labels are regenerated at 2048 samples from the
SAME source as the existing 50-bin depth labels (the manual masks stored in the
cache as `labels_3d`), by placing each void depth-bin at its true time index in
the surface-flattened frame (Voronoi/nearest assignment over the slice indices).
They are NOT an upsample of the 50-bin vector. Verified to land on the void echo
in the apparent (measured) frame, not the CAD depth — see label_align_check.png.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

SEQ_LEN  = 2048      # model input length (A-scans are 2050; crop/pad to this)
N_DEPTHS = 50        # depth bins used by the unified eval harness


# ── data helpers ─────────────────────────────────────────────────────────────
def slice_indices(cache, n_depths=N_DEPTHS):
    """Time index of each of the n_depths depth slices in the flattened frame.

    Mirrors process_to_slices: linspace(target_idx, N_time-1, n_depths).
    """
    n_time = len(cache["tax"])
    target = int(cache["target_idx"])
    return np.linspace(target, n_time - 1, n_depths).astype(int)


def bin_to_time_index(si, seq_len=SEQ_LEN):
    """For each time sample t in [0, seq_len), the nearest depth-bin index.

    Samples before the first / after the last slice map to -1 (always background).
    """
    t = np.arange(seq_len)
    k = np.abs(si[None, :] - t[:, None]).argmin(axis=1)
    k[(t < si[0]) | (t > si[-1])] = -1
    return k


def labels_2048(labels_3d_row, k_map):
    """50-bin void vector → (seq_len,) per-time label via the bin→time map."""
    out = np.zeros(len(k_map), dtype=np.float32)
    valid = k_map >= 0
    out[valid] = labels_3d_row[k_map[valid]]
    return out


def reduce_to_depth_bins(prob_2048, k_map, n_depths=N_DEPTHS):
    """(N, seq_len) per-time probs → (N, n_depths) by max-pool within each bin.

    Inverse of `labels_2048`; lets U-Net-BiLSTM predictions feed the existing
    (50, NY, NX) per-blob unified eval harness for an apples-to-apples F1.
    """
    prob_2048 = np.asarray(prob_2048)
    if prob_2048.ndim == 1:
        prob_2048 = prob_2048[None, :]
    n = prob_2048.shape[0]
    out = np.zeros((n, n_depths), dtype=np.float32)
    for kbin in range(n_depths):
        cols = np.where(k_map == kbin)[0]
        if cols.size:
            out[:, kbin] = prob_2048[:, cols].max(axis=1)
    return out


def crop_or_pad(sig, seq_len=SEQ_LEN):
    """Trim/zero-pad a 1-D signal to seq_len along the last axis."""
    n = sig.shape[-1]
    if n == seq_len:
        return sig
    if n > seq_len:
        return sig[..., :seq_len]
    pad = [(0, 0)] * (sig.ndim - 1) + [(0, seq_len - n)]
    return np.pad(sig, pad, mode="constant")


class AScanSegDataset(Dataset):
    """Each item: (1×2048 A-scan, 2048 per-time void label).

    Reuses the existing per-sample caches; builds the per-time label on the fly
    from `labels_3d` so we never duplicate the (N, 2048) label array in memory.
    Augmentations match the depth model: per-scan z-score, ±5-sample time roll
    (applied to BOTH signal and label), and σ=0.02 Gaussian noise.
    """

    def __init__(self, sample_names, cache_dir, augment=False, seq_len=SEQ_LEN):
        self.items, self.kmaps = [], []
        self.seq_len = seq_len
        for name in sample_names:
            p = cache_dir / f"{name}.npz"
            if not p.exists():
                print(f"  ! missing cache {p}")
                continue
            d = np.load(p)
            ascans = crop_or_pad(d["ascans"].astype(np.float32), seq_len)  # (N,2048)
            self.items.append((ascans, d["labels_3d"], name))
            self.kmaps.append(bin_to_time_index(slice_indices(d), seq_len))
        self.lengths = [a.shape[0] for a, _, _ in self.items]
        self.cumlen  = np.cumsum([0] + self.lengths)
        self.augment = augment

    def pos_fraction(self):
        """Mean positive (void) fraction over all per-time labels — for pos_weight."""
        tot = pos = 0
        for (a, lab, _), k in zip(self.items, self.kmaps):
            L = np.zeros((lab.shape[0], self.seq_len), dtype=np.float32)
            valid = k >= 0
            L[:, valid] = lab[:, k[valid]]
            pos += float(L.sum()); tot += L.size
        return pos / tot

    def __len__(self):
        return int(self.cumlen[-1])

    def __getitem__(self, idx):
        si = int(np.searchsorted(self.cumlen, idx, side="right") - 1)
        pi = idx - int(self.cumlen[si])
        a  = self.items[si][0][pi].copy()                       # (2048,)
        y  = labels_2048(self.items[si][1][pi], self.kmaps[si]) # (2048,)

        m = a.mean(); s = a.std() + 1e-6
        a = (a - m) / s

        if self.augment:
            shift = int(np.random.randint(-5, 6))
            if shift:
                a = np.roll(a, shift); y = np.roll(y, shift)
            a = a + 0.02 * np.random.randn(*a.shape).astype(np.float32)

        return (torch.from_numpy(a).unsqueeze(0).float(),
                torch.from_numpy(y).float())


# ── model ────────────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, 3, padding=1), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
            nn.Conv1d(cout, cout, 3, padding=1), nn.BatchNorm1d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class SkipBiLSTM(nn.Module):
    """BiLSTM applied to a skip tensor (B,C,T): contextualise along time, keep C."""

    def __init__(self, channels, hidden=None):
        super().__init__()
        hidden = hidden or channels
        self.lstm = nn.LSTM(channels, hidden, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(2 * hidden, channels)

    def forward(self, x):                       # x: (B, C, T)
        h = x.transpose(1, 2)                   # (B, T, C)
        h, _ = self.lstm(h)                     # (B, T, 2*hidden)
        h = self.proj(h)                        # (B, T, C)
        return h.transpose(1, 2)                # (B, C, T)


class UNetBiLSTM1D(nn.Module):
    """1D U-Net (channels 16→32→64→128) with BiLSTM on the two deepest skips.

    Input (B,1,L), output logits (B,L). Apply sigmoid externally.
    """

    def __init__(self, base=16, bilstm_deep_skips=True):
        super().__init__()
        c1, c2, c3, c4 = base, base*2, base*4, base*8     # 16,32,64,128
        self.enc1 = DoubleConv(1,  c1)
        self.enc2 = DoubleConv(c1, c2)
        self.enc3 = DoubleConv(c2, c3)
        self.enc4 = DoubleConv(c3, c4)
        self.pool = nn.MaxPool1d(2)
        self.bottleneck = DoubleConv(c4, c4*2)             # 128→256

        # BiLSTM on the two deepest skips only (enc3 @T/4, enc4 @T/8) — the paper's
        # modification; full-length skips (enc1,enc2) are left plain for speed.
        self.use_bilstm = bilstm_deep_skips
        if bilstm_deep_skips:
            self.lstm3 = SkipBiLSTM(c3)
            self.lstm4 = SkipBiLSTM(c4)

        self.up4 = nn.ConvTranspose1d(c4*2, c4, 2, stride=2)
        self.dec4 = DoubleConv(c4*2, c4)
        self.up3 = nn.ConvTranspose1d(c4, c3, 2, stride=2)
        self.dec3 = DoubleConv(c3*2, c3)
        self.up2 = nn.ConvTranspose1d(c3, c2, 2, stride=2)
        self.dec2 = DoubleConv(c2*2, c2)
        self.up1 = nn.ConvTranspose1d(c2, c1, 2, stride=2)
        self.dec1 = DoubleConv(c1*2, c1)
        self.out = nn.Conv1d(c1, 1, 1)

    def forward(self, x):
        s1 = self.enc1(x)                  # (B,16,L)
        s2 = self.enc2(self.pool(s1))      # (B,32,L/2)
        s3 = self.enc3(self.pool(s2))      # (B,64,L/4)
        s4 = self.enc4(self.pool(s3))      # (B,128,L/8)
        b  = self.bottleneck(self.pool(s4))# (B,256,L/16)

        if self.use_bilstm:
            s3 = self.lstm3(s3)
            s4 = self.lstm4(s4)

        d4 = self.dec4(torch.cat([self.up4(b),  s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))
        return self.out(d1).squeeze(1)     # (B, L) logits


if __name__ == "__main__":
    # quick shape sanity check
    m = UNetBiLSTM1D()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(4, 1, SEQ_LEN)
    y = m(x)
    print(f"UNetBiLSTM1D params={n:,}  in={tuple(x.shape)}  out={tuple(y.shape)}")
