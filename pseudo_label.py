"""
pseudo_label.py — Generate pseudo-labels for atlanta2 using a trained U-Net.

Usage:
    thesis_env/bin/python pseudo_label.py [N_SLICES] [MODEL_CKPT] [OFFSET_STEP]

Defaults reproduce the original 500-slice run (atlanta1 model, offsets ±10):
    N_SLICES     = 500
    MODEL_CKPT   = results_v2/unet_v2.pt
    OFFSET_STEP  = 5  → offsets [-10, -5, 0, +5, +10]

Example — 2050-slice with new atlanta2 model and ±40 depth window:
    thesis_env/bin/python pseudo_label.py 2050 \\
        results_v2_atlanta2/unet_pseudo.pt 20

Outputs (per sample) to labels_v2_atlanta2_pseudo_{N_SLICES}/:
  {name}_pseudo_mask.npy   — binary uint8 (N_SLICES, 100, 100)
  {name}_pseudo_prob.npy   — float16 (N_SLICES, 100, 100)  raw probabilities
"""

import sys
import numpy as np
import torch
from pathlib import Path

# ── CLI args (read BEFORE sys.argv is rewritten for the pipeline import) ─────
N_SLICES    = int(sys.argv[1]) if len(sys.argv) > 1 else 500
MODEL_CKPT  = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("results_v2/unet_v2.pt")
OFFSET_STEP = int(sys.argv[3]) if len(sys.argv) > 3 else 5

sys.path.insert(0, str(Path(__file__).parent))
# Force atlanta2 mode for path resolution (we only care about the model class)
sys.argv = [sys.argv[0], "predict", "atlanta2"]
from thz_slice_pipelinev2 import UNet

# ── paths derived from N_SLICES ──────────────────────────────────────────────
# Convention: 500 keeps the original "labels_v2_atlanta2_pseudo" name for back-compat.
SLICES_DIR = (Path("slices_v2_atlanta2") if N_SLICES == 50
              else Path(f"slices_v2_atlanta2_{N_SLICES}"))
OUT_DIR    = (Path("labels_v2_atlanta2_pseudo") if N_SLICES == 500
              else Path(f"labels_v2_atlanta2_pseudo_{N_SLICES}"))
THRESHOLD  = 0.5

CONTEXT_N  = 2     # ±2 channels around centre
OFFSETS    = [o * OFFSET_STEP for o in range(-CONTEXT_N, CONTEXT_N + 1)]
assert len(OFFSETS) == 5, "model expects 5 input channels"


def main():
    print(f"\n=== pseudo_label ===")
    print(f"  model:        {MODEL_CKPT}")
    print(f"  slices in:    {SLICES_DIR}/")
    print(f"  pseudo out:   {OUT_DIR}/")
    print(f"  offsets used: {OFFSETS}   (5 channels, step={OFFSET_STEP})")
    print(f"  threshold:    {THRESHOLD}\n")

    if not SLICES_DIR.exists():
        sys.exit(f"ERROR: {SLICES_DIR} not found — run export_slices.py atlanta2 500 first.")

    # Pick device automatically
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if hasattr(torch.backends, "mps") and
                            torch.backends.mps.is_available()
              else "cpu")
    print(f"  device: {device}")

    ckpt  = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = UNet(in_channels=cfg["in_channels"],
                 base_filters=cfg["base_filters"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    print(f"  model loaded: in_channels={cfg['in_channels']}, "
          f"base_filters={cfg['base_filters']}\n")

    OUT_DIR.mkdir(exist_ok=True)

    sample_files = sorted(SLICES_DIR.glob("*_slices.npy"),
                          key=lambda p: int(p.stem.split("_")[0].rstrip("b")))
    summary = []

    for sp in sample_files:
        name   = sp.stem.replace("_slices", "")
        slices = np.load(sp)                              # (500, 100, 100)
        n_s    = slices.shape[0]
        print(f"  → sample {name}: {slices.shape}")

        # Batch all 500 slices for one big forward pass per sample
        inputs = np.zeros((n_s, 5, 100, 100), dtype=np.float32)
        for i in range(n_s):
            for ch, off in enumerate(OFFSETS):
                j = int(np.clip(i + off, 0, n_s - 1))
                inputs[i, ch] = slices[j]
        x = torch.from_numpy(inputs).to(device)

        with torch.no_grad():
            # Run in chunks to keep memory sane
            probs = np.zeros((n_s, 100, 100), dtype=np.float32)
            CHUNK = 32
            for k in range(0, n_s, CHUNK):
                logits = model(x[k:k + CHUNK])
                probs[k:k + CHUNK] = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        masks = (probs > THRESHOLD).astype(np.uint8)

        np.save(OUT_DIR / f"{name}_pseudo_prob.npy", probs.astype(np.float16))
        np.save(OUT_DIR / f"{name}_pseudo_mask.npy", masks)

        n_pos_slices = int((masks.sum(axis=(1, 2)) > 0).sum())
        total_px     = int(masks.sum())
        summary.append((name, n_pos_slices, total_px))
        print(f"     {n_pos_slices}/{n_s} slices have predicted voids, "
              f"{total_px} total positive px")

    # ── summary table ────────────────────────────────────────────────────────
    print("\n" + "─" * 60)
    print(f"{'sample':>6s}  {'pos slices':>11s}  {'pos px':>10s}")
    print("─" * 60)
    for name, ns, npx in summary:
        print(f"{name:>6s}  {ns:>8d}/500  {npx:>10d}")
    print("─" * 60)
    print(f"\nWrote pseudo-labels to {OUT_DIR}/")


if __name__ == "__main__":
    main()
