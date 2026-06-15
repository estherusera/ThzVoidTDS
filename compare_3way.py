"""
compare_3way.py — Headline 3-way comparison under the same unified STL metric:
  • C-scan U-Net trained on hand-drawn manual labels (results_v2_atlanta2/void_char/)
  • A-scan 1D-CNN  (results_v2_atlanta2/ascan_classifier/vs_stl_unified/)
  • C-scan U-Net trained on STL-derived soft labels  (NEW, results_v2_atlanta2/unet_stl_vs_stl.json)

Produces:
  • three_way_comparison.png  (per-sample bars + aggregate)
  • stl_unet_showcase.png     (6 representative samples for the new STL U-Net)

Run
    thesis_env/bin/python compare_3way.py
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

from thz_slice_pipelinev2 import UNet
from void_characterize import (
    voxelize_gt, extract_pred_blobs, best_alignment_and_match,
    SAMPLE_TO_STL, STL_DIR, PX_MM, NX, NY,
)
from eval_unet_stl import predict_volume, OFFSETS
from ascan_vs_stl import DepthAwareAScanCNN
from ascan_classifier import CACHE_ATL2, N_DEPTHS

ASCAN_JSON  = Path("results_v2_atlanta2/ascan_classifier/vs_stl_unified/summary_unified.json")
UNET_M_JSON = Path("results_v2_atlanta2/void_char/aggregate.json")
UNET_S_JSON = Path("results_v2_atlanta2/unet_stl_vs_stl.json")
SLICES_DIR     = Path("slices_v2_atlanta2")
SLICES_DIR_2050 = Path("slices_v2_atlanta2_2050")
STL_LABELS     = Path("labels_stl_atlanta2")
MANUAL_LBLS    = Path("labels_v2_atlanta2")
MODEL_S        = Path("results_v2_atlanta2/unet_stl.pt")
MODEL_MAN_2050 = Path("results_v2_atlanta2/unet_pseudo_2050.pt")
MODEL_ASCAN    = Path("results_v2_atlanta2/ascan_classifier/ascan_cnn_depth.pt")
OFFSETS_2050   = [-40, -20, 0, 20, 40]
OUT_DIR        = Path("results_v2_atlanta2/three_way")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SHOWCASE = ["1", "4", "5", "8", "9", "12"]

device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def load_per_sample(path):
    with open(path) as fh:
        d = json.load(fh)
    if isinstance(d, dict) and "per_sample" in d:
        d = d["per_sample"]
    return {r["sample"]: r for r in d}


def _f1(r):  return r.get("f1_unified", r.get("f1", 0.0))
def _p(r):   return r.get("p_unified",  r.get("precision", 0.0))
def _r(r):   return r.get("r_unified",  r.get("recall", 0.0))


def plot_three_way():
    ascan  = load_per_sample(ASCAN_JSON)
    unet_m = load_per_sample(UNET_M_JSON)
    unet_s = load_per_sample(UNET_S_JSON)

    samples = sorted(set(ascan) & set(unet_m) & set(unet_s),
                     key=lambda x: int(x.rstrip("b")))
    samples = [s for s in samples
               if ascan[s].get("n_gt", 1) > 0 and unet_m[s].get("n_gt", 1) > 0]

    f1_a = [_f1(ascan[s])  for s in samples]
    f1_m = [_f1(unet_m[s]) for s in samples]
    f1_s = [_f1(unet_s[s]) for s in samples]

    x = np.arange(len(samples)); w = 0.27
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 7), dpi=140,
        gridspec_kw={"height_ratios": [3, 1.1]})

    ax1.bar(x - w, f1_m, w, label="C-scan U-Net (manual labels)", color="#1f78b4")
    ax1.bar(x,     f1_a, w, label="A-scan 1D-CNN",                  color="#33a02c")
    ax1.bar(x + w, f1_s, w, label="C-scan U-Net (STL labels) — NEW", color="#e31a1c")
    ax1.set_xticks(x); ax1.set_xticklabels(samples)
    ax1.set_xlabel("Sample"); ax1.set_ylabel("F1 (per-blob, 8 mm gate)")
    ax1.set_ylim(0, 1.05); ax1.grid(axis="y", alpha=0.3)
    ax1.set_title("STL F1 by approach — same matcher, same atlanta2 samples",
                  fontsize=11, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=9)

    metrics = ["F1", "Precision", "Recall"]
    aggs = []
    for src, label, col in [
        ([unet_m[s] for s in samples], "C-scan U-Net (manual)", "#1f78b4"),
        ([ascan[s]  for s in samples], "A-scan 1D-CNN",          "#33a02c"),
        ([unet_s[s] for s in samples], "C-scan U-Net (STL)",     "#e31a1c"),
    ]:
        aggs.append((label, col, [
            float(np.mean([_f1(r) for r in src])),
            float(np.mean([_p(r)  for r in src])),
            float(np.mean([_r(r)  for r in src])),
        ]))

    xx = np.arange(3)
    for i, (label, col, vals) in enumerate(aggs):
        offs = (i - 1) * w
        ax2.bar(xx + offs, vals, w, color=col)
        for j, v in enumerate(vals):
            ax2.text(xx[j] + offs, v + 0.02, f"{v:.3f}",
                     ha="center", fontsize=8, color=col, fontweight="bold")
    ax2.set_xticks(xx); ax2.set_xticklabels(metrics)
    ax2.set_ylim(0, 1.15); ax2.grid(axis="y", alpha=0.3)
    ax2.set_title(f"Aggregate over {len(samples)} void samples",
                  fontsize=10, fontweight="bold")

    plt.tight_layout()
    out = OUT_DIR / "three_way_comparison.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def _predict_ascan(model_ascan, name):
    """Run the depth-aware A-scan model on a sample → (50, 100, 100) probs."""
    cache_path = CACHE_ATL2 / f"{name}.npz"
    if not cache_path.exists():
        return None
    d = np.load(cache_path)
    a = d["ascans"]
    m = a.mean(axis=1, keepdims=True); s = a.std(axis=1, keepdims=True) + 1e-6
    a_t = torch.from_numpy(((a - m) / s).astype(np.float32)).unsqueeze(1)  # CPU
    probs = np.zeros((a_t.shape[0], N_DEPTHS), dtype=np.float32)
    CHUNK = 512
    with torch.no_grad():
        for k in range(0, a_t.shape[0], CHUNK):
            probs[k:k+CHUNK] = torch.sigmoid(model_ascan(a_t[k:k+CHUNK])).cpu().numpy()
    return probs.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)


def _predict_unet(model_unet, name, slices_dir, offsets):
    slices_p = slices_dir / f"{name}_slices.npy"
    if not slices_p.exists():
        return None
    slices = np.load(slices_p)
    return predict_volume(model_unet, slices, offsets)


def plot_stl_unet_showcase():
    """For each showcase sample, side-by-side panels:
       STL labels | manual rectangles |
       Manual-U-Net prob | A-scan prob | STL-U-Net prob |
       RGB overlay (R=A-scan, G=STL-U-Net, B=Manual-U-Net) over STL outline
    """
    # ── load all three models ──
    ck_s   = torch.load(MODEL_S,        map_location=device, weights_only=False)
    cfg_s  = ck_s.get("config", {})
    model_stl = UNet(in_channels=cfg_s.get("in_channels", 5),
                     base_filters=cfg_s.get("base_filters", 32)).to(device)
    model_stl.load_state_dict(ck_s["model_state"]); model_stl.eval()

    ck_m   = torch.load(MODEL_MAN_2050, map_location=device, weights_only=False)
    cfg_m  = ck_m.get("config", {})
    model_man = UNet(in_channels=cfg_m.get("in_channels", 5),
                     base_filters=cfg_m.get("base_filters", 32)).to(device)
    model_man.load_state_dict(ck_m["model_state"]); model_man.eval()

    ck_a   = torch.load(MODEL_ASCAN, map_location="cpu", weights_only=False)
    model_ascan = DepthAwareAScanCNN()
    model_ascan.load_state_dict(ck_a["model_state"]); model_ascan.eval()
    # A-scan stays on CPU (MPS doesn't support its AdaptiveAvgPool1d(50))

    ncols = 6
    nrows = len(SHOWCASE)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.2 * nrows),
                             dpi=140)

    for row, name in enumerate(SHOWCASE):
        stl    = np.load(STL_LABELS / f"{name}_slice_masks.npy")
        man_p  = MANUAL_LBLS / f"{name}_slice_masks.npy"
        manual = np.load(man_p) if man_p.exists() else np.zeros_like(stl)

        probs_stl   = _predict_unet(model_stl, name, SLICES_DIR, OFFSETS)
        probs_man   = _predict_unet(model_man, name, SLICES_DIR_2050, OFFSETS_2050)
        probs_ascan = _predict_ascan(model_ascan, name)

        stl_proj   = stl.max(axis=0)
        man_proj   = (manual.max(axis=0) > 0).astype(np.float32)
        pred_stl   = probs_stl.max(axis=0)   if probs_stl   is not None else np.zeros((NY, NX))
        pred_man   = probs_man.max(axis=0)   if probs_man   is not None else np.zeros((NY, NX))
        pred_ascan = probs_ascan.max(axis=0) if probs_ascan is not None else np.zeros((NY, NX))

        # Column 0 — STL soft labels
        ax = axes[row, 0]
        ax.imshow(stl_proj, cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title(f"S{name}  STL labels", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # Column 1 — manual rectangles
        ax = axes[row, 1]
        ax.imshow(man_proj, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("Manual rectangles", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # Column 2 — manual-trained U-Net (2050-slice)
        ax = axes[row, 2]
        ax.imshow(pred_man, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("Manual-U-Net pred", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # Column 3 — A-scan 1D-CNN
        ax = axes[row, 3]
        ax.imshow(pred_ascan, cmap="Purples", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("A-scan 1D-CNN pred", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # Column 4 — STL-trained U-Net
        ax = axes[row, 4]
        ax.imshow(pred_stl, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
        ax.set_title("STL-U-Net pred", fontsize=9, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

        # Column 5 — RGB overlay of all three predictions (>0.5)
        ax = axes[row, 5]
        rgb = np.zeros((NY, NX, 3), dtype=np.float32)
        rgb[..., 0] = (pred_ascan > 0.5).astype(np.float32)   # A-scan red
        rgb[..., 1] = (pred_stl   > 0.5).astype(np.float32)   # STL-U-Net green
        rgb[..., 2] = (pred_man   > 0.5).astype(np.float32)   # Manual-U-Net blue
        ax.imshow(rgb, interpolation="nearest")
        # outline STL truth on top in white
        ax.contour(stl_proj > 0.5, levels=[0.5], colors="white",
                   linewidths=0.8, linestyles="--")
        ax.set_title("Overlay (R=Ascan G=STLunet B=Manunet) + STL outline",
                     fontsize=7, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])

    plt.suptitle("Per-sample predictions — STL labels vs all three models "
                 "(threshold 0.5)",
                 fontsize=12, fontweight="bold", y=1.0)
    plt.tight_layout()
    out = OUT_DIR / "stl_unet_showcase.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def main():
    plot_three_way()
    plot_stl_unet_showcase()


if __name__ == "__main__":
    main()
