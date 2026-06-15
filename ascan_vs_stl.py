"""
ascan_vs_stl.py — Apples-to-apples STL evaluation of the A-scan 1D-CNN classifier
using the SAME matching machinery as void_characterize.py:

  • mesh.contains() GT footprint (vs the bbox projection used internally by
    ascan_classifier.stl_projection())
  • 8-symmetry rotation alignment (vs no alignment at all)
  • Per-blob Hungarian matching with an 8 mm centroid gate (vs per-pixel F1)

This lets us compare the A-scan classifier's STL F1 directly with the C-scan
U-Net's number reported in void_char/. For each sample we also recompute the
old-style per-pixel F1 vs the bbox projection so the metric-shift contribution
is quantified on the SAME model output.

Run
    thesis_env/bin/python ascan_vs_stl.py            # all samples + aggregate
    thesis_env/bin/python ascan_vs_stl.py 4          # just sample 4
"""
import sys, json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Capture user's CLI BEFORE the imports below rewrite sys.argv.
_CLI_TARGET = sys.argv[1] if len(sys.argv) > 1 else None

sys.path.insert(0, str(Path(__file__).parent))

# Importing void_characterize re-uses voxelize_gt + matcher + constants.
# Both ascan_classifier and void_characterize rewrite sys.argv to a fixed
# dummy invocation as a side-effect of their pipeline imports — that's fine
# here, we never call their main().
from void_characterize import (
    voxelize_gt, extract_pred_blobs, best_alignment_and_match,
    SAMPLE_TO_STL, STL_DIR, PX_MM, NX, NY,
    DIST_GATE_MM, THRESHOLDS,
)
from ascan_classifier import (
    CACHE_ATL2, N_DEPTHS,
    TRAIN_SAMPLES, VAL_SAMPLES, NOVOID_SANITY,
    stl_projection as bbox_projection,        # the old projection
)
import torch.nn as nn

# Force CPU: the saved model's AdaptiveAvgPool1d(50) has input T'=256 and 256/50
# is not divisible, which MPS doesn't support. The model is small enough that
# CPU is fine.
device = "cpu"


# ── Model class matching the saved ascan_cnn_depth.pt state dict ────────────
# encoder: same conv stack as ascan_classifier.AScanCNN
# head:    AdaptiveAvgPool1d(50) → Conv1d(128, 1, 1)   (not Linear)
# forward returns (B, 50) per-depth logits.
class DepthAwareAScanCNN(nn.Module):
    def __init__(self, n_depths=N_DEPTHS):
        super().__init__()
        self.n_depths = n_depths
        self.encoder = nn.Sequential(
            nn.Conv1d(1,   32, 7, padding=3),  nn.BatchNorm1d(32),  nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32,  64, 5, padding=2),  nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2),  nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(128,128, 3, padding=1),  nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(n_depths),     # (B, 128, T') → (B, 128, n_depths)
            nn.Conv1d(128, 1, kernel_size=1),   # → (B, 1, n_depths)
        )

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)   # (B, n_depths)

# ── config ───────────────────────────────────────────────────────────────────
MODEL_CKPT = Path("results_v2_atlanta2/ascan_classifier/ascan_cnn_depth.pt")
DEPTHS_DIR = Path("slices_v2_atlanta2")        # per-sample {name}_depths.npy (50,)
OUT_DIR    = Path("results_v2_atlanta2/ascan_classifier/vs_stl_unified")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ──────────────────────────────────────────────────────────────────
def load_model():
    if not MODEL_CKPT.exists():
        sys.exit(f"missing checkpoint {MODEL_CKPT}")
    ckpt  = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    model = DepthAwareAScanCNN().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict_prob_volume(model, cache_path):
    """Return (50, NY, NX) probability volume for one sample."""
    d = np.load(cache_path)
    a = d["ascans"]                                    # (10000, 2050)
    m = a.mean(axis=1, keepdims=True)
    s = a.std(axis=1, keepdims=True) + 1e-6
    a_t = torch.from_numpy(((a - m) / s).astype(np.float32)).unsqueeze(1).to(device)
    probs = np.zeros((a_t.shape[0], N_DEPTHS), dtype=np.float32)
    CHUNK = 512
    with torch.no_grad():
        for k in range(0, a_t.shape[0], CHUNK):
            probs[k:k+CHUNK] = torch.sigmoid(model(a_t[k:k+CHUNK])).cpu().numpy()
    # (10000, 50) → (NY, NX, 50) → (50, NY, NX)
    return probs.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)


def pixel_f1(gt, pred):
    g = gt.astype(bool); p = pred.astype(bool)
    tp = int((g & p).sum()); fp = int((~g & p).sum()); fn = int((g & ~p).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1   = 2*tp / max(2*tp + fp + fn, 1)
    return f1, prec, rec, tp, fp, fn


def split_of(name):
    if name in TRAIN_SAMPLES:   return "TRAIN"
    if name in VAL_SAMPLES:     return "VAL"
    if name in NOVOID_SANITY:   return "NOVOID"
    return "?"


# ── per-sample driver ────────────────────────────────────────────────────────
def evaluate_sample(name, model):
    cache_path  = CACHE_ATL2 / f"{name}.npz"
    depths_path = DEPTHS_DIR / f"{name}_depths.npy"
    if not cache_path.exists() or not depths_path.exists():
        return None

    depths   = np.load(depths_path).astype(np.float32)        # (50,)
    stl_path = STL_DIR / SAMPLE_TO_STL[name]
    if not stl_path.exists():
        return None

    # ── GT footprints ────────────────────────────────────────────────────────
    # (a) mesh-contains z-projection per void, used by the new matcher
    t0 = time.time()
    gt_masks, gt_meta = voxelize_gt(stl_path, depths)
    n_gt = len(gt_meta)
    t_gt = time.time() - t0

    # (b) old bbox z-projection — single 2D mask, used by per-pixel F1
    gt_bbox = bbox_projection(name)                            # (NY, NX) bool

    # ── model output ─────────────────────────────────────────────────────────
    t0 = time.time()
    probs = predict_prob_volume(model, cache_path)             # (50, NY, NX)
    t_pred = time.time() - t0
    pred_proj = (probs.max(axis=0) > 0.5).astype(bool)         # for old metric

    # ── new metric: per-blob Hungarian matching with symmetry sweep ──────────
    sweep = []
    for thr in THRESHOLDS:
        blobs = extract_pred_blobs(probs, thr, depths)
        n_blob = len(blobs)
        if n_gt == 0 or n_blob == 0:
            sweep.append(dict(thr=float(thr), n_blobs=n_blob, matches=0,
                              precision=0.0, recall=0.0, f1=0.0, sym=-1))
            continue
        best = best_alignment_and_match(gt_meta, blobs)
        nm   = len(best["matches"])
        prec = nm / n_blob
        rec  = nm / n_gt
        f1   = (2*prec*rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        sweep.append(dict(thr=float(thr), n_blobs=n_blob, matches=nm,
                          precision=prec, recall=rec, f1=f1,
                          sym=best["sym_id"]))

    if n_gt == 0:
        # No-void: best threshold = the one that produces the fewest pred blobs
        best_row = min(sweep, key=lambda r: r["n_blobs"])
    else:
        best_row = max(sweep, key=lambda r: (r["f1"], -r["n_blobs"]))

    # ── old metric: per-pixel F1 vs bbox projection ──────────────────────────
    f1_px, p_px, r_px, _, _, _ = pixel_f1(gt_bbox, pred_proj)

    return dict(
        sample=name,
        n_gt=int(n_gt),
        n_pred_blobs=int(best_row["n_blobs"]),
        n_pred_pixels=int(pred_proj.sum()),
        n_gt_pixels=int(gt_bbox.sum()),
        best_threshold=float(best_row["thr"]),
        best_symmetry=int(best_row["sym"]),
        f1_unified=float(best_row["f1"]),
        p_unified=float(best_row["precision"]),
        r_unified=float(best_row["recall"]),
        f1_pixel=float(f1_px),
        p_pixel=float(p_px),
        r_pixel=float(r_px),
        sweep=sweep,
        t_gt_s=float(t_gt), t_pred_s=float(t_pred),
    )


# ── reporting ────────────────────────────────────────────────────────────────
def print_table(rows, header):
    print(f"\n── {header} ──")
    print(f"{'sample':>6s} {'split':>7s} {'n_gt':>5s} {'n_pr':>5s} {'thr':>5s} "
          f"{'sym':>4s} {'F1u':>6s} {'Pu':>6s} {'Ru':>6s} | "
          f"{'F1px':>6s} {'Ppx':>6s} {'Rpx':>6s}")
    for r in rows:
        print(f"{r['sample']:>6s} {r['split']:>7s} {r['n_gt']:>5d} "
              f"{r['n_pred_blobs']:>5d} {r['best_threshold']:>5.2f} "
              f"{r['best_symmetry']:>4d} "
              f"{r['f1_unified']:>6.3f} {r['p_unified']:>6.3f} {r['r_unified']:>6.3f} | "
              f"{r['f1_pixel']:>6.3f} {r['p_pixel']:>6.3f} {r['r_pixel']:>6.3f}")


def aggregate(rows, label):
    if not rows:
        return
    f1u = np.mean([r["f1_unified"] for r in rows])
    pu  = np.mean([r["p_unified"]  for r in rows])
    ru  = np.mean([r["r_unified"]  for r in rows])
    f1p = np.mean([r["f1_pixel"]   for r in rows])
    pp  = np.mean([r["p_pixel"]    for r in rows])
    rp  = np.mean([r["r_pixel"]    for r in rows])
    print(f"  {label:>11s} (n={len(rows)})  "
          f"F1u={f1u:.3f} (P={pu:.3f} R={ru:.3f}) | "
          f"F1px={f1p:.3f} (P={pp:.3f} R={rp:.3f})  Δ={f1u-f1p:+.3f}")


# ── comparison figure ────────────────────────────────────────────────────────
def make_comparison_plot(results, out_path):
    rows = [r for r in results if r["n_gt"] > 0]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: int(r["sample"].rstrip("b")))
    names = [r["sample"] for r in rows]
    f1u   = [r["f1_unified"] for r in rows]
    f1p   = [r["f1_pixel"]   for r in rows]
    x = np.arange(len(names))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=140)
    fig.patch.set_facecolor("white")
    b1 = ax.bar(x - w/2, f1p, w, label="Old (per-pixel vs bbox)", color="#c0392b")
    b2 = ax.bar(x + w/2, f1u, w, label="New (per-blob, mesh + sym + 8 mm)", color="#27ae60")
    for r, xi in zip(rows, x):
        c = "#1f5fa8" if r["split"] == "VAL" else "#444"
        ax.text(xi, -0.08, r["split"][:1], ha="center", va="top",
                fontsize=8, color=c, transform=ax.get_xaxis_transform())

    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.0)
    ax.set_title("A-scan 1D-CNN: STL F1 under old vs unified metric",
                 fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    model, ckpt = load_model()
    print(f"Loaded model: {MODEL_CKPT.name} (epoch {ckpt.get('epoch', '?')})")
    print(f"Device: {device}")

    # Pick samples
    if _CLI_TARGET and _CLI_TARGET not in ("all", "--all"):
        targets = [_CLI_TARGET]
    else:
        targets = sorted([k for k in SAMPLE_TO_STL if k != "5b"],
                         key=lambda x: int(x.rstrip("b")))

    results = []
    for name in targets:
        print(f"\n· sample {name} → {SAMPLE_TO_STL[name]}")
        r = evaluate_sample(name, model)
        if r is None:
            print("  (skip: missing cache/depths/STL)"); continue
        r["split"]   = split_of(name)
        r["dataset"] = "atlanta2"
        results.append(r)
        print(f"  n_gt={r['n_gt']}  n_pred={r['n_pred_blobs']} blobs  "
              f"thr={r['best_threshold']:.2f}  sym={r['best_symmetry']}  "
              f"F1u={r['f1_unified']:.3f}  F1px={r['f1_pixel']:.3f}")

    if not results:
        return

    # Table + aggregates
    print_table(results, header="A-scan classifier vs STL (unified vs old)")
    print("\n── aggregates ──")
    aggregate([r for r in results if r["split"] == "TRAIN" and r["n_gt"] > 0], "TRAIN(void)")
    aggregate([r for r in results if r["split"] == "VAL"   and r["n_gt"] > 0], "VAL(void)")
    aggregate([r for r in results if r["n_gt"] > 0],                              "ALL(void)")
    # No-void sanity reported separately (F1u and F1px are both 0 by construction)
    nov = [r for r in results if r["split"] == "NOVOID"]
    if nov:
        r = nov[0]
        print(f"  {'NOVOID':>11s} (n=1, sample {r['sample']})  "
              f"n_pred_blobs={r['n_pred_blobs']}  n_pred_pixels={r['n_pred_pixels']}")

    # JSON dump
    out_json = OUT_DIR / "summary_unified.json"
    with open(out_json, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved: {out_json}")

    make_comparison_plot(results, OUT_DIR / "old_vs_unified_f1.png")


if __name__ == "__main__":
    main()
