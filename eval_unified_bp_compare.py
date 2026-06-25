"""
eval_unified_bp_compare.py — band-pass vs unfiltered, unified per-blob F1
========================================================================
Scores both A-scan architectures, each trained on unfiltered AND band-pass
caches, through the existing unified per-blob harness (mesh GT + 8-symmetry +
8 mm gate). Reports same-architecture filtered-vs-unfiltered deltas so the
band-pass effect is isolated, plus same-domain (all-void), held-out VAL, and
cross-domain (atlanta1).

Architectures:
  • 1D-CNN depth classifier = ascan_classifier.AScanCNN  (head: pool→Flatten→Linear)
    ckpts: ascan_cnn_depth_nofilt.pt / ascan_cnn_depth_bp.pt  (from the band-pass A/B)
  • U-Net-BiLSTM = ascan_unet_bilstm.UNetBiLSTM1D
    ckpts: runs/unet_bilstm/unet_bilstm_best.pt / unet_bilstm_bp_best.pt

Run: thesis_env/bin/python eval_unified_bp_compare.py
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from ascan_unet_bilstm import (UNetBiLSTM1D, SEQ_LEN, N_DEPTHS,
                               slice_indices, bin_to_time_index,
                               reduce_to_depth_bins, crop_or_pad)
from ascan_classifier import (AScanCNN, TRAIN_SAMPLES, VAL_SAMPLES)
from void_characterize import (voxelize_gt, extract_pred_blobs,
                               best_alignment_and_match,
                               SAMPLE_TO_STL, STL_DIR, NX, NY, THRESHOLDS)

device = "cpu"
A2_DEPTHS = Path("slices_v2_atlanta2")
A1_DEPTHS = Path("slices_v2")
ADIR = Path("results_v2_atlanta2/ascan_classifier")
BDIR = Path("runs/unet_bilstm")

# (label, ckpt, model_kind, atlanta2_cache, atlanta1_cache)
CONFIGS = [
    ("1D-CNN  unfilt", ADIR/"ascan_cnn_depth_nofilt.pt", "cnn",
     Path("ascan_cache_atlanta2"),    Path("ascan_cache_atlanta1")),
    ("1D-CNN  bp",     ADIR/"ascan_cnn_depth_bp.pt",     "cnn",
     Path("ascan_cache_atlanta2_bp"), Path("ascan_cache_atlanta1_bp")),
    ("BiLSTM  unfilt", BDIR/"unet_bilstm_best.pt",       "bilstm",
     Path("ascan_cache_atlanta2"),    Path("ascan_cache_atlanta1")),
    ("BiLSTM  bp",     BDIR/"unet_bilstm_bp_best.pt",    "bilstm",
     Path("ascan_cache_atlanta2_bp"), Path("ascan_cache_atlanta1_bp")),
]


def load_model(ckpt, kind):
    c = torch.load(ckpt, map_location=device, weights_only=False)
    if kind == "cnn":
        m = AScanCNN().to(device)
    else:
        cfg = c["config"]; m = UNetBiLSTM1D(base=cfg.get("base", 16),
                                            bilstm_deep_skips=cfg.get("bilstm", True)).to(device)
    m.load_state_dict(c["model_state"]); m.eval()
    return m


def vol_cnn(model, cache):
    d = np.load(cache); a = d["ascans"].astype(np.float32)
    m, s = a.mean(1, keepdims=True), a.std(1, keepdims=True) + 1e-6
    xt = torch.from_numpy((a - m) / s).unsqueeze(1).to(device)
    p = np.zeros((a.shape[0], N_DEPTHS), np.float32)
    with torch.no_grad():
        for k in range(0, a.shape[0], 512):
            p[k:k+512] = torch.sigmoid(model(xt[k:k+512])).cpu().numpy()
    return p.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)


def vol_bilstm(model, cache):
    d = np.load(cache); a = crop_or_pad(d["ascans"].astype(np.float32), SEQ_LEN)
    m, s = a.mean(1, keepdims=True), a.std(1, keepdims=True) + 1e-6
    xt = torch.from_numpy((a - m) / s).unsqueeze(1).to(device)
    p = np.zeros((a.shape[0], SEQ_LEN), np.float32)
    with torch.no_grad():
        for k in range(0, a.shape[0], 512):
            p[k:k+512] = torch.sigmoid(model(xt[k:k+512])).cpu().numpy()
    p50 = reduce_to_depth_bins(p, bin_to_time_index(slice_indices(d), SEQ_LEN), N_DEPTHS)
    return p50.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)


def unified_sample(name, vol_fn, cache_dir, depths_dir):
    cache = cache_dir / f"{name}.npz"; dpath = depths_dir / f"{name}_depths.npy"
    if not cache.exists() or not dpath.exists() or name not in SAMPLE_TO_STL:
        return None
    stl = STL_DIR / SAMPLE_TO_STL[name]
    if not stl.exists():
        return None
    depths = np.load(dpath).astype(np.float32)
    _, gt_meta = voxelize_gt(stl, depths); n_gt = len(gt_meta)
    if n_gt == 0:
        return None
    vol = vol_fn(cache)
    sweep = []
    for thr in THRESHOLDS:
        blobs = extract_pred_blobs(vol, thr, depths); nb = len(blobs)
        if nb == 0:
            sweep.append(dict(f1=0., p=0., r=0., nb=0)); continue
        best = best_alignment_and_match(gt_meta, blobs); nm = len(best["matches"])
        p = nm/nb; r = nm/n_gt; f1 = 2*p*r/(p+r) if p+r > 0 else 0.
        sweep.append(dict(f1=f1, p=p, r=r, nb=nb))
    br = max(sweep, key=lambda s: (s["f1"], -s["nb"]))
    return dict(sample=name, n_gt=int(n_gt), f1=float(br["f1"]), p=float(br["p"]), r=float(br["r"]))


def agg(rows):
    rows = [r for r in rows if r]
    if not rows: return None
    return dict(f1=float(np.mean([r["f1"] for r in rows])),
                p=float(np.mean([r["p"] for r in rows])),
                r=float(np.mean([r["r"] for r in rows])), n=len(rows))


def eval_config(label, ckpt, kind, c2, c1):
    if not ckpt.exists():
        print(f"  ! missing {ckpt}"); return None
    model = load_model(ckpt, kind)
    vol_fn = (lambda c: vol_cnn(model, c)) if kind == "cnn" else (lambda c: vol_bilstm(model, c))
    a2 = [unified_sample(n, vol_fn, c2, A2_DEPTHS)
          for n in sorted(set(TRAIN_SAMPLES+VAL_SAMPLES), key=lambda x:int(x.rstrip("b")))]
    a1 = [unified_sample(p.stem, vol_fn, c1, A1_DEPTHS)
          for p in sorted(c1.glob("*.npz"), key=lambda p:int(p.stem.rstrip("b")))] if c1.exists() else []
    out = dict(label=label,
               all=agg(a2),
               val=agg([r for r in a2 if r and r["sample"] in VAL_SAMPLES]),
               cross=agg(a1))
    a = out["all"]; v = out["val"]; cr = out["cross"]
    print(f"  {label:<16} all F1={a['f1']:.3f}(P{a['p']:.2f}/R{a['r']:.2f})"
          + (f"  VAL F1={v['f1']:.3f}" if v else "")
          + (f"  cross F1={cr['f1']:.3f}" if cr else ""))
    return out


def main():
    results = {}
    for label, ckpt, kind, c2, c1 in CONFIGS:
        r = eval_config(label, ckpt, kind, c2, c1)
        if r: results[label] = r

    def line(name, unf, bp):
        if unf and bp:
            print(f"{name:<14}{unf['all']['f1']:>9.3f}{bp['all']['f1']:>9.3f}{bp['all']['f1']-unf['all']['f1']:>+8.3f}"
                  f"   | cross {(unf['cross']['f1'] if unf['cross'] else float('nan')):.3f}→"
                  f"{(bp['cross']['f1'] if bp['cross'] else float('nan')):.3f}")

    print("\n" + "="*72)
    print("BAND-PASS vs UNFILTERED — unified per-blob F1 (atlanta2 all-void)")
    print("="*72)
    print(f"{'model':<14}{'unfilt':>9}{'bp':>9}{'Δ':>8}   | cross-domain")
    line("1D-CNN", results.get("1D-CNN  unfilt"), results.get("1D-CNN  bp"))
    line("U-Net-BiLSTM", results.get("BiLSTM  unfilt"), results.get("BiLSTM  bp"))
    print("\nC-scan U-Net (50-slice proxy, separate run): 0.561 → 0.615  (Δ +0.054)")
    (BDIR/"eval_bp_compare.json").write_text(json.dumps(results, indent=2, default=float))
    print(f"\nSaved: {BDIR}/eval_bp_compare.json")


if __name__ == "__main__":
    main()
