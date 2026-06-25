"""
eval_ascan_unet_bilstm.py — unified per-blob evaluation of the U-Net-BiLSTM
===========================================================================
Reconstructs a (50, NY, NX) prediction volume from the model's per-time-point
A-scan masks (reducing 2048→50 via the same slice-index map used to build the
labels), then scores it with the EXISTING unified per-blob harness
(voxelize_gt + extract_pred_blobs + 8-symmetry Hungarian matching) so the F1 is
directly comparable to the 1D-CNN depth classifier (0.703) and the C-scan U-Net
(0.44). Also reports the per-time-point (2048) F1 to expose the per-point vs
per-blob metric gap, and a cross-domain pass on atlanta1.

Run:
    thesis_env/bin/python eval_ascan_unet_bilstm.py
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

_ARGV = sys.argv[1:]
sys.path.insert(0, str(Path(__file__).parent))
from ascan_unet_bilstm import (UNetBiLSTM1D, SEQ_LEN, N_DEPTHS,
                               slice_indices, bin_to_time_index,
                               reduce_to_depth_bins, labels_2048, crop_or_pad)
from void_characterize import (voxelize_gt, extract_pred_blobs,
                               best_alignment_and_match,
                               SAMPLE_TO_STL, STL_DIR, NX, NY, THRESHOLDS)
from ascan_classifier import (CACHE_ATL2, CACHE_ATL1,
                              TRAIN_SAMPLES, VAL_SAMPLES, NOVOID_SANITY)

CKPT       = Path("runs/unet_bilstm/unet_bilstm_best.pt")
DEPTHS_ATL2 = Path("slices_v2_atlanta2")
DEPTHS_ATL1 = Path("slices_v2")
OUT_DIR    = Path("runs/unet_bilstm")
device     = "cpu"   # match ascan_vs_stl; tiny model, avoids MPS pool quirks


def load_model():
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    model = UNetBiLSTM1D(base=cfg.get("base", 16),
                         bilstm_deep_skips=cfg.get("bilstm", True)).to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    return model, ckpt


def predict(model, cache_path):
    """Return ((50,NY,NX) prob volume, per-point TP/FP/FN vs 2048 labels)."""
    d   = np.load(cache_path)
    a   = crop_or_pad(d["ascans"].astype(np.float32), SEQ_LEN)        # (N,2048)
    m, s = a.mean(1, keepdims=True), a.std(1, keepdims=True) + 1e-6
    xt  = torch.from_numpy(((a - m) / s)).unsqueeze(1).to(device)
    probs = np.zeros((a.shape[0], SEQ_LEN), dtype=np.float32)
    with torch.no_grad():
        for k in range(0, a.shape[0], 512):
            probs[k:k+512] = torch.sigmoid(model(xt[k:k+512])).cpu().numpy()

    kmap = bin_to_time_index(slice_indices(d), SEQ_LEN)
    # per-point F1 vs the 2048 labels
    lab2048 = np.zeros_like(probs)
    valid = kmap >= 0
    lab2048[:, valid] = d["labels_3d"][:, kmap[valid]]
    pred = probs > 0.5
    pp = dict(tp=float((pred & (lab2048 > 0)).sum()),
              fp=float((pred & (lab2048 == 0)).sum()),
              fn=float((~pred & (lab2048 > 0)).sum()))

    prob50 = reduce_to_depth_bins(probs, kmap, N_DEPTHS)              # (N,50)
    vol = prob50.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)         # (50,NY,NX)
    return vol, pp


def unified_eval_sample(name, model, cache_dir, depths_dir):
    cache = cache_dir / f"{name}.npz"
    dpath = depths_dir / f"{name}_depths.npy"
    if not cache.exists() or not dpath.exists() or name not in SAMPLE_TO_STL:
        return None
    stl = STL_DIR / SAMPLE_TO_STL[name]
    if not stl.exists():
        return None
    depths = np.load(dpath).astype(np.float32)
    gt_masks, gt_meta = voxelize_gt(stl, depths)
    n_gt = len(gt_meta)

    vol, pp = predict(model, cache)

    sweep = []
    for thr in THRESHOLDS:
        blobs = extract_pred_blobs(vol, thr, depths)
        nb = len(blobs)
        if n_gt == 0 or nb == 0:
            sweep.append(dict(thr=float(thr), n_blobs=nb, f1=0.0, precision=0.0, recall=0.0, sym=-1)); continue
        best = best_alignment_and_match(gt_meta, blobs)
        nm = len(best["matches"]); prec = nm/nb; rec = nm/n_gt
        f1 = 2*prec*rec/(prec+rec) if prec+rec > 0 else 0.0
        sweep.append(dict(thr=float(thr), n_blobs=nb, f1=f1, precision=prec, recall=rec, sym=best["sym_id"]))
    best_row = (min(sweep, key=lambda r: r["n_blobs"]) if n_gt == 0
                else max(sweep, key=lambda r: (r["f1"], -r["n_blobs"])))
    return dict(sample=name, n_gt=int(n_gt),
                f1_unified=float(best_row["f1"]), p_unified=float(best_row["precision"]),
                r_unified=float(best_row["recall"]), best_threshold=float(best_row["thr"]),
                best_symmetry=int(best_row["sym"]), n_pred_blobs=int(best_row["n_blobs"]),
                pp=pp)


def agg_unified(rows):
    rows = [r for r in rows if r["n_gt"] > 0]
    if not rows: return None
    return dict(f1=np.mean([r["f1_unified"] for r in rows]),
                p=np.mean([r["p_unified"] for r in rows]),
                r=np.mean([r["r_unified"] for r in rows]), n=len(rows))


def perpoint_f1(rows):
    tp = sum(r["pp"]["tp"] for r in rows); fp = sum(r["pp"]["fp"] for r in rows); fn = sum(r["pp"]["fn"] for r in rows)
    prec = tp/max(tp+fp, 1); rec = tp/max(tp+fn, 1)
    return 2*prec*rec/max(prec+rec, 1e-9), prec, rec


def main():
    if not CKPT.exists():
        sys.exit(f"missing checkpoint {CKPT} — train first")
    model, ckpt = load_model()
    print(f"Loaded {CKPT.name} (epoch {ckpt.get('epoch','?')}, "
          f"val per-point F1 {ckpt.get('val',{}).get('f1',float('nan')):.3f})")

    # ── atlanta2 same-domain ──
    a2 = []
    for name in sorted(set(TRAIN_SAMPLES+VAL_SAMPLES+NOVOID_SANITY), key=lambda x:int(x.rstrip("b"))):
        r = unified_eval_sample(name, model, CACHE_ATL2, DEPTHS_ATL2)
        if r is None: continue
        r["split"] = ("VAL" if name in VAL_SAMPLES else "NOVOID" if name in NOVOID_SANITY else "TRAIN")
        a2.append(r)
        print(f"  a2 {name:>3} [{r['split']:>6}] n_gt={r['n_gt']} blobs={r['n_pred_blobs']} "
              f"thr={r['best_threshold']:.2f} F1u={r['f1_unified']:.3f} "
              f"P={r['p_unified']:.3f} R={r['r_unified']:.3f}")

    # ── atlanta1 cross-domain ──
    a1 = []
    a1_names = sorted([p.stem for p in CACHE_ATL1.glob("*.npz")], key=lambda x:int(x.rstrip("b")))
    for name in a1_names:
        r = unified_eval_sample(name, model, CACHE_ATL1, DEPTHS_ATL1)
        if r is None: continue
        r["split"] = "CROSSDOMAIN"; a1.append(r)
        print(f"  a1 {name:>3} [cross] n_gt={r['n_gt']} blobs={r['n_pred_blobs']} "
              f"F1u={r['f1_unified']:.3f} P={r['p_unified']:.3f} R={r['r_unified']:.3f}")

    val   = agg_unified([r for r in a2 if r["split"] == "VAL"])
    train = agg_unified([r for r in a2 if r["split"] == "TRAIN"])
    allv  = agg_unified([r for r in a2 if r["split"] in ("TRAIN","VAL")])
    cross = agg_unified(a1)
    pp_f1_a2, pp_p_a2, pp_r_a2 = perpoint_f1([r for r in a2 if r["n_gt"] > 0])

    # ── existing baselines for the 3-way table ──
    def load_existing_unified():
        f = Path("results_v2_atlanta2/ascan_classifier/vs_stl_unified/summary_unified.json")
        if not f.exists(): return None
        rows = [r for r in json.load(open(f)) if r["n_gt"] > 0]
        return dict(f1=np.mean([r["f1_unified"] for r in rows]),
                    p=np.mean([r["p_unified"] for r in rows]),
                    r=np.mean([r["r_unified"] for r in rows]), n=len(rows))
    def load_cscan():
        f = Path("results_v2_atlanta2/void_char/aggregate.json")
        if not f.exists(): return None
        d = json.load(open(f))
        return d
    cnn = load_existing_unified(); cscan = load_cscan()

    print("\n" + "="*78)
    print("UNIFIED PER-BLOB F1 — 3-WAY COMPARISON (same matcher: mesh + 8-sym + 8mm gate)")
    print("="*78)
    print(f"{'model':<34}{'F1':>7}{'P':>7}{'R':>7}   domain/notes")
    if cnn:
        print(f"{'A-scan 1D-CNN (depth classifier)':<34}{cnn['f1']:>7.3f}{cnn['p']:>7.3f}{cnn['r']:>7.3f}   atlanta2 all-void (existing)")
    if cscan:
        cf = cscan.get('mean_f1', cscan.get('f1', float('nan')))
        cp = cscan.get('mean_precision', cscan.get('precision', float('nan')))
        cr = cscan.get('mean_recall', cscan.get('recall', float('nan')))
        print(f"{'C-scan U-Net (manual labels)':<34}{cf:>7.3f}{cp:>7.3f}{cr:>7.3f}   atlanta2 all-void (existing)")
    if allv:
        print(f"{'U-Net-BiLSTM (NEW, all-void)':<34}{allv['f1']:>7.3f}{allv['p']:>7.3f}{allv['r']:>7.3f}   atlanta2 TRAIN+VAL")
    if val:
        print(f"{'U-Net-BiLSTM (NEW, held-out VAL)':<34}{val['f1']:>7.3f}{val['p']:>7.3f}{val['r']:>7.3f}   atlanta2 VAL only")
    if cross:
        print(f"{'U-Net-BiLSTM (NEW, cross-domain)':<34}{cross['f1']:>7.3f}{cross['p']:>7.3f}{cross['r']:>7.3f}   atlanta1 (held out)")

    print("\n── per-point vs per-blob gap (U-Net-BiLSTM, atlanta2 void samples) ──")
    print(f"  per-point (2048) F1 = {pp_f1_a2:.3f}  (P={pp_p_a2:.3f} R={pp_r_a2:.3f})")
    if allv:
        print(f"  per-blob  (unified) F1 = {allv['f1']:.3f}  →  gap = {allv['f1']-pp_f1_a2:+.3f}")

    out = dict(atlanta2=a2, atlanta1=a1,
               agg=dict(train=train, val=val, all=allv, cross=cross),
               perpoint=dict(f1=pp_f1_a2, p=pp_p_a2, r=pp_r_a2),
               baselines=dict(ascan_1dcnn=cnn, cscan_unet=cscan))
    (OUT_DIR/"eval_unified.json").write_text(json.dumps(out, indent=2, default=float))
    print(f"\nSaved: {OUT_DIR}/eval_unified.json")


if __name__ == "__main__":
    main()
