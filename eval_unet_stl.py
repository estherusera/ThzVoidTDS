"""
eval_unet_stl.py — Evaluate the STL-trained U-Net (unet_stl.pt, 50-slice) on the
unified STL metric (per-blob Hungarian + 8 mm gate, mesh.contains GT,
8-symmetry alignment) so it's directly comparable to:
  • unet_pseudo_2050.pt    (C-scan U-Net trained on manual labels, 2050-slice)
  • ascan_cnn_depth.pt     (A-scan 1D-CNN, unified eval in ascan_vs_stl.py)

Run
    thesis_env/bin/python eval_unet_stl.py
"""
import sys, json, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
sys.argv = [sys.argv[0], "predict", "atlanta2"]

from thz_slice_pipelinev2 import UNet
from void_characterize import (
    voxelize_gt, extract_pred_blobs, best_alignment_and_match,
    SAMPLE_TO_STL, STL_DIR, THRESHOLDS,
)

SLICES_DIR = Path("slices_v2_atlanta2")
MODEL_CKPT = Path("results_v2_atlanta2/unet_stl.pt")
OUT_JSON   = Path("results_v2_atlanta2/unet_stl_vs_stl.json")
OFFSETS    = [-2, -1, 0, 1, 2]

device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    ck = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    cfg = ck.get("config", {})
    model = UNet(in_channels=cfg.get("in_channels", 5),
                 base_filters=cfg.get("base_filters", 32)).to(device)
    model.load_state_dict(ck["model_state"]); model.eval()
    return model


def predict_volume(model, slices, offsets):
    n_s = slices.shape[0]
    inputs = np.zeros((n_s, len(offsets), 100, 100), dtype=np.float32)
    for i in range(n_s):
        for k, off in enumerate(offsets):
            inputs[i, k] = slices[int(np.clip(i + off, 0, n_s - 1))]
    x = torch.from_numpy(inputs).to(device)
    probs = np.zeros((n_s, 100, 100), dtype=np.float32)
    CHUNK = 32
    with torch.no_grad():
        for k in range(0, n_s, CHUNK):
            probs[k:k+CHUNK] = torch.sigmoid(
                model(x[k:k+CHUNK])).squeeze(1).cpu().numpy()
    return probs


def evaluate_sample(name, model):
    slices_p = SLICES_DIR / f"{name}_slices.npy"
    depths_p = SLICES_DIR / f"{name}_depths.npy"
    if not slices_p.exists() or not depths_p.exists():
        return None
    slices = np.load(slices_p)
    depths = np.load(depths_p).astype(np.float32)

    stl_path = STL_DIR / SAMPLE_TO_STL[name]
    if not stl_path.exists():
        return None
    gt_masks, gt_meta = voxelize_gt(stl_path, depths)
    n_gt = len(gt_meta)

    probs = predict_volume(model, slices, OFFSETS)

    sweep = []
    for thr in THRESHOLDS:
        blobs = extract_pred_blobs(probs, thr, depths)
        n_blob = len(blobs)
        if n_gt == 0 or n_blob == 0:
            sweep.append(dict(thr=float(thr), n_blobs=n_blob, matches=0,
                              precision=0.0, recall=0.0, f1=0.0, sym=-1))
            continue
        best = best_alignment_and_match(gt_meta, blobs)
        nm = len(best["matches"])
        prec = nm / n_blob; rec = nm / n_gt
        f1   = (2*prec*rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        sweep.append(dict(thr=float(thr), n_blobs=n_blob, matches=nm,
                          precision=prec, recall=rec, f1=f1,
                          sym=best["sym_id"]))

    if n_gt == 0:
        best_row = min(sweep, key=lambda r: r["n_blobs"])
    else:
        best_row = max(sweep, key=lambda r: (r["f1"], -r["n_blobs"]))

    return dict(
        sample=name,
        n_gt=int(n_gt),
        n_pred_blobs=int(best_row["n_blobs"]),
        best_threshold=float(best_row["thr"]),
        best_symmetry=int(best_row["sym"]),
        f1=float(best_row["f1"]),
        precision=float(best_row["precision"]),
        recall=float(best_row["recall"]),
        sweep=sweep,
    )


def main():
    if not MODEL_CKPT.exists():
        sys.exit(f"missing {MODEL_CKPT} — run train_unet_stl.py first")
    model = load_model()
    print(f"Loaded {MODEL_CKPT.name}    device={device}")

    names = sorted([k for k in SAMPLE_TO_STL if k != "5b"],
                   key=lambda x: int(x.rstrip("b")))
    results = []
    print(f"\n{'sample':>6s} {'n_gt':>5s} {'n_pr':>5s} {'thr':>5s} {'sym':>4s} "
          f"{'F1':>6s} {'P':>6s} {'R':>6s}")
    for name in names:
        r = evaluate_sample(name, model)
        if r is None:
            print(f"{name:>6s}  (skip)"); continue
        results.append(r)
        print(f"{name:>6s} {r['n_gt']:>5d} {r['n_pred_blobs']:>5d} "
              f"{r['best_threshold']:>5.2f} {r['best_symmetry']:>4d} "
              f"{r['f1']:>6.3f} {r['precision']:>6.3f} {r['recall']:>6.3f}")

    void_rows = [r for r in results if r["n_gt"] > 0]
    if void_rows:
        f1m = float(np.mean([r["f1"] for r in void_rows]))
        pm  = float(np.mean([r["precision"] for r in void_rows]))
        rm  = float(np.mean([r["recall"] for r in void_rows]))
        print(f"\n  Mean over {len(void_rows)} void samples: "
              f"F1={f1m:.3f}  P={pm:.3f}  R={rm:.3f}")

    with open(OUT_JSON, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
