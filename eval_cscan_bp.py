"""
eval_cscan_bp.py — C-scan U-Net band-pass proxy (Option A)
==========================================================
Scores the 50-slice C-scan U-Net trained on band-pass-filtered slices
(unet_50_bp.pt) vs its unfiltered twin (unet_50_nofilt.pt) through the SAME
unified per-blob harness used for the A-scan models, giving an honest
filtered-vs-unfiltered C-scan delta.

NOTE: 50-slice resolution — absolute F1 will sit below the 0.437 headline (which
uses the 2050-slice pseudo model); the DELTA is the meaningful number here.

Run: thesis_env/bin/python eval_cscan_bp.py
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import void_characterize as vc
vc.device = "cpu"   # don't contend with the BiLSTM training on MPS
from void_characterize import (UNet, predict_full_volume, voxelize_gt,
                               extract_pred_blobs, best_alignment_and_match,
                               SAMPLE_TO_STL, STL_DIR, THRESHOLDS)

OFFSETS = [-2, -1, 0, 1, 2]
VOID_SAMPLES = ["1","2","3","4","5","6","7","8","9","11","12","13","15"]  # all-void (10,14 empty)
RUNS = {
    "unfiltered": dict(ckpt="results_v2_atlanta2/unet_50_nofilt.pt", slices="slices_v2_atlanta2"),
    "band-pass":  dict(ckpt="results_v2_atlanta2/unet_50_bp.pt",     slices="slices_v2_atlanta2_bp"),
}


def load_unet(ckpt):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = c["config"]
    m = UNet(in_channels=cfg["in_channels"], base_filters=cfg["base_filters"])
    m.load_state_dict(c["model_state"]); m.to("cpu").eval()
    return m


def eval_run(ckpt, slices_dir):
    model = load_unet(ckpt)
    sdir = Path(slices_dir)
    rows = []
    for name in VOID_SAMPLES:
        sp = sdir / f"{name}_slices.npy"
        if not sp.exists() or name not in SAMPLE_TO_STL:
            continue
        stl = STL_DIR / SAMPLE_TO_STL[name]
        if not stl.exists():
            continue
        slices = np.load(sp)
        depths = np.load(sdir / f"{name}_depths.npy").astype(np.float32)
        gt_masks, gt_meta = voxelize_gt(stl, depths); n_gt = len(gt_meta)
        if n_gt == 0:
            continue
        vol = predict_full_volume(model, slices, OFFSETS)   # (50,100,100)
        sweep = []
        for thr in THRESHOLDS:
            blobs = extract_pred_blobs(vol, thr, depths); nb = len(blobs)
            if nb == 0:
                sweep.append(dict(f1=0., p=0., r=0., nb=0)); continue
            best = best_alignment_and_match(gt_meta, blobs); nm = len(best["matches"])
            p = nm/nb; r = nm/n_gt; f1 = 2*p*r/(p+r) if p+r > 0 else 0.
            sweep.append(dict(f1=f1, p=p, r=r, nb=nb))
        br = max(sweep, key=lambda s: (s["f1"], -s["nb"]))
        rows.append(dict(sample=name, n_gt=n_gt, **br))
        print(f"  {name}: n_gt={n_gt} F1={br['f1']:.3f} P={br['p']:.3f} R={br['r']:.3f}")
    f1 = float(np.mean([r["f1"] for r in rows])); p = float(np.mean([r["p"] for r in rows]))
    r_ = float(np.mean([r["r"] for r in rows]))
    return dict(f1=f1, p=p, r=r_, n=len(rows), per_sample=rows)


def main():
    out = {}
    for tag, cfg in RUNS.items():
        if not Path(cfg["ckpt"]).exists():
            print(f"! missing {cfg['ckpt']}"); continue
        print(f"\n=== C-scan U-Net (50-slice) — {tag} ===")
        out[tag] = eval_run(cfg["ckpt"], cfg["slices"])

    print("\n" + "="*60)
    print("C-scan U-Net (50-slice) — band-pass vs unfiltered (per-blob)")
    print("="*60)
    print(f"{'run':<14}{'F1':>8}{'P':>8}{'R':>8}{'n':>4}")
    for tag in ("unfiltered", "band-pass"):
        if tag in out:
            o = out[tag]; print(f"{tag:<14}{o['f1']:>8.3f}{o['p']:>8.3f}{o['r']:>8.3f}{o['n']:>4d}")
    if "unfiltered" in out and "band-pass" in out:
        print(f"{'Δ (bp−unf)':<14}{out['band-pass']['f1']-out['unfiltered']['f1']:>+8.3f}")
    Path("runs/unet_bilstm/eval_cscan_bp.json").write_text(json.dumps(out, indent=2, default=float))
    print("\nSaved: runs/unet_bilstm/eval_cscan_bp.json")
    print("NOTE: 50-slice resolution; absolute F1 is below the 0.437 (2050-slice pseudo) headline.")


if __name__ == "__main__":
    main()
