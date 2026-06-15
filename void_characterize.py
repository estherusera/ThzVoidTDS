"""
void_characterize.py — Void characterization pipeline (all 15 atlanta2 samples).

For each sample:
  1. Voxelize STL ground truth voids at the 2050-slice THz grid.
  2. Run unet_pseudo_2050.pt over the sample → (2050, 100, 100) prob volume.
  3. Threshold sweep on 2D max-projection → 2D connected components → per-blob depth.
  4. Try all 8 in-plane symmetries (4 rotations × 2 flips), pick min-cost alignment.
  5. Hungarian-match GT void centroids ↔ predicted centroids (gated at 8 mm).
  6. Report: per-void position/depth error and pred/GT volume ratio.

Sample → STL mapping (off-by-one above sample 7 because CV07 was not made):
  1→CV01, 2→CV02, 3→CV03, 4→CV04, 5→CV05, 6→CV06, 7→CV08, 8→CV09, 9→CV10,
  10→CV11, 11→CV12, 12→CV13, 13→CV14, 14→CV15, 15→CV16.

Watertight STLs use per-component contains() voxelization.
Non-watertight components fall back to axis-aligned bbox metadata only.

Usage:
    thesis_env/bin/python void_characterize.py 4        # one sample
    thesis_env/bin/python void_characterize.py all      # all 15 + aggregate
"""
import sys, json, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
import trimesh
from pathlib import Path
from scipy.ndimage import label as cc_label
from scipy.optimize import linear_sum_assignment

# Capture CLI arg BEFORE rewriting sys.argv for the pipeline import
_CLI_TARGET = sys.argv[1] if len(sys.argv) > 1 else None

sys.argv = [sys.argv[0], "predict", "atlanta2"]
sys.path.insert(0, ".")
from thz_slice_pipelinev2 import UNet

# ── config ───────────────────────────────────────────────────────────────────
STL_DIR     = Path("AllSamples/stl")
SLICES_DIR  = Path("slices_v2_atlanta2_2050")
MODEL_CKPT  = Path("results_v2_atlanta2/unet_pseudo_2050.pt")
OUT_DIR     = Path("results_v2_atlanta2/void_char")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PX_MM            = 0.2
NX = NY          = 100
MIN_AREA_PX      = 100        # min 2D blob area to keep (~0.4 mm² floor)
THRESHOLDS       = np.arange(0.30, 0.96, 0.05)
DIST_GATE_MM     = 8.0
# Coarser depth grid for GT voxelization (matching happens in mm anyway)
GT_DEPTH_PTS     = 200
# Skip mesh.contains() for components above this face count: trimesh's
# ray-casting becomes O(faces × points) and memory-leaks badly past ~2k faces.
CONTAINS_MAX_FACES = 2000
# Drop GT voids whose bbox or mesh volume is below this floor — they are
# mesh fragments (sub-mm edges, single faces) from non-watertight STLs,
# not actual void cavities. THz resolution makes anything < ~0.05 mm³ undetectable.
MIN_GT_VOLUME_MM3 = 0.05

# Scan sample → STL filename
SAMPLE_TO_STL = {
    "1":  "CV01.stl",
    "2":  "CV02.stl",
    "3":  "CV03.stl",
    "4":  "CV04.stl",
    "5":  "CV05.stl",
    "6":  "CV06.stl",
    "7":  "CV08.stl",
    "8":  "SampleRayas_CV09.stl",
    "9":  "SampleRayasDiagonal_CV10.stl",
    "10": "ManyStripes_CV11.stl",
    "11": "PrismVoids_CV12.stl",
    "12": "random_voids_CV13.stl",
    "13": "Cilindros_CV14.stl",
    "14": "NoVoid_CV15.stl",
    "15": "Inclinedplanes_CV16.stl",
}

device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")


# ── Stage 1: voxelize STL voids ──────────────────────────────────────────────
def voxelize_gt(stl_path, depths_mm, px_mm=PX_MM, ny=NY, nx=NX):
    """Return (void_masks, void_meta) for one STL.

    Watertight components → per-mesh contains() voxelization (gives mask + meta).
    Non-watertight components → axis-aligned bbox metadata only (no mask).

    Voids whose centroid is outside the THz depth range are dropped.
    """
    mesh  = trimesh.load(stl_path, force="mesh")
    comps = mesh.split(only_watertight=False)
    if not comps:
        return [], []

    # Identify outer block = largest bbox volume
    bbox_vols = np.array([np.prod(c.bounding_box.extents) for c in comps])
    block_i   = int(np.argmax(bbox_vols))
    block     = comps[block_i]
    voids     = [c for i, c in enumerate(comps) if i != block_i]

    if not voids:
        return [], []     # e.g. NoVoid sample

    shift = -block.bounds[0]

    # Per-axis voxel-centre coords for the full grid
    xs = (np.arange(nx) + 0.5) * px_mm     # (nx,)
    ys = (np.arange(ny) + 0.5) * px_mm     # (ny,)

    z_min, z_max = float(depths_mm[0]), float(depths_mm[-1])
    nz = len(depths_mm)
    void_masks, void_meta = [], []

    for vi, v in enumerate(voids):
        v_shift = v.copy()
        v_shift.apply_translation(shift)
        bb_min  = v_shift.bounds[0]
        bb_max  = v_shift.bounds[1]
        cx_bb, cy_bb, cz_bb = ((bb_min + bb_max) / 2).tolist()

        # Derive everything from a SMALL subgrid; never materialise the full
        # (2050, 100, 100) mask (would be 20 MB × n_voids and blow memory).
        proj2d = np.zeros((ny, nx), dtype=bool)
        inside_sub = None

        if v_shift.is_watertight and len(v_shift.faces) <= CONTAINS_MAX_FACES:
            x_lo = max(0, int(bb_min[0] / px_mm))
            x_hi = min(nx, int(np.ceil(bb_max[0] / px_mm)) + 1)
            y_lo = max(0, int(bb_min[1] / px_mm))
            y_hi = min(ny, int(np.ceil(bb_max[1] / px_mm)) + 1)
            z_in = (depths_mm >= bb_min[2]) & (depths_mm <= bb_max[2])

            if x_hi > x_lo and y_hi > y_lo and z_in.any():
                z_idx = np.where(z_in)[0]
                xs_v  = xs[x_lo:x_hi]
                ys_v  = ys[y_lo:y_hi]
                zs_v  = depths_mm[z_idx]
                X, Y, Z = np.meshgrid(xs_v, ys_v, zs_v, indexing="xy")
                pts_v = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
                try:
                    sub = (v_shift.contains(pts_v)
                           .reshape(len(ys_v), len(xs_v), len(zs_v))
                           .transpose(2, 0, 1))           # (nz_v, ny_v, nx_v)
                    if sub.any():
                        inside_sub = (sub, z_idx, y_lo, y_hi, x_lo, x_hi)
                except Exception:
                    inside_sub = None

        if inside_sub is not None:
            sub, z_idx, y_lo, y_hi, x_lo, x_hi = inside_sub
            # Centroid from SUBGRID coords (cheap)
            zyx = np.array(np.where(sub))                 # (3, N)
            n_vox = int(zyx.shape[1])
            cz_idx = int(round(zyx[0].mean()))
            cy_idx = int(round(zyx[1].mean()))
            cx_idx = int(round(zyx[2].mean()))
            cz_mm = float(depths_mm[z_idx[cz_idx]])
            cy_mm = float(ys[y_lo + cy_idx])
            cx_mm = float(xs[x_lo + cx_idx])
            z_range = (float(depths_mm[z_idx[zyx[0].min()]]),
                       float(depths_mm[z_idx[zyx[0].max()]]))
            dz_mm = float(np.diff(depths_mm).mean())
            vol_mm3 = float(n_vox * (px_mm ** 2) * dz_mm)
            # 2D z-projection of the subgrid → into the small 2D array
            proj_sub = sub.any(axis=0)
            proj2d[y_lo:y_hi, x_lo:x_hi] = proj_sub
            source = "mesh"
            sub = None
        else:
            # Non-watertight or contains failed — use axis-aligned bbox
            cx_mm, cy_mm, cz_mm = float(cx_bb), float(cy_bb), float(cz_bb)
            z_range = (float(bb_min[2]), float(bb_max[2]))
            vol_mm3 = float(np.prod(bb_max - bb_min))
            x_lo = max(0, int(bb_min[0] / px_mm))
            x_hi = min(nx, int(np.ceil(bb_max[0] / px_mm)) + 1)
            y_lo = max(0, int(bb_min[1] / px_mm))
            y_hi = min(ny, int(np.ceil(bb_max[1] / px_mm)) + 1)
            proj2d[y_lo:y_hi, x_lo:x_hi] = True
            source  = "bbox"

        # Drop voids whose centroid is outside the THz depth range
        if cz_mm < z_min or cz_mm > z_max:
            continue
        # Drop spurious mesh fragments (sub-mm edges) — they are not real voids
        if vol_mm3 < MIN_GT_VOLUME_MM3:
            continue

        void_meta.append(dict(
            id=int(vi),
            source=source,
            centroid_mm=(cx_mm, cy_mm, cz_mm),
            bbox_mm=tuple(float(b) for b in (bb_max - bb_min)),
            volume_mm3=vol_mm3,
            depth_range_mm=z_range,
        ))
        void_masks.append(proj2d)

    return void_masks, void_meta


# ── Stage 2: model predictions ───────────────────────────────────────────────
def predict_full_volume(model, slices, offsets):
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
            logits = model(x[k:k + CHUNK])
            probs[k:k + CHUNK] = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    return probs


# ── Stage 3: extract pred blobs (2D projection + per-blob depth) ─────────────
def extract_pred_blobs(prob_volume, threshold, depths_mm,
                       px_mm=PX_MM, min_area_px=MIN_AREA_PX):
    proj2d = prob_volume.max(axis=0)
    mask2d = (proj2d > threshold).astype(np.uint8)
    labels, n = cc_label(mask2d, structure=np.ones((3, 3), dtype=bool))
    dz_mm = float(np.diff(depths_mm).mean())

    blobs = []
    for bid in range(1, n + 1):
        ys, xs = np.where(labels == bid)
        if ys.size < min_area_px:
            continue
        cy_mm = float((ys.mean() + 0.5) * px_mm)
        cx_mm = float((xs.mean() + 0.5) * px_mm)
        profile = prob_volume[:, ys, xs].max(axis=1)
        depth_ok = profile > threshold
        if not depth_ok.any():
            continue
        z_idx = np.where(depth_ok)[0]
        cz_mm = float(np.average(depths_mm[z_idx], weights=profile[z_idx]))
        z_range = (float(depths_mm[z_idx.min()]),
                   float(depths_mm[z_idx.max()]))
        z_extent = z_range[1] - z_range[0]
        area_mm2 = ys.size * (px_mm ** 2)
        vol_mm3  = area_mm2 * (z_extent if z_extent > 0 else dz_mm)
        blobs.append(dict(
            area_mm2=area_mm2,
            centroid_mm=(cx_mm, cy_mm, cz_mm),
            bbox_mm=(
                float((xs.max() - xs.min() + 1) * px_mm),
                float((ys.max() - ys.min() + 1) * px_mm),
                float(z_extent),
            ),
            volume_mm3=vol_mm3,
            depth_range_mm=z_range,
        ))
    return blobs


# ── Stage 4: rotation alignment + matching ───────────────────────────────────
def apply_symmetry(centroid_xyz, sym_id, box_mm=20.0):
    x, y, z = centroid_xyz
    rot = sym_id // 2
    mir = sym_id % 2
    if mir:
        x = box_mm - x
    if rot == 1:
        x, y = y, box_mm - x
    elif rot == 2:
        x, y = box_mm - x, box_mm - y
    elif rot == 3:
        x, y = box_mm - y, x
    return (x, y, z)


def best_alignment_and_match(gt_meta, pred_meta, dist_gate_mm=DIST_GATE_MM):
    if not gt_meta or not pred_meta:
        return dict(sym_id=0, total_cost=float("inf"), matches=[],
                    G_aligned=[m["centroid_mm"] for m in gt_meta])
    P = np.array([m["centroid_mm"] for m in pred_meta])
    best = None
    for sym_id in range(8):
        G = np.array([apply_symmetry(m["centroid_mm"], sym_id) for m in gt_meta])
        C = np.linalg.norm(G[:, None, :] - P[None, :, :], axis=-1)
        C_gated = np.where(C <= dist_gate_mm, C, 1e6)
        row, col = linear_sum_assignment(C_gated)
        good = C_gated[row, col] <= dist_gate_mm
        matches = [(int(r), int(c), float(C[r, c]))
                   for r, c, g in zip(row, col, good) if g]
        total_cost = (sum(d for _, _, d in matches)
                      + dist_gate_mm * (len(gt_meta) - len(matches)))
        if best is None or total_cost < best["total_cost"]:
            best = dict(sym_id=sym_id, total_cost=total_cost,
                        matches=matches, G_aligned=G.tolist())
    return best


# ── per-sample driver ────────────────────────────────────────────────────────
def characterize(sample, model, offsets):
    print(f"\n{'='*70}\nSample {sample}  →  {SAMPLE_TO_STL[sample]}\n{'='*70}")
    stl_path = STL_DIR / SAMPLE_TO_STL[sample]

    slices_p = SLICES_DIR / f"{sample}_slices.npy"
    depths_p = SLICES_DIR / f"{sample}_depths.npy"
    if not slices_p.exists():
        print(f"  ✗ {slices_p} missing"); return None
    slices = np.load(slices_p)
    depths = np.load(depths_p)

    # Stage 1 — use a coarser depth grid for GT to bound memory
    t = time.time()
    gt_depths = np.linspace(depths[0], depths[-1], GT_DEPTH_PTS, dtype=np.float32)
    gt_masks, gt_meta = voxelize_gt(stl_path, gt_depths)
    n_gt = len(gt_meta)
    n_mesh   = sum(1 for m in gt_meta if m["source"] == "mesh")
    n_bbox   = n_gt - n_mesh
    print(f"  Stage 1: {n_gt} GT voids  ({n_mesh} mesh, {n_bbox} bbox)  "
          f"in {time.time()-t:.1f}s")

    # Stage 2
    t = time.time()
    probs = predict_full_volume(model, slices, offsets)
    print(f"  Stage 2: model on 2050 slices in {time.time()-t:.1f}s "
          f"(prob range {probs.min():.3f}–{probs.max():.3f})")

    # Stage 3: threshold sweep
    sweep = []
    for thr in THRESHOLDS:
        blobs = extract_pred_blobs(probs, thr, depths)
        if not blobs:
            sweep.append(dict(thr=float(thr), n_blobs=0, matches=0,
                              precision=0.0, recall=0.0, f1=0.0,
                              sym=-1, mean_dist_mm=None))
            continue
        best = best_alignment_and_match(gt_meta, blobs)
        nm   = len(best["matches"])
        prec = nm / len(blobs)
        rec  = nm / n_gt if n_gt else 0.0
        f1   = (2*prec*rec / (prec+rec)) if (prec+rec) > 0 else 0.0
        md   = float(np.mean([d for _, _, d in best["matches"]])) if nm else None
        sweep.append(dict(thr=float(thr), n_blobs=len(blobs), matches=nm,
                          precision=prec, recall=rec, f1=f1,
                          sym=best["sym_id"], mean_dist_mm=md))

    # Pick best threshold by F1 (tie-break: fewer blobs)
    if n_gt == 0:
        # No-void sample: best threshold = one that produces fewest predicted blobs
        best_row = min(sweep, key=lambda r: r["n_blobs"])
    else:
        best_row = max(sweep, key=lambda r: (r["f1"], -r["n_blobs"]))
    best_thr = best_row["thr"]
    print(f"  Stage 3: best threshold {best_thr:.2f}  F1 {best_row['f1']:.3f}  "
          f"P {best_row['precision']:.3f}  R {best_row['recall']:.3f}  "
          f"({best_row['n_blobs']} blobs)")

    # Stage 4: final match table at best threshold
    pred_meta = extract_pred_blobs(probs, best_thr, depths)
    best      = best_alignment_and_match(gt_meta, pred_meta)
    sym       = best["sym_id"]
    G_aligned = best["G_aligned"]

    match_rows = []
    for gi, pi, d in sorted(best["matches"], key=lambda x: x[2]):
        gv = gt_meta[gi]["volume_mm3"]
        pv = pred_meta[pi]["volume_mm3"]
        ratio = pv / gv if gv > 0 else float("nan")
        match_rows.append(dict(
            gt_id=gi, pred_id=pi, dist_mm=float(d),
            gt_centroid_mm=tuple(float(c) for c in G_aligned[gi]),
            pred_centroid_mm=tuple(float(c) for c in pred_meta[pi]["centroid_mm"]),
            depth_error_mm=float(pred_meta[pi]["centroid_mm"][2] - G_aligned[gi][2]),
            gt_volume_mm3=gv, pred_volume_mm3=pv, vol_ratio=ratio,
        ))

    # Per-sample output dir
    sd = OUT_DIR / sample
    sd.mkdir(exist_ok=True)
    out_json = sd / "result.json"
    with open(out_json, "w") as fh:
        json.dump({
            "sample": sample,
            "stl": str(stl_path),
            "best_threshold": best_thr,
            "best_symmetry": sym,
            "n_gt": n_gt,
            "n_pred": len(pred_meta),
            "precision": best_row["precision"],
            "recall": best_row["recall"],
            "f1": best_row["f1"],
            "mean_centroid_distance_mm": best_row["mean_dist_mm"],
            "gt_voids": gt_meta,
            "pred_voids": pred_meta,
            "matches": match_rows,
            "sweep": sweep,
        }, fh, indent=2, default=float)

    # Per-sample summary plot
    _plot_sample(sample, depths, probs, best_thr, gt_meta, gt_masks,
                 pred_meta, sym, G_aligned, match_rows, sweep, sd)

    return dict(
        sample=sample,
        n_gt=n_gt,
        n_pred=len(pred_meta),
        f1=best_row["f1"],
        precision=best_row["precision"],
        recall=best_row["recall"],
        mean_dist_mm=best_row["mean_dist_mm"],
        best_threshold=best_thr,
        best_sym=sym,
        match_rows=match_rows,
    )


# ── visualisation ────────────────────────────────────────────────────────────
def _plot_sample(name, depths, probs, thr, gt_meta, gt_masks, pred_meta,
                 sym, G_aligned, match_rows, sweep, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), dpi=140)
    fig.patch.set_facecolor("white")
    pred_proj = probs.max(axis=0) > thr
    gt_proj   = np.zeros((NY, NX), dtype=bool)
    for m in gt_masks:          # m is the per-void 2D z-projection (ny, nx)
        if m is not None:
            gt_proj |= m

    axes[0, 0].imshow(gt_proj, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 0].set_title("STL voids (z-projection)", fontsize=9, fontweight="bold")
    axes[0, 1].imshow(pred_proj, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 1].set_title(f"Predicted (thr={thr:.2f})", fontsize=9, fontweight="bold")

    axes[0, 2].imshow(np.full((NY, NX), 0.95), cmap="gray", vmin=0, vmax=1)
    for g in G_aligned:
        axes[0, 2].plot(g[0]/PX_MM, g[1]/PX_MM, "o", ms=8, mfc="none",
                        mec="#27ae60", lw=2)
    for m in pred_meta:
        axes[0, 2].plot(m["centroid_mm"][0]/PX_MM, m["centroid_mm"][1]/PX_MM,
                        "x", ms=7, color="#c0392b")
    for r in match_rows:
        gx, gy, _ = r["gt_centroid_mm"]
        px, py, _ = r["pred_centroid_mm"]
        axes[0, 2].plot([gx/PX_MM, px/PX_MM], [gy/PX_MM, py/PX_MM],
                        "-", color="#7f8c8d", lw=1)
    axes[0, 2].set_xlim(0, NX); axes[0, 2].set_ylim(NY, 0)
    axes[0, 2].set_aspect("equal")
    axes[0, 2].set_title(f"XY (sym {sym}): GT (green o) / Pred (red x)",
                         fontsize=9, fontweight="bold")

    yc = NY // 2
    axes[0, 3].imshow(probs[:, yc, :], cmap="Reds", vmin=0, vmax=1,
                      aspect="auto", extent=[0, 20, depths[-1], 0])
    axes[0, 3].set_xlabel("X (mm)"); axes[0, 3].set_ylabel("Depth (mm)")
    axes[0, 3].set_title("Prob slice y=mid", fontsize=9, fontweight="bold")

    thrs = [s["thr"] for s in sweep]
    axes[1, 0].plot(thrs, [s["f1"] for s in sweep], "o-", color="#2980b9")
    axes[1, 0].set_xlabel("Threshold"); axes[1, 0].set_ylabel("F1")
    axes[1, 0].axvline(thr, color="red", lw=0.7, ls="--")
    axes[1, 0].set_title("F1 vs threshold", fontsize=9); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(thrs, [s["precision"] for s in sweep], "o-",
                    label="precision", color="#27ae60")
    axes[1, 1].plot(thrs, [s["recall"] for s in sweep], "s-",
                    label="recall", color="#c0392b")
    axes[1, 1].set_xlabel("Threshold"); axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axvline(thr, color="red", lw=0.7, ls="--")
    axes[1, 1].set_title("Precision / Recall", fontsize=9)
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    if match_rows:
        axes[1, 2].bar(range(len(match_rows)),
                       [m["vol_ratio"] for m in match_rows], color="#c0392b")
        axes[1, 2].axhline(1, color="black", lw=0.7, ls="--")
        axes[1, 2].set_xlabel("Match index")
        axes[1, 2].set_ylabel("Pred / GT volume")
        axes[1, 2].set_title("Volume ratio per matched void", fontsize=9)
        axes[1, 2].grid(alpha=0.3)

        axes[1, 3].bar(range(len(match_rows)),
                       [m["depth_error_mm"] for m in match_rows], color="#2980b9")
        axes[1, 3].axhline(0, color="black", lw=0.7, ls="--")
        axes[1, 3].set_xlabel("Match index")
        axes[1, 3].set_ylabel("z_pred − z_gt (mm)")
        axes[1, 3].set_title("Depth error per matched void", fontsize=9)
        axes[1, 3].grid(alpha=0.3)
    else:
        for ax in (axes[1, 2], axes[1, 3]):
            ax.text(0.5, 0.5, "no matches", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#888")
            ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"Void characterization — sample {name}",
                 fontsize=11, fontweight="bold", y=0.997)
    plt.tight_layout()
    plt.savefig(out_dir / "summary.png", dpi=140,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── aggregate report ─────────────────────────────────────────────────────────
def aggregate_report(results, agg_dir):
    rows = [r for r in results if r is not None]
    print(f"\n{'='*70}\nAggregate ({len(rows)} samples)\n{'='*70}")
    print(f"{'sample':>6s}  {'#gt':>4s}  {'#pred':>5s}  {'F1':>5s}  "
          f"{'P':>5s}  {'R':>5s}  {'dist':>5s}  {'thr':>4s}  {'sym':>3s}")
    print("─" * 60)
    f1s, precs, recs, dists, ratios, depth_errs = [], [], [], [], [], []
    for r in rows:
        d = r["mean_dist_mm"]
        print(f"{r['sample']:>6s}  {r['n_gt']:>4d}  {r['n_pred']:>5d}  "
              f"{r['f1']:>5.3f}  {r['precision']:>5.3f}  {r['recall']:>5.3f}  "
              f"{(f'{d:.2f}' if d is not None else '   —'):>5s}  "
              f"{r['best_threshold']:>4.2f}  {r['best_sym']:>3d}")
        f1s.append(r["f1"]); precs.append(r["precision"]); recs.append(r["recall"])
        if d is not None: dists.append(d)
        for m in r["match_rows"]:
            if m["gt_volume_mm3"] > 0:
                ratios.append(m["vol_ratio"])
            depth_errs.append(m["depth_error_mm"])
    print("─" * 60)
    print(f"  mean   F1={np.mean(f1s):.3f}  P={np.mean(precs):.3f}  "
          f"R={np.mean(recs):.3f}  "
          f"mean centroid dist={np.mean(dists):.2f}mm  "
          f"blow-up mean={np.mean(ratios):.2f}x (n={len(ratios)})  "
          f"depth |err| mean={np.mean(np.abs(depth_errs)):.2f}mm")

    # Aggregate JSON
    with open(agg_dir / "aggregate.json", "w") as fh:
        json.dump({
            "n_samples": len(rows),
            "mean_f1": float(np.mean(f1s)),
            "mean_precision": float(np.mean(precs)),
            "mean_recall": float(np.mean(recs)),
            "mean_centroid_dist_mm": float(np.mean(dists)) if dists else None,
            "mean_volume_ratio": float(np.mean(ratios)) if ratios else None,
            "mean_abs_depth_err_mm": float(np.mean(np.abs(depth_errs))) if depth_errs else None,
            "per_sample": [{
                "sample": r["sample"], "n_gt": r["n_gt"], "n_pred": r["n_pred"],
                "f1": r["f1"], "precision": r["precision"], "recall": r["recall"],
                "mean_dist_mm": r["mean_dist_mm"], "best_threshold": r["best_threshold"],
                "best_sym": r["best_sym"],
            } for r in rows],
        }, fh, indent=2, default=float)

    # Aggregate plot
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), dpi=140)
    fig.patch.set_facecolor("white")
    names = [r["sample"] for r in rows]
    xpos  = np.arange(len(rows))

    axes[0, 0].bar(xpos, [r["f1"] for r in rows], color="#2980b9")
    axes[0, 0].set_xticks(xpos); axes[0, 0].set_xticklabels(names)
    axes[0, 0].set_ylabel("F1"); axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_title("F1 per sample", fontsize=10, fontweight="bold")
    axes[0, 0].axhline(np.mean(f1s), color="red", ls="--", lw=0.8,
                       label=f"mean = {np.mean(f1s):.2f}")
    axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    width = 0.35
    axes[0, 1].bar(xpos - width/2, [r["precision"] for r in rows], width,
                   label="Precision", color="#27ae60")
    axes[0, 1].bar(xpos + width/2, [r["recall"] for r in rows], width,
                   label="Recall", color="#c0392b")
    axes[0, 1].set_xticks(xpos); axes[0, 1].set_xticklabels(names)
    axes[0, 1].set_ylim(0, 1); axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_title("Precision / Recall per sample", fontsize=10, fontweight="bold")
    axes[0, 1].grid(alpha=0.3)

    if ratios:
        axes[1, 0].hist(ratios, bins=20, color="#c0392b", alpha=0.85,
                        edgecolor="black", linewidth=0.5)
        axes[1, 0].axvline(1, color="black", ls="--", lw=0.8, label="ideal=1×")
        axes[1, 0].axvline(np.mean(ratios), color="blue", ls="--", lw=0.8,
                           label=f"mean = {np.mean(ratios):.2f}×")
        axes[1, 0].set_xlabel("Pred / GT volume")
        axes[1, 0].set_ylabel("Matched voids")
        axes[1, 0].set_title("Volume blow-up factor (THz physics)",
                             fontsize=10, fontweight="bold")
        axes[1, 0].legend(fontsize=8); axes[1, 0].grid(alpha=0.3)

    if depth_errs:
        axes[1, 1].hist(depth_errs, bins=20, color="#2980b9", alpha=0.85,
                        edgecolor="black", linewidth=0.5)
        axes[1, 1].axvline(0, color="black", ls="--", lw=0.8)
        axes[1, 1].axvline(np.mean(depth_errs), color="red", ls="--", lw=0.8,
                           label=f"mean = {np.mean(depth_errs):.2f}mm")
        axes[1, 1].set_xlabel("z_pred − z_gt (mm)")
        axes[1, 1].set_ylabel("Matched voids")
        axes[1, 1].set_title("Depth error", fontsize=10, fontweight="bold")
        axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    fig.suptitle("Void characterization — aggregate across 15 samples",
                 fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(agg_dir / "aggregate.png", dpi=140,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nSaved: {agg_dir/'aggregate.json'}  +  aggregate.png")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    if _CLI_TARGET is None:
        print(__doc__); sys.exit(1)
    target = _CLI_TARGET

    print(f"device: {device}")
    ckpt  = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = UNet(in_channels=cfg["in_channels"], base_filters=cfg["base_filters"])
    model.load_state_dict(ckpt["model_state"]); model.to(device).eval()
    offsets = cfg["offsets"]
    print(f"Model: {MODEL_CKPT.name}  offsets {offsets}\n")

    if target.lower() == "all":
        names = sorted(SAMPLE_TO_STL.keys(),
                       key=lambda s: int(s.rstrip("b")))
    else:
        if target not in SAMPLE_TO_STL:
            print(f"Unknown sample {target}"); sys.exit(1)
        names = [target]

    results = []
    for n in names:
        try:
            r = characterize(n, model, offsets)
            results.append(r)
        except Exception as e:
            print(f"  ✗ sample {n} failed: {e}")
            results.append(None)

    if len(names) > 1:
        aggregate_report(results, OUT_DIR)


if __name__ == "__main__":
    main()
