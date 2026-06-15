"""
void_characterize_s4.py — Proof-of-concept void characterization pipeline.
Sample 4 (CV04, 4 cylindrical voids).

Stages
------
1. Load CV04.stl, split into block + void components, voxelize each void at
   the 2050-slice THz depth grid (0.2 mm/px laterally).
2. Run unet_pseudo_2050.pt over sample 4 → (2050, 100, 100) probability volume.
3. Threshold sweep (binarise, 3D connected components on predictions) and pick
   the threshold that maximises void-level F1 against the STL voids.
4. For each of the 8 in-plane symmetries (4 rotations × 2 flips) of the STL
   voids, compute matching cost to predictions — pick the best alignment.
5. Hungarian-match GT void centroids ↔ predicted centroids.
6. Report: per-void position error, depth error, volume ratio; aggregate F1.

Run:
    thesis_env/bin/python void_characterize_s4.py
"""
import sys, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import trimesh
from pathlib import Path
from scipy.ndimage import label as cc_label
from scipy.optimize import linear_sum_assignment

# ── pipeline import ──────────────────────────────────────────────────────────
sys.argv = [sys.argv[0], "predict", "atlanta2"]
sys.path.insert(0, ".")
from thz_slice_pipelinev2 import UNet

# ── config ───────────────────────────────────────────────────────────────────
SAMPLE       = "4"
STL_PATH     = Path("AllSamples/stl/CV04.stl")
SLICES_DIR   = Path("slices_v2_atlanta2_2050")
MODEL_CKPT   = Path("results_v2_atlanta2/unet_pseudo_2050.pt")
OUT_DIR      = Path("results_v2_atlanta2/void_char_s4")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Spatial grid: 100×100 px laterally @ 0.2 mm/px = 20×20 mm
PX_MM        = 0.2
NX = NY      = 100
MIN_BLOB_PX  = 100           # min 2D area (px) — ~0.4 mm² floor for noise filtering
THRESHOLDS   = np.arange(0.3, 0.96, 0.05)
DIST_GATE_MM = 8.0           # don't match pairs farther apart than this in mm

device = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}\n")


# ── Stage 1: voxelize STL voids ──────────────────────────────────────────────
def voxelize_stl_voids(stl_path, depths_mm, px_mm=0.2, ny=100, nx=100):
    """Return (n_voids, nz, ny, nx) binary masks + per-void metadata.

    Identifies the outer block (largest bbox volume), treats the other
    connected components as individual voids, and tests grid points against
    each void mesh with trimesh.contains.
    """
    mesh = trimesh.load(stl_path, force="mesh")
    print(f"  STL: {stl_path.name}  bbox extents {mesh.bounding_box.extents} mm")

    comps = mesh.split(only_watertight=False)
    print(f"  components: {len(comps)}")

    # Outer block = largest bounding-box volume
    bbox_vols = np.array([np.prod(c.bounding_box.extents) for c in comps])
    block_i   = int(np.argmax(bbox_vols))
    voids     = [c for i, c in enumerate(comps) if i != block_i]
    block     = comps[block_i]
    print(f"  outer block bbox: {block.bounding_box.extents}, "
          f"{len(voids)} void(s)")

    # Build grid of points (x, y, z) in mm at the THz voxel centres
    # x grid in [px_mm/2, 20-px_mm/2], same for y
    xs = (np.arange(nx) + 0.5) * px_mm
    ys = (np.arange(ny) + 0.5) * px_mm
    zs = depths_mm.astype(np.float32)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="xy")   # (ny, nx, nz)
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)  # (N, 3)

    # Test each void mesh
    void_masks = []
    void_meta  = []
    for vi, v in enumerate(voids):
        # Center the void test in the same coordinate frame as the block: subtract block bbox min
        # so STL origin sits at (0,0,0).
        v_shift  = v.copy()
        v_shift.apply_translation(-block.bounds[0])
        inside   = v_shift.contains(pts).reshape(ny, nx, len(zs)).transpose(2, 0, 1)  # (nz, ny, nx)
        if inside.sum() == 0:
            print(f"    void #{vi}: no voxels inside grid (out of depth range)")
            continue
        # Per-void metadata (in mm)
        zyx = np.array(np.where(inside))   # (3, N)
        cz, cy, cx = zyx.mean(axis=1)
        cz_mm = zs[int(round(cz))]
        cy_mm = ys[int(round(cy))]
        cx_mm = xs[int(round(cx))]
        vol_mm3 = inside.sum() * (px_mm**2) * np.diff(zs).mean()
        z_range = (zs[zyx[0].min()], zs[zyx[0].max()])
        void_masks.append(inside)
        void_meta.append(dict(
            id=vi,
            n_voxels=int(inside.sum()),
            centroid_mm=(float(cx_mm), float(cy_mm), float(cz_mm)),
            bbox_mm=tuple(float(b) for b in v.bounding_box.extents),
            volume_mm3=float(vol_mm3),
            depth_range_mm=(float(z_range[0]), float(z_range[1])),
        ))
        print(f"    void #{vi}: vox={int(inside.sum()):>6d}  "
              f"centroid=({cx_mm:.2f},{cy_mm:.2f},{cz_mm:.2f})mm  "
              f"vol={vol_mm3:.2f}mm³  depth {z_range[0]:.2f}-{z_range[1]:.2f}mm")
    return void_masks, void_meta


# ── Stage 2: model predictions ───────────────────────────────────────────────
def predict_full_volume(model, slices, offsets):
    n_s = slices.shape[0]
    probs = np.zeros_like(slices, dtype=np.float32)
    CHUNK = 32
    inputs = np.zeros((n_s, len(offsets), 100, 100), dtype=np.float32)
    for i in range(n_s):
        for k, off in enumerate(offsets):
            inputs[i, k] = slices[int(np.clip(i + off, 0, n_s - 1))]
    x = torch.from_numpy(inputs).to(device)
    with torch.no_grad():
        for k in range(0, n_s, CHUNK):
            logits = model(x[k:k + CHUNK])
            probs[k:k + CHUNK] = torch.sigmoid(logits).squeeze(1).cpu().numpy()
    return probs


# ── Helper: extract blob metadata via 2D projection + per-blob depth profile ─
def extract_blobs(prob_volume, threshold, depths_mm, px_mm=0.2,
                  min_area_px=MIN_BLOB_PX):
    """Project (nz, ny, nx) prob volume to 2D max, threshold, find 2D blobs.

    For each blob, derive depth metadata from the original 3D probabilities
    inside that blob's footprint (avoids 3D connectivity fragmentation).
    """
    # 2D max-projection of probabilities, threshold there
    proj2d = prob_volume.max(axis=0)                       # (ny, nx)
    mask2d = (proj2d > threshold).astype(np.uint8)

    # 2D connected components (8-connectivity)
    labels, n = cc_label(mask2d, structure=np.ones((3, 3), dtype=bool))

    dz_mm = float(np.diff(depths_mm).mean())
    blobs = []
    for bid in range(1, n + 1):
        ys, xs = np.where(labels == bid)
        n_px = ys.size
        if n_px < min_area_px:
            continue

        # XY centroid
        cy_mm = float((ys.mean() + 0.5) * px_mm)
        cx_mm = float((xs.mean() + 0.5) * px_mm)

        # Per-blob depth profile: max prob over the blob footprint at each z
        profile = prob_volume[:, ys, xs].max(axis=1)        # (nz,)
        depth_mask = profile > threshold                    # which depths the void spans
        if not depth_mask.any():
            continue
        z_idx = np.where(depth_mask)[0]
        # Weighted depth centroid (weighted by max-prob)
        weights = profile[z_idx]
        cz_mm = float(np.average(depths_mm[z_idx], weights=weights))
        z_range = (float(depths_mm[z_idx.min()]),
                   float(depths_mm[z_idx.max()]))
        z_extent = z_range[1] - z_range[0]

        # Volume estimate: 2D area × depth extent
        area_mm2 = n_px * (px_mm ** 2)
        vol_mm3  = area_mm2 * (z_extent if z_extent > 0 else dz_mm)

        # Lateral bbox
        bbox_mm = (
            float((xs.max() - xs.min() + 1) * px_mm),
            float((ys.max() - ys.min() + 1) * px_mm),
            float(z_extent),
        )

        blobs.append(dict(
            n_voxels=int(n_px),                  # 2D area in pixels (semantic kept)
            centroid_mm=(cx_mm, cy_mm, cz_mm),
            bbox_mm=bbox_mm,
            volume_mm3=vol_mm3,
            depth_range_mm=z_range,
            area_mm2=area_mm2,
        ))
    return blobs


# ── Stage 3-4: rotation alignment + matching ─────────────────────────────────
def apply_symmetry(centroid_xyz, sym_id):
    """Map (x, y, z) mm via one of 8 in-plane symmetries (z untouched)."""
    x, y, z = centroid_xyz
    # In-plane 4 rotations × {identity, mirror-x}
    rot = sym_id // 2
    mir = sym_id % 2
    if mir:
        x = 20.0 - x
    if rot == 1:                # 90°
        x, y = y, 20.0 - x
    elif rot == 2:              # 180°
        x, y = 20.0 - x, 20.0 - y
    elif rot == 3:              # 270°
        x, y = 20.0 - y, x
    return (x, y, z)


def best_alignment_and_match(gt_meta, pred_meta, dist_gate_mm=DIST_GATE_MM):
    """Try 8 symmetries, return best (sym_id, cost, assignment)."""
    P = np.array([m["centroid_mm"] for m in pred_meta])     # (Np, 3)
    best = None
    for sym_id in range(8):
        G = np.array([apply_symmetry(m["centroid_mm"], sym_id) for m in gt_meta])
        # Cost matrix (mm) with a large gate beyond dist_gate
        C = np.linalg.norm(G[:, None, :] - P[None, :, :], axis=-1)   # (Ng, Np)
        C_gated = np.where(C <= dist_gate_mm, C, 1e6)
        row, col = linear_sum_assignment(C_gated)
        # Real matches = those with cost <= gate
        good = C_gated[row, col] <= dist_gate_mm
        matches = [(int(r), int(c), float(C[r, c]))
                   for r, c, g in zip(row, col, good) if g]
        total_cost = sum(d for _, _, d in matches) + dist_gate_mm * (len(gt_meta) - len(matches))
        if best is None or total_cost < best["total_cost"]:
            best = dict(sym_id=sym_id, total_cost=total_cost,
                        matches=matches, G_aligned=G.tolist())
    return best


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    print(f"=== Void characterization (sample {SAMPLE}) ===\n")

    # ---- Load model ----
    ckpt  = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    cfg   = ckpt["config"]
    model = UNet(in_channels=cfg["in_channels"], base_filters=cfg["base_filters"])
    model.load_state_dict(ckpt["model_state"]); model.to(device).eval()
    OFFSETS = cfg["offsets"]
    print(f"Model: {MODEL_CKPT.name} (offsets {OFFSETS})\n")

    # ---- Load THz slices + depths ----
    slices  = np.load(SLICES_DIR / f"{SAMPLE}_slices.npy")
    depths  = np.load(SLICES_DIR / f"{SAMPLE}_depths.npy")
    print(f"THz: {slices.shape}, depth 0–{depths[-1]:.2f} mm\n")

    # ---- Stage 1: voxelize STL ----
    print("Stage 1 — STL voxelization …")
    gt_masks, gt_meta = voxelize_stl_voids(STL_PATH, depths, PX_MM, NY, NX)
    n_gt = len(gt_meta)
    print(f"  → {n_gt} GT voids\n")

    # ---- Stage 2: prediction volume ----
    print("Stage 2 — running model on 2050 slices …")
    probs = predict_full_volume(model, slices, OFFSETS)
    print(f"  prob volume {probs.shape}  range {probs.min():.3f}–{probs.max():.3f}\n")

    # ---- Stage 3: threshold sweep ----
    print("Stage 3 — threshold sweep …")
    print(f"  {'thr':>5s}  {'#blobs':>6s}  {'matches':>7s}  "
          f"{'precision':>9s}  {'recall':>7s}  {'F1':>6s}  {'sym':>4s}  {'mean d':>7s}")
    print("  " + "─" * 70)
    sweep_rows = []
    for thr in THRESHOLDS:
        pred_meta = extract_blobs(probs, thr, depths, PX_MM)
        if not pred_meta:
            sweep_rows.append((float(thr), 0, 0, 0.0, 0.0, 0.0, -1, np.nan))
            continue
        best = best_alignment_and_match(gt_meta, pred_meta)
        n_match = len(best["matches"])
        precision = n_match / len(pred_meta)
        recall    = n_match / n_gt
        f1        = (2*precision*recall / (precision+recall)) if (precision+recall) > 0 else 0.0
        mean_d    = (np.mean([d for _, _, d in best["matches"]])
                     if best["matches"] else np.nan)
        sweep_rows.append((float(thr), len(pred_meta), n_match,
                           precision, recall, f1, best["sym_id"], mean_d))
        print(f"  {thr:.2f}   {len(pred_meta):>6d}   {n_match:>7d}   "
              f"{precision:>9.3f}  {recall:>7.3f}  {f1:>6.3f}   {best['sym_id']:>4d}  "
              f"{mean_d:>6.2f}mm")

    # Pick best threshold (highest F1; on tie prefer fewer FP, i.e. fewer blobs)
    sweep_arr = np.array(sweep_rows, dtype=object)
    best_row  = max(sweep_rows, key=lambda r: (r[5], -r[1]))
    best_thr  = best_row[0]
    print(f"\n  → best threshold: {best_thr:.2f}  "
          f"(F1 {best_row[5]:.3f}, sym {best_row[6]}, mean dist {best_row[7]:.2f} mm)\n")

    # ---- Stage 4: final analysis at best threshold ----
    pred_meta = extract_blobs(probs, best_thr, depths, PX_MM)
    binv      = (probs.max(axis=0) > best_thr)  # for visualisation only
    best      = best_alignment_and_match(gt_meta, pred_meta)
    sym       = best["sym_id"]
    G_aligned = best["G_aligned"]

    print(f"Stage 4 — match table (symmetry id {sym}):\n")
    print(f"  {'pair':>5s}  {'gt_centroid (mm)':>20s}  {'pred_centroid (mm)':>22s}  "
          f"{'dist':>5s}  {'gt_vol':>7s}  {'pred_vol':>9s}  ratio")
    print("  " + "─" * 90)
    match_rows = []
    for gi, pi, d in sorted(best["matches"], key=lambda x: x[2]):
        gxyz = G_aligned[gi]
        pxyz = pred_meta[pi]["centroid_mm"]
        gv   = gt_meta[gi]["volume_mm3"]
        pv   = pred_meta[pi]["volume_mm3"]
        ratio = pv / gv if gv > 0 else float('nan')
        print(f"  {gi}→{pi:<3d}  "
              f"({gxyz[0]:5.2f},{gxyz[1]:5.2f},{gxyz[2]:5.2f})  "
              f"({pxyz[0]:5.2f},{pxyz[1]:5.2f},{pxyz[2]:5.2f})  "
              f"{d:>5.2f}  {gv:>7.2f}  {pv:>9.2f}  {ratio:>5.2f}x")
        match_rows.append(dict(
            gt_id=gi, pred_id=pi, dist_mm=d,
            gt_centroid_mm=tuple(gxyz),
            pred_centroid_mm=tuple(pxyz),
            depth_error_mm=float(pxyz[2] - gxyz[2]),
            gt_volume_mm3=gv, pred_volume_mm3=pv, vol_ratio=ratio,
        ))

    unmatched_gt   = [g for g in range(n_gt)
                      if g not in [m[0] for m in best["matches"]]]
    unmatched_pred = [p for p in range(len(pred_meta))
                      if p not in [m[1] for m in best["matches"]]]
    print(f"\n  unmatched GT voids (missed): {unmatched_gt}")
    print(f"  unmatched predicted blobs (extra): {unmatched_pred}")

    # ---- Save outputs ----
    out_json = OUT_DIR / "result.json"
    with open(out_json, "w") as fh:
        json.dump({
            "sample": SAMPLE,
            "stl": str(STL_PATH),
            "model": str(MODEL_CKPT),
            "best_threshold": best_thr,
            "best_symmetry": sym,
            "n_gt": n_gt,
            "n_pred": len(pred_meta),
            "precision": best_row[3],
            "recall": best_row[4],
            "f1": best_row[5],
            "mean_centroid_distance_mm": best_row[7],
            "gt_voids": gt_meta,
            "pred_voids": pred_meta,
            "matches": match_rows,
            "sweep": [{"thr": r[0], "n_blobs": r[1], "matches": r[2],
                       "precision": r[3], "recall": r[4], "f1": r[5],
                       "sym": r[6], "mean_dist_mm": float(r[7]) if not np.isnan(r[7]) else None}
                      for r in sweep_rows],
        }, fh, indent=2, default=float)
    print(f"\nSaved: {out_json}")

    # ---- Visualisation ----
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), dpi=140)
    fig.patch.set_facecolor("white")
    # Top row: max-projection (z) of GT voids vs prediction
    gt_proj   = np.zeros((NY, NX), dtype=bool)
    for m in gt_masks:
        gt_proj |= m.any(axis=0)
    pred_proj = binv   # already a 2D max-projection threshold
    axes[0, 0].imshow(gt_proj, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 0].set_title("STL voids (z-projection)", fontsize=9, fontweight="bold")
    axes[0, 1].imshow(pred_proj, cmap="Reds", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 1].set_title(f"Predicted (thr={best_thr:.2f})", fontsize=9, fontweight="bold")
    # Top-down overlay with matches
    axes[0, 2].imshow(np.zeros((NY, NX)) + 0.95, cmap="gray", vmin=0, vmax=1)
    for m in gt_meta:
        ax = axes[0, 2]
        cx, cy, _ = apply_symmetry(m["centroid_mm"], sym)
        ax.plot(cx / PX_MM, cy / PX_MM, "o", ms=9, mfc="none", mec="#27ae60", lw=2)
    for m in pred_meta:
        cx, cy, _ = m["centroid_mm"]
        axes[0, 2].plot(cx / PX_MM, cy / PX_MM, "x", ms=8, color="#c0392b")
    for gi, pi, _ in best["matches"]:
        gx, gy, _ = G_aligned[gi]
        px, py, _ = pred_meta[pi]["centroid_mm"]
        axes[0, 2].plot([gx/PX_MM, px/PX_MM], [gy/PX_MM, py/PX_MM],
                        "-", color="#7f8c8d", lw=1)
    axes[0, 2].set_xlim(0, NX); axes[0, 2].set_ylim(NY, 0)
    axes[0, 2].set_title("XY: GT (green o) vs Pred (red x)", fontsize=9, fontweight="bold")
    axes[0, 2].set_aspect("equal")

    # Depth side-view: cross-section at y = midline
    yc = NY // 2
    axes[0, 3].imshow(probs[:, yc, :], cmap="Reds", vmin=0, vmax=1,
                      aspect="auto",
                      extent=[0, 20, depths[-1], 0])
    axes[0, 3].set_xlabel("X (mm)"); axes[0, 3].set_ylabel("Depth (mm)")
    axes[0, 3].set_title(f"Prob slice y=mid", fontsize=9, fontweight="bold")

    # Bottom row: sweep curves
    thrs = [r[0] for r in sweep_rows]
    axes[1, 0].plot(thrs, [r[5] for r in sweep_rows], "o-", color="#2980b9")
    axes[1, 0].set_xlabel("Threshold"); axes[1, 0].set_ylabel("F1")
    axes[1, 0].set_title("F1 vs threshold", fontsize=9); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].axvline(best_thr, color="red", lw=0.7, ls="--")

    axes[1, 1].plot(thrs, [r[3] for r in sweep_rows], "o-", label="precision", color="#27ae60")
    axes[1, 1].plot(thrs, [r[4] for r in sweep_rows], "s-", label="recall", color="#c0392b")
    axes[1, 1].set_xlabel("Threshold"); axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_title("Precision / Recall", fontsize=9)
    axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].axvline(best_thr, color="red", lw=0.7, ls="--")

    # Volume ratio per matched void
    if match_rows:
        axes[1, 2].bar(range(len(match_rows)),
                       [m["vol_ratio"] for m in match_rows], color="#c0392b")
        axes[1, 2].axhline(1, color="black", lw=0.7, ls="--")
        axes[1, 2].set_xlabel("Match index")
        axes[1, 2].set_ylabel("Pred / GT volume")
        axes[1, 2].set_title("Volume ratio per matched void", fontsize=9)
        axes[1, 2].grid(alpha=0.3)

    # Depth error
    if match_rows:
        axes[1, 3].bar(range(len(match_rows)),
                       [m["depth_error_mm"] for m in match_rows], color="#2980b9")
        axes[1, 3].axhline(0, color="black", lw=0.7, ls="--")
        axes[1, 3].set_xlabel("Match index")
        axes[1, 3].set_ylabel("z_pred − z_gt (mm)")
        axes[1, 3].set_title("Depth error per matched void", fontsize=9)
        axes[1, 3].grid(alpha=0.3)

    fig.suptitle(f"Void characterization — sample {SAMPLE} (CV04, 4 cylindrical voids)",
                 fontsize=12, fontweight="bold", y=0.995)
    plt.tight_layout()
    out_png = OUT_DIR / "summary.png"
    plt.savefig(out_png, dpi=140, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
