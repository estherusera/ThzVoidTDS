"""
stl_to_labels.py — Build per-slice void masks for U-Net training directly from
the STL CAD voids (instead of the hand-drawn rectangular masks in
labels_v2_atlanta2/). Uses the same machinery as void_characterize.py.

For each atlanta2 sample:
  • per-slice 3D mask via mesh.contains() (watertight voids, ≤2k faces)
    or extruded axis-aligned bbox over depth range (non-watertight fallback)
  • per-sample best symmetry from summary_unified.json (CAD → scan alignment)
  • optional 3D Gaussian blur (σ_xy, σ_z) — adds boundary slack for THz beam
  • s13 + s14: zero masks (no detectable scan voids despite STL design)

Output:
  labels_stl_atlanta2/{phys}_slice_masks.npy   (50, 100, 100) float32 in [0,1]
  results_v2_atlanta2/stl_labels/stl_vs_manual_sigxyA_sigzB.png   QC figure

Run
    thesis_env/bin/python stl_to_labels.py            # default σ_xy=1, σ_z=1
    thesis_env/bin/python stl_to_labels.py 0 0        # no blur
    thesis_env/bin/python stl_to_labels.py 2 1        # heavier xy blur
"""
import sys, json, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import trimesh
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

# Capture CLI BEFORE void_characterize import rewrites sys.argv.
# Flags use the "--" prefix; "-0.7" must NOT be misread as a flag.
_CLI_ARGS  = [a for a in sys.argv[1:] if not a.startswith("--")]
_CLI_FLAGS = [a for a in sys.argv[1:] if a.startswith("--")]

sys.path.insert(0, str(Path(__file__).parent))
from void_characterize import (
    SAMPLE_TO_STL, STL_DIR, PX_MM, NX, NY,
    CONTAINS_MAX_FACES, MIN_GT_VOLUME_MM3,
    voxelize_gt, best_alignment_and_match,
)
from scipy.ndimage import label as cc_label

SLICES_DIR    = Path("slices_v2_atlanta2")
UNIFIED_JSON  = Path("results_v2_atlanta2/ascan_classifier/vs_stl_unified/summary_unified.json")
MANUAL_LABELS = Path("labels_v2_atlanta2")
OUT_LABELS    = Path("labels_stl_atlanta2")
OUT_QC_DIR    = Path("results_v2_atlanta2/stl_labels")
ASCAN_CKPT    = Path("results_v2_atlanta2/ascan_classifier/ascan_cnn_depth.pt")
ASCAN_CACHE   = Path("ascan_cache_atlanta2")
OUT_LABELS.mkdir(exist_ok=True)
OUT_QC_DIR.mkdir(parents=True, exist_ok=True)

N_DEPTHS = 50

# Per-sample alignment overrides applied when --joint-align is OFF. When
# --joint-align is ON, the joint A-scan + manual IoU search overrides these.
# Z-shift in mm, sym in [0, 7].
Z_SHIFT_OVERRIDE = {}
SYM_OVERRIDE_VIZ = {}


# ── Depth-aware A-scan classifier (for joint sym + z-shift search) ──────────
class DepthAwareAScanCNN(nn.Module):
    """Matches the saved state dict of ascan_cnn_depth.pt; see ascan_vs_stl.py."""
    def __init__(self, n_depths=N_DEPTHS):
        super().__init__()
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
            nn.AdaptiveAvgPool1d(n_depths),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def forward(self, x):
        return self.head(self.encoder(x)).squeeze(1)


_ASCAN_MODEL_CACHE = {"loaded": False, "model": None}

def _load_ascan_model():
    if _ASCAN_MODEL_CACHE["loaded"]:
        return _ASCAN_MODEL_CACHE["model"]
    _ASCAN_MODEL_CACHE["loaded"] = True
    if not ASCAN_CKPT.exists():
        print(f"  ! {ASCAN_CKPT} not found — joint-align disabled.")
        return None
    ckpt = torch.load(ASCAN_CKPT, map_location="cpu", weights_only=False)
    m = DepthAwareAScanCNN().to("cpu")
    m.load_state_dict(ckpt["model_state"])
    m.eval()
    _ASCAN_MODEL_CACHE["model"] = m
    return m


def predict_ascan_volume(name):
    """Return (50, NY, NX) probability volume from the depth-aware A-scan CNN,
    or None if the cache / model is unavailable."""
    cache_path = ASCAN_CACHE / f"{name}.npz"
    if not cache_path.exists():
        return None
    model = _load_ascan_model()
    if model is None:
        return None
    d = np.load(cache_path)
    a = d["ascans"]
    mean = a.mean(axis=1, keepdims=True)
    std  = a.std(axis=1, keepdims=True) + 1e-6
    a_norm = (a - mean) / std
    a_t = torch.from_numpy(a_norm.astype(np.float32)).unsqueeze(1).to("cpu")
    probs = np.zeros((a_t.shape[0], N_DEPTHS), dtype=np.float32)
    CHUNK = 512
    with torch.no_grad():
        for k in range(0, a_t.shape[0], CHUNK):
            probs[k:k+CHUNK] = torch.sigmoid(model(a_t[k:k+CHUNK])).numpy()
    return probs.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)


def shift_z(m, z_slices):
    """Roll a (NZ, NY, NX) mask along depth, zeroing wrap-around."""
    if z_slices == 0:
        return m.copy()
    out = np.roll(m, z_slices, axis=0)
    if z_slices > 0:
        out[:z_slices] = 0
    else:
        out[z_slices:] = 0
    return out


def joint_align(stl_mask_cad, manual_mask, ascan_pred, dz_mm,
                z_shift_range=(-1.5, 0.05, 0.1)):
    """Jointly search over (sym, z_shift) using A-scan 3D IoU + manual 2D IoU.

    Returns dict with keys: sym, z_shift_mm, score, iou_ascan, iou_manual,
    source (string description).
    """
    if not stl_mask_cad.any():
        return dict(sym=0, z_shift_mm=0.0, score=-1.0,
                    iou_ascan=0.0, iou_manual=0.0, source="empty-stl")

    ascan_bin = (ascan_pred > 0.5) if ascan_pred is not None else None
    have_ascan = ascan_bin is not None and ascan_bin.any()

    man_proj = None
    have_manual = False
    if manual_mask is not None and manual_mask.any():
        man_proj = (manual_mask.max(axis=0) > 0).astype(bool)
        have_manual = True

    z_shifts = np.arange(z_shift_range[0], z_shift_range[1], z_shift_range[2])
    best = dict(sym=0, z_shift_mm=0.0, score=-1.0,
                iou_ascan=0.0, iou_manual=0.0, source="default")

    for z_shift in z_shifts:
        z_slices = int(round(z_shift / dz_mm))
        m_z = shift_z(stl_mask_cad, z_slices)
        for sym_id in range(8):
            m = apply_sym_mask(m_z, sym_id)
            m_bin = (m > 0.5)

            iou_a = 0.0
            if have_ascan:
                inter = int((m_bin & ascan_bin).sum())
                union = int((m_bin | ascan_bin).sum())
                iou_a = inter / max(union, 1)

            iou_m = 0.0
            if have_manual:
                m_proj = m_bin.max(axis=0)
                inter = int((m_proj & man_proj).sum())
                union = int((m_proj | man_proj).sum())
                iou_m = inter / max(union, 1)

            # Weight A-scan higher (per-depth disambiguation) but never zero
            # out manual (it carries the true scan orientation).
            score = 2.0 * iou_a + iou_m
            if score > best["score"]:
                best = dict(sym=int(sym_id), z_shift_mm=float(z_shift),
                            score=float(score),
                            iou_ascan=float(iou_a), iou_manual=float(iou_m),
                            source=("joint(ascan+manual)" if (have_ascan and have_manual)
                                    else "joint(ascan)" if have_ascan
                                    else "joint(manual)" if have_manual
                                    else "default"))
    return best

# STL-positive samples that we still treat as no-void for TRAINING,
# because the THz scan shows no detectable voids:
#   s13 → Cilindros_CV14.stl (cylinders by design, but scan is blank)
#   s14 → NoVoid_CV15.stl    (no voids by design)
# In viz mode (--viz) the s13 designed geometry is kept so the viewer can
# show what the CAD specifies, regardless of THz detectability.
NO_VOID_STL_TRAIN = {"13", "14"}
NO_VOID_STL_VIZ   = {"14"}

# Per-sample symmetry overrides for samples where the IoU-vs-manual pick is
# unreliable (sparse manual labels, single-blob manual, etc.). Values here
# beat both the IoU picker and the unified fallback. Override key = sample
# name (string); value = sym_id in [0, 7].
SYM_OVERRIDE = {
    # s3 has a single-blob manual mask → IoU picker ties at 0.08 across
    # multiple syms. The A-scan unified eval picked sym=7 with F1u=0.857.
    "3":  7,
}

# Threshold below which we consider the manual-IoU pick unreliable and
# fall back to the A-scan-unified sym. With IoU < 0.10 the geometry barely
# overlaps anywhere; the matcher's choice is then essentially random.
SYM_IOU_FALLBACK_THR = 0.10


# ── per-sample STL → 3D mask ────────────────────────────────────────────────
def voxelize_gt_3d(stl_path, depths_mm, ny=NY, nx=NX, px_mm=PX_MM,
                   z_shift_mm=0.0):
    """Return (NZ, NY, NX) hard 0/1 mask of STL voids on the supplied depth
    grid. Re-uses the per-component logic from void_characterize.voxelize_gt
    but keeps the 3D mask instead of just the z-projection.

    Parameters
    ----------
    z_shift_mm : float, default 0.0
        Additive shift applied to each void's CAD-frame z bounds before they
        are placed on the apparent THz depth grid `depths_mm`. The pipeline's
        surface detector picks a sub-surface peak (see Wiki §11.6), so the
        apparent depth axis is offset relative to CAD; a negative z_shift_mm
        moves STL voids shallower to compensate.
    """
    mesh  = trimesh.load(stl_path, force="mesh")
    comps = mesh.split(only_watertight=False)
    if not comps:
        return np.zeros((len(depths_mm), ny, nx), dtype=np.float32)
    bbox_vols = np.array([np.prod(c.bounding_box.extents) for c in comps])
    block_i   = int(np.argmax(bbox_vols))
    block     = comps[block_i]
    voids     = [c for i, c in enumerate(comps) if i != block_i]
    if not voids:
        return np.zeros((len(depths_mm), ny, nx), dtype=np.float32)

    shift = -block.bounds[0]
    xs = (np.arange(nx) + 0.5) * px_mm
    ys = (np.arange(ny) + 0.5) * px_mm
    z_min, z_max = float(depths_mm[0]), float(depths_mm[-1])
    nz = len(depths_mm)
    mask3d = np.zeros((nz, ny, nx), dtype=np.float32)

    for v in voids:
        v_shift = v.copy(); v_shift.apply_translation(shift)
        bb_min, bb_max = v_shift.bounds
        # CAD-frame centroid filtered against the apparent depth window after
        # accounting for the z-shift.
        cz_app = float((bb_min[2] + bb_max[2]) / 2) + z_shift_mm
        if cz_app < z_min or cz_app > z_max:
            continue
        vol_bbox = float(np.prod(bb_max - bb_min))
        if vol_bbox < MIN_GT_VOLUME_MM3:
            continue

        x_lo = max(0, int(bb_min[0] / px_mm))
        x_hi = min(nx, int(np.ceil(bb_max[0] / px_mm)) + 1)
        y_lo = max(0, int(bb_min[1] / px_mm))
        y_hi = min(ny, int(np.ceil(bb_max[1] / px_mm)) + 1)
        # The CAD void at CAD-z = z_cad lands at apparent-z = z_cad + z_shift.
        # Equivalently, an apparent depth d corresponds to CAD depth d - z_shift.
        z_in = ((depths_mm - z_shift_mm) >= bb_min[2]) & \
               ((depths_mm - z_shift_mm) <= bb_max[2])
        if not (x_hi > x_lo and y_hi > y_lo and z_in.any()):
            continue
        z_idx = np.where(z_in)[0]

        used_mesh = False
        if v_shift.is_watertight and len(v_shift.faces) <= CONTAINS_MAX_FACES:
            xs_v = xs[x_lo:x_hi]; ys_v = ys[y_lo:y_hi]
            # Query mesh.contains() in CAD coordinates: take apparent depths
            # and undo the shift.
            zs_v = depths_mm[z_idx] - z_shift_mm
            X, Y, Z = np.meshgrid(xs_v, ys_v, zs_v, indexing="xy")
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
            try:
                sub = (v_shift.contains(pts)
                       .reshape(len(ys_v), len(xs_v), len(zs_v))
                       .transpose(2, 0, 1)).astype(np.float32)   # (nz_v, ny_v, nx_v)
                if sub.any():
                    for k, zi in enumerate(z_idx):
                        mask3d[zi, y_lo:y_hi, x_lo:x_hi] = np.maximum(
                            mask3d[zi, y_lo:y_hi, x_lo:x_hi], sub[k])
                    used_mesh = True
            except Exception:
                pass

        if not used_mesh:
            for zi in z_idx:
                mask3d[zi, y_lo:y_hi, x_lo:x_hi] = 1.0

    return mask3d


# ── XY symmetry on a (Z, Y, X) mask, matching void_characterize.apply_symmetry ─
# apply_symmetry on centroids:
#   if mir: x = box - x
#   rot 0: identity
#   rot 1: (x, y) → (y, box - x)
#   rot 2: (x, y) → (box - x, box - y)
#   rot 3: (x, y) → (box - y, x)
def apply_sym_mask(m, sym_id):
    rot = sym_id // 2; mir = sym_id % 2
    if mir:
        m = m[:, :, ::-1]                              # flip X
    if rot == 1:
        m = m.transpose(0, 2, 1)[:, ::-1, :]           # 90° CW
    elif rot == 2:
        m = m[:, ::-1, ::-1]                            # 180°
    elif rot == 3:
        m = m.transpose(0, 2, 1)[:, :, ::-1]            # 90° CCW
    return np.ascontiguousarray(m)


def manual_blob_centroids(manual_mask, depths_mm, px_mm=PX_MM):
    """Connected components of the manual mask's z-projection → centroids in
    mm, packaged as a list of {'centroid_mm': (cx, cy, cz)} dicts so that
    void_characterize.best_alignment_and_match can consume them.
    """
    proj2d = (manual_mask.max(axis=0) > 0).astype(np.uint8)
    labels, n = cc_label(proj2d, structure=np.ones((3, 3), dtype=bool))
    meta = []
    for bid in range(1, n + 1):
        ys, xs = np.where(labels == bid)
        if ys.size < 4:           # drop sub-mm noise blobs
            continue
        cx_mm = float((xs.mean() + 0.5) * px_mm)
        cy_mm = float((ys.mean() + 0.5) * px_mm)
        z_any = manual_mask[:, ys, xs].max(axis=1) > 0
        if not z_any.any():
            continue
        z_idx = np.where(z_any)[0]
        cz_mm = float(depths_mm[z_idx].mean())
        meta.append(dict(centroid_mm=(cx_mm, cy_mm, cz_mm)))
    return meta


def pick_sym(stl_mask_3d, manual_mask, unified_sym, name=None,
             iou_thr=SYM_IOU_FALLBACK_THR):
    """Pick the in-plane symmetry that aligns STL voids with the scan frame.

    Priority order:
      1. SYM_OVERRIDE[name] if present (manual override for problem samples).
      2. The symmetry that maximises STL z-proj ∩ manual z-proj IoU,
         *if* that IoU exceeds `iou_thr`.
      3. The A-scan unified sym for `name` (passed as `unified_sym`).
      4. CAD-frame identity (sym=0) if everything above is unavailable.
    """
    if name is not None and name in SYM_OVERRIDE:
        return int(SYM_OVERRIDE[name]), f"manual-override={SYM_OVERRIDE[name]}"

    if manual_mask is None or not manual_mask.any() or not stl_mask_3d.any():
        return unified_sym, "no-manual"

    man_proj = (manual_mask.max(axis=0) > 0).astype(bool)
    best_sym_id = 0
    best_iou    = -1.0
    for sym_id in range(8):
        m_sym = apply_sym_mask(stl_mask_3d, sym_id)
        stl_proj = (m_sym.max(axis=0) > 0.5)
        inter = int((stl_proj & man_proj).sum())
        union = int((stl_proj | man_proj).sum())
        iou   = inter / max(union, 1)
        if iou > best_iou:
            best_iou = iou; best_sym_id = sym_id

    if best_iou <= 0:
        return unified_sym, "no-overlap-fallback"
    if best_iou < iou_thr:
        # Manual is too sparse to disambiguate; trust the A-scan unified sym
        # (it had access to the model's blob predictions, which are usually
        # more reliable than a 1–2 blob manual mask).
        return unified_sym, f"low-iou={best_iou:.2f}-ascan-fallback"
    return best_sym_id, f"manual-IoU={best_iou:.2f}"


def _verify_symmetry():
    """Unit test: applying apply_sym_mask matches the centroid transform
    apply_symmetry from void_characterize (which the unified matcher uses)."""
    from void_characterize import apply_symmetry as cent_sym
    rng = np.random.default_rng(0)
    for _ in range(8):
        z, r, c = int(rng.integers(0, 50)), int(rng.integers(0, NY)), int(rng.integers(0, NX))
        m = np.zeros((50, NY, NX), dtype=np.float32)
        m[z, r, c] = 1.0
        for sym_id in range(8):
            m_t = apply_sym_mask(m, sym_id)
            # Expected centroid after sym
            cx = (c + 0.5) * PX_MM; cy = (r + 0.5) * PX_MM; cz = 0.0
            ex, ey, _ = cent_sym((cx, cy, cz), sym_id, box_mm=NX * PX_MM)
            er = int(ey / PX_MM); ec = int(ex / PX_MM)
            ys, xs = np.where(m_t[z] > 0.5)
            assert ys.size == 1, f"sym {sym_id} (z={z},r={r},c={c}): {ys.size} hits"
            assert (ys[0] == er) and (xs[0] == ec), (
                f"sym {sym_id}: expected ({er},{ec}), got ({ys[0]},{xs[0]})")
    print("✓ symmetry mask transform matches centroid transform (8/8 sym, 8 random points)")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    args = _CLI_ARGS
    # Parse positional then optional named flags
    sig_xy = float(args[0]) if len(args) >= 1 else 1.0
    sig_z  = float(args[1]) if len(args) >= 2 else 1.0
    z_shift_default = float(args[2]) if len(args) >= 3 else 0.0
    # --viz / --train flags toggle which output dir + NO_VOID set we use
    viz_mode = "--viz" in _CLI_FLAGS
    joint_mode = "--joint-align" in _CLI_FLAGS
    # --cad-frame disables ALL transforms so the STL panel matches the raw
    # .stl file exactly (sym=0, z_shift=0, no symmetry pick). Used for visual
    # verification of the design geometry against the CAD source.
    cad_frame = "--cad-frame" in _CLI_FLAGS
    if viz_mode:
        out_dir = Path("labels_stl_atlanta2_viz")
        no_void_set = NO_VOID_STL_VIZ
        out_dir.mkdir(exist_ok=True)
    else:
        out_dir = OUT_LABELS
        no_void_set = NO_VOID_STL_TRAIN

    print(f"σ_xy = {sig_xy:.2f} px ({sig_xy*PX_MM:.2f} mm),  "
          f"σ_z  = {sig_z:.2f} slices,  z_shift_default = {z_shift_default:+.2f} mm,  "
          f"mode = {'viz' if viz_mode else 'train'}{' +joint-align' if joint_mode else ''}")
    print(f"output dir = {out_dir}/   no-void set = {sorted(no_void_set)}")

    _verify_symmetry()

    with open(UNIFIED_JSON) as fh:
        unified = json.load(fh)
    best_sym = {r["sample"]: int(r.get("best_symmetry", 0)) for r in unified}

    rows = []
    for name in sorted(SAMPLE_TO_STL, key=lambda x: (int(x.rstrip("b")), x)):
        if name == "5b":
            continue
        depths_path = SLICES_DIR / f"{name}_depths.npy"
        if not depths_path.exists():
            print(f"  skip {name}: no depths file"); continue
        depths = np.load(depths_path).astype(np.float32)
        out_path = out_dir / f"{name}_slice_masks.npy"

        if name in no_void_set:
            mask = np.zeros((len(depths), NY, NX), dtype=np.float32)
            np.save(out_path, mask)
            rows.append(dict(name=name, n_pos_slices=0, frac=0.0, sym=-1,
                             source="NO_VOID"))
            print(f"  {name}: NO_VOID → all-zero mask")
            continue

        t0 = time.time()
        stl_path = STL_DIR / SAMPLE_TO_STL[name]

        man_mask = None
        man_path = MANUAL_LABELS / f"{name}_slice_masks.npy"
        if man_path.exists():
            man_mask = np.load(man_path).astype(np.uint8)

        if cad_frame:
            # Pure CAD frame: no sym, no z-shift, no symmetry pick.
            m3d = voxelize_gt_3d(stl_path, depths, z_shift_mm=0.0)
            sym = 0
            z_shift_used = 0.0
            sym_reason = "cad-frame (sym=0 z=0)"
        elif joint_mode:
            # Voxelize at CAD frame; the joint search applies sym + z-shift.
            m_cad = voxelize_gt_3d(stl_path, depths, z_shift_mm=0.0)
            ascan_pred = predict_ascan_volume(name)
            dz_mm = float(np.diff(depths).mean())
            best = joint_align(m_cad, man_mask, ascan_pred, dz_mm)
            sym = best["sym"]
            z_shift_used = best["z_shift_mm"]
            sym_reason = (f"joint sym={sym} z={z_shift_used:+.2f}mm "
                          f"score={best['score']:.2f} "
                          f"(iou_a={best['iou_ascan']:.2f}, "
                          f"iou_m={best['iou_manual']:.2f})")
            # Apply z-shift + sym
            z_slices = int(round(z_shift_used / dz_mm))
            m3d = apply_sym_mask(shift_z(m_cad, z_slices), sym)
        else:
            z_shift_used = Z_SHIFT_OVERRIDE.get(name, z_shift_default)
            m3d = voxelize_gt_3d(stl_path, depths, z_shift_mm=z_shift_used)
            if viz_mode and name in SYM_OVERRIDE_VIZ:
                sym = int(SYM_OVERRIDE_VIZ[name])
                sym_reason = f"manual-override-viz={sym}"
            else:
                sym, sym_reason = pick_sym(m3d, man_mask,
                                           best_sym.get(name, 0), name=name)
            if sym >= 0:
                m3d = apply_sym_mask(m3d, sym)

        if sig_xy > 0 or sig_z > 0:
            m3d = gaussian_filter(m3d, sigma=(sig_z, sig_xy, sig_xy))
            m3d = np.clip(m3d, 0.0, 1.0)

        np.save(out_path, m3d.astype(np.float32))
        n_pos_slices = int((m3d.max(axis=(1, 2)) > 0.5).sum())
        frac = float((m3d > 0.5).sum()) / m3d.size
        rows.append(dict(name=name, n_pos_slices=n_pos_slices, frac=frac,
                         sym=sym, z_shift_mm=z_shift_used, source=sym_reason))
        print(f"  {name}: {sym_reason}  "
              f"pos_slices={n_pos_slices:>3d}/{len(depths)}  "
              f"frac={frac*100:.2f}%  in {time.time()-t0:.1f}s")

    # ── QC figure: red = STL, blue = manual (z-projection) ────────────────
    fig, axes = plt.subplots(3, 5, figsize=(13, 7.8), dpi=140)
    fig.patch.set_facecolor("white")
    order = sorted([n for n in SAMPLE_TO_STL if n != "5b"],
                   key=lambda x: int(x.rstrip("b")))
    for i, name in enumerate(order):
        ax = axes.flat[i]
        stl_p = out_dir / f"{name}_slice_masks.npy"
        man_p = MANUAL_LABELS / f"{name}_slice_masks.npy"
        if not stl_p.exists():
            ax.axis("off"); continue
        stl_mask = np.load(stl_p)
        stl_proj = stl_mask.max(axis=0)
        man_proj = np.zeros((NY, NX), dtype=np.float32)
        if man_p.exists():
            mm = np.load(man_p).astype(np.float32)
            man_proj = (mm.max(axis=0) > 0).astype(np.float32)
        rgb = np.zeros((NY, NX, 3), dtype=np.float32)
        rgb[..., 0] = stl_proj
        rgb[..., 2] = man_proj
        ax.imshow(rgb, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        sym = next((r["sym"] for r in rows if r["name"] == name), "?")
        ax.set_title(f"S{name}  sym={sym}", fontsize=9, fontweight="bold")

    z_shift_label = "per-sample" if joint_mode else f"{z_shift_default:+.2f}mm"
    plt.suptitle(f"STL (red) vs manual (blue) — z-projection   "
                 f"σ_xy={sig_xy:.1f}px σ_z={sig_z:.1f}sl z_shift={z_shift_label} "
                 f"({'viz' if viz_mode else 'train'}{' +joint' if joint_mode else ''})",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.02, 1, 0.97])
    suffix = (f"sigxy{sig_xy:.1f}_sigz{sig_z:.1f}_"
              f"{'joint' if joint_mode else f'zsh{z_shift_default:+.2f}'}_"
              f"{'viz' if viz_mode else 'train'}")
    fig_path = OUT_QC_DIR / f"stl_vs_manual_{suffix}.png"
    plt.savefig(fig_path, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"\nQC figure: {fig_path}")
    print(f"STL labels: {out_dir}/  ({len(rows)} samples)")


if __name__ == "__main__":
    main()
