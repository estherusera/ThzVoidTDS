"""
ascan_classifier.py — Depth-aware 1D-CNN void classifier on raw A-scans.

Input  : one surface-flattened A-scan per pixel (length 2050 time points).
Output : 50-dim void probability vector — one per depth bin, aligned with the
         50 depth slices already used for the U-Net training labels.

Datasets
  - atlanta2 (clean, isotropic 0.2 mm)  — used for TRAIN / VAL
  - atlanta1 (mixed-resolution)         — held out entirely for cross-domain test

Splits (atlanta2)
  - TRAIN  : samples 1,2,3,4,5,6,7,8,9,10,12,15
  - VAL    : samples 11, 13
  - NOVOID : sample 14 (sanity)
Cross-domain test (atlanta1): all samples with manual labels.

Run
  thesis_env/bin/python ascan_classifier.py prepare-atlanta2  # build atlanta2 cache
  thesis_env/bin/python ascan_classifier.py prepare-atlanta1  # build atlanta1 cache
  thesis_env/bin/python ascan_classifier.py train             # train depth-aware CNN
  thesis_env/bin/python ascan_classifier.py eval              # eval atlanta2 + atlanta1
"""

import sys, json, time, os
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import hilbert
import trimesh

# ── CLI ──────────────────────────────────────────────────────────────────────
_CLI = sys.argv[1] if len(sys.argv) > 1 else None
sys.argv = [sys.argv[0], "predict", "atlanta2"]
sys.path.insert(0, str(Path(__file__).parent))
from thz_slice_pipeline import load_all_volumes, bandpass_filter, BANDPASS_ENABLED

# ── paths ────────────────────────────────────────────────────────────────────
TPRJ_ATLANTA2 = "3D_print_esther_atlanta2.tprj"
TPRJ_ATLANTA1 = "3D_print_esther_atlanta.tprj"
ROI_JSON_2    = "sample_rois_atlanta2.json"
ROI_JSON_1    = "sample_rois.json"
MANUAL_DIR_2  = Path("labels_v2_atlanta2")
MANUAL_DIR_1  = Path("labels_v2")
STL_DIR       = Path("AllSamples/stl")

# ASCAN_VARIANT labels a run (e.g. "_bp" band-pass-filtered, "_nofilt" baseline)
# so the checkpoint and eval outputs never clobber another run. The cache dirs
# are overridable separately (ASCAN_CACHE2/1) so a run can reuse an existing
# cache instead of rebuilding it (e.g. the baseline reuses the unfiltered cache).
VARIANT       = os.environ.get("ASCAN_VARIANT", "")
CACHE_ATL2    = Path(os.environ.get("ASCAN_CACHE2", "ascan_cache_atlanta2"))
CACHE_ATL1    = Path(os.environ.get("ASCAN_CACHE1", "ascan_cache_atlanta1"))
OUT_DIR       = Path("results_v2_atlanta2/ascan_classifier")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_ATL2.mkdir(parents=True, exist_ok=True)
CACHE_ATL1.mkdir(parents=True, exist_ok=True)
MODEL_CKPT    = OUT_DIR / f"ascan_cnn_depth{VARIANT}.pt"

# ── name mappings (consistent with rest of project) ──────────────────────────
NAME_MAP_2 = {**{str(i): str(i) for i in range(1, 16)}, "sample 6": "6"}
# atlanta1 — physical-name remap
NAME_MAP_1 = {
    "1":"1","2":"2","3":"3","4":"4","5":"5","5b":"5b",
    "6-9":"6","7":"7","8":"8","9-6.":"9",
    "10":"10","11":"11","12":"12","13":"13","14":"14","15":"15",
}
SPACING_OVERRIDES_1 = {"1": dict(dx_mm=0.2, dy_mm=0.5)}

# Scan-sample → STL filename (off-by-one above CV07)
SAMPLE_TO_STL = {
    "1":"CV01.stl","2":"CV02.stl","3":"CV03.stl","4":"CV04.stl","5":"CV05.stl",
    "6":"CV06.stl","7":"CV08.stl","8":"SampleRayas_CV09.stl",
    "9":"SampleRayasDiagonal_CV10.stl","10":"ManyStripes_CV11.stl",
    "11":"PrismVoids_CV12.stl","12":"random_voids_CV13.stl",
    "13":"Cilindros_CV14.stl","14":"NoVoid_CV15.stl","15":"Inclinedplanes_CV16.stl",
    "5b":"CV05.stl",
}

# ── splits ───────────────────────────────────────────────────────────────────
TRAIN_SAMPLES = ["1","2","3","4","5","6","7","8","9","10","12","15"]
VAL_SAMPLES   = ["11","13"]
NOVOID_SANITY = ["14"]

# ── config ───────────────────────────────────────────────────────────────────
NX = NY        = 100
N_TIME         = 2050
N_DEPTHS       = 50
BATCH_SIZE     = 256
N_EPOCHS       = int(os.environ.get("ASCAN_EPOCHS", "30"))
LR             = 1e-3
POS_WEIGHT     = 8.0

device = ("cuda" if torch.cuda.is_available()
          else "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
          else "cpu")


# ── Surface flatten ──────────────────────────────────────────────────────────
def flatten_volume(sample):
    vol = sample["volume"]; tax = sample["time_axis"]
    nt, ny, nx = vol.shape
    # Surface alignment is detected on the RAW envelope (sharp, unambiguous
    # front-surface pulse). Band-pass would turn that pulse into a ringing wavelet
    # and make argmax snap regions onto different lobes → misaligned A-scans. We
    # align on the raw pulse, then store the (optionally) band-pass-denoised
    # waveform rolled by those same shifts.
    env = np.abs(hilbert(vol, axis=0))
    surf = int(np.argmax(env.mean(axis=(1, 2))))
    m = int(0.2 * nt)
    s0, s1 = max(0, surf - m), min(nt, surf + m)
    local  = s0 + np.argmax(env[s0:s1], axis=0)
    target = int(np.median(local))
    if BANDPASS_ENABLED:
        vol = bandpass_filter(vol, tax)   # denoise the values; alignment unchanged
    flat = np.empty_like(vol, dtype=np.float32)
    for iy in range(ny):
        for ix in range(nx):
            flat[:, iy, ix] = np.roll(vol[:, iy, ix], target - local[iy, ix])
    return flat, tax, target


# ── Cache builder ────────────────────────────────────────────────────────────
def prepare_cache(dataset):
    """Cache per-sample (ascans, labels_3d) for one dataset."""
    if dataset == "atlanta2":
        tprj, roi_json = TPRJ_ATLANTA2, ROI_JSON_2
        name_map, manual_dir, cache = (NAME_MAP_2, MANUAL_DIR_2, CACHE_ATL2)
        spacing_overrides = {}
        zoom_needed = False  # atlanta2 is already isotropic
    elif dataset == "atlanta1":
        tprj, roi_json = TPRJ_ATLANTA1, ROI_JSON_1
        name_map, manual_dir, cache = (NAME_MAP_1, MANUAL_DIR_1, CACHE_ATL1)
        spacing_overrides = SPACING_OVERRIDES_1
        zoom_needed = True
    else:
        sys.exit(f"unknown dataset: {dataset}")

    print(f"Loading {dataset} tprj…")
    raw = load_all_volumes([tprj])
    with open(roi_json) as fh:
        rois = json.load(fh)

    from scipy.ndimage import zoom as nd_zoom

    for s in raw:
        tprj_name = s["name"].split("/")[-1]
        if tprj_name.lower().startswith("test"):
            continue
        phys = name_map.get(tprj_name, tprj_name)
        if phys not in rois:
            print(f"  ✗ {phys}: no ROI, skipping"); continue

        cache_path = cache / f"{phys}.npz"
        if cache_path.exists():
            print(f"  ✓ {phys} cached"); continue

        t = time.time()
        if phys in spacing_overrides:
            s = dict(s); s.update(spacing_overrides[phys])

        flat, tax, target = flatten_volume(s)

        # Isotropic zoom (atlanta1 only)
        if zoom_needed:
            dx, dy = s.get("dx_mm", 0.2), s.get("dy_mm", 0.2)
            if abs(dx - dy) / max(dx, dy) > 0.05:
                t0 = min(dx, dy)
                flat = nd_zoom(flat, (1, dy / t0, dx / t0), order=1)

        roi = rois[phys]
        flat = flat[:, roi["r0"]:roi["r1"], roi["c0"]:roi["c1"]]   # (2050, 100, 100)

        # Per-pixel A-scans
        ascans = flat.transpose(1, 2, 0).reshape(-1, flat.shape[0])  # (10000, 2050)

        # 3D manual mask: (50, 100, 100) → per-pixel (10000, 50)
        mask_path = manual_dir / f"{phys}_slice_masks.npy"
        if mask_path.exists():
            m3d = np.load(mask_path).astype(np.uint8)           # (50, 100, 100)
            labels_3d = m3d.transpose(1, 2, 0).reshape(-1, N_DEPTHS)
        else:
            labels_3d = np.zeros((NY * NX, N_DEPTHS), dtype=np.uint8)

        np.savez_compressed(
            cache_path,
            ascans=ascans.astype(np.float32),
            labels_3d=labels_3d.astype(np.uint8),
            tax=tax.astype(np.float32),
            target_idx=np.int32(target),
        )
        n_void_2d = int((labels_3d.max(axis=1) > 0).sum())
        n_void_3d = int(labels_3d.sum())
        print(f"  ✓ {phys}  ascans={ascans.shape}  "
              f"void_pix_2d={n_void_2d}  void_voxels_3d={n_void_3d}  "
              f"in {time.time()-t:.1f}s")
    print(f"\nCache written to {cache}/")


# ── Dataset ──────────────────────────────────────────────────────────────────
class AScanDataset(Dataset):
    """Each item: (1×2050 A-scan, 50-dim binary depth label)."""

    def __init__(self, sample_names, cache_dir, augment=False):
        self.items = []
        for name in sample_names:
            p = cache_dir / f"{name}.npz"
            if not p.exists():
                print(f"  ! missing cache {p}")
                continue
            d = np.load(p)
            self.items.append((d["ascans"], d["labels_3d"], name))
        self.lengths = [a.shape[0] for a, _, _ in self.items]
        self.cumlen  = np.cumsum([0] + self.lengths)
        self.augment = augment

    def __len__(self):
        return int(self.cumlen[-1])

    def __getitem__(self, idx):
        si = int(np.searchsorted(self.cumlen, idx, side="right") - 1)
        pi = idx - int(self.cumlen[si])
        a = self.items[si][0][pi].astype(np.float32)
        y = self.items[si][1][pi].astype(np.float32)        # (50,)

        # Per-A-scan z-score
        m = a.mean(); s = a.std() + 1e-6
        a = (a - m) / s

        if self.augment:
            shift = int(np.random.randint(-5, 6))
            if shift:
                a = np.roll(a, shift)
            a = a + 0.02 * np.random.randn(*a.shape).astype(np.float32)

        x = torch.from_numpy(a).unsqueeze(0)                # (1, 2050)
        return x, torch.from_numpy(y)


# ── Depth-aware Model ────────────────────────────────────────────────────────
class AScanCNN(nn.Module):
    """Encoder → AdaptiveAvgPool1d(50) → 1×1 Conv → 50-dim logits."""

    def __init__(self, n_depths=N_DEPTHS):
        super().__init__()
        self.n_depths = n_depths
        self.encoder = nn.Sequential(
            nn.Conv1d(1,   32, 7, padding=3),  nn.BatchNorm1d(32),  nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                                    # 2050 → 1025
            nn.Conv1d(32,  64, 5, padding=2),  nn.BatchNorm1d(64),  nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                                    # 1025 →  512
            nn.Conv1d(64, 128, 5, padding=2),  nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.MaxPool1d(2),                                                    #  512 →  256
            nn.Conv1d(128,128, 3, padding=1),  nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        # Global average pool over the temporal dim → (B, 128), then Linear → 50 depths.
        # AdaptiveAvgPool1d(1) is divisible-compatible on MPS (output size 1 is always valid).
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),                       # (B, 128, 1) → (B, 128)
            nn.Dropout(0.3),
            nn.Linear(128, n_depths),           # (B, 50)
        )

    def forward(self, x):
        # x: (B, 1, 2050)
        feats  = self.encoder(x)                 # (B, 128, T'=256)
        pooled = self.pool(feats)                # (B, 128, 1)
        return self.head(pooled)                 # (B, 50)


# ── Train ────────────────────────────────────────────────────────────────────
def train():
    train_ds = AScanDataset(TRAIN_SAMPLES, CACHE_ATL2, augment=True)
    val_ds   = AScanDataset(VAL_SAMPLES,   CACHE_ATL2, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    n_train_pos_2d = sum(int((d[1].max(axis=1) > 0).sum()) for d in train_ds.items)
    n_train_tot    = len(train_ds)
    n_train_pos_3d = sum(int(d[1].sum()) for d in train_ds.items)
    n_train_neg_3d = n_train_tot * N_DEPTHS - n_train_pos_3d
    print(f"\nTrain: {n_train_tot} A-scans  "
          f"({n_train_pos_2d} XY void = {n_train_pos_2d/n_train_tot:.1%}, "
          f"{n_train_pos_3d} void voxels = {n_train_pos_3d/(n_train_tot*N_DEPTHS):.1%})")
    print(f"Val  : {len(val_ds)} A-scans across {VAL_SAMPLES}")

    model = AScanCNN().to(device)
    n_p = sum(p.numel() for p in model.parameters())
    print(f"Model: AScanCNN (depth-aware), {n_p:,} params, device={device}")

    pos_w = torch.tensor([POS_WEIGHT], device=device)
    crit  = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, N_EPOCHS)

    history = {"train_loss": [], "train_f1_3d": [],
               "val_loss":   [], "val_f1_3d":   [],
               "val_p_3d":   [], "val_r_3d":    []}
    best_val_f1 = -1.0
    t0 = time.time()
    metrics_path = OUT_DIR / f"metrics{VARIANT}.jsonl"
    # truncate from previous runs
    with open(metrics_path, "w") as fh:
        fh.write("")

    print("─" * 90)
    for epoch in range(N_EPOCHS):
        model.train()
        ep_loss = ep_tp = ep_fp = ep_fn = 0; nb = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)                       # (B, 50)
            loss = crit(logits, y)
            optim.zero_grad(); loss.backward(); optim.step()
            with torch.no_grad():
                p = (torch.sigmoid(logits) > 0.5).float()
                ep_tp += float(((p == 1) & (y == 1)).sum())
                ep_fp += float(((p == 1) & (y == 0)).sum())
                ep_fn += float(((p == 0) & (y == 1)).sum())
            ep_loss += loss.item(); nb += 1
        sched.step()
        train_f1 = 2*ep_tp / max(2*ep_tp + ep_fp + ep_fn, 1.0)
        history["train_loss"].append(ep_loss/nb)
        history["train_f1_3d"].append(train_f1)

        # ── Validation with per-depth metric accumulation ────────────────
        model.eval()
        v_loss = 0; vb = 0
        # Per-depth TP/FP/FN (50,)
        per_d_tp = np.zeros(N_DEPTHS, dtype=np.int64)
        per_d_fp = np.zeros(N_DEPTHS, dtype=np.int64)
        per_d_fn = np.zeros(N_DEPTHS, dtype=np.int64)
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x); loss = crit(logits, y)
                p = (torch.sigmoid(logits) > 0.5).float()
                tp_d = ((p == 1) & (y == 1)).sum(dim=0).cpu().numpy()
                fp_d = ((p == 1) & (y == 0)).sum(dim=0).cpu().numpy()
                fn_d = ((p == 0) & (y == 1)).sum(dim=0).cpu().numpy()
                per_d_tp += tp_d.astype(np.int64)
                per_d_fp += fp_d.astype(np.int64)
                per_d_fn += fn_d.astype(np.int64)
                v_loss += loss.item(); vb += 1

        # Aggregate (sum over depths) metrics
        v_tp = int(per_d_tp.sum()); v_fp = int(per_d_fp.sum()); v_fn = int(per_d_fn.sum())
        val_f1 = 2*v_tp / max(2*v_tp + v_fp + v_fn, 1.0)
        val_p  = v_tp / max(v_tp + v_fp, 1.0)
        val_r  = v_tp / max(v_tp + v_fn, 1.0)
        # Per-depth F1
        denom_d = 2*per_d_tp + per_d_fp + per_d_fn
        per_d_f1 = np.where(denom_d > 0,
                            (2 * per_d_tp) / np.maximum(denom_d, 1),
                            0.0).astype(float).tolist()

        history["val_loss"].append(v_loss/vb)
        history["val_f1_3d"].append(val_f1)
        history["val_p_3d"].append(val_p); history["val_r_3d"].append(val_r)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"model_state": model.state_dict(),
                        "history": history,
                        "train_samples": TRAIN_SAMPLES,
                        "val_samples": VAL_SAMPLES,
                        "epoch": epoch + 1}, MODEL_CKPT)

        # Append metrics.jsonl row for this epoch (one self-contained line)
        with open(metrics_path, "a") as fh:
            fh.write(json.dumps({
                "epoch": epoch + 1,
                "elapsed_min": (time.time() - t0) / 60,
                "train_loss": ep_loss / nb,
                "train_f1_3d": train_f1,
                "val_loss":  v_loss / vb,
                "val_f1_3d": val_f1,
                "val_p_3d":  val_p,
                "val_r_3d":  val_r,
                "val_per_depth_f1": per_d_f1,
                "best_so_far_val_f1": best_val_f1,
                "is_best": (val_f1 == best_val_f1),
            }) + "\n")

        print(f"  Epoch {epoch+1:3d}/{N_EPOCHS} | "
              f"train loss {ep_loss/nb:.4f}  F1_3D {train_f1:.3f} | "
              f"val loss {v_loss/vb:.4f}  F1_3D {val_f1:.3f}  P {val_p:.3f}  R {val_r:.3f} | "
              f"{(time.time()-t0)/60:.1f} min")
    print("─" * 90)
    print(f"Best val F1_3D: {best_val_f1:.3f}  →  saved {MODEL_CKPT}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=140)
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"],   label="val")
    axes[0].set_yscale("log"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].set_title("Loss")
    axes[1].plot(history["train_f1_3d"], label="train F1 (3D)")
    axes[1].plot(history["val_f1_3d"],   label="val F1 (3D)")
    axes[1].plot(history["val_p_3d"],    label="val P", ls=":")
    axes[1].plot(history["val_r_3d"],    label="val R", ls=":")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].set_title("3D voxel metrics")
    plt.suptitle("Depth-aware A-scan CNN — training", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"training_curves_depth{VARIANT}.png", dpi=140, bbox_inches="tight")
    print(f"Saved: {OUT_DIR}/training_curves_depth{VARIANT}.png")


# ── STL z-projection (for evaluation only) ───────────────────────────────────
def stl_projection(name):
    if name not in SAMPLE_TO_STL:
        return np.zeros((NY, NX), dtype=bool)
    stl_path = STL_DIR / SAMPLE_TO_STL[name]
    if not stl_path.exists():
        return np.zeros((NY, NX), dtype=bool)
    mesh  = trimesh.load(stl_path, force="mesh")
    comps = mesh.split(only_watertight=False)
    if not comps:
        return np.zeros((NY, NX), dtype=bool)
    bbox_vols = np.array([np.prod(c.bounding_box.extents) for c in comps])
    block_i   = int(np.argmax(bbox_vols))
    block     = comps[block_i]
    voids     = [c for i, c in enumerate(comps) if i != block_i]
    shift = -block.bounds[0]
    proj = np.zeros((NY, NX), dtype=bool)
    for v in voids:
        v_shift = v.copy(); v_shift.apply_translation(shift)
        bb_min, bb_max = v_shift.bounds
        if np.prod(bb_max - bb_min) < 0.05:
            continue
        px_mm = 0.2
        x_lo = max(0, int(bb_min[0] / px_mm))
        x_hi = min(NX, int(np.ceil(bb_max[0] / px_mm)) + 1)
        y_lo = max(0, int(bb_min[1] / px_mm))
        y_hi = min(NY, int(np.ceil(bb_max[1] / px_mm)) + 1)
        proj[y_lo:y_hi, x_lo:x_hi] = True
    return proj


# ── Eval helper ──────────────────────────────────────────────────────────────
def predict_sample(model, cache_path):
    d = np.load(cache_path)
    ascans = d["ascans"]; labels_3d = d["labels_3d"]
    mean = ascans.mean(axis=1, keepdims=True)
    std  = ascans.std(axis=1, keepdims=True) + 1e-6
    a_norm = (ascans - mean) / std
    a_t = torch.from_numpy(a_norm).unsqueeze(1).to(device)
    probs = np.zeros((a_t.shape[0], N_DEPTHS), dtype=np.float32)
    CHUNK = 512
    with torch.no_grad():
        for k in range(0, a_t.shape[0], CHUNK):
            logits = model(a_t[k:k+CHUNK])
            probs[k:k+CHUNK] = torch.sigmoid(logits).cpu().numpy()
    return probs, labels_3d                       # (N_pix, 50), (N_pix, 50)


def metrics(gt, pred):
    """Return P, R, F1, tp, fp, fn for boolean arrays of arbitrary shape."""
    g = gt.astype(bool); p = pred.astype(bool)
    tp = int((g & p).sum())
    fp = int((~g & p).sum())
    fn = int((g & ~p).sum())
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1   = 2*tp / max(2*tp + fp + fn, 1)
    return prec, rec, f1, tp, fp, fn


def eval_one(name, model, cache_dir):
    if not (cache_dir / f"{name}.npz").exists():
        return None
    probs_3d, gt_3d = predict_sample(model, cache_dir / f"{name}.npz")  # (N_pix, 50)
    pred_3d = (probs_3d > 0.5).astype(np.uint8)

    # 3D voxel metrics (every (pixel, depth) voxel)
    p3, r3, f1_3, *_ = metrics(gt_3d, pred_3d)

    # 2D XY metrics — predicted void at any depth vs GT void at any depth
    gt_2d   = (gt_3d.max(axis=1) > 0).reshape(NY, NX)
    pred_2d = (pred_3d.max(axis=1) > 0).reshape(NY, NX)
    p2, r2, f1_2, *_ = metrics(gt_2d, pred_2d)

    # STL z-projection comparison (only when STL exists, atlanta2 style names)
    stl = stl_projection(name)
    ps, rs, f1_s, *_ = metrics(stl, pred_2d)

    # Probability volume per sample (for visualization)
    prob_vol_2d = probs_3d.max(axis=1).reshape(NY, NX)    # max-projection
    prob_3d     = probs_3d.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)  # (50,Y,X)

    return dict(
        name=name,
        n_pix=int(probs_3d.shape[0]),
        gt_voxels_3d=int(gt_3d.sum()),
        pred_voxels_3d=int(pred_3d.sum()),
        gt_2d_pos=int(gt_2d.sum()), pred_2d_pos=int(pred_2d.sum()),
        f1_3d=f1_3, p_3d=p3, r_3d=r3,
        f1_2d=f1_2, p_2d=p2, r_2d=r2,
        f1_stl=f1_s, p_stl=ps, r_stl=rs,
        prob_vol_2d=prob_vol_2d, prob_3d=prob_3d,
        gt_2d=gt_2d.astype(np.uint8), stl=stl.astype(np.uint8),
    )


def evaluate():
    ckpt  = torch.load(MODEL_CKPT, map_location=device, weights_only=False)
    model = AScanCNN().to(device)
    model.load_state_dict(ckpt["model_state"]); model.eval()
    print(f"Model loaded: epoch {ckpt['epoch']}")

    sd_root = OUT_DIR / "per_sample"; sd_root.mkdir(exist_ok=True)
    summary = []

    print("\n── atlanta2 (train/val/no-void) ──")
    print(f"{'sample':>6s} {'split':>7s} {'F1_3D':>7s} {'F1_2D':>7s} {'F1_STL':>7s} "
          f"{'P_3D':>6s} {'R_3D':>6s}")
    for name in TRAIN_SAMPLES + VAL_SAMPLES + NOVOID_SANITY:
        r = eval_one(name, model, CACHE_ATL2)
        if r is None: continue
        split = "TRAIN" if name in TRAIN_SAMPLES else ("VAL" if name in VAL_SAMPLES
                                                       else "NOVOID")
        r["split"] = split; r["dataset"] = "atlanta2"
        print(f"{name:>6s} {split:>7s} "
              f"{r['f1_3d']:>7.3f} {r['f1_2d']:>7.3f} {r['f1_stl']:>7.3f} "
              f"{r['p_3d']:>6.3f} {r['r_3d']:>6.3f}")
        # save probability volume per sample
        sd = sd_root / f"atlanta2_{name}"; sd.mkdir(exist_ok=True)
        np.save(sd / "prob_2d.npy",  r["prob_vol_2d"])
        np.save(sd / "prob_3d.npy",  r["prob_3d"])
        summary.append(r)

    # ── atlanta1 cross-domain ────────────────────────────────────────────────
    cross_results = []
    if CACHE_ATL1.exists() and any(CACHE_ATL1.glob("*.npz")):
        print("\n── atlanta1 cross-domain (held-out entirely) ──")
        print(f"{'sample':>6s} {'F1_3D':>7s} {'F1_2D':>7s} {'F1_STL':>7s} "
              f"{'P_3D':>6s} {'R_3D':>6s}")
        atl1_names = sorted([p.stem for p in CACHE_ATL1.glob("*.npz")],
                            key=lambda x: (int(x.rstrip("b")), x))
        for name in atl1_names:
            r = eval_one(name, model, CACHE_ATL1)
            if r is None: continue
            r["split"] = "CROSSDOMAIN"; r["dataset"] = "atlanta1"
            print(f"{name:>6s} {r['f1_3d']:>7.3f} {r['f1_2d']:>7.3f} "
                  f"{r['f1_stl']:>7.3f} {r['p_3d']:>6.3f} {r['r_3d']:>6.3f}")
            sd = sd_root / f"atlanta1_{name}"; sd.mkdir(exist_ok=True)
            np.save(sd / "prob_2d.npy",  r["prob_vol_2d"])
            np.save(sd / "prob_3d.npy",  r["prob_3d"])
            cross_results.append(r)
            summary.append(r)

    # ── aggregate ──
    print("\n── aggregate ──")
    def mean_split(rows, key):
        vals = [r[key] for r in rows]
        return float(np.mean(vals)) if vals else float("nan")
    for split, rows in [
        ("TRAIN",        [r for r in summary if r["split"] == "TRAIN"]),
        ("VAL",          [r for r in summary if r["split"] == "VAL"]),
        ("CROSSDOMAIN",  [r for r in summary if r["split"] == "CROSSDOMAIN"]),
    ]:
        if rows:
            print(f"  {split:>11s}  F1_3D={mean_split(rows,'f1_3d'):.3f}  "
                  f"F1_2D={mean_split(rows,'f1_2d'):.3f}  "
                  f"F1_STL={mean_split(rows,'f1_stl'):.3f}  (n={len(rows)})")

    # JSON dump (drop heavy numpy arrays)
    light = []
    for r in summary:
        light.append({k: v for k, v in r.items()
                      if k not in ("prob_vol_2d", "prob_3d", "gt_2d", "stl")})
    with open(OUT_DIR / f"summary_depth{VARIANT}.json", "w") as fh:
        json.dump(light, fh, indent=2, default=float)
    print(f"\nSaved: {OUT_DIR}/summary_depth{VARIANT}.json")


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    if _CLI == "prepare-atlanta2":
        prepare_cache("atlanta2")
    elif _CLI == "prepare-atlanta1":
        prepare_cache("atlanta1")
    elif _CLI == "train":
        if not CACHE_ATL2.exists() or not any(CACHE_ATL2.glob("*.npz")):
            prepare_cache("atlanta2")
        train()
        evaluate()
    elif _CLI == "eval":
        evaluate()
    else:
        print(__doc__); sys.exit(1)


if __name__ == "__main__":
    main()
