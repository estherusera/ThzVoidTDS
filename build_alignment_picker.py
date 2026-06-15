"""
build_alignment_picker.py — Interactive STL-alignment picker

Lets you pick the (symmetry, z-shift) per sample by visual inspection in a
self-contained HTML page. For each sample the page shows:

  • Input slice (red) — the actual THz scan
  • STL slice (orange) under the currently-selected (sym, z_shift)
  • Manual mask (blue) — for context

Controls:
  • Sample selector buttons (1..15)
  • 8 sym buttons (0..7) using the void_characterize convention
  • Z-shift slider in mm
  • Slice slider (depth)
  • Picks per sample are stored in the page; export to JSON when done.

The 8 sym versions are pre-rendered per slice in Python (so the JS is just
image-swap), and z-shift is implemented client-side as a slice-index offset.

Run:
    thesis_env/bin/python build_alignment_picker.py
"""
import sys, io, base64, json
from pathlib import Path
import numpy as np
import torch
import trimesh
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
sys.argv = [sys.argv[0], "predict", "atlanta2"]

from void_characterize import (SAMPLE_TO_STL, STL_DIR, voxelize_gt,
                                CONTAINS_MAX_FACES, MIN_GT_VOLUME_MM3)
from stl_to_labels import apply_sym_mask, DepthAwareAScanCNN
from ascan_classifier import CACHE_ATL2, N_DEPTHS
from thz_slice_pipelinev2 import UNet


# Per-sample pre-voxelization rotation of the .stl mesh around Z axis (deg).
# Used when the .stl's CAD XY orientation doesn't match the THz scan XY
# orientation.
#   s13 (CV14 Cilindros) — cylinders run along CAD X but the THz scan
#                          convention shows them vertical (along Y).
#   s8  (CV09 SampleRayas) — three stripes ordered short→long top-to-bottom
#                            in CAD; the scan shows them as vertical stripes
#                            with the LONGEST on the left ("longer rectangle
#                            first"). A 90° CCW rotation maps:
#                              void 2 (15 mm, was bottom) → left column
#                              void 1 (10 mm, was middle) → middle column
#                              void 0  (8 mm, was top)    → right column
#                            equivalent to sym=6, visually ≈ sym=7 (IoU 0.138
#                            vs 0.148 — they differ only by an X-mirror).
MESH_ROTATE_Z_DEG = {
    "8":  90.0,
    "13": 90.0,
}


def voxelize_gt_3d_rotated(stl_path, depths_mm, ny=100, nx=100, px_mm=0.2,
                            pre_rotate_z_deg=0.0):
    """Voxelize STL into a (NZ, NY, NX) hard mask with an optional pre-
    rotation of the mesh around its Z-axis. Re-implements the per-component
    logic of stl_to_labels.voxelize_gt_3d so the rotation can be applied
    BEFORE the block/void decomposition (otherwise the block bbox shift
    would be computed in the wrong frame).
    """
    mesh = trimesh.load(stl_path, force="mesh")
    if pre_rotate_z_deg != 0.0:
        angle_rad = np.deg2rad(pre_rotate_z_deg)
        center = mesh.centroid
        R = trimesh.transformations.rotation_matrix(
            angle_rad, [0.0, 0.0, 1.0], point=center)
        mesh.apply_transform(R)

    comps = mesh.split(only_watertight=False)
    if not comps:
        return np.zeros((len(depths_mm), ny, nx), dtype=np.float32)
    bbox_vols = np.array([np.prod(c.bounding_box.extents) for c in comps])
    block_i = int(np.argmax(bbox_vols))
    block = comps[block_i]
    voids = [c for i, c in enumerate(comps) if i != block_i]
    if not voids:
        return np.zeros((len(depths_mm), ny, nx), dtype=np.float32)

    shift = -block.bounds[0]
    xs = (np.arange(nx) + 0.5) * px_mm
    ys = (np.arange(ny) + 0.5) * px_mm
    z_min, z_max = float(depths_mm[0]), float(depths_mm[-1])
    nz = len(depths_mm)
    mask3d = np.zeros((nz, ny, nx), dtype=np.float32)

    for v in voids:
        v_s = v.copy(); v_s.apply_translation(shift)
        bb_min, bb_max = v_s.bounds
        cz_bb = float((bb_min[2] + bb_max[2]) / 2)
        if cz_bb < z_min or cz_bb > z_max:
            continue
        if float(np.prod(bb_max - bb_min)) < MIN_GT_VOLUME_MM3:
            continue
        x_lo = max(0, int(bb_min[0] / px_mm))
        x_hi = min(nx, int(np.ceil(bb_max[0] / px_mm)) + 1)
        y_lo = max(0, int(bb_min[1] / px_mm))
        y_hi = min(ny, int(np.ceil(bb_max[1] / px_mm)) + 1)
        z_in = (depths_mm >= bb_min[2]) & (depths_mm <= bb_max[2])
        if not (x_hi > x_lo and y_hi > y_lo and z_in.any()):
            continue
        z_idx = np.where(z_in)[0]
        used_mesh = False
        if v_s.is_watertight and len(v_s.faces) <= CONTAINS_MAX_FACES:
            xs_v, ys_v, zs_v = xs[x_lo:x_hi], ys[y_lo:y_hi], depths_mm[z_idx]
            X, Y, Z = np.meshgrid(xs_v, ys_v, zs_v, indexing="xy")
            pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
            try:
                sub = (v_s.contains(pts)
                       .reshape(len(ys_v), len(xs_v), len(zs_v))
                       .transpose(2, 0, 1))
                if sub.any():
                    Z_idx, Y_idx, X_idx = np.ix_(z_idx,
                                                  np.arange(y_lo, y_hi),
                                                  np.arange(x_lo, x_hi))
                    mask3d[Z_idx, Y_idx, X_idx] = np.maximum(
                        mask3d[Z_idx, Y_idx, X_idx], sub.astype(np.float32))
                    used_mesh = True
            except Exception:
                pass
        if not used_mesh:
            for zi in z_idx:
                mask3d[zi, y_lo:y_hi, x_lo:x_hi] = 1.0
    return mask3d

SLICES_DIR      = Path("slices_v2_atlanta2")
SLICES_DIR_2050 = Path("slices_v2_atlanta2_2050")
MANUAL_LABELS   = Path("labels_v2_atlanta2")
MODEL_STL       = Path("results_v2_atlanta2/unet_stl.pt")
MODEL_MAN_2050  = Path("results_v2_atlanta2/unet_pseudo_2050.pt")
MODEL_ASCAN     = Path("results_v2_atlanta2/ascan_classifier/ascan_cnn_depth.pt")
OUT_HTML        = Path("results_v2_atlanta2/alignment_picker.html")
OUT_HTML.parent.mkdir(exist_ok=True)

OFFSETS_50   = [-2, -1, 0, 1, 2]
OFFSETS_2050 = [-40, -20, 0, 20, 40]
NY = NX = 100

device_unet = ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")


# ── model loaders / predictors ───────────────────────────────────────────────
def load_unet(path):
    ck = torch.load(path, map_location=device_unet, weights_only=False)
    cfg = ck.get("config", {})
    m = UNet(in_channels=cfg.get("in_channels", 5),
             base_filters=cfg.get("base_filters", 32)).to(device_unet)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


def load_ascan():
    ck = torch.load(MODEL_ASCAN, map_location="cpu", weights_only=False)
    m = DepthAwareAScanCNN().to("cpu")
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


def predict_unet(model, slices, offsets):
    n_s = slices.shape[0]
    inp = np.zeros((n_s, len(offsets), NY, NX), dtype=np.float32)
    for i in range(n_s):
        for k, o in enumerate(offsets):
            inp[i, k] = slices[int(np.clip(i + o, 0, n_s - 1))]
    x = torch.from_numpy(inp).to(device_unet)
    probs = np.zeros((n_s, NY, NX), dtype=np.float32)
    CHUNK = 32
    with torch.no_grad():
        for k in range(0, n_s, CHUNK):
            probs[k:k+CHUNK] = torch.sigmoid(
                model(x[k:k+CHUNK])).squeeze(1).cpu().numpy()
    return probs


def predict_ascan(model, name):
    cache = CACHE_ATL2 / f"{name}.npz"
    if not cache.exists():
        return np.zeros((N_DEPTHS, NY, NX), dtype=np.float32)
    d = np.load(cache)
    a = d["ascans"]
    m_ = a.mean(axis=1, keepdims=True); s_ = a.std(axis=1, keepdims=True) + 1e-6
    a_t = torch.from_numpy(((a - m_) / s_).astype(np.float32)).unsqueeze(1)
    probs = np.zeros((a_t.shape[0], N_DEPTHS), dtype=np.float32)
    CHUNK = 512
    with torch.no_grad():
        for k in range(0, a_t.shape[0], CHUNK):
            probs[k:k+CHUNK] = torch.sigmoid(model(a_t[k:k+CHUNK])).cpu().numpy()
    return probs.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)


# ── colour ramps + base64 PNG ────────────────────────────────────────────────
def reds(v):
    v = np.clip(v, 0, 1)
    return np.stack([1.0 - 0.28 * v, 1.0 - 0.97 * v, 1.0 - 0.97 * v], axis=-1)


def oranges(v):
    v = np.clip(v, 0, 1)
    return np.stack([1.0 - 0.05 * v, 1.0 - 0.55 * v, 1.0 - 0.95 * v], axis=-1)


def blues(v):
    v = np.clip(v, 0, 1)
    return np.stack([1.0 - 0.92 * v, 1.0 - 0.72 * v, 1.0 - 0.05 * v], axis=-1)


def purples(v):
    v = np.clip(v, 0, 1)
    return np.stack([1.0 - 0.58 * v, 1.0 - 0.90 * v, 1.0 - 0.42 * v], axis=-1)


def greens(v):
    v = np.clip(v, 0, 1)
    return np.stack([1.0 - 0.90 * v, 1.0 - 0.45 * v, 1.0 - 0.80 * v], axis=-1)


def darkblues(v):
    v = np.clip(v, 0, 1)
    return np.stack([1.0 - 0.88 * v, 1.0 - 0.73 * v, 1.0 - 0.34 * v], axis=-1)


def to_b64(rgb):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── load models ──────────────────────────────────────────────────────────────
print(f"device(U-Net) = {device_unet}")
print(f"Loading models…")
model_stl = load_unet(MODEL_STL) if MODEL_STL.exists() else None
model_man = load_unet(MODEL_MAN_2050) if MODEL_MAN_2050.exists() else None
model_a   = load_ascan() if MODEL_ASCAN.exists() else None
print(f"  ✓ STL U-Net:    {'loaded' if model_stl else 'MISSING'}")
print(f"  ✓ Manual U-Net: {'loaded' if model_man else 'MISSING'}")
print(f"  ✓ A-scan CNN:   {'loaded' if model_a else 'MISSING'}")


# ── per-sample data ──────────────────────────────────────────────────────────
samples_data = []
order = sorted([n for n in SAMPLE_TO_STL if n != "5b"],
               key=lambda x: int(x.rstrip("b")))

for name in order:
    slices_p = SLICES_DIR / f"{name}_slices.npy"
    depths_p = SLICES_DIR / f"{name}_depths.npy"
    if not slices_p.exists() or not depths_p.exists():
        print(f"  skip {name}: missing slices/depths")
        continue

    slices = np.load(slices_p).astype(np.float32)        # (50, 100, 100)
    depths = np.load(depths_p).astype(np.float32)        # (50,)
    n_s, H, W = slices.shape

    # Normalise input to [0,1] for display
    s_norm = (slices - slices.min()) / max(slices.max() - slices.min(), 1e-6)

    # Voxelize STL at CAD frame (sym=0, z_shift=0). Apply per-sample mesh
    # pre-rotation around Z if MESH_ROTATE_Z_DEG specifies one for this sample.
    stl_path = STL_DIR / SAMPLE_TO_STL[name]
    if not stl_path.exists():
        print(f"  skip {name}: STL missing")
        continue
    rot_z_deg = MESH_ROTATE_Z_DEG.get(name, 0.0)
    m_cad = voxelize_gt_3d_rotated(stl_path, depths,
                                    pre_rotate_z_deg=rot_z_deg)
    if rot_z_deg != 0.0:
        print(f"     ↪ pre-rotated mesh by {rot_z_deg}° around Z")

    # Manual mask (binary z-projection per slice for blue overlay)
    man_p = MANUAL_LABELS / f"{name}_slice_masks.npy"
    manual = (np.load(man_p).astype(np.float32) if man_p.exists()
              else np.zeros_like(m_cad))

    # ── Model predictions ────────────────────────────────────────────────
    # 1. STL-trained U-Net (50-slice native)
    pred_stl = (predict_unet(model_stl, slices, OFFSETS_50)
                if model_stl is not None
                else np.zeros((n_s, H, W), dtype=np.float32))

    # 2. Manual-trained U-Net (2050-slice native → mapped to 50-slice via
    #    nearest depth so it lines up with the input slice axis)
    slices_2050_p = SLICES_DIR_2050 / f"{name}_slices.npy"
    depths_2050_p = SLICES_DIR_2050 / f"{name}_depths.npy"
    pred_man = np.zeros((n_s, H, W), dtype=np.float32)
    if model_man is not None and slices_2050_p.exists() and depths_2050_p.exists():
        slices_2050 = np.load(slices_2050_p).astype(np.float32)
        depths_2050 = np.load(depths_2050_p).astype(np.float32)
        probs_2050  = predict_unet(model_man, slices_2050, OFFSETS_2050)
        # Map each 50-slice depth to nearest 2050-slice index
        for i, d in enumerate(depths):
            j = int(np.argmin(np.abs(depths_2050 - d)))
            pred_man[i] = probs_2050[j]

    # 3. A-scan 1D-CNN (50-slice native)
    pred_ascan = (predict_ascan(model_a, name)
                  if model_a is not None
                  else np.zeros((n_s, H, W), dtype=np.float32))

    # ── Encode per-slice panels ─────────────────────────────────────────
    input_b64    = [to_b64(reds(s_norm[i]))            for i in range(n_s)]
    manual_b64   = [to_b64(blues(manual[i]))           for i in range(n_s)]
    pred_man_b64 = [to_b64(darkblues(pred_man[i]))     for i in range(n_s)]
    pred_asc_b64 = [to_b64(purples(pred_ascan[i]))     for i in range(n_s)]
    pred_stl_b64 = [to_b64(greens(pred_stl[i]))        for i in range(n_s)]
    stl_b64_by_sym = [
        [to_b64(oranges(apply_sym_mask(m_cad, sym)[i])) for i in range(n_s)]
        for sym in range(8)
    ]
    # 8 sym z-projections for the comparison grid (one PNG per sym)
    sym_projs = [
        to_b64(oranges(apply_sym_mask(m_cad, sym).max(axis=0)))
        for sym in range(8)
    ]

    samples_data.append({
        "name":        name,
        "n_slices":    int(n_s),
        "dz_mm":       float(np.diff(depths).mean()),
        "depths":      [round(float(d), 4) for d in depths],
        "input":       input_b64,
        "manual":      manual_b64,
        "pred_man":    pred_man_b64,
        "pred_ascan":  pred_asc_b64,
        "pred_stl":    pred_stl_b64,
        "stl_by_sym":  stl_b64_by_sym,
        "sym_projs":   sym_projs,
    })
    print(f"  ✓ {name}: {n_s} slices, dz={samples_data[-1]['dz_mm']:.3f} mm, "
          f"manual={'yes' if manual.any() else 'no'}, "
          f"preds={'all 3' if (model_stl and model_man and model_a) else 'partial'}")


# ── HTML + JS ────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>THz STL Alignment Picker</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #181828; color: #e0e0e0; padding-bottom: 40px; }
  .header { background: #11192a; padding: 12px 18px; display: flex;
            align-items: center; gap: 16px; flex-wrap: wrap; }
  .header h1 { font-size: 17px; }
  .tag { background: #e94560; padding: 2px 8px; border-radius: 3px;
         font-size: 11px; font-weight: bold; }
  .btn { padding: 5px 12px; border: 1px solid #444; border-radius: 4px;
         background: #0f3460; color: #e0e0e0; cursor: pointer; font-size: 12px; }
  .btn:hover { background: #1a5276; }
  .btn.active { background: #e94560; border-color: #e94560; font-weight: bold; }
  .btn.export { background: #27ae60; border-color: #27ae60; font-weight: bold; }
  .row { padding: 10px 18px; display: flex; align-items: center; gap: 10px;
         flex-wrap: wrap; border-top: 1px solid #2c2c40; }
  .row.controls { background: #1a2238; }
  .row label { font-size: 12px; color: #aaa; margin-right: 4px; }
  input[type=range] { flex: 1; max-width: 420px; }
  .info { font-size: 12px; color: #aaa; }
  .pill { background: #2a3450; padding: 3px 8px; border-radius: 12px;
          font-family: monospace; font-size: 12px; }
  .panels { display: flex; gap: 14px; padding: 18px; flex-wrap: wrap;
            justify-content: center; }
  .panel { text-align: center; position: relative; }
  .panel img, .panel canvas {
    width: 280px; height: 280px; image-rendering: pixelated;
    border: 2px solid #2c2c40; border-radius: 4px; background: #ffffff;
  }
  .panel .lbl { margin-top: 6px; font-size: 12px; color: #bbb; }
  .panel .lbl b { color: #e0e0e0; }
  .overlay-wrap { position: relative; width: 280px; height: 280px; }
  .overlay-wrap img { position: absolute; left: 0; top: 0;
                      mix-blend-mode: multiply; }
  .sym-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 6px;
              padding: 14px 18px; }
  .sym-thumb { cursor: pointer; text-align: center; padding: 4px;
               border-radius: 4px; border: 2px solid transparent; }
  .sym-thumb img { width: 100%; image-rendering: pixelated;
                   border: 1px solid #2c2c40; background: #ffffff; }
  .sym-thumb.active { border-color: #e94560; }
  .sym-thumb .n { font-size: 11px; color: #aaa; margin-top: 2px; }
  .picks-bar { background: #11192a; padding: 12px 18px; display: flex;
               align-items: center; gap: 8px; flex-wrap: wrap;
               border-top: 1px solid #2c2c40; }
  .pick-chip { background: #2a3450; padding: 4px 8px; border-radius: 4px;
               font-family: monospace; font-size: 11px;
               border: 1px solid #2c2c40; cursor: pointer; }
  .pick-chip.has-pick { background: #27ae60; border-color: #27ae60; }
  textarea { width: 100%; height: 110px; background: #0a1020; color: #d0e0d0;
             border: 1px solid #2c2c40; border-radius: 4px; padding: 8px;
             font-family: monospace; font-size: 11px; margin-top: 6px; }
  kbd { padding: 1px 6px; background: #0f3460; border: 1px solid #444;
        border-radius: 3px; font-family: monospace; font-size: 11px; }
</style></head>
<body>

<div class="header">
  <h1>THz STL Alignment Picker</h1>
  <span class="tag">atlanta2</span>
  <span class="info">scrub sym + z-shift + slice; export picks as JSON</span>
  <button class="btn export" onclick="exportPicks()">Export picks ▼</button>
</div>

<div class="row controls">
  <label>Sample:</label>
  <div id="sample-btns" style="display:flex; gap:4px; flex-wrap:wrap;"></div>
</div>

<div class="row controls">
  <label>Symmetry (sym):</label>
  <div id="sym-btns" style="display:flex; gap:4px;"></div>
  <span class="info">— or click a thumbnail below</span>
</div>

<div class="row controls">
  <label>Z-shift (mm):</label>
  <input type="range" id="zshift" min="-3.5" max="1.0" step="0.05" value="0">
  <span class="pill" id="zshift-val">+0.00 mm</span>
  <span class="info">(<span id="zshift-slices">0</span> slices)</span>
</div>

<div class="row controls">
  <label>Slice:</label>
  <input type="range" id="slice" min="0" max="49" value="25">
  <span class="pill" id="slice-val">25</span>
  <span class="pill" id="depth-val">depth — mm</span>
  <span class="info">| <kbd>←</kbd><kbd>→</kbd> slice &nbsp; <kbd>n</kbd>/<kbd>p</kbd> sample &nbsp; <kbd>1-8</kbd> sym &nbsp; <kbd>s</kbd> save</span>
</div>

<div class="panels">
  <div class="panel">
    <img id="img-input">
    <div class="lbl"><b>Input</b> (THz scan)</div>
  </div>
  <div class="panel">
    <img id="img-stl">
    <div class="lbl"><b>STL</b> (sym + z-shift applied)</div>
  </div>
  <div class="panel">
    <img id="img-manual">
    <div class="lbl"><b>Manual</b> (rectangles)</div>
  </div>
  <div class="panel">
    <div class="overlay-wrap">
      <img id="img-ov-input">
      <img id="img-ov-stl">
    </div>
    <div class="lbl"><b>Overlay</b> (input × STL)</div>
  </div>
</div>

<div class="panels">
  <div class="panel">
    <img id="img-pred-man">
    <div class="lbl"><b>Manual U-Net</b> (unet_pseudo_2050)</div>
  </div>
  <div class="panel">
    <img id="img-pred-ascan">
    <div class="lbl"><b>A-scan 1D-CNN</b> (ascan_cnn_depth)</div>
  </div>
  <div class="panel">
    <img id="img-pred-stl">
    <div class="lbl"><b>STL U-Net</b> (unet_stl)</div>
  </div>
  <div class="panel">
    <div class="overlay-wrap">
      <img id="img-ov-input2">
      <img id="img-ov-pred-man" style="opacity:0.5;">
      <img id="img-ov-pred-ascan" style="opacity:0.5;">
      <img id="img-ov-pred-stl" style="opacity:0.5;">
    </div>
    <div class="lbl"><b>All preds overlay</b> (input × 3 models)</div>
  </div>
</div>

<div class="row" style="background:#11192a;">
  <label>Compare all 8 syms (z-projection):</label>
  <span class="info">click to set sym</span>
</div>
<div class="sym-grid" id="sym-grid"></div>

<div class="picks-bar">
  <label>Per-sample picks:</label>
  <div id="picks-chips" style="display:flex; gap:6px; flex-wrap:wrap;"></div>
  <button class="btn" onclick="resetPick()">Reset current</button>
</div>
<div style="padding: 0 18px;">
  <label class="info">Exported JSON (also writeable to clipboard with <kbd>Export picks</kbd>):</label>
  <textarea id="export-out" readonly placeholder="Click Export picks to fill"></textarea>
</div>

<script>
const DATA = __DATA__;
const PICKS = {};                 // {name: {sym, z_shift_mm}}
let curSample = 0;
let curSym    = 0;
let curZShift = 0.0;              // mm
let curSlice  = 25;

function init() {
  const sb = document.getElementById("sample-btns");
  DATA.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "btn";
    b.textContent = "S" + s.name;
    b.onclick = () => selectSample(i);
    sb.appendChild(b);
  });
  const sym = document.getElementById("sym-btns");
  for (let i = 0; i < 8; i++) {
    const b = document.createElement("button");
    b.className = "btn";
    b.textContent = i;
    b.onclick = () => selectSym(i);
    sym.appendChild(b);
  }
  document.getElementById("zshift").addEventListener("input", e => {
    curZShift = parseFloat(e.target.value);
    storePick();
    render();
  });
  document.getElementById("slice").addEventListener("input", e => {
    curSlice = parseInt(e.target.value);
    render();
  });
  document.addEventListener("keydown", onKey);
  selectSample(0);
  renderPicksChips();
}

function selectSample(i) {
  curSample = i;
  const s = DATA[i];
  // Restore prior pick if any
  if (PICKS[s.name]) {
    curSym = PICKS[s.name].sym;
    curZShift = PICKS[s.name].z_shift_mm;
  } else {
    curSym = 0;
    curZShift = 0.0;
  }
  curSlice = Math.floor(s.n_slices / 2);
  document.getElementById("slice").max = s.n_slices - 1;
  document.getElementById("slice").value = curSlice;
  document.getElementById("zshift").value = curZShift;
  document.querySelectorAll("#sample-btns .btn").forEach((b, j) =>
    b.classList.toggle("active", j === i));
  renderSymGrid();
  render();
}

function selectSym(s) {
  curSym = s;
  storePick();
  render();
}

function storePick() {
  PICKS[DATA[curSample].name] = { sym: curSym, z_shift_mm: curZShift };
  renderPicksChips();
}

function resetPick() {
  delete PICKS[DATA[curSample].name];
  curSym = 0; curZShift = 0;
  document.getElementById("zshift").value = 0;
  renderPicksChips();
  render();
}

function renderSymGrid() {
  const s = DATA[curSample];
  const g = document.getElementById("sym-grid");
  g.innerHTML = "";
  for (let i = 0; i < 8; i++) {
    const c = document.createElement("div");
    c.className = "sym-thumb" + (i === curSym ? " active" : "");
    c.innerHTML =
      "<img src='data:image/png;base64," + s.sym_projs[i] + "'>" +
      "<div class='n'>sym " + i + "</div>";
    c.onclick = () => selectSym(i);
    g.appendChild(c);
  }
}

function render() {
  const s = DATA[curSample];
  const zSlices = Math.round(curZShift / s.dz_mm);
  const cadIdx  = curSlice - zSlices;
  const stlSrc  = (cadIdx >= 0 && cadIdx < s.n_slices)
                  ? s.stl_by_sym[curSym][cadIdx]
                  : null;

  const inputSrc = "data:image/png;base64," + s.input[curSlice];
  document.getElementById("img-input").src  = inputSrc;
  document.getElementById("img-manual").src = "data:image/png;base64," + s.manual[curSlice];
  document.getElementById("img-pred-man").src   = "data:image/png;base64," + s.pred_man[curSlice];
  document.getElementById("img-pred-ascan").src = "data:image/png;base64," + s.pred_ascan[curSlice];
  document.getElementById("img-pred-stl").src   = "data:image/png;base64," + s.pred_stl[curSlice];
  document.getElementById("img-ov-input2").src    = inputSrc;
  document.getElementById("img-ov-pred-man").src   = "data:image/png;base64," + s.pred_man[curSlice];
  document.getElementById("img-ov-pred-ascan").src = "data:image/png;base64," + s.pred_ascan[curSlice];
  document.getElementById("img-ov-pred-stl").src   = "data:image/png;base64," + s.pred_stl[curSlice];
  const stl = document.getElementById("img-stl");
  const ovs = document.getElementById("img-ov-stl");
  const ovi = document.getElementById("img-ov-input");
  ovi.src = inputSrc;
  if (stlSrc) {
    stl.src = "data:image/png;base64," + stlSrc;
    stl.style.opacity = 1.0;
    ovs.src = "data:image/png;base64," + stlSrc;
    ovs.style.opacity = 1.0;
  } else {
    // out-of-range: show grey
    stl.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAAEElEQVR42mP8/5+hHgAGgwL/CDLfHwwAAAABJRU5ErkJggg==";
    stl.style.opacity = 0.3;
    ovs.src = stl.src;
    ovs.style.opacity = 0.0;
  }
  document.getElementById("slice-val").textContent = curSlice + "/" + (s.n_slices - 1);
  document.getElementById("depth-val").textContent = "depth " + s.depths[curSlice] + " mm";
  document.getElementById("zshift-val").textContent =
    (curZShift >= 0 ? "+" : "") + curZShift.toFixed(2) + " mm";
  document.getElementById("zshift-slices").textContent =
    (zSlices >= 0 ? "+" : "") + zSlices;
  document.querySelectorAll("#sym-btns .btn").forEach((b, j) =>
    b.classList.toggle("active", j === curSym));
  document.querySelectorAll("#sym-grid .sym-thumb").forEach((b, j) =>
    b.classList.toggle("active", j === curSym));
}

function renderPicksChips() {
  const c = document.getElementById("picks-chips");
  c.innerHTML = "";
  DATA.forEach((s, i) => {
    const chip = document.createElement("span");
    const p = PICKS[s.name];
    chip.className = "pick-chip" + (p ? " has-pick" : "");
    chip.textContent = "S" + s.name + (p
      ? "  sym=" + p.sym + "  z=" + (p.z_shift_mm >= 0 ? "+" : "")
        + p.z_shift_mm.toFixed(2)
      : "  —");
    chip.onclick = () => selectSample(i);
    c.appendChild(chip);
  });
}

function exportPicks() {
  const out = {};
  DATA.forEach(s => {
    out[s.name] = PICKS[s.name] || null;
  });
  const text = JSON.stringify(out, null, 2);
  document.getElementById("export-out").value = text;
  // Try clipboard
  if (navigator.clipboard) {
    navigator.clipboard.writeText(text).catch(() => {});
  }
}

function onKey(e) {
  const s = DATA[curSample];
  if (e.key === "ArrowRight" && curSlice < s.n_slices - 1) {
    curSlice++; document.getElementById("slice").value = curSlice; render();
  } else if (e.key === "ArrowLeft" && curSlice > 0) {
    curSlice--; document.getElementById("slice").value = curSlice; render();
  } else if (e.key === "n") {
    selectSample((curSample + 1) % DATA.length);
  } else if (e.key === "p") {
    selectSample((curSample - 1 + DATA.length) % DATA.length);
  } else if (/^[1-8]$/.test(e.key)) {
    selectSym(parseInt(e.key) - 1);
  } else if (e.key === "s") {
    storePick();
  }
}

init();
</script></body></html>
"""

OUT_HTML.write_text(HTML.replace("__DATA__", json.dumps(samples_data)))
mb = OUT_HTML.stat().st_size / 1e6
print(f"\n  done — {OUT_HTML}")
print(f"  {mb:.1f} MB embedded ({len(samples_data)} samples × "
      f"{50 * 10} PNGs ≈ {len(samples_data) * 500} PNGs)")
