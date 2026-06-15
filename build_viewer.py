"""
build_viewer.py — Build the interactive prediction viewer HTML
===============================================================
Loads slices, GT masks, runs the trained U-Net, and bundles everything
into a single self-contained viewer.html (all images embedded as base64).

Usage:
    thesis_env/bin/python build_viewer.py                       # atlanta1 (default, stride=1)
    thesis_env/bin/python build_viewer.py atlanta2              # atlanta2, 50-slice model
    thesis_env/bin/python build_viewer.py atlanta2_pseudo       # atlanta2, 500-slice pseudo model
    thesis_env/bin/python build_viewer.py atlanta2_pseudo 5     # subsample every 5th slice
"""

import sys, io, base64
import numpy as np
from pathlib import Path
from PIL import Image
from scipy.ndimage import binary_dilation
import torch

sys.path.insert(0, str(Path(__file__).parent))

# ── viewer configs ───────────────────────────────────────────────────────────
VIEWER_CONFIGS = {
    "atlanta1": dict(
        slices_dir="slices_v2",
        labels_dir="labels_v2",                 # 50-slice manual labels
        model_path="results_v2/unet_v2.pt",
        out_html="results_v2/viewer.html",
        offsets=[-2, -1, 0, 1, 2],
    ),
    "atlanta2": dict(
        slices_dir="slices_v2_atlanta2",
        labels_dir="labels_v2_atlanta2",        # 50-slice manual labels
        model_path="results_v2_atlanta2/unet_v2.pt",   # if exists; else fall back
        out_html="results_v2_atlanta2/viewer.html",
        offsets=[-2, -1, 0, 1, 2],
    ),
    "atlanta2_pseudo": dict(
        slices_dir="slices_v2_atlanta2_500",
        labels_dir="labels_v2_atlanta2",        # 50-slice manual labels (sparse GT)
        manual_depth_dir="slices_v2_atlanta2",  # 50-slice depths for GT matching
        model_path="results_v2_atlanta2/unet_pseudo.pt",
        out_html="results_v2_atlanta2/viewer_pseudo.html",
        offsets=[-10, -5, 0, 5, 10],
    ),
    "atlanta2_2050": dict(
        slices_dir="slices_v2_atlanta2_2050",
        labels_dir="labels_v2_atlanta2",        # 50-slice manual labels (sparse GT)
        manual_depth_dir="slices_v2_atlanta2",  # 50-slice depths for GT matching
        model_path="results_v2_atlanta2/unet_pseudo.pt",
        out_html="results_v2_atlanta2/viewer_2050.html",
        offsets=[-40, -20, 0, 20, 40],
    ),
    "atlanta2_2050_tight": dict(
        slices_dir="slices_v2_atlanta2_2050",
        labels_dir="labels_v2_atlanta2",
        manual_depth_dir="slices_v2_atlanta2",
        model_path="results_v2_atlanta2/unet_pseudo.pt",
        out_html="results_v2_atlanta2/viewer_2050_tight.html",
        offsets=[-10, -5, 0, 5, 10],       # tight depth window (±0.021 mm)
    ),
    "atlanta2_2050_v2": dict(
        # Retrained on 2050-slice pseudo-labels with offsets ±40
        slices_dir="slices_v2_atlanta2_2050",
        labels_dir="labels_v2_atlanta2",
        manual_depth_dir="slices_v2_atlanta2",
        model_path="results_v2_atlanta2/unet_pseudo_2050.pt",
        out_html="results_v2_atlanta2/viewer_2050_v2.html",
        offsets=[-40, -20, 0, 20, 40],
    ),
    "crossdomain": dict(
        # atlanta2-trained 2050 model applied to OLD atlanta1 data
        slices_dir="slices_v2_2050",            # atlanta1 @ 2050 slices
        labels_dir="labels_v2",                 # atlanta1 manual masks (50-slice)
        manual_depth_dir="slices_v2",           # atlanta1 50-slice depths
        model_path="results_v2_atlanta2/unet_pseudo_2050.pt",
        out_html="results_v2/viewer_crossdomain.html",
        offsets=[-40, -20, 0, 20, 40],
    ),
    "crossdomain_tight": dict(
        # Same as crossdomain but tight ±10 window (model trained at ±40)
        slices_dir="slices_v2_2050",
        labels_dir="labels_v2",
        manual_depth_dir="slices_v2",
        model_path="results_v2_atlanta2/unet_pseudo_2050.pt",
        out_html="results_v2/viewer_crossdomain_tight.html",
        offsets=[-10, -5, 0, 5, 10],
    ),
}

DATASET = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in VIEWER_CONFIGS else "atlanta1"
DEFAULT_STRIDE = {
    "atlanta1":             1,
    "atlanta2":             1,
    "atlanta2_pseudo":      5,
    "atlanta2_2050":        40,   # ~51 evenly-spaced + ~50 manual = ~100 total
    "atlanta2_2050_tight":  40,
    "atlanta2_2050_v2":     40,
    "crossdomain":          40,
    "crossdomain_tight":    40,
}
STRIDE  = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_STRIDE.get(DATASET, 1)
CFG     = VIEWER_CONFIGS[DATASET]

SLICES_DIR = Path(CFG["slices_dir"])
LABELS_DIR = Path(CFG["labels_dir"])
MODEL_PATH = Path(CFG["model_path"])
OUT_HTML   = Path(CFG["out_html"])
OFFSETS    = CFG["offsets"]
MANUAL_DEPTH_DIR = Path(CFG.get("manual_depth_dir", CFG["slices_dir"]))

print(f"Viewer config: {DATASET}  (stride={STRIDE})")
print(f"  slices:    {SLICES_DIR}/")
print(f"  manual GT: {LABELS_DIR}/")
print(f"  model:     {MODEL_PATH}")
print(f"  offsets:   {OFFSETS}")
print(f"  out HTML:  {OUT_HTML}\n")

# ── load model ───────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    fallback = Path("results_v2/unet_v2.pt")
    print(f"  {MODEL_PATH} not found; falling back to {fallback}")
    MODEL_PATH = fallback

ckpt  = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
cfg   = ckpt["config"]
# UNet class
sys.argv = [sys.argv[0], "predict", "atlanta1"]   # any valid mode for import
from thz_slice_pipelinev2 import UNet
model = UNet(in_channels=cfg["in_channels"], base_filters=cfg["base_filters"])
model.load_state_dict(ckpt["model_state"])
model.eval()
print(f"  model loaded from {MODEL_PATH}\n")

# ── enumerate samples ────────────────────────────────────────────────────────
sample_names = sorted(
    [p.stem.replace("_slices", "") for p in SLICES_DIR.glob("*_slices.npy")],
    key=lambda x: (int(x.rstrip("b")), x),
)
print(f"  {len(sample_names)} samples: {sample_names}\n")


# ── base64 PNG helpers ───────────────────────────────────────────────────────
def _rgb_to_b64(rgb):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _reds(v):
    v = np.clip(v, 0, 1)
    return np.stack([1.0 - 0.28 * v, 1.0 - 0.97 * v, 1.0 - 0.97 * v], axis=-1)


def edges(m):
    m = m.astype(bool)
    return binary_dilation(m, iterations=1) ^ m


def render_panels(slice_img, gt, has_gt, pred_soft, pred_bin):
    H, W = slice_img.shape

    raw_b64 = _rgb_to_b64(_reds(slice_img))

    if has_gt:
        gt_rgb = np.ones((H, W, 3))
        gt_rgb[gt > 0] = [0.72, 0.11, 0.11]
        gt_b64 = _rgb_to_b64(gt_rgb)
    else:
        # blank slate; UI will overlay a "no manual GT" label
        gt_b64 = _rgb_to_b64(np.ones((H, W, 3)) * 0.94)

    pred_soft_b64 = _rgb_to_b64(_reds(pred_soft))

    pb_rgb = np.ones((H, W, 3))
    pb_rgb[pred_bin > 0] = [0.72, 0.11, 0.11]
    pred_b64 = _rgb_to_b64(pb_rgb)

    overlay = np.zeros((H, W, 3))
    overlay[..., 0] = slice_img
    overlay[..., 1] = slice_img * 0.3
    overlay[..., 2] = slice_img * 0.3
    if has_gt:
        overlay[edges(gt > 0)] = [0.0, 1.0, 0.0]
    overlay[edges(pred_bin > 0)] = [0.0, 0.83, 1.0]
    overlay_b64 = _rgb_to_b64(overlay)

    if has_gt:
        gt_pos = gt > 0
        pred_pos = pred_bin > 0
        tp = gt_pos & pred_pos
        fp = ~gt_pos & pred_pos
        fn = gt_pos & ~pred_pos
        diff = np.zeros((H, W, 3))
        diff[..., :] = 0.13
        diff[tp] = [0.0, 1.0, 0.0]
        diff[fp] = [1.0, 0.0, 0.0]
        diff[fn] = [0.0, 0.4, 1.0]
        diff_b64 = _rgb_to_b64(diff)
        gt_px = int(gt_pos.sum()); pred_px = int(pred_pos.sum())
        tp_c, fp_c, fn_c = int(tp.sum()), int(fp.sum()), int(fn.sum())
    else:
        diff_b64 = _rgb_to_b64(np.ones((H, W, 3)) * 0.94)
        gt_px = -1; pred_px = int((pred_bin > 0).sum())
        tp_c = fp_c = fn_c = -1

    return dict(
        raw=raw_b64, gt=gt_b64, pred=pred_b64,
        pred_soft=pred_soft_b64, overlay=overlay_b64, diff=diff_b64,
        has_gt=bool(has_gt),
        gt_px=gt_px, pred_px=pred_px,
        tp=tp_c, fp=fp_c, fn=fn_c,
    )


# ── per-sample processing ────────────────────────────────────────────────────
import json
all_data = []
for name in sample_names:
    slices = np.load(SLICES_DIR / f"{name}_slices.npy")
    depths_p = SLICES_DIR / f"{name}_depths.npy"
    depths = (np.load(depths_p) if depths_p.exists()
              else np.linspace(0, 4.27, slices.shape[0]))
    n_s, H, W = slices.shape

    # Manual GT mapping
    manual_path = LABELS_DIR / f"{name}_slice_masks.npy"
    manual_depths_path = MANUAL_DEPTH_DIR / f"{name}_depths.npy"
    has_manual = manual_path.exists()
    if has_manual:
        manual = np.load(manual_path)
        if manual.shape[0] == n_s:
            # same resolution — GT at every slice
            gt_map = {i: manual[i] for i in range(n_s)}
        elif manual_depths_path.exists():
            # 50-slice manual mapped onto N-slice grid by closest depth
            m_depths = np.load(manual_depths_path)
            gt_map = {}
            for j, d in enumerate(m_depths):
                i = int(np.argmin(np.abs(depths - d)))
                gt_map[i] = manual[j]
        else:
            gt_map = {}
    else:
        gt_map = {}

    # Indices to render
    show_idx = list(range(0, n_s, STRIDE))
    # Always include slices that have manual GT so the user can find them
    for i in gt_map.keys():
        if i not in show_idx:
            show_idx.append(i)
    show_idx = sorted(set(show_idx))

    print(f"  processing sample {name}  ({len(show_idx)} of {n_s} slices)…")

    slice_list = []
    with torch.no_grad():
        for i in show_idx:
            chans = [slices[int(np.clip(i + o, 0, n_s - 1))] for o in OFFSETS]
            inp   = torch.FloatTensor(np.stack(chans)).unsqueeze(0)
            soft  = torch.sigmoid(model(inp)).squeeze().numpy()
            binv  = (soft > 0.5).astype(np.uint8)

            if i in gt_map:
                p = render_panels(slices[i], gt_map[i], True, soft, binv)
            else:
                p = render_panels(slices[i], np.zeros((H, W), dtype=np.uint8),
                                  False, soft, binv)
            p["depth"]      = round(float(depths[i]), 4)
            p["slice_idx"]  = int(i)
            slice_list.append(p)

    all_data.append(dict(name=name, n_slices=n_s,
                         stride=STRIDE, slices=slice_list))

# ── HTML ─────────────────────────────────────────────────────────────────────
html_head = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>THz Void Detection — Prediction Viewer (__DATASET__)</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #1a1a2e; color: #e0e0e0; }
  .header { background: #16213e; padding: 12px 20px; display: flex;
            align-items: center; gap: 20px; flex-wrap: wrap; }
  .header h1 { font-size: 18px; white-space: nowrap; }
  .header .tag { background: #e94560; padding: 2px 8px; border-radius: 3px;
                 font-size: 11px; font-weight: bold; }
  .sample-btn { padding: 6px 14px; border: 1px solid #444; border-radius: 4px;
                background: #0f3460; color: #e0e0e0; cursor: pointer;
                font-size: 13px; }
  .sample-btn:hover { background: #1a5276; }
  .sample-btn.active { background: #e94560; border-color: #e94560; font-weight: bold; }
  .controls { background: #16213e; padding: 10px 20px; display: flex;
              align-items: center; gap: 15px; border-top: 1px solid #333; }
  .controls label { font-size: 13px; }
  .controls input[type=range] { flex: 1; max-width: 500px; }
  .controls .info { font-size: 13px; color: #aaa; margin-left: auto; }
  .toggle-btn { padding: 5px 12px; border: 1px solid #444; border-radius: 4px;
                background: #0f3460; color: #e0e0e0; cursor: pointer; font-size: 12px; }
  .toggle-btn.active { background: #e94560; border-color: #e94560; }
  .gt-badge { display: inline-block; padding: 2px 8px; border-radius: 3px;
              font-size: 11px; font-weight: bold; }
  .gt-badge.has { background: #27ae60; color: white; }
  .gt-badge.no  { background: #555; color: #ccc; }
  .panels { display: flex; justify-content: center; gap: 8px; padding: 15px;
            flex-wrap: wrap; }
  .panel { text-align: center; position: relative; }
  .panel img { width: 200px; height: 200px; image-rendering: pixelated;
               border: 2px solid #333; border-radius: 4px; }
  .panel .label { font-size: 12px; margin-top: 4px; color: #aaa; }
  .panel .label b { color: #e0e0e0; }
  .panel .nogt-overlay { position: absolute; top: 8px; left: 8px;
                         background: rgba(0,0,0,0.6); color: #ccc;
                         padding: 3px 8px; font-size: 11px; border-radius: 3px;
                         pointer-events: none; }
  .legend { display: flex; justify-content: center; gap: 20px; padding: 8px;
            font-size: 12px; }
  .legend .dot { display: inline-block; width: 10px; height: 10px;
                 border-radius: 50%; margin-right: 4px; vertical-align: middle; }
  kbd { padding: 1px 6px; background: #0f3460; border: 1px solid #444;
        border-radius: 3px; font-family: monospace; font-size: 11px; }
</style></head>
<body>

<div class="header">
  <h1>THz Void Detection — Prediction Viewer</h1>
  <span class="tag">__DATASET__</span>
  <span class="info">stride=__STRIDE__</span>
  <div id="sample-btns" style="display:flex; gap:6px; flex-wrap:wrap;"></div>
</div>

<div class="controls">
  <label>Slice: <span id="slice-idx">0</span></label>
  <input type="range" id="slider" min="0" max="0" value="0">
  <span id="depth-info" class="info"></span>
  <span id="gt-status" class="gt-badge no">no manual GT</span>
  <button class="toggle-btn" id="soft-toggle" onclick="toggleSoft()">Show Soft</button>
  <span class="info">Keys: <kbd>&larr;</kbd><kbd>&rarr;</kbd> slice &nbsp;
        <kbd>n</kbd><kbd>p</kbd> sample &nbsp; <kbd>t</kbd> toggle</span>
</div>

<div class="panels">
  <div class="panel"><img id="img-raw"><div class="label"><b>Input</b></div></div>
  <div class="panel">
    <img id="img-gt">
    <div class="nogt-overlay" id="nogt-overlay" style="display:none;">no manual GT</div>
    <div class="label" id="lbl-gt"><b>GT Mask</b></div>
  </div>
  <div class="panel"><img id="img-pred"><div class="label" id="lbl-pred"><b>Prediction</b></div></div>
  <div class="panel"><img id="img-overlay"><div class="label"><b>Overlay</b></div></div>
  <div class="panel"><img id="img-diff"><div class="label" id="lbl-diff"><b>Difference</b></div></div>
</div>

<div class="legend">
  <span><span class="dot" style="background:#00ff00"></span> True Positive</span>
  <span><span class="dot" style="background:#ff0000"></span> False Positive</span>
  <span><span class="dot" style="background:#0066ff"></span> False Negative</span>
  <span><span class="dot" style="background:#222"></span> True Negative</span>
</div>

<script>
const DATA = """

html_tail = """;

let curSample = 0;
let curIdx    = 0;       // index into DATA[curSample].slices
let showSoft  = false;

function init() {
  const btns = document.getElementById("sample-btns");
  DATA.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "sample-btn";
    b.textContent = s.name;
    b.onclick = () => selectSample(i);
    btns.appendChild(b);
  });
  document.getElementById("slider").addEventListener("input", e => {
    curIdx = parseInt(e.target.value);
    render();
  });
  document.addEventListener("keydown", onKey);
  selectSample(0);
}

function selectSample(i) {
  curSample = i; curIdx = 0;
  const s = DATA[i];
  const slider = document.getElementById("slider");
  slider.max = s.slices.length - 1;
  slider.value = 0;
  document.querySelectorAll(".sample-btn").forEach((b, j) =>
    b.classList.toggle("active", j === i));
  render();
}

function render() {
  const s  = DATA[curSample];
  const sl = s.slices[curIdx];
  document.getElementById("slice-idx").textContent =
    sl.slice_idx + "/" + (s.n_slices - 1) + "  (view " + curIdx + "/" + (s.slices.length - 1) + ")";
  document.getElementById("depth-info").textContent =
    "depth = " + sl.depth + " mm  |  Sample " + s.name;
  document.getElementById("img-raw").src = "data:image/png;base64," + sl.raw;
  document.getElementById("img-gt").src = "data:image/png;base64," + sl.gt;
  document.getElementById("img-pred").src = "data:image/png;base64," +
    (showSoft ? sl.pred_soft : sl.pred);
  document.getElementById("img-overlay").src = "data:image/png;base64," + sl.overlay;
  document.getElementById("img-diff").src = "data:image/png;base64," + sl.diff;

  const gtBadge = document.getElementById("gt-status");
  const nogtOver = document.getElementById("nogt-overlay");
  if (sl.has_gt) {
    gtBadge.textContent = "manual GT"; gtBadge.className = "gt-badge has";
    nogtOver.style.display = "none";
    document.getElementById("lbl-gt").innerHTML = "<b>GT Mask</b> (" + sl.gt_px + " px)";
    document.getElementById("lbl-diff").innerHTML =
      "<b>TP=" + sl.tp + " FP=" + sl.fp + " FN=" + sl.fn + "</b>";
  } else {
    gtBadge.textContent = "no manual GT"; gtBadge.className = "gt-badge no";
    nogtOver.style.display = "block";
    document.getElementById("lbl-gt").innerHTML = "<b>GT Mask</b> (n/a)";
    document.getElementById("lbl-diff").innerHTML = "<b>—</b>";
  }
  document.getElementById("lbl-pred").innerHTML =
    "<b>" + (showSoft ? "Pred (soft)" : "Pred (&gt;0.5)") +
    "</b> (" + sl.pred_px + " px)";
}

function toggleSoft() {
  showSoft = !showSoft;
  document.getElementById("soft-toggle").classList.toggle("active", showSoft);
  document.getElementById("soft-toggle").textContent =
    showSoft ? "Show Binary" : "Show Soft";
  render();
}

function onKey(e) {
  const s = DATA[curSample];
  if (e.key === "ArrowRight" && curIdx < s.slices.length - 1) {
    curIdx++; document.getElementById("slider").value = curIdx; render();
  } else if (e.key === "ArrowLeft" && curIdx > 0) {
    curIdx--; document.getElementById("slider").value = curIdx; render();
  } else if (e.key === "n") {
    selectSample((curSample + 1) % DATA.length);
  } else if (e.key === "p") {
    selectSample((curSample - 1 + DATA.length) % DATA.length);
  } else if (e.key === "t") {
    toggleSoft();
  }
}

init();
</script></body></html>"""

OUT_HTML.parent.mkdir(exist_ok=True)
with open(OUT_HTML, "w") as fh:
    fh.write(html_head.replace("__DATASET__", DATASET).replace("__STRIDE__", str(STRIDE)))
    fh.write(json.dumps(all_data))
    fh.write(html_tail)

size_mb = OUT_HTML.stat().st_size / 1e6
total_slices = sum(len(s["slices"]) for s in all_data)
print(f"\n  done — {OUT_HTML}")
print(f"  {size_mb:.1f} MB  ·  {total_slices} embedded slices across {len(all_data)} samples")
