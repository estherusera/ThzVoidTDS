"""
build_viewer_3way.py — Interactive HTML viewer comparing three void detectors
on atlanta2 (50-slice grid), side-by-side per slice and per sample.

Per slice we render 7 panels:
  Input · STL labels · Manual rectangles · Manual-U-Net pred · A-scan pred ·
  STL-U-Net pred · Overlay (R=A-scan, G=STL-U-Net, B=Manual-U-Net + STL outline)

Models:
  • Manual-U-Net : results_v2_atlanta2/unet_pseudo_2050.pt (2050-slice; mapped
                   to the 50-slice depths by closest-depth lookup)
  • A-scan       : results_v2_atlanta2/ascan_classifier/ascan_cnn_depth.pt
                   (depth-aware 50-dim head — native 50-slice output)
  • STL-U-Net    : results_v2_atlanta2/unet_stl.pt (50-slice, NEW)

Output: results_v2_atlanta2/viewer_3way.html (self-contained, base64 PNGs).

Run
    thesis_env/bin/python build_viewer_3way.py            # full 50 slices
    thesis_env/bin/python build_viewer_3way.py 2          # stride 2
"""
import sys, io, json, base64
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation
import torch

_CLI = [a for a in sys.argv[1:] if not a.startswith("-")]
STRIDE = int(_CLI[0]) if _CLI else 1

sys.path.insert(0, str(Path(__file__).parent))
sys.argv = [sys.argv[0], "predict", "atlanta2"]
from thz_slice_pipelinev2 import UNet
from ascan_vs_stl import DepthAwareAScanCNN
from ascan_classifier import CACHE_ATL2, N_DEPTHS

SLICES_DIR_50   = Path("slices_v2_atlanta2")
SLICES_DIR_2050 = Path("slices_v2_atlanta2_2050")
# Use the VIZ STL labels (z-shifted, s13 kept, sym overrides applied) for the
# viewer, not the training labels.
STL_LABELS_DIR  = Path("labels_stl_atlanta2_viz")
MANUAL_LBLS     = Path("labels_v2_atlanta2")
MODEL_STL       = Path("results_v2_atlanta2/unet_stl.pt")
MODEL_MAN_2050  = Path("results_v2_atlanta2/unet_pseudo_2050.pt")
MODEL_ASCAN     = Path("results_v2_atlanta2/ascan_classifier/ascan_cnn_depth.pt")
OUT_HTML        = Path("results_v2_atlanta2/viewer_3way.html")

OFFSETS_50   = [-2, -1, 0, 1, 2]
OFFSETS_2050 = [-40, -20, 0, 20, 40]
NX = NY      = 100

device_unet  = ("mps" if hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                else "cuda" if torch.cuda.is_available() else "cpu")
# A-scan model lives on CPU (its AdaptiveAvgPool1d(50) doesn't work on MPS).


# ── helpers ──────────────────────────────────────────────────────────────────
def _rgb_to_b64(rgb):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _reds(v):
    v = np.clip(v, 0, 1)
    return np.stack([1.0 - 0.28 * v, 1.0 - 0.97 * v, 1.0 - 0.97 * v], axis=-1)


def _solid(mask, color):
    """White background, fill mask>0 with color."""
    rgb = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    rgb[mask > 0.5] = color
    return rgb


def _cmap(mask, color):
    """White → color gradient, scaling with mask value [0,1]."""
    rgb = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.float32)
    v = np.clip(mask, 0, 1)
    for c in range(3):
        rgb[..., c] = 1.0 - v * (1.0 - color[c])
    return rgb


def edges(m):
    m = m.astype(bool)
    return binary_dilation(m, iterations=1) ^ m


# ── model loaders ────────────────────────────────────────────────────────────
def load_unet(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = ck.get("config", {})
    m = UNet(in_channels=cfg.get("in_channels", 5),
             base_filters=cfg.get("base_filters", 32)).to(dev)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


def load_ascan():
    ck = torch.load(MODEL_ASCAN, map_location="cpu", weights_only=False)
    m = DepthAwareAScanCNN()
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


def predict_unet(model, slices, offsets, dev):
    n_s = slices.shape[0]
    inp = np.zeros((n_s, len(offsets), NY, NX), dtype=np.float32)
    for i in range(n_s):
        for k, o in enumerate(offsets):
            inp[i, k] = slices[int(np.clip(i + o, 0, n_s - 1))]
    x = torch.from_numpy(inp).to(dev)
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
        return None
    d = np.load(cache)
    a = d["ascans"]
    m = a.mean(axis=1, keepdims=True); s = a.std(axis=1, keepdims=True) + 1e-6
    a_t = torch.from_numpy(((a - m) / s).astype(np.float32)).unsqueeze(1)
    probs = np.zeros((a_t.shape[0], N_DEPTHS), dtype=np.float32)
    CHUNK = 512
    with torch.no_grad():
        for k in range(0, a_t.shape[0], CHUNK):
            probs[k:k+CHUNK] = torch.sigmoid(model(a_t[k:k+CHUNK])).cpu().numpy()
    return probs.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)


# ── per-slice panel renderer ─────────────────────────────────────────────────
def render_slice_panels(slc, stl_label, man_label,
                        pred_man, pred_ascan, pred_stl):
    """All six maps are (H, W) in [0, 1] (labels can be soft). Predictions are
    soft. Binary masks are computed at > 0.5. Returns base64 PNGs as a dict.
    """
    H, W = slc.shape
    has_man   = man_label is not None and man_label.any()
    has_stl   = stl_label.any()

    raw_b64 = _rgb_to_b64(_reds(slc))

    stl_rgb = _cmap(stl_label, color=(0.92, 0.42, 0.05))         # orange
    stl_b64 = _rgb_to_b64(stl_rgb)

    if has_man:
        man_b64 = _rgb_to_b64(_solid((man_label > 0).astype(np.float32),
                                     (0.12, 0.27, 0.66)))         # blue
    else:
        man_b64 = _rgb_to_b64(np.ones((H, W, 3)) * 0.94)

    pred_man_soft = _rgb_to_b64(_cmap(pred_man,   (0.12, 0.27, 0.66)))
    pred_asc_soft = _rgb_to_b64(_cmap(pred_ascan, (0.42, 0.10, 0.58)))   # purple
    pred_stl_soft = _rgb_to_b64(_cmap(pred_stl,   (0.10, 0.55, 0.20)))   # green

    pred_man_bin = _rgb_to_b64(_solid((pred_man   > 0.5).astype(np.float32),
                                      (0.12, 0.27, 0.66)))
    pred_asc_bin = _rgb_to_b64(_solid((pred_ascan > 0.5).astype(np.float32),
                                      (0.42, 0.10, 0.58)))
    pred_stl_bin = _rgb_to_b64(_solid((pred_stl   > 0.5).astype(np.float32),
                                      (0.10, 0.55, 0.20)))

    # Overlay: R=A-scan, G=STL-U-Net, B=Manual-U-Net (binary>0.5) on dark bg,
    # with STL footprint outlined in white dashes (approximated by 1-px edge).
    overlay = np.zeros((H, W, 3), dtype=np.float32)
    overlay[..., 0] = (pred_ascan > 0.5).astype(np.float32)
    overlay[..., 1] = (pred_stl   > 0.5).astype(np.float32)
    overlay[..., 2] = (pred_man   > 0.5).astype(np.float32)
    stl_edge = edges(stl_label > 0.5)
    overlay[stl_edge] = [1.0, 1.0, 1.0]
    overlay_b64 = _rgb_to_b64(overlay)

    n_stl   = int((stl_label > 0.5).sum())
    n_man_l = int((man_label > 0).sum()) if has_man else -1
    n_pm    = int((pred_man   > 0.5).sum())
    n_pa    = int((pred_ascan > 0.5).sum())
    n_ps    = int((pred_stl   > 0.5).sum())

    return dict(
        raw=raw_b64,
        stl=stl_b64, man=man_b64,
        pred_man=pred_man_bin, pred_man_soft=pred_man_soft,
        pred_ascan=pred_asc_bin, pred_ascan_soft=pred_asc_soft,
        pred_stl=pred_stl_bin,  pred_stl_soft=pred_stl_soft,
        overlay=overlay_b64,
        has_man=bool(has_man), has_stl=bool(has_stl),
        n_stl=n_stl, n_man=n_man_l, n_pm=n_pm, n_pa=n_pa, n_ps=n_ps,
    )


# ── main build ───────────────────────────────────────────────────────────────
def main():
    print(f"device(U-Net)={device_unet}   stride={STRIDE}")
    if not MODEL_STL.exists():
        sys.exit(f"missing {MODEL_STL} — train it first")
    if not MODEL_ASCAN.exists():
        sys.exit(f"missing {MODEL_ASCAN}")
    if not MODEL_MAN_2050.exists():
        sys.exit(f"missing {MODEL_MAN_2050}")

    model_stl = load_unet(MODEL_STL,       device_unet)
    model_man = load_unet(MODEL_MAN_2050,  device_unet)
    model_a   = load_ascan()
    print("  ✓ models loaded")

    sample_names = sorted([p.stem.replace("_slices", "")
                           for p in SLICES_DIR_50.glob("*_slices.npy")],
                          key=lambda x: int(x.rstrip("b")))
    print(f"  samples: {sample_names}")

    all_data = []
    for name in sample_names:
        slices_p_50 = SLICES_DIR_50  / f"{name}_slices.npy"
        depths_p_50 = SLICES_DIR_50  / f"{name}_depths.npy"
        slices_p_2050 = SLICES_DIR_2050 / f"{name}_slices.npy"
        depths_p_2050 = SLICES_DIR_2050 / f"{name}_depths.npy"
        if not slices_p_50.exists() or not depths_p_50.exists():
            print(f"  skip {name}: no 50-slice data"); continue

        slices_50 = np.load(slices_p_50)
        depths_50 = np.load(depths_p_50)
        n_s, H, W = slices_50.shape

        # STL labels (soft) and manual labels (binary, 50-slice)
        stl = np.load(STL_LABELS_DIR / f"{name}_slice_masks.npy")
        man_p = MANUAL_LBLS / f"{name}_slice_masks.npy"
        man = (np.load(man_p) if man_p.exists()
               else np.zeros_like(stl))

        # STL-U-Net (50-slice native) and A-scan (50-slice native)
        probs_stl   = predict_unet(model_stl, slices_50, OFFSETS_50, device_unet)
        probs_ascan = predict_ascan(model_a, name)
        if probs_ascan is None:
            probs_ascan = np.zeros_like(probs_stl)

        # Manual U-Net (2050-slice) → map to 50-slice depths
        probs_man_50 = np.zeros_like(probs_stl)
        if slices_p_2050.exists() and depths_p_2050.exists():
            slices_2050 = np.load(slices_p_2050)
            depths_2050 = np.load(depths_p_2050)
            probs_man_2050 = predict_unet(model_man, slices_2050,
                                          OFFSETS_2050, device_unet)
            for i, d in enumerate(depths_50):
                j = int(np.argmin(np.abs(depths_2050 - d)))
                probs_man_50[i] = probs_man_2050[j]
        else:
            print(f"  {name}: no 2050-slice data, manual U-Net pred will be empty")

        show_idx = list(range(0, n_s, STRIDE))
        # Always include slices that have STL or manual signal
        for i in range(n_s):
            if (stl[i] > 0.5).any() or (man[i] > 0).any():
                if i not in show_idx:
                    show_idx.append(i)
        show_idx = sorted(set(show_idx))

        print(f"  · {name}  rendering {len(show_idx)}/{n_s} slices…")

        slice_list = []
        for i in show_idx:
            panels = render_slice_panels(
                slc=slices_50[i],
                stl_label=stl[i],
                man_label=man[i],
                pred_man=probs_man_50[i],
                pred_ascan=probs_ascan[i],
                pred_stl=probs_stl[i],
            )
            panels["depth"]     = round(float(depths_50[i]), 4)
            panels["slice_idx"] = int(i)
            slice_list.append(panels)

        all_data.append(dict(name=name, n_slices=int(n_s), slices=slice_list))

    # ── HTML ──
    html_head = """<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>THz Void Detection — 3-Way Viewer (atlanta2)</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
       background:#1a1a2e;color:#e0e0e0}
  .header{background:#16213e;padding:12px 20px;display:flex;align-items:center;
          gap:20px;flex-wrap:wrap}
  .header h1{font-size:17px}
  .tag{background:#e94560;padding:2px 8px;border-radius:3px;font-size:11px;
       font-weight:bold}
  .sample-btn{padding:5px 11px;border:1px solid #444;border-radius:4px;
              background:#0f3460;color:#e0e0e0;cursor:pointer;font-size:12px}
  .sample-btn:hover{background:#1a5276}
  .sample-btn.active{background:#e94560;border-color:#e94560;font-weight:bold}
  .controls{background:#16213e;padding:10px 20px;display:flex;
            align-items:center;gap:15px;border-top:1px solid #333}
  .controls input[type=range]{flex:1;max-width:600px}
  .info{font-size:12px;color:#aaa}
  .toggle-btn{padding:4px 10px;border:1px solid #444;border-radius:4px;
              background:#0f3460;color:#e0e0e0;cursor:pointer;font-size:11px}
  .toggle-btn.active{background:#e94560;border-color:#e94560}
  .panels{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;padding:15px}
  .panel{text-align:center}
  .panel img{width:100%;max-width:170px;aspect-ratio:1;image-rendering:pixelated;
             border:2px solid #333;border-radius:4px}
  .panel .label{font-size:11px;margin-top:4px;color:#aaa}
  .panel .label b{color:#e0e0e0}
  .panel .nogt{position:relative;display:inline-block}
  .panel .nogt::after{content:"no manual GT";position:absolute;top:6px;left:6px;
                      background:rgba(0,0,0,.65);color:#ccc;padding:2px 6px;
                      font-size:10px;border-radius:3px}
  .legend{display:flex;justify-content:center;gap:24px;padding:8px;font-size:12px}
  .legend .dot{display:inline-block;width:10px;height:10px;border-radius:50%;
               margin-right:4px;vertical-align:middle}
  kbd{padding:1px 6px;background:#0f3460;border:1px solid #444;border-radius:3px;
      font-family:monospace;font-size:11px}
</style></head><body>
<div class="header">
  <h1>THz Void Detection — 3-Way Viewer</h1>
  <span class="tag">atlanta2</span>
  <span class="info">Manual-U-Net (2050) · A-scan 1D-CNN · STL-U-Net (50)</span>
  <div id="sample-btns" style="display:flex;gap:5px;flex-wrap:wrap;margin-left:auto"></div>
</div>
<div class="controls">
  <label>Slice: <span id="slice-idx">0</span></label>
  <input type="range" id="slider" min="0" max="0" value="0">
  <span id="depth-info" class="info"></span>
  <button class="toggle-btn" id="soft-toggle" onclick="toggleSoft()">Show Soft</button>
  <span class="info">Keys: <kbd>&larr;</kbd><kbd>&rarr;</kbd> slice &nbsp;
        <kbd>n</kbd><kbd>p</kbd> sample &nbsp;<kbd>t</kbd> binary/soft</span>
</div>
<div class="panels">
  <div class="panel"><img id="img-raw"><div class="label"><b>Input</b></div></div>
  <div class="panel"><img id="img-stl"><div class="label"><b>STL labels</b> <span id="cnt-stl"></span></div></div>
  <div class="panel"><img id="img-man"><div class="label"><b>Manual rectangles</b> <span id="cnt-man"></span></div></div>
  <div class="panel"><img id="img-pm"><div class="label"><b>Manual-U-Net pred</b> <span id="cnt-pm"></span></div></div>
  <div class="panel"><img id="img-pa"><div class="label"><b>A-scan 1D-CNN pred</b> <span id="cnt-pa"></span></div></div>
  <div class="panel"><img id="img-ps"><div class="label"><b>STL-U-Net pred</b> <span id="cnt-ps"></span></div></div>
  <div class="panel"><img id="img-ov"><div class="label"><b>Overlay</b> (RGB)</div></div>
</div>
<div class="legend">
  <span><span class="dot" style="background:rgb(31,69,168)"></span> Manual-U-Net (blue)</span>
  <span><span class="dot" style="background:rgb(108,26,148)"></span> A-scan (purple)</span>
  <span><span class="dot" style="background:rgb(26,140,52)"></span> STL-U-Net (green)</span>
  <span><span class="dot" style="background:rgb(255,255,255);border:1px solid #888"></span> STL outline (overlay)</span>
</div>
<script>
const DATA = """

    html_tail = """;
let curSample = 0, curIdx = 0, showSoft = false;

function init() {
  const btns = document.getElementById("sample-btns");
  DATA.forEach((s, i) => {
    const b = document.createElement("button");
    b.className = "sample-btn"; b.textContent = "S" + s.name;
    b.onclick = () => selectSample(i);
    btns.appendChild(b);
  });
  document.getElementById("slider").addEventListener("input", e => {
    curIdx = parseInt(e.target.value); render();
  });
  document.addEventListener("keydown", onKey);
  selectSample(0);
}
function selectSample(i) {
  curSample = i; curIdx = 0;
  const s = DATA[i];
  const slider = document.getElementById("slider");
  slider.max = s.slices.length - 1; slider.value = 0;
  document.querySelectorAll(".sample-btn").forEach((b, j) =>
    b.classList.toggle("active", j === i));
  render();
}
function render() {
  const s = DATA[curSample], sl = s.slices[curIdx];
  document.getElementById("slice-idx").textContent =
    sl.slice_idx + "/" + (s.n_slices - 1) +
    "  (view " + curIdx + "/" + (s.slices.length - 1) + ")";
  document.getElementById("depth-info").textContent =
    "depth = " + sl.depth + " mm  ·  Sample " + s.name;
  document.getElementById("img-raw").src = "data:image/png;base64," + sl.raw;
  document.getElementById("img-stl").src = "data:image/png;base64," + sl.stl;
  document.getElementById("img-man").src = "data:image/png;base64," + sl.man;
  document.getElementById("img-pm").src  = "data:image/png;base64," +
    (showSoft ? sl.pred_man_soft   : sl.pred_man);
  document.getElementById("img-pa").src  = "data:image/png;base64," +
    (showSoft ? sl.pred_ascan_soft : sl.pred_ascan);
  document.getElementById("img-ps").src  = "data:image/png;base64," +
    (showSoft ? sl.pred_stl_soft   : sl.pred_stl);
  document.getElementById("img-ov").src  = "data:image/png;base64," + sl.overlay;
  document.getElementById("cnt-stl").textContent = "(" + sl.n_stl + " px)";
  document.getElementById("cnt-man").textContent =
    sl.has_man ? "(" + sl.n_man + " px)" : "(n/a)";
  document.getElementById("cnt-pm").textContent  = "(" + sl.n_pm + " px)";
  document.getElementById("cnt-pa").textContent  = "(" + sl.n_pa + " px)";
  document.getElementById("cnt-ps").textContent  = "(" + sl.n_ps + " px)";
}
function toggleSoft() {
  showSoft = !showSoft;
  const btn = document.getElementById("soft-toggle");
  btn.classList.toggle("active", showSoft);
  btn.textContent = showSoft ? "Show Binary" : "Show Soft";
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
  } else if (e.key === "t") { toggleSoft(); }
}
init();
</script></body></html>"""

    OUT_HTML.parent.mkdir(exist_ok=True, parents=True)
    with open(OUT_HTML, "w") as fh:
        fh.write(html_head)
        fh.write(json.dumps(all_data))
        fh.write(html_tail)

    sz_mb = OUT_HTML.stat().st_size / 1e6
    total_slices = sum(len(s["slices"]) for s in all_data)
    print(f"\n  done — {OUT_HTML}")
    print(f"  {sz_mb:.1f} MB  ·  {total_slices} embedded slices across "
          f"{len(all_data)} samples")


if __name__ == "__main__":
    main()
