"""
build_ascan_compare_viewer.py — interactive HTML to compare the A-scan models
=============================================================================
Side-by-side depth-slice viewer: Input · GT · 1D-CNN (depth classifier) ·
U-Net-BiLSTM. Both A-scan models are run per-pixel and reduced to the SAME 50
depth bins so the spatial comparison is apples-to-apples. Scrub depth + sample.

Output: runs/unet_bilstm/viewer_ascan_compare.html  (self-contained, base64).
Run:    thesis_env/bin/python build_ascan_compare_viewer.py
"""
import sys, io, base64, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch

import os
_ARGV = sys.argv[1:]
sys.path.insert(0, str(Path(__file__).parent))
from ascan_unet_bilstm import (UNetBiLSTM1D, SEQ_LEN, N_DEPTHS,
                               slice_indices, bin_to_time_index,
                               reduce_to_depth_bins, crop_or_pad)
from ascan_classifier import (AScanCNN, CACHE_ATL2,
                              TRAIN_SAMPLES, VAL_SAMPLES, NOVOID_SANITY)
CNN_CKPT_DEFAULT = "results_v2_atlanta2/ascan_classifier/ascan_cnn_depth_nofilt.pt"

# Env overrides let this build the band-pass viewer without code changes:
#   ASCAN_CACHE2=ascan_cache_atlanta2_bp (cache, read by ascan_classifier),
#   ASCAN_CNN_CKPT, ASCAN_BILSTM_CKPT, ASCAN_SLICES_DIR, ASCAN_VIEWER_OUT
CNN_CKPT    = Path(os.environ.get("ASCAN_CNN_CKPT", str(CNN_CKPT_DEFAULT)))
BILSTM_CKPT = Path(os.environ.get("ASCAN_BILSTM_CKPT", "runs/unet_bilstm/unet_bilstm_best.pt"))
SLICES_DIR  = Path(os.environ.get("ASCAN_SLICES_DIR", "slices_v2_atlanta2"))
LABELS_DIR  = Path("labels_v2_atlanta2")
OUT         = Path(os.environ.get("ASCAN_VIEWER_OUT", "runs/unet_bilstm/viewer_ascan_compare.html"))
device = "cpu"   # 1D-CNN pool needs CPU; BiLSTM also fine on CPU for 15 samples


def b64(rgb):
    img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    img = img.resize((200, 200), Image.NEAREST)
    buf = io.BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def reds(v):
    v = np.clip(v, 0, 1)
    return np.stack([1 - 0.28*v, 1 - 0.97*v, 1 - 0.97*v], axis=-1)


def binary_red(m):
    rgb = np.ones((*m.shape, 3))
    rgb[m > 0] = [0.72, 0.11, 0.11]
    return rgb


def load_models():
    cnn = AScanCNN().to(device)
    cnn.load_state_dict(torch.load(CNN_CKPT, map_location=device, weights_only=False)["model_state"])
    cnn.eval()
    bk = torch.load(BILSTM_CKPT, map_location=device, weights_only=False)
    bil = UNetBiLSTM1D(base=bk["config"].get("base", 16),
                       bilstm_deep_skips=bk["config"].get("bilstm", True)).to(device)
    bil.load_state_dict(bk["model_state"]); bil.eval()
    return cnn, bil


def predict_volumes(name, cnn, bil):
    d = np.load(CACHE_ATL2 / f"{name}.npz")
    a = d["ascans"].astype(np.float32)
    m, s = a.mean(1, keepdims=True), a.std(1, keepdims=True) + 1e-6
    az = (a - m) / s
    # 1D-CNN: native 50 bins
    xt = torch.from_numpy(az).unsqueeze(1).to(device)
    cnn_p = np.zeros((a.shape[0], N_DEPTHS), np.float32)
    bil_p2048 = np.zeros((a.shape[0], SEQ_LEN), np.float32)
    azc = crop_or_pad(az, SEQ_LEN)
    xtc = torch.from_numpy(azc).unsqueeze(1).to(device)
    with torch.no_grad():
        for k in range(0, a.shape[0], 512):
            cnn_p[k:k+512] = torch.sigmoid(cnn(xt[k:k+512])).cpu().numpy()
            bil_p2048[k:k+512] = torch.sigmoid(bil(xtc[k:k+512])).cpu().numpy()
    kmap = bin_to_time_index(slice_indices(d), SEQ_LEN)
    bil_p = reduce_to_depth_bins(bil_p2048, kmap, N_DEPTHS)
    NY = NX = 100
    cnn_vol = cnn_p.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)
    bil_vol = bil_p.reshape(NY, NX, N_DEPTHS).transpose(2, 0, 1)
    return cnn_vol, bil_vol


def main():
    cnn, bil = load_models()
    names = sorted(set(TRAIN_SAMPLES + VAL_SAMPLES + NOVOID_SANITY),
                   key=lambda x: int(x.rstrip("b")))
    split = {**{n: "TRAIN" for n in TRAIN_SAMPLES},
             **{n: "VAL" for n in VAL_SAMPLES}, **{n: "NOVOID" for n in NOVOID_SANITY}}

    samples = []
    for name in names:
        sp = SLICES_DIR / f"{name}_slices.npy"
        if not sp.exists():
            continue
        inp = np.load(sp)                                  # (50,100,100) [0,1]
        depths = np.load(SLICES_DIR / f"{name}_depths.npy")
        gt = (np.load(LABELS_DIR / f"{name}_slice_masks.npy") > 0
              if (LABELS_DIR / f"{name}_slice_masks.npy").exists()
              else np.zeros_like(inp, bool))
        cnn_vol, bil_vol = predict_volumes(name, cnn, bil)
        print(f"  {name} [{split[name]}]: gt_vox={int(gt.sum())} "
              f"cnn>{0.5}={int((cnn_vol>0.5).sum())} bilstm>0.5={int((bil_vol>0.5).sum())}")
        slices = []
        for i in range(inp.shape[0]):
            slices.append(dict(
                d=float(depths[i]),
                inp=b64(reds(inp[i])),
                gt=b64(binary_red(gt[i])),
                cnn=b64(reds(cnn_vol[i])),
                bil=b64(reds(bil_vol[i])),
                gtpx=int(gt[i].sum()),
                cnnpx=int((cnn_vol[i] > 0.5).sum()),
                bilpx=int((bil_vol[i] > 0.5).sum()),
            ))
        samples.append(dict(name=name, split=split[name], slices=slices))

    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(samples))
    OUT.write_text(html)
    mb = OUT.stat().st_size / 1e6
    print(f"\nSaved: {OUT}  ({mb:.1f} MB, {len(samples)} samples)")


HTML_TEMPLATE = r"""<!doctype html><html><head><meta charset=utf-8>
<title>A-scan model comparison</title><style>
body{font-family:system-ui,Arial;margin:0;background:#111;color:#eee}
.bar{padding:10px 16px;background:#1c1c1c;position:sticky;top:0}
select,input{font-size:15px}
.panels{display:flex;gap:18px;padding:18px;flex-wrap:wrap}
.panel{text-align:center}
.panel img{width:200px;height:200px;image-rendering:pixelated;border:1px solid #333;background:#fff}
.panel h3{margin:6px 0 2px;font-size:14px;font-weight:600}
.panel .sub{font-size:12px;color:#9bd}
.cnn h3{color:#f5a} .bil h3{color:#7df}
small{color:#888}
</style></head><body>
<div class=bar>
  Sample <select id=samp></select>
  &nbsp; Depth slice <input id=sl type=range min=0 max=49 value=25 style="width:360px">
  <span id=info></span>
  <br><small>← → change slice · n/p change sample · 1D-CNN = depth classifier (50-bin) · U-Net-BiLSTM = per-time, reduced to 50 bins · soft probability (red = void)</small>
</div>
<div class=panels>
  <div class=panel><h3>Input (envelope)</h3><div class=sub id=s0>&nbsp;</div><img id=i0></div>
  <div class=panel><h3 style="color:#c33">Manual GT</h3><div class=sub id=s1>&nbsp;</div><img id=i1></div>
  <div class="panel cnn"><h3>1D-CNN pred</h3><div class=sub id=s2>&nbsp;</div><img id=i2></div>
  <div class="panel bil"><h3>U-Net-BiLSTM pred</h3><div class=sub id=s3>&nbsp;</div><img id=i3></div>
</div>
<script>
const DATA=__DATA__;
const samp=document.getElementById('samp'), sl=document.getElementById('sl');
DATA.forEach((s,i)=>{let o=document.createElement('option');o.value=i;o.text=`${s.name} [${s.split}]`;samp.add(o);});
function draw(){
  const s=DATA[+samp.value], i=+sl.value, sc=s.slices[i];
  i0.src='data:image/png;base64,'+sc.inp; i1.src='data:image/png;base64,'+sc.gt;
  i2.src='data:image/png;base64,'+sc.cnn; i3.src='data:image/png;base64,'+sc.bil;
  s1.textContent=`GT px: ${sc.gtpx}`; s2.textContent=`pred px: ${sc.cnnpx}`;
  s3.textContent=`pred px: ${sc.bilpx}`; s0.textContent=`depth ${sc.d.toFixed(2)} mm`;
  info.textContent=`  slice ${i}/49  ·  depth ${sc.d.toFixed(2)} mm`;
}
samp.onchange=draw; sl.oninput=draw;
document.onkeydown=e=>{
  if(e.key==='ArrowRight')sl.value=Math.min(49,+sl.value+1);
  else if(e.key==='ArrowLeft')sl.value=Math.max(0,+sl.value-1);
  else if(e.key==='n')samp.value=Math.min(DATA.length-1,+samp.value+1);
  else if(e.key==='p')samp.value=Math.max(0,+samp.value-1);
  else return; draw();
};
draw();
</script></body></html>"""


if __name__ == "__main__":
    main()
