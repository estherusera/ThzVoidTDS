"""
export_slices.py — Pre-export ROI-cropped depth slices to .npy files
=====================================================================
Run this ONCE on the machine that has the .tprj raw data.
Outputs go to slices_v2/ and can be committed to git so the training
server never needs the raw .tprj file.

Output per sample:
  slices_v2/{name}_slices.npy   — float32 (N_SLICES, 100, 100)
  slices_v2/{name}_depths.npy   — float32 (N_SLICES,) depth in mm

Run:
    thesis_env/bin/python export_slices.py
"""

import sys, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from thz_slice_pipeline import load_all_volumes, process_to_slices

TPRJ     = "3D_print_esther_atlanta.tprj"
ROI_JSON = "sample_rois.json"
N_SLICES = 50
OUT_DIR  = Path("slices_v2")

PHYSICAL_NAMES = {
    "1":    "1",  "2":  "2",  "3":  "3",  "4":   "4",
    "5":    "5",  "5b": "5b", "6-9":"6",  "7":   "7",
    "8":    "8",  "9-6.":"9", "10": "10", "11":  "11",
    "12":   "12", "13": "13", "14": "14", "15":  "15",
}

SPACING_OVERRIDES = {
    "1": dict(dx_mm=0.2, dy_mm=0.5),
}


def main():
    with open(ROI_JSON) as fh:
        rois = json.load(fh)

    OUT_DIR.mkdir(exist_ok=True)

    print("Loading volumes from tprj…")
    raw_samples = load_all_volumes([TPRJ])

    print(f"\nExporting {N_SLICES}-slice ROI crops to {OUT_DIR}/\n")
    exported = []
    for s in raw_samples:
        tprj_name = s["name"].split("/")[-1]
        if tprj_name.lower().startswith("test"):
            continue

        phys = PHYSICAL_NAMES.get(tprj_name, tprj_name)
        if phys not in rois:
            print(f"  –  {phys}: no ROI, skipping")
            continue

        if phys in SPACING_OVERRIDES:
            s = dict(s)
            s.update(SPACING_OVERRIDES[phys])

        try:
            slices, depths, _, _ = process_to_slices(s, n_slices=N_SLICES)
            roi = rois[phys]
            slices = slices[:, roi["r0"]:roi["r1"], roi["c0"]:roi["c1"]].copy()

            np.save(OUT_DIR / f"{phys}_slices.npy", slices.astype(np.float32))
            np.save(OUT_DIR / f"{phys}_depths.npy", depths.astype(np.float32))
            print(f"  ✓  {phys}  {slices.shape}  depth 0–{depths[-1]:.2f} mm")
            exported.append(phys)
        except Exception as e:
            print(f"  ✗  {phys}: {e}")

    print(f"\nDone. {len(exported)} samples exported to {OUT_DIR}/")
    print("Now run:  git add slices_v2/ && git commit -m 'add exported slices' && git push")


if __name__ == "__main__":
    main()
