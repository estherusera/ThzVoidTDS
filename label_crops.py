"""
label_crops.py — Interactive crop labeler
==========================================
Shows depth slices for each sample one at a time.
Draw a rectangle with click-and-drag to define the crop box.
Press ENTER or click "Save & Next" to confirm and move on.
Press 'r' to reset the rectangle.
Results are written to crop_coords.json.

Run:
    thesis_env/bin/python label_crops.py
"""

import json, os, sys
import h5py
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # needs Tk — falls back to Qt5Agg if absent
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
from matplotlib.widgets import RectangleSelector, Button
import scipy.ndimage

# ── config ────────────────────────────────────────────────────────────────────
TPRJ       = "3D_print_esther_atlanta.tprj"
OUT_JSON   = "crop_coords.json"
N_PLA      = 1.5
C_MM_PS    = 0.29979

SAMPLES = [
    ("2",  0.2, 0.5),
    ("3",  0.2, 0.5),
    ("4",  0.2, 0.5),
    ("5",  0.2, 0.5),
    ("5b", 0.2, 0.5),
    ("8",  0.2, 0.2),
    ("10", 0.2, 0.2),
    ("11", 0.5, 0.2),
    ("12", 0.5, 0.2),
    ("13", 0.5, 0.2),
    ("14", 0.5, 0.2),
    ("15", 0.5, 0.2),
]
# ──────────────────────────────────────────────────────────────────────────────


def load_and_flatten(f, sname):
    img_base = "TerapulseDocument/Measurements/Image Data"
    for img_id in f[img_base]:
        sg = f[f"{img_base}/{img_id}/sample"]
        attr = sg.attrs.get("SampleName", None)
        if attr is None:
            continue
        try:
            n = str(attr[0, 0]).strip()
        except Exception:
            n = str(attr).strip()
        if n != sname:
            continue

        data = sg["data"][:]
        line = sg["line"][:].ravel()
        x_off = float(sg.attrs["X_Offset"][0, 0])
        x_spc = float(sg.attrs["X_Spacing"][0, 0])
        wfm   = int(sg.attrs["WfmLength"][0, 0])
        tax   = x_off + x_spc * np.arange(wfm)
        us    = sg.attrs["UserScanSettings"][0, 0]
        sc    = json.loads(us)["MeasurementConfig"]["Token"]["scanner_config"]
        dx_   = float(sc["axis1_spacing"])
        dy_   = float(sc["axis2_spacing"])
        x_max = float(sc["axis1_max"]); x_min_ = float(sc["axis1_min"])
        y_max = float(sc["axis2_max"]); y_min_ = float(sc["axis2_min"])

        nt = len(tax)
        ns = len(line)
        nf = int(line.max())

        # correct wrong metadata spacing (e.g. sample 10)
        if nf > 1:
            inferred = (x_max - x_min_) / (nf - 1)
            if abs(inferred - dx_) / max(dx_, 1e-6) > 0.10:
                dx_ = inferred
        if ns > 1:
            inferred = (y_max - y_min_) / (ns - 1)
            if abs(inferred - dy_) / max(dy_, 1e-6) > 0.10:
                dy_ = inferred
        vol = np.zeros((ns, nf, nt), dtype=data.dtype)
        col = 0
        for i in range(ns):
            ni = int(line[i])
            vol[i, :ni, :] = data[:nt, col:col + ni].T
            col += ni

        # surface flatten
        amp  = np.max(np.abs(vol), axis=2)
        sig  = amp > np.percentile(amp[amp > 0], 10) if np.any(amp > 0) else np.ones(amp.shape, bool)
        t0   = np.argmax(np.abs(vol), axis=2)
        t0[~sig] = 0
        ti   = (np.arange(nt)[None, None, :] + t0[:, :, None]) % nt
        flat = vol[np.arange(ns)[:, None, None],
                   np.arange(nf)[None, :, None], ti]
        flat = flat[:, :, :nt - int(t0.max())]

        dt   = float(tax[1] - tax[0])
        dz   = C_MM_PS / (2 * N_PLA) * dt
        return flat, dz, dx_, dy_

    raise RuntimeError(f"Sample {sname!r} not found")


def run_labeler():
    # Load existing coords if the file already exists
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as fh:
            saved = json.load(fh)
    else:
        saved = {}

    with h5py.File(TPRJ, "r") as f:
        for sname, dx_cfg, dy_cfg in SAMPLES:

            if sname in saved:
                print(f"Sample {sname}: already labelled ({saved[sname]}) — skipping")
                continue

            print(f"\nLoading sample {sname} …", end=" ", flush=True)
            try:
                flat, dz, dx, dy = load_and_flatten(f, sname)
            except Exception as e:
                print(f"ERROR: {e}")
                continue
            print("done")

            ns, nf, nz = flat.shape
            x_ax   = np.arange(nf) * dx     # fast axis (cols) in mm
            y_ax   = np.arange(ns) * dy     # slow axis (rows) in mm
            extent = [x_ax[0], x_ax[-1], y_ax[-1], y_ax[0]]

            # 6 depth slices up to ~1 mm
            z_max  = min(nz - 1, int(round(1.0 / dz)))
            z_idxs = np.linspace(10, z_max, 6, dtype=int)
            amp    = np.abs(flat)

            # ── build figure ─────────────────────────────────────────────────
            fig = plt.figure(figsize=(18, 11))
            fig.suptitle(
                f"Sample {sname}  |  {ns*dy:.0f}mm(slow) × {nf*dx:.0f}mm(fast)"
                f"  |  dx={dx} dy={dy} mm\n"
                "DRAG a rectangle on any panel, then click  [ Save & Next ]",
                fontsize=11, fontweight="bold",
            )

            gs   = fig.add_gridspec(3, 3,
                                    left=0.06, right=0.97,
                                    top=0.88,  bottom=0.12,
                                    hspace=0.35, wspace=0.3)
            axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
            btn_ax  = fig.add_axes([0.35, 0.02, 0.15, 0.06])
            skip_ax = fig.add_axes([0.52, 0.02, 0.12, 0.06])

            btn_save = Button(btn_ax,  "Save & Next", color="limegreen")
            btn_skip = Button(skip_ax, "Skip",        color="lightsalmon")

            # draw slices
            for i, zi in enumerate(z_idxs):
                ax  = axes[i]
                slc = amp[:, :, zi]
                ax.imshow(slc, cmap="inferno", extent=extent,
                          origin="upper", aspect="equal",
                          vmin=np.percentile(slc, 2),
                          vmax=np.percentile(slc, 98))
                ax.set_title(f"z = {zi*dz:.3f} mm", fontsize=8)
                ax.set_xlabel("Fast (mm)", fontsize=7)
                ax.set_ylabel("Slow (mm)", fontsize=7)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
                ax.grid(color="cyan", linewidth=0.4, alpha=0.6)
                ax.tick_params(labelsize=6)

            # shared state
            state = {"rect": None, "coords": None, "action": None}

            # highlight rectangle patches on all axes
            rect_patches = [None] * 6

            def on_select(eclick, erelease):
                # eclick/erelease are in data coordinates (mm)
                x0 = min(eclick.xdata, erelease.xdata)
                x1 = max(eclick.xdata, erelease.xdata)
                y0 = min(eclick.ydata, erelease.ydata)
                y1 = max(eclick.ydata, erelease.ydata)
                state["coords"] = dict(fast_min=round(x0, 2), fast_max=round(x1, 2),
                                       slow_min=round(y0, 2), slow_max=round(y1, 2))
                # draw rectangle on all panels
                for pi, ax in enumerate(axes):
                    if rect_patches[pi] is not None:
                        rect_patches[pi].remove()
                    rp = mpatches.Rectangle(
                        (x0, y0), x1 - x0, y1 - y0,
                        linewidth=2, edgecolor="lime", facecolor="lime",
                        alpha=0.25)
                    ax.add_patch(rp)
                    rect_patches[pi] = rp
                fig.canvas.draw_idle()
                print(f"  Preview: fast=[{x0:.1f}, {x1:.1f}] mm  "
                      f"slow=[{y0:.1f}, {y1:.1f}] mm")

            # attach RectangleSelector to the first (top-left) axis
            rs = RectangleSelector(
                axes[0], on_select,
                useblit=True,
                button=[1],
                minspanx=1, minspany=1,
                spancoords="data",
                interactive=True,
            )

            def save_cb(event):
                state["action"] = "save"
                plt.close(fig)

            def skip_cb(event):
                state["action"] = "skip"
                plt.close(fig)

            btn_save.on_clicked(save_cb)
            btn_skip.on_clicked(skip_cb)

            # also close on Enter key
            def on_key(event):
                if event.key in ("enter", "return"):
                    state["action"] = "save"
                    plt.close(fig)
                elif event.key == "r":
                    state["coords"] = None
                    for pi, ax in enumerate(axes):
                        if rect_patches[pi] is not None:
                            rect_patches[pi].remove()
                            rect_patches[pi] = None
                    fig.canvas.draw_idle()

            fig.canvas.mpl_connect("key_press_event", on_key)

            plt.show(block=True)

            # ── process result ────────────────────────────────────────────────
            if state["action"] == "save" and state["coords"] is not None:
                saved[sname] = state["coords"]
                with open(OUT_JSON, "w") as fh:
                    json.dump(saved, fh, indent=2)
                c = state["coords"]
                print(f"  ✓  Saved: fast=[{c['fast_min']}, {c['fast_max']}]  "
                      f"slow=[{c['slow_min']}, {c['slow_max']}]")
            elif state["action"] == "skip":
                print(f"  –  Skipped sample {sname}")
            else:
                print(f"  !  No selection made for sample {sname} — skipped")

    print(f"\nAll done. Coordinates written to {OUT_JSON}")
    print("Run   python pipeline_batch.py   to process with these crops.")


if __name__ == "__main__":
    run_labeler()
