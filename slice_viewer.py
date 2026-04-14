"""
slice_viewer.py — Interactive depth-slice viewer
=================================================
Shows each cropped THz volume with its boundary mask overlay.
Controls:
  • Depth slider  — scrub through z-slices
  • ← / →  keys  — step one slice at a time
  • Prev / Next buttons — switch sample
  • 'm' key — toggle mask overlay on/off

Run:
    thesis_env/bin/python slice_viewer.py
"""

import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button

# ── samples to view ───────────────────────────────────────────────────────────
SAMPLES = [
    dict(name="2",   label="CV02",           dx=0.2, dy=0.5, outdir="output_atlanta_s2"),
    dict(name="3",   label="CV03",           dx=0.2, dy=0.5, outdir="output_atlanta_s3"),
    dict(name="4",   label="CV04",           dx=0.2, dy=0.5, outdir="output_atlanta_s4"),
    dict(name="5",   label="CV05",           dx=0.2, dy=0.5, outdir="output_atlanta_s5"),
    dict(name="5b",  label="CV05 (repeat)",  dx=0.2, dy=0.5, outdir="output_atlanta_s5b"),
    dict(name="8",   label="CV08",           dx=0.2, dy=0.2, outdir="output_atlanta_s8"),
    dict(name="10",  label="CV10-Diagonal",  dx=0.493, dy=0.2, outdir="output_atlanta_s10"),
    dict(name="11",  label="CV11-Stripes",   dx=0.5, dy=0.2, outdir="output_atlanta_s11"),
    dict(name="12",  label="CV12-Prism",     dx=0.5, dy=0.2, outdir="output_atlanta_s12"),
    dict(name="13",  label="CV13-Random",    dx=0.5, dy=0.2, outdir="output_atlanta_s13"),
    dict(name="14",  label="CV14-Cilindros", dx=0.5, dy=0.2, outdir="output_atlanta_s14"),
    dict(name="15",  label="CV15-NoVoid",    dx=0.5, dy=0.2, outdir="output_atlanta_s15"),
]
# ──────────────────────────────────────────────────────────────────────────────


def load_sample(s):
    thz  = np.load(os.path.join(s["outdir"], "thz_volume.npy"))
    mask = np.load(os.path.join(s["outdir"], "boundary_mask.npy"))
    meta = json.load(open(os.path.join(s["outdir"], "metadata.json")))
    return thz, mask, float(meta["dz_mm"])


class Viewer:
    def __init__(self):
        self.idx       = 0          # current sample index
        self.z_idx     = 0          # current depth slice
        self.show_mask = True

        # pre-load all samples
        print("Loading samples …")
        self.data = []
        for s in SAMPLES:
            try:
                thz, mask, dz = load_sample(s)
                self.data.append(dict(thz=thz, mask=mask, dz=dz, **s))
                print(f"  ✓  {s['name']:4s}  shape={thz.shape}")
            except Exception as e:
                print(f"  ✗  {s['name']:4s}  {e}")
                self.data.append(None)

        self._build_figure()
        self._draw()

    # ── figure layout ────────────────────────────────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(11, 9))
        self.fig.patch.set_facecolor("#1e1e1e")

        gs = gridspec.GridSpec(
            3, 3,
            figure=self.fig,
            left=0.08, right=0.97,
            bottom=0.18, top=0.92,
            hspace=0.05, wspace=0.05,
        )

        # main image (large, spans 3 cols × 2 rows)
        self.ax_main = self.fig.add_subplot(gs[0:2, :])
        self.ax_main.set_facecolor("black")

        # slider axis
        ax_sl = self.fig.add_axes([0.12, 0.10, 0.76, 0.03],
                                   facecolor="#333333")
        self.slider = Slider(ax_sl, "Depth (mm)", 0.0, 1.0,
                             valinit=0.0, color="#4caf50")
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")
        self.slider.on_changed(self._on_slider)

        # buttons
        ax_prev = self.fig.add_axes([0.12, 0.03, 0.13, 0.05])
        ax_next = self.fig.add_axes([0.75, 0.03, 0.13, 0.05])
        ax_mask = self.fig.add_axes([0.44, 0.03, 0.13, 0.05])

        self.btn_prev = Button(ax_prev, "◀  Prev",
                               color="#37474f", hovercolor="#546e7a")
        self.btn_next = Button(ax_next, "Next  ▶",
                               color="#37474f", hovercolor="#546e7a")
        self.btn_mask = Button(ax_mask, "Mask: ON",
                               color="#1b5e20", hovercolor="#2e7d32")

        for btn in (self.btn_prev, self.btn_next, self.btn_mask):
            btn.label.set_color("white")

        self.btn_prev.on_clicked(lambda e: self._change_sample(-1))
        self.btn_next.on_clicked(lambda e: self._change_sample(+1))
        self.btn_mask.on_clicked(self._toggle_mask)

        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

        # title placeholder
        self.title = self.fig.text(
            0.5, 0.955, "", ha="center", va="top",
            fontsize=13, fontweight="bold", color="white",
        )
        self.subtitle = self.fig.text(
            0.5, 0.925, "", ha="center", va="top",
            fontsize=9, color="#aaaaaa",
        )

        self.im_thz  = None
        self.im_mask = None

    # ── drawing ──────────────────────────────────────────────────────────────
    def _current(self):
        return self.data[self.idx]

    def _draw(self):
        d = self._current()
        if d is None:
            self.ax_main.cla()
            self.ax_main.text(0.5, 0.5, "No data", color="white",
                              ha="center", va="center",
                              transform=self.ax_main.transAxes)
            self.fig.canvas.draw_idle()
            return

        thz  = d["thz"]
        mask = d["mask"]
        dz   = d["dz"]
        dx   = d["dx"]
        dy   = d["dy"]
        ns, nf, nz = thz.shape

        # update slider range
        max_mm = (nz - 1) * dz
        self.slider.valmin = 0.0
        self.slider.valmax = max_mm
        self.slider.ax.set_xlim(0.0, max_mm)
        self.z_idx = min(self.z_idx, nz - 1)
        self.slider.set_val(self.z_idx * dz)

        self._redraw_slice()

    def _redraw_slice(self):
        d = self._current()
        if d is None:
            return

        thz  = d["thz"]
        mask = d["mask"]
        dz   = d["dz"]
        dx   = d["dx"]
        dy   = d["dy"]
        ns, nf, nz = thz.shape
        zi = max(0, min(self.z_idx, nz - 1))

        extent = [0, nf * dx, ns * dy, 0]
        slc    = np.abs(thz[:, :, zi])
        msk    = mask[:, :, zi].astype(float)
        msk[msk == 0] = np.nan

        self.ax_main.cla()
        self.ax_main.set_facecolor("black")

        self.ax_main.imshow(
            slc, cmap="inferno", extent=extent, origin="upper",
            aspect="equal",
            vmin=np.percentile(slc, 2), vmax=np.percentile(slc, 98),
        )

        if self.show_mask:
            self.ax_main.imshow(
                msk, cmap="cool", alpha=0.50, extent=extent,
                origin="upper", aspect="equal", vmin=0, vmax=1,
            )

        self.ax_main.set_xlabel("Fast axis (mm)", color="white", fontsize=9)
        self.ax_main.set_ylabel("Slow axis (mm)", color="white", fontsize=9)
        self.ax_main.tick_params(colors="white", labelsize=8)
        for sp in self.ax_main.spines.values():
            sp.set_edgecolor("#555555")

        # titles
        self.title.set_text(
            f"Sample {d['name']}  —  {d['label']}"
            f"  ({self.idx + 1}/{len(self.data)})"
        )
        self.subtitle.set_text(
            f"z = {zi * dz:.4f} mm   (slice {zi} / {nz - 1})   |   "
            f"volume: {ns*dy:.1f}×{nf*dx:.1f} mm   |   "
            f"dz = {dz:.5f} mm   |   "
            f"mask voxels at this slice: {int(mask[:, :, zi].sum())}"
        )

        self.fig.canvas.draw_idle()

    # ── callbacks ────────────────────────────────────────────────────────────
    def _on_slider(self, val):
        d = self._current()
        if d is None:
            return
        dz = d["dz"]
        self.z_idx = int(round(val / dz))
        self._redraw_slice()

    def _change_sample(self, delta):
        self.idx   = (self.idx + delta) % len(self.data)
        self.z_idx = 0
        self._draw()

    def _toggle_mask(self, event):
        self.show_mask = not self.show_mask
        self.btn_mask.label.set_text("Mask: ON" if self.show_mask else "Mask: OFF")
        self.btn_mask.ax.set_facecolor("#1b5e20" if self.show_mask else "#4a148c")
        self._redraw_slice()

    def _on_key(self, event):
        if event.key == "right":
            d = self._current()
            if d:
                self.z_idx = min(self.z_idx + 1, d["thz"].shape[2] - 1)
                self.slider.set_val(self.z_idx * d["dz"])
        elif event.key == "left":
            self.z_idx = max(self.z_idx - 1, 0)
            d = self._current()
            if d:
                self.slider.set_val(self.z_idx * d["dz"])
        elif event.key == "m":
            self._toggle_mask(None)
        elif event.key == "n":
            self._change_sample(+1)
        elif event.key == "p":
            self._change_sample(-1)

    def show(self):
        plt.show()


if __name__ == "__main__":
    v = Viewer()
    v.show()
