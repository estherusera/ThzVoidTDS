"""
roi_labeler.py — Interactive sample ROI labeler
================================================
Uses identical loading and processing as thz_slice_pipeline.py so the
visualization is exactly the same.

For each sample the scan area may be larger than the 20×20mm physical sample.
Click anywhere on the image to place a fixed 20×20mm box over the sample.
For scans that are already ~20×20mm the box is auto-placed.

Controls:
  • Depth slider / ← → keys — scrub through depth slices
  • Click on image    — centre the 20×20mm ROI box at that point
  • 'r'               — reset / remove ROI
  • ENTER or "Save & Next" — save and advance
  • "Skip"            — skip this sample
  • "◀ Prev"          — go back

Output:
  sample_rois.json — {physical_name: {r0,r1,c0,c1,zoom_ny,zoom_nx,dx_mm,dy_mm}}
  Pixel coordinates in isotropic-zoomed space (after min(dx,dy) resampling).

Run:
    thesis_env/bin/python roi_labeler.py
"""

import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.widgets import Slider, Button
from pathlib import Path

# ── use the exact same load + process as the pipeline ──────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from thz_slice_pipeline import load_all_volumes, process_to_slices

TPRJ     = "3D_print_esther_atlanta.tprj"
OUT_JSON = "sample_rois.json"
N_SLICES = 250

# Physical name mapping: tprj sample_name → human label
# (tprj names are like "3D_print_esther_atlanta/2", "…/6-9", etc.)
PHYSICAL_NAMES = {
    "1":    "1",
    "2":    "2",
    "3":    "3",
    "4":    "4",
    "5":    "5",
    "5b":   "5b",
    "6-9":  "6",     # accidental duplicate of sample 2
    "7":    "7",
    "8":    "8",
    "9-6.": "9",     # rotated rectangles
    "10":   "10",
    "11":   "11",
    "12":   "12",
    "13":   "13",
    "14":   "14",
    "15":   "15",
}

CMAP = plt.get_cmap("Reds", 200)
NORM = mcolors.Normalize(vmin=0, vmax=1)
ROI_MM = 20.0   # all physical samples are 20×20mm

# Override spacing for samples whose tprj range metadata is wrong.
# Sample 1 has ±30mm range stored instead of ±15mm (same Ny=61, Nx=152 as samples 6-9, 7).
SPACING_OVERRIDES = {
    "1": dict(dx_mm=0.2, dy_mm=0.5),
}


class ROILabeler:
    def __init__(self, samples_data, saved):
        self.samples     = samples_data   # list of pipeline dicts with 'slices', 'depths_mm', etc.
        self.saved       = saved
        self.sample_idx  = 0
        self.z_idx       = 0
        self.roi         = None   # (r0, r1, c0, c1) zoomed-px

        self._find_next_unsaved()
        self._build_figure()
        self._load_current()
        self._draw()

    # ── helpers ────────────────────────────────────────────────────────────────
    def _current_sd(self):
        return self.samples[self.sample_idx]

    def _roi_px(self):
        """Fixed 20×20mm box size in zoomed pixels."""
        sd = self._current_sd()
        pitch = min(sd['dx_mm'], sd['dy_mm'])
        return int(round(ROI_MM / pitch))

    def _find_next_unsaved(self):
        for i, sd in enumerate(self.samples):
            if sd['physical_name'] not in self.saved:
                self.sample_idx = i
                return
        self.sample_idx = len(self.samples)

    # ── figure ─────────────────────────────────────────────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(11, 9))

        gs = gridspec.GridSpec(
            1, 1, figure=self.fig,
            left=0.08, right=0.97,
            bottom=0.18, top=0.92,
        )
        self.ax_main = self.fig.add_subplot(gs[0])

        ax_sl = self.fig.add_axes([0.12, 0.11, 0.76, 0.03])
        self.slider = Slider(ax_sl, "Depth (mm)", 0.0, 1.0,
                             valinit=0.0, color="#c0392b")
        self.slider.on_changed(self._on_slider)

        ax_prev  = self.fig.add_axes([0.08, 0.03, 0.12, 0.05])
        ax_save  = self.fig.add_axes([0.38, 0.03, 0.18, 0.05])
        ax_skip  = self.fig.add_axes([0.58, 0.03, 0.10, 0.05])
        ax_reset = self.fig.add_axes([0.70, 0.03, 0.10, 0.05])

        self.btn_prev  = Button(ax_prev,  "◀  Prev",        color="#dddddd", hovercolor="#bbbbbb")
        self.btn_save  = Button(ax_save,  "Save & Next ▶",  color="#a8d5a2", hovercolor="#7fbf7b")
        self.btn_skip  = Button(ax_skip,  "Skip",            color="#f5c6c6", hovercolor="#e89090")
        self.btn_reset = Button(ax_reset, "Reset ROI",       color="#fde8c8", hovercolor="#f5c88a")

        self.btn_prev.on_clicked(self._prev_sample)
        self.btn_save.on_clicked(self._save_cb)
        self.btn_skip.on_clicked(self._skip_cb)
        self.btn_reset.on_clicked(self._reset_roi)

        self.title = self.fig.text(0.5, 0.955, "", ha="center", va="top",
                                   fontsize=13, fontweight="bold", color="black")
        self.subtitle = self.fig.text(0.5, 0.925, "", ha="center", va="top",
                                      fontsize=8, color="#444444")

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",   self._on_key)

    # ── data ───────────────────────────────────────────────────────────────────
    def _load_current(self):
        if self.sample_idx >= len(self.samples):
            return
        sd     = self._current_sd()
        slices = sd['slices']          # already (n_s, Ny, Nx) normalised [0,1]
        depths = sd['depths_mm']
        n_s, zoom_ny, zoom_nx = slices.shape

        self.z_idx = 0
        self.roi   = None

        # Auto-place if scan is already ~20×20mm
        roi_px = self._roi_px()
        if zoom_ny <= roi_px + 5 and zoom_nx <= roi_px + 5:
            self.roi = (0, zoom_ny - 1, 0, zoom_nx - 1)
            print(f"  → scan ~20×20mm, ROI auto-set to full image")

    # ── drawing ────────────────────────────────────────────────────────────────
    def _draw(self):
        if self.sample_idx >= len(self.samples):
            self.ax_main.cla()
            self.ax_main.text(0.5, 0.5, "All samples done!",
                              ha="center", va="center",
                              transform=self.ax_main.transAxes, fontsize=16)
            self.title.set_text("Done")
            self.subtitle.set_text("")
            self.fig.canvas.draw_idle()
            return

        sd = self._current_sd()
        depths = sd['depths_mm']
        n_s    = len(depths)
        max_mm = float(depths[-1])

        self.slider.valmin = 0.0
        self.slider.valmax = max_mm
        self.slider.ax.set_xlim(0.0, max_mm)
        self.z_idx = min(self.z_idx, n_s - 1)
        self.slider.set_val(depths[self.z_idx])

        n_done = sum(1 for s in self.samples if s['physical_name'] in self.saved)
        self.title.set_text(
            f"Sample {sd['physical_name']}  "
            f"({self.sample_idx+1}/{len(self.samples)})  [{n_done} saved]"
        )
        self._redraw_slice()

    def _redraw_slice(self):
        if self.sample_idx >= len(self.samples):
            return
        sd     = self._current_sd()
        slices = sd['slices']
        depths = sd['depths_mm']
        n_s, zoom_ny, zoom_nx = slices.shape
        zi  = max(0, min(self.z_idx, n_s - 1))
        slc = slices[zi]   # already normalised [0,1] by the pipeline

        self.ax_main.cla()
        self.ax_main.imshow(slc, cmap=CMAP, norm=NORM, origin="upper")

        if self.roi is not None:
            r0, r1, c0, c1 = self.roi
            rp = mpatches.Rectangle(
                (c0, r0), c1 - c0, r1 - r0,
                linewidth=2, edgecolor="#2196f3", facecolor="#2196f3", alpha=0.20)
            self.ax_main.add_patch(rp)

        # mm tick labels
        pitch = min(sd['dx_mm'], sd['dy_mm'])
        self.ax_main.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{x*pitch:.1f}"))
        self.ax_main.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda y, _: f"{y*pitch:.1f}"))
        self.ax_main.set_xlabel("Fast axis (mm)", fontsize=9)
        self.ax_main.set_ylabel("Slow axis (mm)", fontsize=9)
        self.ax_main.tick_params(labelsize=8)
        self.ax_main.axis("on")

        roi_px = self._roi_px()
        scan_mm = f"{zoom_ny*pitch:.0f}×{zoom_nx*pitch:.0f} mm"
        if self.roi is not None:
            r0, r1, c0, c1 = self.roi
            roi_str = (f"ROI [{r0}:{r1}, {c0}:{c1}]  "
                       f"= {(r1-r0)*pitch:.1f}×{(c1-c0)*pitch:.1f} mm  ✓")
        else:
            roi_str = f"Click to place {ROI_MM:.0f}×{ROI_MM:.0f}mm box  ({roi_px}×{roi_px} px)"
        self.subtitle.set_text(
            f"z = {depths[zi]:.3f} mm  (slice {zi}/{n_s-1})  |  "
            f"scan={scan_mm}  |  {roi_str}"
        )
        self.fig.canvas.draw_idle()

    # ── callbacks ──────────────────────────────────────────────────────────────
    def _on_slider(self, val):
        sd = self._current_sd()
        self.z_idx = int(np.argmin(np.abs(sd['depths_mm'] - val)))
        self._redraw_slice()

    def _on_click(self, event):
        if event.inaxes is not self.ax_main or self.sample_idx >= len(self.samples):
            return
        sd = self._current_sd()
        _, zoom_ny, zoom_nx = sd['slices'].shape
        roi_px = self._roi_px()
        half   = roi_px // 2
        cx = int(round(event.xdata))
        cy = int(round(event.ydata))
        c0, c1 = cx - half, cx - half + roi_px
        r0, r1 = cy - half, cy - half + roi_px
        if c0 < 0:           c0, c1 = 0, roi_px
        if c1 > zoom_nx - 1: c0, c1 = zoom_nx - roi_px, zoom_nx - 1
        if r0 < 0:           r0, r1 = 0, roi_px
        if r1 > zoom_ny - 1: r0, r1 = zoom_ny - roi_px, zoom_ny - 1
        self.roi = (r0, r1, c0, c1)
        pitch = min(sd['dx_mm'], sd['dy_mm'])
        print(f"  ROI: rows [{r0},{r1}] cols [{c0},{c1}]  "
              f"= {(r1-r0)*pitch:.1f}×{(c1-c0)*pitch:.1f} mm")
        self._redraw_slice()

    def _save_cb(self, event):
        if self.sample_idx >= len(self.samples):
            return
        if self.roi is None:
            print("  ! Click the image to place the ROI first")
            return
        sd = self._current_sd()
        _, zoom_ny, zoom_nx = sd['slices'].shape
        r0, r1, c0, c1 = self.roi
        self.saved[sd['physical_name']] = dict(
            r0=r0, r1=r1, c0=c0, c1=c1,
            zoom_ny=zoom_ny, zoom_nx=zoom_nx,
            dx_mm=sd['dx_mm'], dy_mm=sd['dy_mm'],
        )
        with open(OUT_JSON, "w") as fh:
            json.dump(self.saved, fh, indent=2)
        pitch = min(sd['dx_mm'], sd['dy_mm'])
        print(f"  ✓  Saved {sd['physical_name']}: "
              f"rows=[{r0},{r1}] cols=[{c0},{c1}]  "
              f"= {(r1-r0)*pitch:.1f}×{(c1-c0)*pitch:.1f} mm")
        self._advance()

    def _skip_cb(self, event):
        print(f"  –  Skipped {self._current_sd()['physical_name']}")
        self._advance()

    def _reset_roi(self, event):
        self.roi = None
        self._redraw_slice()

    def _prev_sample(self, event):
        if self.sample_idx > 0:
            self.sample_idx -= 1
            self._load_current()
            self._draw()

    def _advance(self):
        self.sample_idx += 1
        while (self.sample_idx < len(self.samples) and
               self.samples[self.sample_idx]['physical_name'] in self.saved):
            self.sample_idx += 1
        self._load_current()
        self._draw()

    def _on_key(self, event):
        if event.key == "right":
            sd = self._current_sd()
            self.z_idx = min(self.z_idx + 1, len(sd['depths_mm']) - 1)
            self.slider.set_val(sd['depths_mm'][self.z_idx])
        elif event.key == "left":
            self.z_idx = max(self.z_idx - 1, 0)
            sd = self._current_sd()
            self.slider.set_val(sd['depths_mm'][self.z_idx])
        elif event.key in ("enter", "return"):
            self._save_cb(None)
        elif event.key == "r":
            self._reset_roi(None)

    def show(self):
        plt.show()


def main():
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as fh:
            saved = json.load(fh)
    else:
        saved = {}

    print("Loading volumes from tprj (same as pipeline)…")
    raw_samples = load_all_volumes([TPRJ])

    print(f"\nProcessing {len(raw_samples)} samples to {N_SLICES} depth slices…")
    samples_data = []
    for s in raw_samples:
        tprj_name = s['name'].split('/')[-1]
        if tprj_name.lower().startswith('test'):
            continue
        phys = PHYSICAL_NAMES.get(tprj_name, tprj_name)
        # Apply spacing override before processing (fixes bad range metadata)
        if phys in SPACING_OVERRIDES:
            s = dict(s)  # shallow copy so we don't mutate the original
            s.update(SPACING_OVERRIDES[phys])
        try:
            slices, depths, _, _ = process_to_slices(s, n_slices=N_SLICES)
            samples_data.append({
                'physical_name': phys,
                'tprj_name':     tprj_name,
                'slices':        slices,
                'depths_mm':     depths,
                'dx_mm':         s['dx_mm'],
                'dy_mm':         s['dy_mm'],
            })
            print(f"  ✓  {phys} (tprj: {tprj_name})  {slices.shape}")
        except Exception as e:
            print(f"  ✗  {phys}: {e}")

    remaining = [s for s in samples_data if s['physical_name'] not in saved]
    print(f"\n{len(saved)} already saved, {len(remaining)} remaining.")
    if not remaining:
        print("All done. Delete sample_rois.json to redo.")
        return

    labeler = ROILabeler(samples_data, saved)
    labeler.show()
    print(f"\nROIs written to {OUT_JSON}")


if __name__ == "__main__":
    main()
