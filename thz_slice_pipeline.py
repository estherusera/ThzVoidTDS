"""
THz Void Detection — Per-Slice Labeling & Training
====================================================
Instead of one mask per sample, label voids in EACH depth slice 
where they're visible. This is:
  - More natural (label what you see)
  - More data (20 slices × N samples = 20N training pairs)
  - Depth-aware (model learns where voids appear at each depth)

Workflow:
  1. python thz_slice_pipeline.py label 3D_print_Dicky.tprj
     → browse slices, paint void regions, save per-slice masks
  
  2. python thz_slice_pipeline.py train 3D_print_Dicky.tprj
     → train U-Net on labeled slices

  3. python thz_slice_pipeline.py predict 3D_print_Dicky.tprj
     → run trained model, visualize 3D void map
"""

import h5py
import numpy as np
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RectangleSelector
from scipy.signal import hilbert
from pathlib import Path
import sys
import warnings
from PIL import Image

warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# DATA LOADING & PROCESSING
# ============================================================

def load_all_volumes(tprj_paths):
    """Load all samples from .tprj files.

    Supports two storage layouts:
      - DataViews  (e.g. 3D_print_Dicky.tprj): data under
        TerapulseDocument/DataViews/<name>/PC_1/*_C/raw data/sample/
      - Measurements/Image Data  (e.g. 3D_print_esther_atlanta.tprj):
        data under TerapulseDocument/Measurements/Image Data/<id>/sample/
        identified by the SampleName attribute.
    """
    all_samples = []
    for tprj_path in tprj_paths:
        stem = Path(tprj_path).stem
        with h5py.File(tprj_path, 'r') as f:

            # ── Layout 1: DataViews ──────────────────────────────────────────
            data_paths = []
            def find_data(name, obj):
                if isinstance(obj, h5py.Dataset) and name.endswith('raw data/sample/data'):
                    data_paths.append(name)
            f.visititems(find_data)

            for dpath in data_paths:
                parts = dpath.split('/')
                dv_idx = parts.index('DataViews')
                sample_name = parts[dv_idx + 1]
                parent = dpath.replace('/data', '')
                line_path = parent + '/line'
                try:
                    raw = f[dpath][:]
                    line_data = np.array(f[line_path][:]).flatten()
                    attrs = dict(f[parent].attrs)
                    N_time, N_flat = raw.shape
                    Ny = len(line_data)
                    Nx = N_flat // Ny
                    if Ny * Nx > N_flat:
                        continue
                    volume = raw[:, :Ny * Nx].reshape(N_time, Ny, Nx)
                    x_offset = float(np.array(attrs.get('X_Offset', [[0]])).flatten()[0])
                    x_spacing = float(np.array(attrs.get('X_Spacing', [[0.02]])).flatten()[0])
                    time_axis = x_offset + np.arange(N_time) * x_spacing
                    # Extract spatial spacing from scan settings JSON if available
                    import json as _json
                    dx_mm = dy_mm = None
                    sc_raw = attrs.get('UserScanSettings', None)
                    if sc_raw is not None:
                        try:
                            sc_str = str(np.array(sc_raw).flatten()[0])
                            sc = _json.loads(sc_str)["MeasurementConfig"]["Token"]["scanner_config"]
                            dx_mm = float(sc["axis1_spacing"])
                            dy_mm = float(sc["axis2_spacing"])
                        except Exception:
                            pass
                    all_samples.append({
                        'name': f"{stem}/{sample_name}",
                        'volume': volume, 'time_axis': time_axis,
                        'Ny': Ny, 'Nx': Nx,
                        'dx_mm': dx_mm, 'dy_mm': dy_mm,
                    })
                    print(f"  ✓ {sample_name}: ({N_time}, {Ny}, {Nx})")
                except Exception as e:
                    print(f"  ✗ {sample_name}: {e}")

            # ── Layout 2: Measurements/Image Data ───────────────────────────
            # Skip if this file was already fully covered by DataViews
            loaded_names = {s['name'].split('/')[-1] for s in all_samples
                            if s['name'].startswith(stem + '/')}
            img_base = "TerapulseDocument/Measurements/Image Data"
            if img_base in f:
                import json as _json
                for img_id in f[img_base]:
                    sg = f[f"{img_base}/{img_id}/sample"]
                    # Read SampleName attribute
                    attr = sg.attrs.get("SampleName", None)
                    if attr is None:
                        continue
                    try:
                        sample_name = str(attr[0, 0]).strip()
                    except Exception:
                        sample_name = str(attr).strip()

                    # Skip test frames and samples already loaded from DataViews
                    if 'test' in sample_name.lower() or sample_name in loaded_names:
                        continue

                    try:
                        raw   = sg["data"][:]          # (N_time, N_flat)
                        line  = sg["line"][:].ravel()  # (Ny,)
                        N_time, N_flat = raw.shape
                        Ny   = len(line)
                        Nx   = int(line.max())

                        # Time axis from attrs
                        x_off = float(sg.attrs["X_Offset"][0, 0])
                        x_spc = float(sg.attrs["X_Spacing"][0, 0])
                        time_axis = x_off + x_spc * np.arange(N_time)

                        # Spatial spacing — infer from scan config; correct if wrong
                        sc    = _json.loads(sg.attrs["UserScanSettings"][0, 0]
                                            )["MeasurementConfig"]["Token"]["scanner_config"]
                        dx_mm = float(sc["axis1_spacing"])
                        dy_mm = float(sc["axis2_spacing"])
                        x_min = float(sc["axis1_min"]); x_max = float(sc["axis1_max"])
                        y_min = float(sc["axis2_min"]); y_max = float(sc["axis2_max"])
                        if Nx > 1:
                            inferred = (x_max - x_min) / (Nx - 1)
                            if abs(inferred - dx_mm) / max(dx_mm, 1e-6) > 0.10:
                                dx_mm = inferred
                        if Ny > 1:
                            inferred = (y_max - y_min) / (Ny - 1)
                            if abs(inferred - dy_mm) / max(dy_mm, 1e-6) > 0.10:
                                dy_mm = inferred

                        # Reconstruct (N_time, Ny, Nx) volume from packed data
                        volume = np.zeros((N_time, Ny, Nx), dtype=raw.dtype)
                        col = 0
                        for i in range(Ny):
                            ni = int(line[i])
                            volume[:, i, :ni] = raw[:N_time, col:col + ni]
                            col += ni

                        all_samples.append({
                            'name': f"{stem}/{sample_name}",
                            'volume': volume, 'time_axis': time_axis,
                            'Ny': Ny, 'Nx': Nx,
                            'dx_mm': dx_mm, 'dy_mm': dy_mm,
                        })
                        print(f"  ✓ sample {sample_name}: ({N_time}, {Ny}, {Nx})  "
                              f"dy={dy_mm:.3f} dx={dx_mm:.3f} mm")
                    except Exception as e:
                        print(f"  ✗ sample {sample_name}: {e}")

    return all_samples


def process_to_slices(sample, n_slices=20, n_pla=1.57, c=0.29979):
    """
    Converts raw THz waveforms into normalized depth slices
    Process sample → normalized depth slices + metadata.
    
    Returns:
        slices: (n_slices, Ny, Nx) normalized [0,1]
        slice_depths_mm: (n_slices,) depth in mm
        surface_idx: int
    """
    volume = sample['volume']
    N_time, Ny, Nx = volume.shape
    time_axis = sample['time_axis']
    
    # Envelope
    envelope = np.abs(hilbert(volume, axis=0))
    
    # Surface detection + flattening: this is the strongest peak in the averaged A-scan, which finds the time index where the THz pulse first hits the sample surface. 
    mean_ascan = np.mean(envelope, axis=(1, 2))
    surface_idx = np.argmax(mean_ascan)
    
    margin = int(0.2 * N_time)
    s0, s1 = max(0, surface_idx - margin), min(N_time, surface_idx + margin)
    #flatenning is odne so that it aligns all A-scans so the surface is at the same time index everywhere
    local_surface = s0 + np.argmax(envelope[s0:s1], axis=0)
    target_idx = int(np.median(local_surface))
    
    #VER BIEN. 

       
    flat_env = np.zeros_like(envelope)
    for iy in range(Ny):
        for ix in range(Nx):
            shift = target_idx - local_surface[iy, ix]
            flat_env[:, iy, ix] = np.roll(envelope[:, iy, ix], shift)
    
    # Extract evenly-spaced depth slices from surface to end of time window
    slice_indices = np.linspace(target_idx, N_time - 1, n_slices, dtype=int)
    slices = flat_env[slice_indices].copy()
    
    # Depth of each slice in mm
    surface_time = time_axis[target_idx]
    slice_depths_mm = ((time_axis[slice_indices] - surface_time) * c) / (2 * n_pla)
    
    # Depth profile: computed AFTER normalization so the surface spike doesn't
    # swamp everything. Use the 95th-percentile value per slice — this is high
    # when there are bright localised reflectors (voids) and low when the slice
    # is just flat noise. Computed here before the spatial resampling so the
    # variable 'slices' still refers to flat_env[slice_indices].
    # We normalise each slice temporarily just for the profile calculation.
    _profile_slices = np.zeros(slices.shape[0], dtype=np.float32)
    for _i in range(slices.shape[0]):
        _vmax = np.percentile(slices[_i], 99) + 1e-8
        _norm = np.clip(slices[_i] / _vmax, 0, 1)
        _profile_slices[_i] = float(np.percentile(_norm, 95))
    depth_profile = _profile_slices

    # Resample spatial dims to isotropic pixels (same physical mm per pixel in
    # both axes) so the sample looks square regardless of scanner step sizes.
    dx_mm = sample.get('dx_mm') or 1.0
    dy_mm = sample.get('dy_mm') or 1.0
    if abs(dx_mm - dy_mm) / max(dx_mm, dy_mm) > 0.05:
        import scipy.ndimage as _nd
        target = min(dx_mm, dy_mm)          # upsample coarse axis to match the fine one
        zoom_y = dy_mm / target             # <1 if dy is finer than target → shrink
        zoom_x = dx_mm / target
        slices = _nd.zoom(slices, (1, zoom_y, zoom_x), order=1)

    # Normalize each slice to [0, 1] for the neural network input
    for i in range(slices.shape[0]):
        vmax = np.percentile(slices[i], 99) + 1e-8
        slices[i] = np.clip(slices[i] / vmax, 0, 1).astype(np.float32)

    return slices, slice_depths_mm, target_idx, depth_profile


# ============================================================
# INTERACTIVE PER-SLICE LABELER
# ============================================================

class SliceLabeler:
    """
    Label voids in individual depth slices.
    
    For each sample, browse through depth slices and draw rectangles 
    on the ones where voids are visible. Skip slices with no voids.
    
    Controls:
      - Depth slider: browse slices
      - Left-click drag: ADD void rectangle
      - Right-click drag: REMOVE rectangle
      - c: Clear current slice mask
      - s: Save all masks for current sample
      - n: Next sample (auto-saves)
      - p: Previous sample
      - q: Quit (auto-saves all)
    """
    
    def __init__(self, samples_data, output_dir='labels'):
        self.samples = samples_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.current_sample = 0
        self.masks = {}  # {sample_name: (n_slices, Ny, Nx)}
        
        # Load existing or initialize
        for sd in samples_data:
            name_safe = sd['name'].replace('/', '_')
            mask_path = self.output_dir / f'{name_safe}_slice_masks.npy'
            n_s, Ny, Nx = sd['slices'].shape
            if mask_path.exists():
                loaded = np.load(mask_path)
                if loaded.shape == (n_s, Ny, Nx):
                    self.masks[sd['name']] = loaded
                    n_labeled = (loaded.sum(axis=(1,2)) > 0).sum()
                    print(f"  Loaded existing: {name_safe} ({n_labeled} slices labeled)")
                else:
                    print(f"  Shape mismatch for {name_safe} "
                          f"(saved {loaded.shape} vs current ({n_s},{Ny},{Nx})) — reinitializing")
                    self.masks[sd['name']] = np.zeros((n_s, Ny, Nx), dtype=np.uint8)
            else:
                self.masks[sd['name']] = np.zeros((n_s, Ny, Nx), dtype=np.uint8)
    
    def run(self):
        self.fig = plt.figure(figsize=(18, 7))
        gs = self.fig.add_gridspec(2, 5, height_ratios=[1, 0.06],
                                    hspace=0.3, wspace=0.3)
        
        self.ax_slice = self.fig.add_subplot(gs[0, 0:2])
        self.ax_mask = self.fig.add_subplot(gs[0, 2])
        self.ax_overlay = self.fig.add_subplot(gs[0, 3])
        self.ax_summary = self.fig.add_subplot(gs[0, 4])
        
        # Slider
        ax_slider = self.fig.add_subplot(gs[1, 0:3])
        n_slices = self.samples[0]['slices'].shape[0]
        self.slider = Slider(ax_slider, 'Slice', 0, n_slices - 1,
                            valinit=0, valstep=1)
        self.slider.on_changed(self._on_slider)
        
        # Buttons
        ax_clear = self.fig.add_axes([0.65, 0.02, 0.07, 0.04])
        ax_save = self.fig.add_axes([0.73, 0.02, 0.07, 0.04])
        ax_next = self.fig.add_axes([0.81, 0.02, 0.07, 0.04])
        ax_prev = self.fig.add_axes([0.89, 0.02, 0.07, 0.04])
        
        self.btn_clear = Button(ax_clear, 'Clear(c)')
        self.btn_save = Button(ax_save, 'Save(s)')
        self.btn_next = Button(ax_next, 'Next(n)')
        self.btn_prev = Button(ax_prev, 'Prev(p)')
        
        self.btn_clear.on_clicked(lambda e: self._clear_slice())
        self.btn_save.on_clicked(lambda e: self._save_current())
        self.btn_next.on_clicked(lambda e: self._change_sample(1))
        self.btn_prev.on_clicked(lambda e: self._change_sample(-1))
        
        # Rectangle selectors on the slice image
        self.rect_add = RectangleSelector(
            self.ax_slice, self._on_add,
            useblit=True, button=[1],
            props=dict(facecolor='lime', alpha=0.3),
            interactive=False)
        self.rect_remove = RectangleSelector(
            self.ax_slice, self._on_remove,
            useblit=True, button=[3],
            props=dict(facecolor='red', alpha=0.3),
            interactive=False)
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        
        self._display()
        plt.show()
        self._save_all()
    
    def _get_current(self):
        sd = self.samples[self.current_sample]
        si = int(self.slider.val)
        mask_vol = self.masks[sd['name']]
        return sd, si, mask_vol
    
    def _display(self):
        sd, si, mask_vol = self._get_current()
        slices = sd['slices']
        depths = sd['depths_mm']
        n_slices = slices.shape[0]
        
        for ax in [self.ax_slice, self.ax_mask, self.ax_overlay, self.ax_summary]:
            ax.clear()
        
        # Current depth slice
        self.ax_slice.imshow(slices[si], cmap='Reds', vmin=0, vmax=1)
        self.ax_slice.set_title(f"Slice {si}/{n_slices-1} — depth={depths[si]:.2f} mm\n"
                                f"Left-drag=ADD  Right-drag=REMOVE",
                                fontsize=10)
        self.ax_slice.axis('off')
        
        # Current mask
        self.ax_mask.imshow(mask_vol[si], cmap='gray', vmin=0, vmax=1)
        n_px = mask_vol[si].sum()
        self.ax_mask.set_title(f'Mask ({n_px} px)', fontsize=10)
        self.ax_mask.axis('off')
        
        # Overlay
        overlay = np.zeros((*slices[si].shape, 3))
        overlay[..., 0] = slices[si]                    # red = data
        overlay[..., 1] = slices[si] * 0.3 + mask_vol[si] * 0.7  # green = mask
        overlay[..., 2] = slices[si] * 0.3
        self.ax_overlay.imshow(np.clip(overlay, 0, 1))
        self.ax_overlay.set_title('Overlay', fontsize=10)
        self.ax_overlay.axis('off')
        
        # Summary panel: depth profile (signal strength) + labeled-slice markers
        from scipy.signal import find_peaks as _find_peaks
        labeled_per_slice = mask_vol.sum(axis=(1, 2))
        profile = sd.get('depth_profile', np.zeros(n_slices))

        # Normalize profile to [0, 1] for display
        p_min, p_max = profile.min(), profile.max()
        p_norm = (profile - p_min) / (p_max - p_min + 1e-8)

        # Find prominent peaks (at least 10 % of range above neighbours)
        peaks, _ = _find_peaks(p_norm, prominence=0.10, distance=3)

        ax = self.ax_summary
        # Filled profile curve
        ax.fill_betweenx(range(n_slices), 0, p_norm, alpha=0.35, color='steelblue')
        ax.plot(p_norm, range(n_slices), color='steelblue', lw=1)

        # Mark peaks with orange dots + depth label
        for pk in peaks:
            ax.plot(p_norm[pk], pk, 'o', color='orange', ms=6, zorder=5)
            ax.text(p_norm[pk] + 0.03, pk, f'{depths[pk]:.2f}mm',
                    va='center', fontsize=6, color='orange')

        # Green ticks on the right edge for labeled slices
        for i, lps in enumerate(labeled_per_slice):
            if lps > 0:
                ax.axhspan(i - 0.4, i + 0.4, xmin=0.92, xmax=1.0,
                           color='limegreen', alpha=0.7)

        # Current-slice cursor
        ax.axhline(si, color='red', lw=1.5, alpha=0.8, linestyle='--')
        ax.text(0.01, si - 0.8, f'← {depths[si]:.2f}mm',
                fontsize=6, color='red', va='top')

        tick_step = max(1, n_slices // 20)
        ax.set_yticks(range(0, n_slices, tick_step))
        ax.set_yticklabels([f'{depths[i]:.1f}' for i in range(0, n_slices, tick_step)],
                           fontsize=6)
        ax.set_xlim(0, 1.35)
        ax.set_xlabel('Norm. amplitude', fontsize=7)
        ax.set_ylabel('Depth (mm)', fontsize=7)
        ax.set_title(f'Depth profile  ({len(peaks)} peaks)', fontsize=9)
        ax.invert_yaxis()
        
        n_labeled = (labeled_per_slice > 0).sum()
        self.fig.suptitle(
            f"Sample {self.current_sample+1}/{len(self.samples)}: {sd['name']}  |  "
            f"{n_labeled}/{n_slices} slices labeled  |  "
            f"Keys: c=clear, s=save, n/p=next/prev, q=quit",
            fontsize=11, fontweight='bold')
        
        self.fig.canvas.draw_idle()
    
    def _on_slider(self, val):
        self._display()
    
    def _on_add(self, eclick, erelease):
        sd, si, mask_vol = self._get_current()
        x0, x1 = sorted([int(eclick.xdata), int(erelease.xdata)])
        y0, y1 = sorted([int(eclick.ydata), int(erelease.ydata)])
        Ny, Nx = mask_vol.shape[1], mask_vol.shape[2]
        y0, y1 = max(0, y0), min(Ny, y1 + 1)
        x0, x1 = max(0, x0), min(Nx, x1 + 1)
        mask_vol[si, y0:y1, x0:x1] = 1
        self._display()
    
    def _on_remove(self, eclick, erelease):
        sd, si, mask_vol = self._get_current()
        x0, x1 = sorted([int(eclick.xdata), int(erelease.xdata)])
        y0, y1 = sorted([int(eclick.ydata), int(erelease.ydata)])
        Ny, Nx = mask_vol.shape[1], mask_vol.shape[2]
        y0, y1 = max(0, y0), min(Ny, y1 + 1)
        x0, x1 = max(0, x0), min(Nx, x1 + 1)
        mask_vol[si, y0:y1, x0:x1] = 0
        self._display()
    
    def _clear_slice(self):
        sd, si, mask_vol = self._get_current()
        mask_vol[si] = 0
        self._display()
    
    def _on_key(self, event):
        if event.key == 'c':
            self._clear_slice()
        elif event.key == 's':
            self._save_current()
        elif event.key == 'n':
            self._change_sample(1)
        elif event.key == 'p':
            self._change_sample(-1)
        elif event.key == 'q':
            self._save_all()
            plt.close(self.fig)
    
    def _change_sample(self, delta):
        self._save_current()
        self.current_sample = (self.current_sample + delta) % len(self.samples)
        n_slices = self.samples[self.current_sample]['slices'].shape[0]
        self.slider.valmax = n_slices - 1
        self.slider.set_val(0)
        self._display()
    
    def _save_current(self):
        sd = self.samples[self.current_sample]
        name_safe = sd['name'].replace('/', '_')
        np.save(self.output_dir / f'{name_safe}_slice_masks.npy',
                self.masks[sd['name']])
        n_labeled = (self.masks[sd['name']].sum(axis=(1,2)) > 0).sum()
        print(f"  Saved {name_safe}: {n_labeled} slices with labels")
    
    def _save_all(self):
        for sd in self.samples:
            name_safe = sd['name'].replace('/', '_')
            np.save(self.output_dir / f'{name_safe}_slice_masks.npy',
                    self.masks[sd['name']])
        print(f"\nAll masks saved to {self.output_dir}/")


# ============================================================
# U-NET (same as before, but 1-channel in → 1-channel out)
# ============================================================

if HAS_TORCH:
    class DoubleConv(nn.Module):
        def __init__(self, in_ch, out_ch):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        def forward(self, x):
            return self.conv(x)

    class UNet(nn.Module):
        """
        Per-slice U-Net: (B, 1, H, W) → (B, 1, H, W)
        Single depth slice in, void mask out.
        Small model suitable for small spatial dims (67×68).
        """
        
        def __init__(self, in_channels=1, base_filters=32):
            super().__init__()
            f = base_filters
            self.enc1 = DoubleConv(in_channels, f)
            self.enc2 = DoubleConv(f, f*2)
            self.enc3 = DoubleConv(f*2, f*4)
            self.pool = nn.MaxPool2d(2)
            self.bottleneck = DoubleConv(f*4, f*8)
            self.up3 = nn.ConvTranspose2d(f*8, f*4, 2, stride=2)
            self.dec3 = DoubleConv(f*8, f*4)
            self.up2 = nn.ConvTranspose2d(f*4, f*2, 2, stride=2)
            self.dec2 = DoubleConv(f*4, f*2)
            self.up1 = nn.ConvTranspose2d(f*2, f, 2, stride=2)
            self.dec1 = DoubleConv(f*2, f)
            self.out_conv = nn.Conv2d(f, 1, 1)

        def forward(self, x): #??
            _, _, H, W = x.shape
            pad_h = (8 - H % 8) % 8 #how many pixels to add so it is divisible by 8, era 67x68 
            pad_w = (8 - W % 8) % 8
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')
            
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))
            
            d3 = self.up3(b)
            d3 = self._cat(d3, e3)
            d3 = self.dec3(d3)
            d2 = self.up2(d3)
            d2 = self._cat(d2, e2)
            d2 = self.dec2(d2)
            d1 = self.up1(d2)
            d1 = self._cat(d1, e1)
            d1 = self.dec1(d1)
            
            return self.out_conv(d1)[:, :, :H, :W]
        
        def _cat(self, up, skip):
            dh = skip.shape[2] - up.shape[2]
            dw = skip.shape[3] - up.shape[3]
            #si no del mismo tamaño poner padding: 
            up = F.pad(up, [0, dw, 0, dh])
            return torch.cat([up, skip], dim=1)

    class SliceDataset(Dataset):
        """
        Each item is a single depth slice + its mask.
        Optionally includes neighboring slices as extra channels for context.
        """
        def __init__(self, all_slices, all_masks, context_slices=2, augment=False,
                     include_unlabeled=False):
            """
            Args:
                all_slices: list of (n_slices, Ny, Nx) per sample
                all_masks: list of (n_slices, Ny, Nx) per sample
                context_slices: number of neighboring slices above/below to include
                augment: random augmentation
                include_unlabeled: if True, include slices with empty masks (all zeros)
            """
            self.items = []  # (input_tensor, mask_tensor)
            self.context = context_slices
            self.augment = augment
            
            for slices, masks in zip(all_slices, all_masks):
                n_s, Ny, Nx = slices.shape
                for i in range(n_s):
                    has_label = masks[i].sum() > 0 #label of a pxiel is always >0 
                    if not has_label and not include_unlabeled:
                        continue
                    
                    # Build input: center slice + context neighbors
                    channels = []
                    for offset in range(-context_slices, context_slices + 1):
                        j = np.clip(i + offset, 0, n_s - 1) #para no llegar hasta el ultimo slice, y no tener ningún neihgbor, ns-1, i+offset {0,ns+1}
                        ch = Image.fromarray(slices[j]).resize((64, 64), Image.BILINEAR) 
                        channels.append(ch)
                        #channels.append(slices[j])
                    
                    inp = np.stack(channels, axis=0)  # (2*context+1, Ny, Nx)
                    self.items.append((inp.astype(np.float32),
                                      masks[i].astype(np.float32)))
            
            print(f"  Dataset: {len(self.items)} slice-mask pairs "
                  f"({context_slices*2+1} channels each)")
        
        def __len__(self):
            return len(self.items)
        
        def __getitem__(self, idx):
            x, y = self.items[idx]
            x = torch.FloatTensor(x)
            y = torch.FloatTensor(y).unsqueeze(0)  # (1, H, W)
            
            if self.augment:
                # Horizontal flip 
                if torch.rand(1) > 0.5:
                    x = torch.flip(x, [-1])
                    y = torch.flip(y, [-1])
                #vertical flip 
                if torch.rand(1) > 0.5:
                    x = torch.flip(x, [-2])
                    y = torch.flip(y, [-2])
                # Only 180° rotation to preserve dimensions (avoid 90°/270° that swap H/W)
                if torch.rand(1) > 0.5:
                    x = torch.rot90(x, 2, [-2, -1])
                    y = torch.rot90(y, 2, [-2, -1])
                x = x + 0.02 * torch.randn_like(x) #add small noise 
                x = torch.clamp(x, 0, 1)
            
            return x, y

    class DiceBCELoss(nn.Module):
        def __init__(self, dice_weight=0.5):
            super().__init__()
            self.dice_weight = dice_weight
            self.bce = nn.BCEWithLogitsLoss() #penalizes wrong pixel predictions
        
        def forward(self, logits, targets):
            bce = self.bce(logits, targets)
            probs = torch.sigmoid(logits)
            inter = (probs * targets).sum(dim=(2, 3)) 
            union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
            dice = (2 * inter + 1) / (union + 1) # measures overlap between prediction a nd ground truth 
            return (1 - self.dice_weight) * bce + self.dice_weight * (1 - dice.mean()) #loss que quieres minimizar, para actualizar pesos del modelo 


# ============================================================
# TRAINING
# ============================================================

def train(model, train_dataset, n_epochs=300, lr=1e-3, device='cpu'):
    loader = DataLoader(train_dataset, batch_size=min(len(train_dataset), 8),
                        shuffle=True, drop_last=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)
    criterion = DiceBCELoss(0.5)
    model.to(device)
    
    history = {'loss': [], 'dice': [], 'iou': []}
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nTraining: {len(train_dataset)} pairs, {n_epochs} epochs, "
          f"{n_params:,} params, device={device}")
    print(f"{'─' * 60}")
    
    for epoch in range(n_epochs):
        model.train()
        ep_loss, ep_dice, ep_iou, nb = 0, 0, 0, 0
        
        for x, y in loader: # comparando el modelo con el label  
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad() #borarr datos anteriores
            loss.backward() #calcular la derivada de la loss respecto a los parametros
            optimizer.step() #actualizar los pesos
            
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                inter = (preds * y).sum(dim=(2,3))
                dice = ((2*inter+1) / (preds.sum(dim=(2,3))+y.sum(dim=(2,3))+1)).mean()
                iou = ((inter+1) / (preds.sum(dim=(2,3))+y.sum(dim=(2,3))-inter+1)).mean()
            
            ep_loss += loss.item(); ep_dice += dice.item()
            ep_iou += iou.item(); nb += 1
        
        scheduler.step()
        history['loss'].append(ep_loss/nb)
        history['dice'].append(ep_dice/nb)
        history['iou'].append(ep_iou/nb)
        
        if (epoch+1) % 25 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d} | Loss: {ep_loss/nb:.4f} | "
                  f"Dice: {ep_dice/nb:.4f} | IoU: {ep_iou/nb:.4f}")
    
    return history


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_predictions(model, samples_data, device='cpu'):
    """Run model on all slices and show results."""
    model.eval()
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    for sd in samples_data:
        slices = sd['slices']
        masks = sd.get('masks', np.zeros_like(slices))
        name = sd['name']
        depths = sd['depths_mm']
        n_s, Ny, Nx = slices.shape
        
        # Predict all slices
        pred_masks = np.zeros((n_s, Ny, Nx))
        
        with torch.no_grad():
            for i in range(n_s): #build a 5 channel input para que 
                channels = []
                for offset in range(-2, 3):
                    j = np.clip(i + offset, 0, n_s - 1)
                    channels.append(slices[j])
                inp = torch.FloatTensor(np.stack(channels)).unsqueeze(0).to(device)
                logits = model(inp)
                pred_masks[i] = torch.sigmoid(logits).cpu().squeeze().numpy() 
        
        pred_binary = (pred_masks > 0.5).astype(float)
        
        # Plot: grid of slices with predictions
        n_show = min(n_s, 10)
        indices = np.linspace(0, n_s - 1, n_show, dtype=int)
        
        fig, axes = plt.subplots(3, n_show, figsize=(2.5 * n_show, 7), dpi=100)
        
        for j, si in enumerate(indices):
            axes[0, j].imshow(slices[si], cmap='Reds', vmin=0, vmax=1)
            axes[0, j].set_title(f'd={depths[si]:.1f}mm', fontsize=8)
            axes[0, j].axis('off')
            
            axes[1, j].imshow(masks[si], cmap='gray', vmin=0, vmax=1)
            axes[1, j].axis('off')
            
            # Overlay prediction
            overlay = np.zeros((Ny, Nx, 3))
            overlay[..., 0] = slices[si]
            overlay[..., 1] = pred_binary[si] * 0.7
            axes[2, j].imshow(np.clip(overlay, 0, 1))
            axes[2, j].axis('off')
        
        axes[0, 0].set_ylabel('Input', fontsize=10)
        axes[1, 0].set_ylabel('GT Mask', fontsize=10)
        axes[2, 0].set_ylabel('Prediction', fontsize=10)
        
        plt.suptitle(f'{name} — Per-Slice Predictions', fontsize=12, fontweight='bold')
        plt.tight_layout()
        name_safe = name.replace('/', '_')
        plt.savefig(results_dir / f'{name_safe}_predictions.png',
                    dpi=150, bbox_inches='tight')
        plt.show()
    
    print(f"Results saved to {results_dir}/")


def plot_history(history):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
    axes[0].plot(history['loss']); axes[0].set_title('Loss'); axes[0].set_yscale('log')
    axes[1].plot(history['dice']); axes[1].set_title('Dice'); axes[1].set_ylim(0, 1)
    axes[2].plot(history['iou']); axes[2].set_title('IoU'); axes[2].set_ylim(0, 1)
    for ax in axes: ax.set_xlabel('Epoch'); ax.grid(True, alpha=0.3)
    plt.suptitle('Training Curves', fontsize=14)
    plt.tight_layout()
    Path('results').mkdir(exist_ok=True)
    plt.savefig('results/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python thz_slice_pipeline.py label file1.tprj [file2.tprj ...]")
        print("  python thz_slice_pipeline.py train file1.tprj [file2.tprj ...]")
        print("  python thz_slice_pipeline.py predict file1.tprj [file2.tprj ...]")
        sys.exit(1)
    
    mode = sys.argv[1]
    tprj_files = sys.argv[2:] if len(sys.argv) > 2 else ['3D_print_Dicky.tprj']
    
    N_SLICES = 250
    CONTEXT = 2       # ±2 neighboring slices as extra input channels
    N_EPOCHS = 300
    LR = 1e-3
    DEVICE = 'cuda' if torch.cuda.is_available() else \
             'mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'
    
    # --- Load & process ---
    print("Loading volumes...")
    all_samples = load_all_volumes(tprj_files)
    
    print("\nProcessing to depth slices...")
    samples_data = []
    for sample in all_samples:
        try:
            slices, depths, si, profile = process_to_slices(sample, n_slices=N_SLICES)
            samples_data.append({
                'name': sample['name'],
                'slices': slices,
                'depths_mm': depths,
                'surface_idx': si,
                'depth_profile': profile,
            })
        except Exception as e:
            print(f"  SKIP {sample['name']}: {e}")
    
    # --- MODE: LABEL ---
    if mode == 'label':
        print(f"\nLaunching per-slice labeler for {len(samples_data)} samples...")
        print("  Browse slices → draw rectangles on voids → save")
        labeler = SliceLabeler(samples_data)
        labeler.run()
    
    # --- MODE: TRAIN ---
    elif mode == 'train':
        assert HAS_TORCH, "PyTorch required for training"
        
        # Load masks
        labels_dir = Path('labels')
        all_slices, all_masks = [], []
        
        for sd in samples_data:
            name_safe = sd['name'].replace('/', '_')
            mask_path = labels_dir / f'{name_safe}_slice_masks.npy'
            
            if mask_path.exists():
                masks = np.load(mask_path)
                n_labeled = (masks.sum(axis=(1,2)) > 0).sum()
                if n_labeled > 0:
                    sd['masks'] = masks
                    all_slices.append(sd['slices'])
                    all_masks.append(masks)
                    print(f"  {sd['name']}: {n_labeled}/{N_SLICES} slices labeled")
                else:
                    print(f"  {sd['name']}: no labeled slices, skipping")
            else:
                print(f"  {sd['name']}: no mask file found, skipping")
        
        if len(all_slices) == 0:
            print("\nNo labeled data found! Run 'label' mode first.")
            sys.exit(1)
        
        # Count total labeled slices
        total_labeled = sum((m.sum(axis=(1,2)) > 0).sum() for m in all_masks)
        total_empty = sum((m.sum(axis=(1,2)) == 0).sum() for m in all_masks)
        print(f"\nTotal: {total_labeled} slices with voids, "
              f"{total_empty} empty slices")

        # Check shapes and filter to consistent dimensions
        shapes = [s.shape for s in all_slices]
        print(f"\nSample shapes: {shapes}")
        from collections import Counter
        shape_counts = Counter([s[1:] for s in shapes])  # (Ny, Nx)
        most_common_shape = shape_counts.most_common(1)[0][0]
        print(f"Most common shape: {most_common_shape}")

        # Filter to keep only samples with the most common shape
        filtered_slices, filtered_masks = [], []
        for i, (slices, masks) in enumerate(zip(all_slices, all_masks)):
            if slices.shape[1:] == most_common_shape:
                filtered_slices.append(slices)
                filtered_masks.append(masks)
            else:
                print(f"  Skipping sample with shape {slices.shape} (doesn't match {most_common_shape})")

        all_slices, all_masks = filtered_slices, filtered_masks
        print(f"\nUsing {len(all_slices)} samples with consistent shape {most_common_shape}")

        # Build dataset: labeled slices + some empty slices as negatives
        dataset = SliceDataset(all_slices, all_masks,
                               context_slices=CONTEXT, augment=True,
                               include_unlabeled=True)  # include empty as negatives
        
        in_ch = CONTEXT * 2 + 1  # 5 channels
        model = UNet(in_channels=in_ch, base_filters=32)
        
        history = train(model, dataset, n_epochs=N_EPOCHS, lr=LR, device=DEVICE)
        plot_history(history)
        
        # Save
        Path('results').mkdir(exist_ok=True)
        torch.save({
            'model_state': model.state_dict(),
            'config': {'in_channels': in_ch, 'base_filters': 32,
                       'context': CONTEXT, 'n_slices': N_SLICES},
            'history': history,
        }, 'results/unet_slices.pt')
        
        # Visualize
        visualize_predictions(model, [sd for sd in samples_data if 'masks' in sd],
                             device=DEVICE)
        
        print(f"\n{'=' * 60}")
        print(f"Final — Loss: {history['loss'][-1]:.4f} | "
              f"Dice: {history['dice'][-1]:.4f} | IoU: {history['iou'][-1]:.4f}")
        print(f"Model saved: results/unet_slices.pt")
    
    # --- MODE: PREDICT ---
    elif mode == 'predict':
        assert HAS_TORCH, "PyTorch required"
        
        ckpt = torch.load('results/unet_slices.pt', map_location=DEVICE,
                           weights_only=False)
        cfg = ckpt['config']
        model = UNet(in_channels=cfg['in_channels'], base_filters=cfg['base_filters'])
        model.load_state_dict(ckpt['model_state'])
        model.to(DEVICE)
        
        visualize_predictions(model, samples_data, device=DEVICE)
    
    else:
        print(f"Unknown mode: {mode}")
        print("Use: label, train, or predict")
