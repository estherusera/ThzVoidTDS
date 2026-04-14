"""
THz TDS Imaging: C-Scan & B-Scan Visualization for 3D-Printed PLA Void Detection
==================================================================================
Each dataset in the .tprj file is a full 2D raster scan (ImageAcquisitionParams).
The data shape (N_time, N_spatial) contains a flattened Ny × Nx spatial grid.

Samples:
  - Sample 1:       axis1/2: -10 to 10 mm, 0.3 mm spacing
  - Sample 1_run2:  repeat of Sample 1
  - Sample 3:       "microvoid" sample — key void detection target
  - Sample int:     intentional void sample
  - Sample int(4):  intentional void, higher resolution (134 lines)
"""

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.signal import hilbert
import json
import warnings
warnings.filterwarnings('ignore')

# --- Configuration ---
phys_file = '3D_print_Dicky.tprj'
n_pla = 1.57
c_vacuum = 0.29979  # mm/ps

# --- Sample definitions (extracted from HDF5 attributes) ---
SAMPLES = {
    'Sample 1': {
        'data_path': 'TerapulseDocument/DataViews/Sample 1/PC_1/Sample 1_C/raw data/sample/data',
        'line_path': 'TerapulseDocument/DataViews/Sample 1/PC_1/Sample 1_C/raw data/sample/line',
        'sample_path': 'TerapulseDocument/DataViews/Sample 1/PC_1/Sample 1_C/raw data/sample',
        'axis1_min': -10.0, 'axis1_max': 10.0, 'axis1_spacing': 0.3,
        'axis2_min': -10.0, 'axis2_max': 10.0, 'axis2_spacing': 0.3,
    },
    'Sample 1_run2': {
        'data_path': 'TerapulseDocument/DataViews/Sample 1_run2/PC_1/Sample 1_run2_C/raw data/sample/data',
        'line_path': 'TerapulseDocument/DataViews/Sample 1_run2/PC_1/Sample 1_run2_C/raw data/sample/line',
        'sample_path': 'TerapulseDocument/DataViews/Sample 1_run2/PC_1/Sample 1_run2_C/raw data/sample',
        'axis1_min': -10.0, 'axis1_max': 10.0, 'axis1_spacing': 0.3,
        'axis2_min': -10.0, 'axis2_max': 10.0, 'axis2_spacing': 0.3,
    },
    'Sample 3 (microvoid)': {
        'data_path': 'TerapulseDocument/DataViews/Sample 3/PC_1/Sample 3_C/raw data/sample/data',
        'line_path': 'TerapulseDocument/DataViews/Sample 3/PC_1/Sample 3_C/raw data/sample/line',
        'sample_path': 'TerapulseDocument/DataViews/Sample 3/PC_1/Sample 3_C/raw data/sample',
        'axis1_min': -13.5, 'axis1_max': 6.5, 'axis1_spacing': 0.3,
        'axis2_min': -13.5, 'axis2_max': 6.5, 'axis2_spacing': 0.3,
    },
    'Sample int': {
        'data_path': 'TerapulseDocument/DataViews/Sample int/PC_1/Sample int_C/raw data/sample/data',
        'line_path': 'TerapulseDocument/DataViews/Sample int/PC_1/Sample int_C/raw data/sample/line',
        'sample_path': 'TerapulseDocument/DataViews/Sample int/PC_1/Sample int_C/raw data/sample',
        'axis1_min': -10.0, 'axis1_max': 10.0, 'axis1_spacing': 0.3,  # placeholder
        'axis2_min': -10.0, 'axis2_max': 10.0, 'axis2_spacing': 0.3,
    },
    'Sample int(4)': {
        'data_path': 'TerapulseDocument/DataViews/Sample int(4)/PC_1/Sample int(4)_C/raw data/sample/data',
        'line_path': 'TerapulseDocument/DataViews/Sample int(4)/PC_1/Sample int(4)_C/raw data/sample/line',
        'sample_path': 'TerapulseDocument/DataViews/Sample int(4)/PC_1/Sample int(4)_C/raw data/sample',
        'axis1_min': -10.0, 'axis1_max': 10.0, 'axis1_spacing': 0.3,  # placeholder
        'axis2_min': -10.0, 'axis2_max': 10.0, 'axis2_spacing': 0.3,
    },
}


def load_and_reshape(f, sample_info):
    """Load raw data and reshape from (N_time, N_flat) to (N_time, Ny, Nx)."""
    raw = f[sample_info['data_path']][:]
    line_data = np.array(f[sample_info['line_path']][:]).flatten()
    
    N_time, N_flat = raw.shape
    Ny = len(line_data)  # number of raster lines
    Nx = N_flat // Ny     # positions per line
    remainder = N_flat - Ny * Nx
    
    # Try to extract actual scan params from attributes if available
    sample_grp = f[sample_info['sample_path']]
    attrs = dict(sample_grp.attrs)
    
    # Get time axis from X_Offset and X_Spacing
    wfm_len = 2048
    x_offset = None
    x_spacing = None
    if 'X_Offset' in attrs:
        x_offset = float(np.array(attrs['X_Offset']).flatten()[0])
    if 'X_Spacing' in attrs:
        x_spacing = float(np.array(attrs['X_Spacing']).flatten()[0])
    if 'WfmLength' in attrs:
        wfm_len = int(np.array(attrs['WfmLength']).flatten()[0])
    
    # Try to get actual scan settings from UserScanSettings
    try:
        settings_str = str(np.array(attrs['UserScanSettings']).flatten()[0])
        settings = json.loads(settings_str)
        sc = settings['MeasurementConfig']['Token']['scanner_config']
        sample_info.update({
            'axis1_min': sc['axis1_min'], 'axis1_max': sc['axis1_max'],
            'axis1_spacing': sc['axis1_spacing'],
            'axis2_min': sc['axis2_min'], 'axis2_max': sc['axis2_max'],
            'axis2_spacing': sc['axis2_spacing'],
        })
        print(f"    Scan params from metadata: X=[{sc['axis1_min']},{sc['axis1_max']}] "
              f"Y=[{sc['axis2_min']},{sc['axis2_max']}] spacing={sc['axis1_spacing']}mm")
    except:
        print("    Using default scan params")
    
    # Build time axis
    if x_offset is not None and x_spacing is not None:
        time_axis = x_offset + np.arange(N_time) * x_spacing
    else:
        # Fallback: use line_data range
        t_start, t_end = float(line_data[0]), float(line_data[-1])
        time_axis = np.linspace(t_start, t_end, N_time)
    
    # Trim flat data to exact Ny * Nx
    usable = Ny * Nx
    data_trimmed = raw[:, :usable]
    
    # Reshape: (N_time, Ny*Nx) -> (N_time, Ny, Nx)
    # TeraView raster: axis1 (X) is fast axis, axis2 (Y) is slow axis
    # Data is stored as [line0_x0, line0_x1, ..., line0_xN, line1_x0, ...]
    volume = data_trimmed.reshape(N_time, Ny, Nx)
    
    # Build spatial axes
    x_axis = np.linspace(sample_info['axis1_min'], sample_info['axis1_max'], Nx)
    y_axis = np.linspace(sample_info['axis2_min'], sample_info['axis2_max'], Ny)
    
    print(f"    Shape: ({N_time}, {N_flat}) → volume ({N_time}, {Ny}, {Nx})")
    print(f"    Trimmed {remainder} extra columns")
    print(f"    Time: {time_axis[0]:.2f} to {time_axis[-1]:.2f} ps ({N_time} samples)")
    print(f"    X: {x_axis[0]:.1f} to {x_axis[-1]:.1f} mm ({Nx} pts)")
    print(f"    Y: {y_axis[0]:.1f} to {y_axis[-1]:.1f} mm ({Ny} pts)")
    
    return volume, time_axis, x_axis, y_axis


def compute_envelope(volume):
    """Hilbert envelope along time axis."""
    return np.abs(hilbert(volume, axis=0))


def find_surface(envelope):
    """Find surface reflection index from mean A-scan."""
    mean_ascan = np.mean(envelope, axis=(1, 2))
    return np.argmax(mean_ascan)


def plot_sample_full(name, volume_env, time_axis, x_axis, y_axis, save_prefix):
    """Generate comprehensive C-scan and B-scan views for one sample."""
    N_time, Ny, Nx = volume_env.shape
    
    surface_idx = find_surface(volume_env)
    surface_time = time_axis[surface_idx]
    
    # Subsurface: start 15% below surface, stop before back-surface
    margin_top = int(0.15 * N_time)
    margin_bot = int(0.05 * N_time)
    sub_start = min(surface_idx + margin_top, N_time - margin_bot)
    sub_end = max(N_time - margin_bot, sub_start + 10)
    
    print(f"    Surface at t={surface_time:.2f} ps (idx {surface_idx}/{N_time})")
    print(f"    Subsurface window: idx {sub_start}–{sub_end}")
    
    # --- C-Scan views ---
    # Max projection (full depth)
    c_max_full = np.max(volume_env, axis=0)
    # Max projection (subsurface only — void signatures)
    c_max_sub = np.max(volume_env[sub_start:sub_end, :, :], axis=0)
    # Time-of-flight of max reflection (depth map)
    tof_idx = np.argmax(volume_env, axis=0)
    c_tof = time_axis[tof_idx]
    # Depth in mm
    c_depth_mm = ((c_tof - time_axis[0]) * c_vacuum) / (2 * n_pla)
    
    # --- Subsurface ToF (ignoring surface) ---
    sub_vol = volume_env[sub_start:sub_end, :, :]
    sub_tof_idx = np.argmax(sub_vol, axis=0)
    sub_tof_time = time_axis[sub_start:sub_end][sub_tof_idx]
    sub_depth_mm = ((sub_tof_time - surface_time) * c_vacuum) / (2 * n_pla)
    
    # ==========================================
    # FIGURE 1: C-Scan Views
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=100)
    fig.suptitle(f'{name} — C-Scan Views', fontsize=16, fontweight='bold')
    
    extent_xy = [x_axis[0], x_axis[-1], y_axis[-1], y_axis[0]]
    
    # C-scan: full max projection
    v = np.percentile(c_max_full, 99)
    im0 = axes[0, 0].imshow(c_max_full, aspect='equal', cmap='Reds',
                             vmin=0, vmax=v, extent=extent_xy)
    axes[0, 0].set_title('Max Projection (full depth)')
    axes[0, 0].set_xlabel('X (mm)')
    axes[0, 0].set_ylabel('Y (mm)')
    plt.colorbar(im0, ax=axes[0, 0], label='Amplitude', shrink=0.8)
    
    # C-scan: subsurface max projection (VOID DETECTION)
    v_sub = np.percentile(c_max_sub, 99)
    im1 = axes[0, 1].imshow(c_max_sub, aspect='equal', cmap='Reds',
                             vmin=0, vmax=v_sub, extent=extent_xy)
    axes[0, 1].set_title('Subsurface Max Projection (void signatures)')
    axes[0, 1].set_xlabel('X (mm)')
    axes[0, 1].set_ylabel('Y (mm)')
    plt.colorbar(im1, ax=axes[0, 1], label='Amplitude', shrink=0.8)
    
    # Depth map (full)
    im2 = axes[1, 0].imshow(c_depth_mm, aspect='equal', cmap='viridis',
                             extent=extent_xy)
    axes[1, 0].set_title('Depth of Max Reflection')
    axes[1, 0].set_xlabel('X (mm)')
    axes[1, 0].set_ylabel('Y (mm)')
    plt.colorbar(im2, ax=axes[1, 0], label='Depth (mm)', shrink=0.8)
    
    # Subsurface depth map
    im3 = axes[1, 1].imshow(sub_depth_mm, aspect='equal', cmap='viridis',
                             extent=extent_xy)
    axes[1, 1].set_title('Subsurface Depth of Max Reflection')
    axes[1, 1].set_xlabel('X (mm)')
    axes[1, 1].set_ylabel('Y (mm)')
    plt.colorbar(im3, ax=axes[1, 1], label='Depth below surface (mm)', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_cscans.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ==========================================
    # FIGURE 2: B-Scan Slices (XZ and YZ)
    # ==========================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=100)
    fig.suptitle(f'{name} — B-Scan Cross-Sections', fontsize=16, fontweight='bold')
    
    # B-scan XZ at multiple Y positions
    y_positions = [Ny // 4, Ny // 2, 3 * Ny // 4]
    
    for i, yi in enumerate(y_positions[:2]):
        bscan_xz = volume_env[:, yi, :]  # (N_time, Nx)
        v_b = np.percentile(bscan_xz, 99)
        extent_xz = [x_axis[0], x_axis[-1], time_axis[-1], time_axis[0]]
        
        ax = axes[0, i]
        im = ax.imshow(bscan_xz, aspect='auto', cmap='Reds', vmin=0, vmax=v_b,
                       extent=extent_xz, interpolation='bilinear')
        ax.set_title(f'B-Scan XZ at Y={y_axis[yi]:.1f} mm')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Time (ps)')
        plt.colorbar(im, ax=ax, label='Amplitude', shrink=0.8)
    
    # B-scan YZ at multiple X positions
    x_positions = [Nx // 4, Nx // 2, 3 * Nx // 4]
    
    for i, xi in enumerate(x_positions[:2]):
        bscan_yz = volume_env[:, :, xi]  # (N_time, Ny)
        v_b = np.percentile(bscan_yz, 99)
        extent_yz = [y_axis[0], y_axis[-1], time_axis[-1], time_axis[0]]
        
        ax = axes[1, i]
        im = ax.imshow(bscan_yz, aspect='auto', cmap='Reds', vmin=0, vmax=v_b,
                       extent=extent_yz, interpolation='bilinear')
        ax.set_title(f'B-Scan YZ at X={x_axis[xi]:.1f} mm')
        ax.set_xlabel('Y (mm)')
        ax.set_ylabel('Time (ps)')
        plt.colorbar(im, ax=ax, label='Amplitude', shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_bscans.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # ==========================================
    # FIGURE 3: Depth slices (C-scans at specific times)
    # ==========================================
    n_slices = 6
    slice_indices = np.linspace(surface_idx, N_time - margin_bot, n_slices, dtype=int)
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=100)
    fig.suptitle(f'{name} — Time/Depth Slices', fontsize=16, fontweight='bold')
    
    for i, idx in enumerate(slice_indices):
        ax = axes[i // 3, i % 3]
        slc = volume_env[idx, :, :]
        v = np.percentile(slc, 99)
        depth_mm = ((time_axis[idx] - surface_time) * c_vacuum) / (2 * n_pla)
        
        im = ax.imshow(slc, aspect='equal', cmap='Reds', vmin=0, vmax=v,
                       extent=extent_xy)
        ax.set_title(f't={time_axis[idx]:.2f} ps (d≈{depth_mm:.3f} mm)')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        plt.colorbar(im, ax=ax, shrink=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_depth_slices.png', dpi=100, bbox_inches='tight')
    plt.show()
    
    return surface_idx


# ==========================================
# MAIN: Process all samples
# ==========================================
print("=" * 80)
print("THz TDS IMAGING — LOADING AND PROCESSING ALL SAMPLES")
print("=" * 80)

with h5py.File(phys_file, 'r') as f:
    results = {}
    
    for name, info in SAMPLES.items():
        print(f"\n{'─' * 60}")
        print(f"Processing: {name}")
        print(f"{'─' * 60}")
        
        try:
            volume, time_axis, x_axis, y_axis = load_and_reshape(f, info)
            
            print("    Computing Hilbert envelope...")
            envelope = compute_envelope(volume)
            
            prefix = name.replace(' ', '_').replace('(', '').replace(')', '')
            surface_idx = plot_sample_full(name, envelope, time_axis, x_axis, y_axis, prefix)
            
            results[name] = {
                'envelope': envelope,
                'time_axis': time_axis,
                'x_axis': x_axis,
                'y_axis': y_axis,
                'surface_idx': surface_idx,
            }
            
        except Exception as e:
            print(f"    ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # ==========================================
    # COMPARISON: All samples C-scan overview
    # ==========================================
    if len(results) > 1:
        print(f"\n{'=' * 80}")
        print("COMPARISON: All Samples — Subsurface C-Scans")
        print(f"{'=' * 80}")
        
        n_samples = len(results)
        fig, axes = plt.subplots(1, n_samples, figsize=(5 * n_samples, 5), dpi=100)
        if n_samples == 1:
            axes = [axes]
        
        for i, (name, res) in enumerate(results.items()):
            env = res['envelope']
            si = res['surface_idx']
            N_time = env.shape[0]
            margin = int(0.15 * N_time)
            sub_start = min(si + margin, N_time - 10)
            
            c_sub = np.max(env[sub_start:, :, :], axis=0)
            v = np.percentile(c_sub, 99)
            
            x, y = res['x_axis'], res['y_axis']
            extent = [x[0], x[-1], y[-1], y[0]]
            
            axes[i].imshow(c_sub, aspect='equal', cmap='Reds', vmin=0, vmax=v,
                          extent=extent)
            axes[i].set_title(name, fontsize=10)
            axes[i].set_xlabel('X (mm)')
            if i == 0:
                axes[i].set_ylabel('Y (mm)')
        
        plt.suptitle('Subsurface C-Scans — All Samples', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('all_samples_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()

print("\n" + "=" * 80)
print("DONE — Check generated PNG files")
print("=" * 80)
