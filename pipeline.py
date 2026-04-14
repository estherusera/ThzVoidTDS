#!/usr/bin/env python3
"""THz-TDS Boundary Mask Generation Pipeline.

Generates 3D ground-truth boundary masks from STL design files,
spatially aligned to THz-TDS reflection scan data (.tprj files),
for training a 2.5D U-Net.

The output volumes are shape (nx, ny, nz) where each depth slice is an
X-Y image.  The ML training loop picks triplets [z-1, z, z+1] as input
and the central slice's boundary mask as label.
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.ndimage
import trimesh

# Fine Z pitch (mm) used when voxelising the STL.
# Must be fine enough to resolve the smallest void features, then
# resampled to match the THz dz via nearest-neighbour zoom.
_STL_Z_PITCH_MM = 0.01


# ---------------------------------------------------------------------------
# Data Ingestion
# ---------------------------------------------------------------------------

def load_tprj(path, sample_name=None):
    """Load THz-TDS data from a .tprj (HDF5) file.

    Parameters
    ----------
    path : str – path to .tprj file
    sample_name : str, optional – name of the sample to load.
        Searches first in DataViews (e.g. 'Test_frame'), then in
        Measurements/Image Data by SampleName attribute (e.g. '1', '2', ...).
        If None, loads the first C-scan found in DataViews.

    Returns
    -------
    volume : ndarray, shape (n_slow, n_fast, nt)
    dt_ps, dx_mm, dy_mm, origin, time_axis
    """
    with h5py.File(path, "r") as f:
        sample_group = None
        raw_data_group = None

        # ---- 1. Try DataViews first ----
        dv = f["TerapulseDocument/DataViews"]
        dv_names = list(dv.keys())

        def _load_from_dataview(sname):
            nonlocal sample_group, raw_data_group
            pc1_path = f"TerapulseDocument/DataViews/{sname}/PC_1"
            if pc1_path not in f:
                return False
            pc1 = f[pc1_path]
            for sub_name in pc1:
                if sub_name.endswith("_C"):
                    raw_data_group = pc1[sub_name]["raw data"]
                    sample_group = raw_data_group["sample"]
                    return True
            return False

        if sample_name is not None:
            if sample_name in dv_names:
                if not _load_from_dataview(sample_name):
                    raise RuntimeError(
                        f"DataView '{sample_name}' has no C-scan (PC_1/*_C).")
            else:
                # ---- 2. Try Measurements/Image Data by SampleName ----
                img_base = "TerapulseDocument/Measurements/Image Data"
                found_img = None
                if img_base in f:
                    for img_id in f[img_base]:
                        sg = f[f"{img_base}/{img_id}/sample"]
                        attr = sg.attrs.get("SampleName", None)
                        if attr is None:
                            continue
                        try:
                            name_val = str(attr[0, 0]).strip()
                        except Exception:
                            name_val = str(attr).strip()
                        if name_val == sample_name:
                            found_img = sg
                            break
                if found_img is None:
                    available_dv  = dv_names
                    available_img = []
                    if img_base in f:
                        for img_id in f[img_base]:
                            sg = f[f"{img_base}/{img_id}/sample"]
                            a = sg.attrs.get("SampleName", None)
                            if a is not None:
                                try: available_img.append(str(a[0,0]).strip())
                                except: available_img.append(str(a).strip())
                    raise RuntimeError(
                        f"Sample '{sample_name}' not found.\n"
                        f"  DataViews: {available_dv}\n"
                        f"  Image Data SampleNames: {available_img}")
                sample_group = found_img
                raw_data_group = None  # no separate ref group in Image Data
        else:
            # Auto-detect first C-scan in DataViews
            for sname in dv_names:
                if _load_from_dataview(sname):
                    break

        if sample_group is None:
            raise RuntimeError(f"No C-scan found in {path}")

        data = sample_group["data"][:]
        line = sample_group["line"][:].ravel()

        # Time axis: prefer Current Reference in raw_data_group; fall back to
        # X_Offset / X_Spacing attributes on the sample group itself.
        time_axis = None
        if raw_data_group is not None:
            for ref_key in ("Current Baseline", "Current Reference"):
                if ref_key in raw_data_group:
                    time_axis = raw_data_group[ref_key]["sample"]["xdata"][:].ravel()
                    break
        if time_axis is None:
            # Measurements/Image Data path – reconstruct from attrs
            x_off = float(sample_group.attrs["X_Offset"][0, 0])
            x_spc = float(sample_group.attrs["X_Spacing"][0, 0])
            wfm_len = int(sample_group.attrs["WfmLength"][0, 0])
            time_axis = x_off + x_spc * np.arange(wfm_len)

        nt = len(time_axis)

        settings_json = sample_group.attrs["UserScanSettings"][0, 0]
        scan_cfg = json.loads(settings_json)[
            "MeasurementConfig"]["Token"]["scanner_config"]
        dx_mm = float(scan_cfg["axis1_spacing"])
        dy_mm = float(scan_cfg["axis2_spacing"])
        x_min = float(scan_cfg["axis1_min"])
        y_min = float(scan_cfg["axis2_min"])
        x_max = float(scan_cfg["axis1_max"])
        y_max = float(scan_cfg["axis2_max"])
        dt_ps = float(time_axis[1] - time_axis[0])

        n_slow = len(line)
        n_fast = int(np.max(line))

        # Sanity-check: infer spacing from axis range / pixel count and
        # override the metadata value if they disagree by more than 10 %.
        if n_fast > 1:
            inferred_dx = (x_max - x_min) / (n_fast - 1)
            if abs(inferred_dx - dx_mm) / max(dx_mm, 1e-6) > 0.10:
                print(f"  WARNING: axis1_spacing in metadata ({dx_mm} mm) "
                      f"disagrees with pixel count ({inferred_dx:.4f} mm). "
                      f"Using inferred value.")
                dx_mm = inferred_dx
        if n_slow > 1:
            inferred_dy = (y_max - y_min) / (n_slow - 1)
            if abs(inferred_dy - dy_mm) / max(dy_mm, 1e-6) > 0.10:
                print(f"  WARNING: axis2_spacing in metadata ({dy_mm} mm) "
                      f"disagrees with pixel count ({inferred_dy:.4f} mm). "
                      f"Using inferred value.")
                dy_mm = inferred_dy

        volume = np.zeros((n_slow, n_fast, nt), dtype=data.dtype)
        col = 0
        for i in range(n_slow):
            n = int(line[i])
            volume[i, :n, :] = data[:nt, col : col + n].T
            col += n

    origin = (x_min, y_min)

    print(f"  Loaded {path}  (sample={sample_name!r})")
    print(f"    volume shape : {volume.shape}  "
          f"(n_slow={n_slow}, n_fast={n_fast}, nt={nt})")
    print(f"    dt={dt_ps:.4f} ps, dx={dx_mm} mm (fast), dy={dy_mm} mm (slow)")
    print(f"    origin: x={x_min} mm, y={y_min} mm")

    return volume, dt_ps, dx_mm, dy_mm, origin, time_axis


# ---------------------------------------------------------------------------
# Sample Footprint Detection & Crop
# ---------------------------------------------------------------------------

def detect_sample_region(flat, dx_mm, dy_mm,
                          slow_mm=20.0, fast_mm=20.0):
    """Detect the sample's XY footprint from the surface-echo amplitude map
    and return pixel crop indices (s0, s1, f0, f1) that isolate the
    slow_mm × fast_mm region.

    The surface echo (first ~20 samples after flattening) is strong where
    the PLA sample is and near-zero over bare air, giving a clean amplitude
    contrast for localisation.
    """
    # Surface echo amplitude map
    surf_amp = np.max(np.abs(flat[:, :, :20]), axis=2).astype(np.float32)

    # Smooth to suppress per-pixel noise
    smoothed = scipy.ndimage.gaussian_filter(surf_amp, sigma=2.0)

    # Binary: top-50 % amplitude pixels belong to the sample
    thresh   = np.percentile(smoothed, 50)
    binary   = smoothed > thresh

    # Largest connected component → sample centroid
    labeled, n_comp = scipy.ndimage.label(binary)
    if n_comp == 0:
        cy, cx = flat.shape[0] // 2, flat.shape[1] // 2
        print("  Sample detection: no region found, using scan centre")
    else:
        sizes = [(labeled == i).sum() for i in range(1, n_comp + 1)]
        best  = int(np.argmax(sizes)) + 1
        c     = scipy.ndimage.center_of_mass(labeled == best)
        cy, cx = int(round(c[0])), int(round(c[1]))

    print(f"  Sample centre: pixel ({cy}, {cx}) = "
          f"({cy * dy_mm:.1f} mm slow, {cx * dx_mm:.1f} mm fast)")

    # Crop size in pixels
    ns = int(round(slow_mm / dy_mm))
    nf = int(round(fast_mm / dx_mm))

    # Centre the crop window on the detected centroid, clamped to array
    s0 = max(0, cy - ns // 2)
    s1 = min(flat.shape[0], s0 + ns)
    s0 = max(0, s1 - ns)   # re-clamp after upper bound adjustment

    f0 = max(0, cx - nf // 2)
    f1 = min(flat.shape[1], f0 + nf)
    f0 = max(0, f1 - nf)

    print(f"  Crop region  : slow [{s0}:{s1}] ({(s1-s0)*dy_mm:.1f} mm)  "
          f"fast [{f0}:{f1}] ({(f1-f0)*dx_mm:.1f} mm)")
    return s0, s1, f0, f1


# ---------------------------------------------------------------------------
# Surface Alignment
# ---------------------------------------------------------------------------

def find_top_surface(volume, signal_percentile=10):
    """Per-pixel index of the peak reflected amplitude.

    Pixels whose peak amplitude is below *signal_percentile* percent of the
    global max are treated as air / no-signal and get t0 = 0 so they don't
    inflate the crop window.

    Returns
    -------
    t0        : ndarray (nx, ny) – surface sample index
    sig_mask  : ndarray (nx, ny) bool – True where there is real signal
    """
    amp = np.max(np.abs(volume), axis=2)
    sig_mask = amp > (np.percentile(amp[amp > 0], signal_percentile)
                      if np.any(amp > 0) else 0)

    t0 = np.argmax(np.abs(volume), axis=2)
    t0[~sig_mask] = 0  # air pixels don't push the crop boundary
    return t0, sig_mask


def flatten_surface(volume, t0):
    """Roll each waveform so the surface reflection sits at index 0.

    Returns
    -------
    flat : ndarray (nx, ny, nt_crop)
    """
    nx, ny, nt = volume.shape

    t_idx = (np.arange(nt)[None, None, :] + t0[:, :, None]) % nt
    x_idx = np.arange(nx)[:, None, None]
    y_idx = np.arange(ny)[None, :, None]
    flat = volume[x_idx, y_idx, t_idx]

    # Crop using the max t0 of *signal* pixels only
    max_t0 = int(t0.max())
    nt_crop = nt - max_t0
    flat = flat[:, :, :nt_crop]

    print(f"  Surface flattened: {volume.shape} -> {flat.shape}")
    print(f"    max t0 (signal pixels) = {max_t0}, depth samples kept = {nt_crop}")
    return flat


# ---------------------------------------------------------------------------
# Z-Axis Conversion
# ---------------------------------------------------------------------------

def compute_z_resolution(dt_ps, n_pla):
    """Depth per time sample (mm) in reflection geometry."""
    c = 3e8  # m/s
    dz_mm = (c * dt_ps * 1e-12) / (2.0 * n_pla) * 1e3
    print(f"  dz = {dz_mm:.5f} mm/sample  (n={n_pla})")
    return dz_mm


def detect_sample_thickness(flat, dz_mm, sig_mask):
    """Detect actual sample thickness from the bottom-surface echo.

    After surface flattening, the strongest subsurface peak in each A-scan
    is the bottom-surface reflection.  We take the median of those peaks
    across all signal pixels to robustly estimate the sample thickness.

    The bottom-surface echo must be:
    - At least 0.3mm deep (to avoid surface ringing)
    - At least 10% of the surface peak amplitude
    - Consistent across many A-scans (low std)

    Returns
    -------
    thickness_mm : float or None – measured thickness, None if not detected
    """
    from scipy.signal import find_peaks

    nx, ny, nz = flat.shape
    # Skip the first ~0.3mm to avoid surface ringing
    skip = max(50, int(0.3 / dz_mm))

    # Surface amplitude per pixel (sample 0 after flattening)
    surf_amp = np.abs(flat[:, :, 0])

    bottom_depths = []
    # Subsample for speed
    step = max(1, min(nx, ny) // 30)
    for i in range(0, nx, step):
        for j in range(0, ny, step):
            if not sig_mask[i, j]:
                continue
            sa = surf_amp[i, j]
            if sa < 0.01:
                continue
            ascan = np.abs(flat[i, j, skip:])
            # Peak must be at least 10% of the SURFACE peak (not local max)
            peaks, props = find_peaks(
                ascan, height=sa * 0.10, distance=50)
            if len(peaks) > 0:
                best = peaks[np.argmax(props["peak_heights"])]
                bottom_depths.append((best + skip) * dz_mm)

    if len(bottom_depths) < 10:
        print("  No bottom-surface echo detected "
              "(sample may be thicker than THz penetration)")
        return None

    thickness = float(np.median(bottom_depths))
    std = float(np.std(bottom_depths))

    # Reject if too noisy (std > 20% of thickness) — likely picking up noise
    if std > thickness * 0.2:
        print(f"  Bottom-surface echo inconsistent "
              f"(thickness={thickness:.3f}mm, std={std:.3f}mm) — skipping")
        return None

    print(f"  Detected sample thickness: {thickness:.3f} mm "
          f"(std={std:.3f}, n={len(bottom_depths)} A-scans)")
    return thickness


# ---------------------------------------------------------------------------
# STL Voxelization
# ---------------------------------------------------------------------------

def _detect_internal_voids(solid):
    """Detect internal voids using flood-fill from the exterior.

    Flood-fills from the boundary faces of the volume to identify all
    exterior air voxels.  Anything that is not solid AND not reachable
    from the exterior is an internal void.

    Parameters
    ----------
    solid : ndarray bool (Vx, Vy, Vz) – True where material exists

    Returns
    -------
    voids : ndarray bool (Vx, Vy, Vz) – True at internal void voxels
    """
    empty = ~solid

    # Seed: all empty voxels touching any face of the bounding box
    seed = np.zeros_like(empty)
    seed[0, :, :] = empty[0, :, :]
    seed[-1, :, :] = empty[-1, :, :]
    seed[:, 0, :] = empty[:, 0, :]
    seed[:, -1, :] = empty[:, -1, :]
    seed[:, :, 0] = empty[:, :, 0]
    seed[:, :, -1] = empty[:, :, -1]

    # Binary dilation of seed, constrained to empty voxels = flood fill
    exterior = scipy.ndimage.binary_dilation(
        seed, iterations=0, mask=empty
    )

    voids = empty & ~exterior
    return voids


def voxelize_stl(stl_path, dx, dy, dz, max_depth_mm, z_scale=1.0):
    """Voxelize an STL mesh and extract internal voids.

    Samples at (dx, dy) in X-Y and a fine pitch (_STL_Z_PITCH_MM) in Z,
    detects internal voids via flood-fill on the full mesh (before
    trimming), then resamples Z to match the THz *dz* via nearest-
    neighbour zoom.

    Parameters
    ----------
    z_scale : float – scale factor applied to the STL Z coordinates
        before voxelization.  If the STL says 2mm but the actual sample
        is 0.89mm, use z_scale = 0.89/2.0 = 0.445.

    Returns
    -------
    voxels     : ndarray bool (Vx, Vy, Vz_thz) – solid occupancy
    voids      : ndarray bool (Vx, Vy, Vz_thz) – internal void regions
    bounds_min : ndarray (3,)
    """
    mesh = trimesh.load(stl_path)
    if not mesh.is_watertight:
        print("  WARNING: mesh is not watertight – contains() may be unreliable")

    if z_scale != 1.0:
        # Scale mesh Z coordinates around Z-min (top surface stays at same Z)
        z_min = mesh.bounds[0][2]
        verts = mesh.vertices.copy()
        verts[:, 2] = z_min + (verts[:, 2] - z_min) * z_scale
        mesh = trimesh.Trimesh(vertices=verts, faces=mesh.faces)
        print(f"  Z-scale applied: {z_scale:.4f} "
              f"(STL Z range rescaled to {mesh.extents[2]:.3f} mm)")

    bmin, bmax = mesh.bounds[0], mesh.bounds[1]

    coarse_dz = _STL_Z_PITCH_MM
    xs = np.arange(bmin[0] + dx / 2, bmax[0], dx)   # STL X axis
    ys = np.arange(bmin[1] + dy / 2, bmax[1], dy)    # STL Y axis
    zs = np.arange(bmin[2] + coarse_dz / 2, bmax[2], coarse_dz)
    vx, vy, vz = len(xs), len(ys), len(zs)

    print(f"  Voxelising {stl_path}")
    print(f"    mesh bounds   : {bmin} -> {bmax}  (extents {mesh.extents})")
    print(f"    sample grid   : {vx} x {vy} x {vz} = {vx * vy * vz:,} points")
    print(f"    pitches       : dx={dx}, dy={dy}, z_pitch={coarse_dz} mm")

    grid = np.stack(
        np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1
    ).reshape(-1, 3)

    inside = mesh.contains(grid)
    mat = inside.reshape(vx, vy, vz)

    print(f"    filled        : {mat.sum():,} / {mat.size:,}")

    # --- Detect internal voids via flood-fill ---
    void_coarse = _detect_internal_voids(mat)

    print(f"    internal voids: {void_coarse.sum():,} voxels")

    # --- Flip Z so index 0 = top surface (max Z) ---
    mat = mat[:, :, ::-1]
    void_coarse = void_coarse[:, :, ::-1]

    # --- Trim Z to max_depth ---
    max_z_coarse = int(np.ceil(max_depth_mm / coarse_dz))
    end_z = min(max_z_coarse, vz)
    mat = mat[:, :, :end_z]
    void_coarse = void_coarse[:, :, :end_z]

    depth_mm = end_z * coarse_dz
    print(f"    flipped Z (top->bottom), trimmed: {end_z} slices ({depth_mm:.2f} mm)")
    print(f"    internal voids (trimmed): {void_coarse.sum():,} voxels")

    # --- Resample Z from coarse_dz to thz dz ---
    zoom_z = coarse_dz / dz

    voxels = scipy.ndimage.zoom(
        mat.astype(np.float32), (1.0, 1.0, zoom_z), order=0,
    ) > 0.5

    voids = scipy.ndimage.zoom(
        void_coarse.astype(np.float32), (1.0, 1.0, zoom_z), order=0,
    ) > 0.5

    print(f"    resampled Z   : {mat.shape[2]} -> {voxels.shape[2]} slices "
          f"(zoom={zoom_z:.1f}x, effective dz={dz:.5f} mm)")
    print(f"    void voxels (fine): {voids.sum():,}")

    return voxels, voids, bmin


# ---------------------------------------------------------------------------
# Spatial Registration
# ---------------------------------------------------------------------------

def _apply_spatial_transforms(vg, flip=None, rot90=0):
    """Apply flip and rotation to a 3D voxel grid (spatial axes 0,1 only)."""
    if flip:
        if 'x' in flip:
            vg = vg[::-1, :, :]
        if 'y' in flip:
            vg = vg[:, ::-1, :]
    if rot90:
        vg = np.rot90(vg, k=rot90, axes=(0, 1))
    return vg


def _centre_and_paste(vg, thz_shape, dx, dy, stl_offset_mm):
    """Centre a transformed voxel grid onto the THz grid shape.

    stl_offset_mm is (slow_mm, fast_mm) — slow axis first, fast axis second.
    """
    n_slow, n_fast, nz = thz_shape
    vs, vf, vz_t = vg.shape

    aligned = np.zeros((n_slow, n_fast, nz), dtype=bool)

    # offset[0] is slow-axis mm, offset[1] is fast-axis mm
    off_slow_px = round(stl_offset_mm[0] / dy)
    off_fast_px = round(stl_offset_mm[1] / dx)
    off_slow = (n_slow - vs) // 2 + off_slow_px
    off_fast = (n_fast - vf) // 2 + off_fast_px

    src_s0 = max(0, -off_slow)
    src_f0 = max(0, -off_fast)
    dst_s0 = max(0, off_slow)
    dst_f0 = max(0, off_fast)

    cs = min(vs - src_s0, n_slow - dst_s0)
    cf = min(vf - src_f0, n_fast - dst_f0)
    cz = min(vz_t, nz)

    aligned[
        dst_s0 : dst_s0 + cs,
        dst_f0 : dst_f0 + cf,
        :cz,
    ] = vg[
        src_s0 : src_s0 + cs,
        src_f0 : src_f0 + cf,
        :cz,
    ]
    return aligned, (vs, vf, vz_t), (off_slow, off_fast), (cs, cf, cz)


def align_to_thz(voxel_grid, stl_bounds_min, thz_shape,
                  dx, dy, thz_origin, stl_offset_mm=(0.0, 0.0),
                  flip=None, rot90=0):
    """Register the voxel grid to the THz scan.

    Applies optional flip / rotation, then centres on the scan footprint.
    """
    vg = _apply_spatial_transforms(voxel_grid.copy(), flip=flip, rot90=rot90)
    aligned, (vs, vf, vz_t), (off_s, off_f), (cs, cf, cz) = \
        _centre_and_paste(vg, thz_shape, dx, dy, stl_offset_mm)

    if flip or rot90:
        print(f"    transforms: flip={flip}, rot90={rot90}")
    print(f"  Aligned voxels (STL) {voxel_grid.shape} -> (THz) {aligned.shape}")
    print(f"    after transform: ({vs}, {vf}, {vz_t})")
    print(f"    centred: slow offset={off_s}, fast offset={off_f}")
    print(f"    copied region: {cs} (slow) x {cf} (fast) x {cz} (depth) voxels")
    return aligned


def align_voids_to_thz(void_grid, thz_shape, dx, dy,
                        stl_offset_mm=(0.0, 0.0), flip=None, rot90=0):
    """Align void grid to THz grid using same transforms as align_to_thz."""
    vg = _apply_spatial_transforms(void_grid.copy(), flip=flip, rot90=rot90)
    aligned, _, _, _ = _centre_and_paste(vg, thz_shape, dx, dy, stl_offset_mm)
    return aligned


# ---------------------------------------------------------------------------
# Auto-Alignment
# ---------------------------------------------------------------------------

_TRANSFORMS = [
    {"flip": None,  "rot90": 0, "name": "identity"},
    {"flip": "x",   "rot90": 0, "name": "flip_slow"},
    {"flip": "y",   "rot90": 0, "name": "flip_fast"},
    {"flip": "xy",  "rot90": 0, "name": "flip_both"},
    {"flip": None,  "rot90": 1, "name": "rot90_CCW"},
    {"flip": None,  "rot90": 3, "name": "rot90_CW"},
    {"flip": None,  "rot90": 2, "name": "rot180"},
    {"flip": "y",   "rot90": 1, "name": "rot90_CCW+flip_fast"},
]


def _depth_gated_cscan(flat, dz_mm, z_start_mm, z_end_mm):
    """Mean absolute amplitude in a depth gate."""
    i0 = max(0, int(round(z_start_mm / dz_mm)))
    i1 = min(flat.shape[2], int(round(z_end_mm / dz_mm)))
    if i1 <= i0:
        return np.zeros(flat.shape[:2])
    return np.mean(np.abs(flat[:, :, i0:i1]), axis=2)


def auto_align(voids_stl, flat, dx, dy, dz_mm, thz_shape,
               echo_shift_samples=0):
    """Score all 8 flip/rotation transforms and return the best one.

    For each transform, aligns the void grid to the THz grid and computes
    a correlation score: mean C-scan amplitude inside void regions minus
    mean amplitude outside.  Void boundaries create reflections, so
    higher amplitude inside the void footprint indicates better alignment.

    The scoring uses multiple depth gates to be robust across different
    void depths.
    """
    max_depth = flat.shape[2] * dz_mm
    gates = []
    for z0 in np.arange(0.0, min(max_depth, 5.0), 0.25):
        gates.append((z0, z0 + 0.25))

    best_score = -np.inf
    best_tf = _TRANSFORMS[0]

    print(f"  Testing {len(_TRANSFORMS)} transforms across {len(gates)} depth gates...")

    for tf in _TRANSFORMS:
        vg = _apply_spatial_transforms(voids_stl.copy(),
                                       flip=tf["flip"], rot90=tf["rot90"])
        aligned_v, _, _, _ = _centre_and_paste(
            vg, thz_shape, dx, dy, (0.0, 0.0))

        # Apply echo depth shift for scoring
        if echo_shift_samples > 0:
            nz_a = aligned_v.shape[2]
            if echo_shift_samples < nz_a:
                shifted = np.zeros_like(aligned_v)
                shifted[:, :, echo_shift_samples:] = aligned_v[:, :, :nz_a - echo_shift_samples]
                aligned_v = shifted

        total_score = 0.0
        for z0, z1 in gates:
            cscan = _depth_gated_cscan(flat, dz_mm, z0, z1)
            iz0 = max(0, int(round(z0 / dz_mm)))
            iz1 = min(aligned_v.shape[2], int(round(z1 / dz_mm)))
            if iz1 <= iz0:
                continue
            void_xy = np.any(aligned_v[:, :, iz0:iz1], axis=2)
            n_void = void_xy.sum()
            if n_void < 10:
                continue
            non_void = ~void_xy & (cscan > 0)
            if non_void.sum() < 10:
                continue
            # Voids create reflections -> higher amplitude at void boundaries
            score = cscan[void_xy].mean() - cscan[non_void].mean()
            total_score += score

        if total_score > best_score:
            best_score = total_score
            best_tf = tf

        print(f"    {tf['name']:25s} score={total_score:+.4f}")

    print(f"  Best transform: {best_tf['name']} (score={best_score:+.4f})")
    return best_tf["flip"], best_tf["rot90"]


def auto_z_shift(aligned_voids, flat, dz_mm, measured_thickness):
    """Find the optimal Z-shift by correlating void XY footprint with C-scan.

    Scans candidate shifts from 0 to 5×thickness in small steps and scores
    each by how well the shifted void footprint matches C-scan amplitude.
    This is more robust than a fixed echo-multiplier formula.

    Returns
    -------
    best_shift_samples : int – optimal Z-shift in sample indices
    """
    nz = aligned_voids.shape[2]
    void_xy = np.any(aligned_voids, axis=2)
    n_void = void_xy.sum()
    if n_void < 10:
        print("  Auto Z-shift: too few void voxels, skipping")
        return 0

    # Depth range of voids in the un-shifted mask
    void_z = np.any(aligned_voids, axis=(0, 1))
    z_indices = np.where(void_z)[0]
    void_span = (z_indices[-1] - z_indices[0]) * dz_mm

    # Test shifts from 1×thickness to 5×thickness.
    # We skip 0-1× because the bottom-surface echo (at ~thickness) is bright
    # everywhere and would give false-positive scores.
    min_shift_mm = measured_thickness
    max_shift_mm = 5.0 * measured_thickness
    step_mm = max(0.02, measured_thickness * 0.05)
    best_score = -np.inf
    best_shift = 0

    for shift_mm in np.arange(min_shift_mm, max_shift_mm, step_mm):
        shift_samp = int(round(shift_mm / dz_mm))
        if shift_samp + z_indices[0] >= nz:
            break

        # Compute depth range of shifted voids
        z_start = (z_indices[0] + shift_samp) * dz_mm
        z_end = min((z_indices[-1] + shift_samp) * dz_mm, nz * dz_mm)
        cscan = _depth_gated_cscan(flat, dz_mm, z_start, z_end)

        # Score: mean amplitude inside void footprint vs outside
        non_void = ~void_xy & (cscan > 0)
        if non_void.sum() < 10:
            continue
        score = cscan[void_xy].mean() - cscan[non_void].mean()
        if score > best_score:
            best_score = score
            best_shift = shift_samp

    best_mm = best_shift * dz_mm
    print(f"  Auto Z-shift: {best_mm:.2f} mm ({best_shift} samples), "
          f"score={best_score:+.4f}")
    return best_shift


# ---------------------------------------------------------------------------
# Boundary Extraction
# ---------------------------------------------------------------------------

def extract_boundaries(voids):
    """Convert solid void regions into hollow boundary shells.

    Uses a 3D morphological gradient: dilation XOR erosion with a 3x3x3
    structuring element.  The result contains only the interface voxels
    (top, bottom, and side walls of each void).

    This matches THz reflection physics: only refractive-index boundaries
    produce reflected signal, not the empty volume inside voids.
    """
    struct = scipy.ndimage.generate_binary_structure(3, 1)
    dilated = scipy.ndimage.binary_dilation(voids, structure=struct)
    eroded = scipy.ndimage.binary_erosion(voids, structure=struct)
    boundaries = dilated ^ eroded

    print(f"  Boundary extraction: {voids.sum():,} void voxels -> "
          f"{boundaries.sum():,} boundary voxels")
    return boundaries


# ---------------------------------------------------------------------------
# PSF Blur + Boundary Extraction
# ---------------------------------------------------------------------------

def apply_psf_blur(voids, dx, dy, dz, sigma_xy_mm, sigma_z_mm, threshold):
    """Blur the solid void mask with a Gaussian PSF and binarise.

    The blurred mask directly represents where the THz beam detects
    void-related signal — larger voids produce bigger blobs, small voids
    produce smaller ones, matching the physical response.

    Parameters
    ----------
    sigma_xy_mm : float – lateral (XY) blur sigma in mm
    sigma_z_mm  : float – axial (depth) blur sigma in mm
    threshold   : float – binarisation threshold (relative to peak)
    """
    if sigma_xy_mm == 0 and sigma_z_mm == 0:
        mask = voids.astype(np.uint8)
        print(f"  PSF sigma: 0 (no blur)")
        print(f"  Mask voxels: {mask.sum():,} / {mask.size:,}")
        return mask

    sigma_vox = [
        sigma_xy_mm / dy,   # slow axis
        sigma_xy_mm / dx,   # fast axis
        sigma_z_mm / dz if sigma_z_mm > 0 else 0.0,  # depth axis
    ]

    blurred = scipy.ndimage.gaussian_filter(
        voids.astype(np.float32), sigma=sigma_vox
    )

    peak = blurred.max()
    if peak > 0:
        blurred /= peak

    mask = (blurred > threshold).astype(np.uint8)

    print(f"  PSF sigma (mm) : XY={sigma_xy_mm}, Z={sigma_z_mm}")
    print(f"  PSF sigma (vox): [{sigma_vox[0]:.1f}, {sigma_vox[1]:.1f}, {sigma_vox[2]:.1f}]")
    print(f"  Mask voxels: {mask.sum():,} / {mask.size:,}")
    return mask


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export(thz_flat, mask, output_dir, dz_mm):
    """Save the flattened THz volume, boundary mask, and metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    assert thz_flat.shape == mask.shape, (
        f"Shape mismatch: thz {thz_flat.shape} vs mask {mask.shape}"
    )

    vol_path = output_dir / "thz_volume.npy"
    mask_path = output_dir / "boundary_mask.npy"
    meta_path = output_dir / "metadata.json"

    np.save(vol_path, thz_flat)
    np.save(mask_path, mask)

    metadata = {
        "dz_mm": dz_mm,
        "shape": list(thz_flat.shape),
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    nz = thz_flat.shape[2]
    print(f"\n  Saved:")
    print(f"    {vol_path}  shape={thz_flat.shape}  "
          f"min={thz_flat.min():.4f}  max={thz_flat.max():.4f}")
    print(f"    {mask_path}  shape={mask.shape}  "
          f"sum={mask.sum():,}  dtype={mask.dtype}")
    print(f"    {meta_path}")
    print(f"\n  Depth slices: {nz}  (each training sample = central + 2 neighbours)")
    void_slices = np.any(mask, axis=(0, 1))
    print(f"  Slices with void boundaries : {void_slices.sum()}")
    print(f"  Slices without (solid PLA)  : {nz - int(void_slices.sum())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="THz-TDS Boundary Mask Generation Pipeline"
    )
    parser.add_argument("--tprj", required=True, help="Path to .tprj file")
    parser.add_argument("--stl", required=True, help="Path to .stl file")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument(
        "--sample-name", default=None,
        help="Name of sample view inside .tprj (e.g. 'Sample int(4)')")
    parser.add_argument(
        "--n-pla", type=float, default=1.5,
        help="Refractive index of PLA (default: 1.5)")
    parser.add_argument(
        "--sigma-mm", type=float, default=1.5,
        help="PSF lateral (XY) Gaussian sigma in mm (default: 1.5)")
    parser.add_argument(
        "--sigma-z-mm", type=float, default=None,
        help="PSF axial (Z) Gaussian sigma in mm. Default: 0.2mm.")
    parser.add_argument(
        "--threshold", type=float, default=0.1,
        help="Binarisation threshold after PSF blur (default: 0.1)")
    parser.add_argument(
        "--stl-offset", type=float, nargs=2, default=[0.0, 0.0],
        metavar=("SLOW", "FAST"),
        help="Manual STL offset in mm (slow fast) applied after centring")
    parser.add_argument(
        "--stl-flip", default=None, choices=["x", "y", "xy"],
        help="Flip STL axes before alignment (e.g. 'y' or 'xy')")
    parser.add_argument(
        "--stl-rot90", type=int, default=0,
        help="Rotate STL mask by N*90 degrees anticlockwise (1=90, 2=180, 3=270)")
    parser.add_argument(
        "--auto-align", action="store_true",
        help="Automatically select best flip/rotation by correlating with C-scan")
    parser.add_argument(
        "--sample-thickness", type=float, default=None,
        help="Override measured sample thickness in mm (used to rescale STL Z-axis)")
    parser.add_argument(
        "--crop-to-sample", action="store_true",
        help="Auto-detect sample footprint and crop the THz volume to it")
    parser.add_argument(
        "--sample-footprint", type=float, nargs=2, default=[20.0, 20.0],
        metavar=("SLOW_MM", "FAST_MM"),
        help="Sample size in mm for cropping (default: 20 20)")
    args = parser.parse_args()

    print("=" * 60)
    print("THz-TDS Boundary Mask Generation Pipeline")
    print("=" * 60)

    # 1. Load THz data
    print("\n[1/7] Loading THz data...")
    volume, dt_ps, dx_mm, dy_mm, thz_origin, _ = load_tprj(
        args.tprj, args.sample_name)

    # 2. Surface alignment
    print("\n[2/7] Aligning to top surface...")
    t0, sig_mask = find_top_surface(volume)
    flat = flatten_surface(volume, t0)

    # 2b. Crop to sample footprint (optional)
    if args.crop_to_sample:
        print("\n[2b] Cropping to sample footprint "
              f"({args.sample_footprint[0]}×{args.sample_footprint[1]} mm)...")
        s0, s1, f0, f1 = detect_sample_region(
            flat, dx_mm, dy_mm,
            slow_mm=args.sample_footprint[0],
            fast_mm=args.sample_footprint[1])
        flat     = flat[s0:s1, f0:f1, :]
        sig_mask = sig_mask[s0:s1, f0:f1]
        print(f"  Volume after crop: {flat.shape}")

    # 3. Z-axis resolution and sample thickness
    print("\n[3/7] Computing Z resolution...")
    dz_mm = compute_z_resolution(dt_ps, args.n_pla)
    max_depth_mm = flat.shape[2] * dz_mm
    print(f"  THz penetration depth: {max_depth_mm:.2f} mm "
          f"({flat.shape[2]} samples)")

    measured_thickness = (args.sample_thickness
                          or detect_sample_thickness(flat, dz_mm, sig_mask))

    # 4. Voxelise STL (includes void detection via flood-fill)
    print("\n[4/7] Voxelising STL...")

    # Compute Z-scale: compare STL thickness to measured thickness
    stl_mesh = trimesh.load(args.stl)
    stl_thickness = stl_mesh.extents[2]

    z_scale = 1.0
    if measured_thickness is not None and abs(stl_thickness - measured_thickness) > 0.05:
        z_scale = measured_thickness / stl_thickness
        print(f"  STL thickness: {stl_thickness:.2f} mm, "
              f"measured: {measured_thickness:.2f} mm -> z_scale={z_scale:.4f}")
    else:
        print(f"  STL thickness: {stl_thickness:.2f} mm "
              f"(matches measured, no Z rescale needed)")

    voxels, stl_voids, stl_origin = voxelize_stl(
        args.stl, dx_mm, dy_mm, dz_mm, max_depth_mm, z_scale=z_scale)

    # 5. Spatial alignment
    print("\n[5/7] Aligning STL to THz grid...")

    # Thin sample: measured thickness << STL thickness. The void signal
    # is buried in surface ringing and only visible in a later echo pass.
    # Shift the mask deeper by ~2.5× measured thickness.
    is_thin = (measured_thickness is not None and z_scale < 0.9)
    echo_shift_samples = 0
    if is_thin:
        echo_offset_mm = 2.5 * measured_thickness
        echo_shift_samples = int(round(echo_offset_mm / dz_mm))
        print(f"  Thin sample: echo shift = 2.5 x {measured_thickness:.2f} mm "
              f"= {echo_offset_mm:.2f} mm ({echo_shift_samples} samples)")

    if args.auto_align:
        print("  Auto-alignment enabled")
        flip, rot90 = auto_align(
            stl_voids, flat, dx_mm, dy_mm, dz_mm, flat.shape,
            echo_shift_samples=echo_shift_samples)
    else:
        flip = args.stl_flip
        rot90 = args.stl_rot90

    stl_offset = tuple(args.stl_offset)

    aligned = align_to_thz(voxels, stl_origin, flat.shape,
                            dx_mm, dy_mm, thz_origin,
                            stl_offset_mm=stl_offset,
                            flip=flip, rot90=rot90)
    aligned_voids = align_voids_to_thz(stl_voids, flat.shape,
                                         dx_mm, dy_mm,
                                         stl_offset_mm=stl_offset,
                                         flip=flip, rot90=rot90)
    print(f"  Aligned void voxels: {aligned_voids.sum():,}")

    # Apply echo depth shift
    if echo_shift_samples > 0:
        nz = aligned_voids.shape[2]
        if echo_shift_samples < nz:
            shifted = np.zeros_like(aligned_voids)
            shifted[:, :, echo_shift_samples:] = aligned_voids[:, :, :nz - echo_shift_samples]
            aligned_voids = shifted
            print(f"  Shifted void voxels: {aligned_voids.sum():,}")

    # 6. PSF blur
    print("\n[6/7] Applying PSF blur...")
    if args.sigma_z_mm is not None:
        sigma_z = args.sigma_z_mm
    elif is_thin:
        sigma_z = measured_thickness * 0.4
        print(f"  Auto sigma_z (thin): {sigma_z:.3f} mm")
    else:
        sigma_z = 0.05
        print(f"  Auto sigma_z (thick): {sigma_z:.3f} mm")
    mask = apply_psf_blur(
        aligned_voids, dx_mm, dy_mm, dz_mm,
        args.sigma_mm, sigma_z, args.threshold)

    # 7. Export
    print("\n[7/7] Exporting...")
    export(flat, mask, args.output_dir, dz_mm)

    print("\nDone.")


if __name__ == "__main__":
    main()
