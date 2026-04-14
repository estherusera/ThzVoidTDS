"""
manual_inspect_atlanta.py
=========================
Visual inspection of all samples in 3D_print_esther_atlanta.tprj.
For each sample: C-scan max projections + B-scan cross-sections + depth slices.
No STL / auto-masking involved — pure THz data review.
Saves one PNG folder per sample under output_manual_inspect/.
"""

import json, os
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ── config ────────────────────────────────────────────────────────────────────
TPRJ        = "3D_print_esther_atlanta.tprj"
OUT_ROOT    = "output_manual_inspect"
N_PLA       = 1.5          # refractive index PLA
C_MM_PS     = 0.29979      # mm / ps
DEPTH_SKIP  = 0.10         # fraction of window to skip at top (surface ringing)
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def depth_mm(t_ps, surface_ps):
    return (t_ps - surface_ps) * C_MM_PS / (2.0 * N_PLA)


def load_sample(f, sample_name):
    """Return (volume, dt_ps, dx_mm, dy_mm, time_axis) for one measurement."""
    img_base = "TerapulseDocument/Measurements/Image Data"
    sg = None
    for img_id in f[img_base]:
        grp = f[f"{img_base}/{img_id}/sample"]
        attr = grp.attrs.get("SampleName", None)
        if attr is None:
            continue
        try:
            n = str(attr[0, 0]).strip()
        except Exception:
            n = str(attr).strip()
        if n == sample_name:
            sg = grp
            break

    if sg is None:
        raise RuntimeError(f"Sample '{sample_name}' not found in Measurements/Image Data")

    data = sg["data"][:]
    line = sg["line"][:].ravel()

    x_off = float(sg.attrs["X_Offset"][0, 0])
    x_spc = float(sg.attrs["X_Spacing"][0, 0])
    wfm   = int(sg.attrs["WfmLength"][0, 0])
    time_axis = x_off + x_spc * np.arange(wfm)

    us  = sg.attrs["UserScanSettings"][0, 0]
    sc  = json.loads(us)["MeasurementConfig"]["Token"]["scanner_config"]
    dx  = float(sc["axis1_spacing"])
    dy  = float(sc["axis2_spacing"])
    xmin = float(sc["axis1_min"])
    ymin = float(sc["axis2_min"])

    nt     = len(time_axis)
    n_slow = len(line)
    n_fast = int(line.max())
    vol    = np.zeros((n_slow, n_fast, nt), dtype=data.dtype)
    col    = 0
    for i in range(n_slow):
        n = int(line[i])
        vol[i, :n, :] = data[:nt, col:col + n].T
        col += n

    dt = float(time_axis[1] - time_axis[0])
    return vol, dt, dx, dy, xmin, ymin, time_axis


def find_surface_idx(vol):
    """Index of the dominant surface reflection (per-pixel argmax, then median)."""
    amp  = np.max(np.abs(vol), axis=2)
    sig  = amp > np.percentile(amp[amp > 0], 10) if np.any(amp > 0) else np.ones(amp.shape, bool)
    t0s  = np.argmax(np.abs(vol), axis=2)
    return int(np.median(t0s[sig]))


def envelope(vol):
    return np.abs(hilbert(vol, axis=2))


# ---------------------------------------------------------------------------
# Per-sample plotting
# ---------------------------------------------------------------------------

def plot_sample(name, vol, dx, dy, xmin, ymin, time_axis, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    n_slow, n_fast, nt = vol.shape

    env  = envelope(vol)
    si   = find_surface_idx(vol)
    skip = max(si, int(DEPTH_SKIP * nt))

    x_ax = xmin + dx * np.arange(n_fast)
    y_ax = ymin + dy * np.arange(n_slow)
    t_ax = time_axis
    surface_ps = t_ax[si]
    extent_xy  = [x_ax[0], x_ax[-1], y_ax[-1], y_ax[0]]

    # ── 1. C-scan overview ────────────────────────────────────────────────────
    c_full = np.max(env, axis=2)
    c_sub  = np.max(env[:, :, skip:], axis=2)
    tof    = np.argmax(env, axis=2)
    d_full = depth_mm(t_ax[tof], surface_ps)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=130)
    fig.suptitle(f"Sample {name}  —  C-scan views\n"
                 f"({n_slow}×{n_fast} px, dx={dx} dy={dy} mm)", fontsize=12)

    for ax, img, ttl, cm in [
        (axes[0], c_full, "Max projection (full depth)", "gray"),
        (axes[1], c_sub,  f"Sub-surface max (skip {skip} samples = {depth_mm(t_ax[skip], surface_ps):.2f} mm)", "inferno"),
        (axes[2], d_full, "Depth of max reflection (mm)", "viridis"),
    ]:
        im = ax.imshow(img, cmap=cm, extent=extent_xy, origin="upper", aspect="equal",
                       vmin=np.percentile(img, 2), vmax=np.percentile(img, 98))
        ax.set_title(ttl, fontsize=9)
        ax.set_xlabel("X / fast (mm)"); ax.set_ylabel("Y / slow (mm)")
        plt.colorbar(im, ax=ax, shrink=0.75)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cscan.png"), dpi=130, bbox_inches="tight")
    plt.close()

    # ── 2. B-scans (XZ and YZ at 25 / 50 / 75 %) ─────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), dpi=120)
    fig.suptitle(f"Sample {name}  —  B-scans", fontsize=12)

    for col_i, frac in enumerate([0.25, 0.50, 0.75]):
        # XZ at y-fraction
        yi = int(frac * n_slow)
        bxz = env[yi, :, :].T          # (nt, n_fast)
        ax = axes[0, col_i]
        im = ax.imshow(bxz, cmap="inferno", aspect="auto",
                       extent=[x_ax[0], x_ax[-1], t_ax[-1], t_ax[0]],
                       vmin=np.percentile(bxz, 2), vmax=np.percentile(bxz, 98))
        ax.set_title(f"XZ  at Y={y_ax[yi]:.1f} mm", fontsize=9)
        ax.set_xlabel("X (mm)"); ax.set_ylabel("Time (ps)")
        plt.colorbar(im, ax=ax, shrink=0.75)

        # YZ at x-fraction
        xi = int(frac * n_fast)
        byz = env[:, xi, :].T          # (nt, n_slow)
        ax = axes[1, col_i]
        im = ax.imshow(byz, cmap="inferno", aspect="auto",
                       extent=[y_ax[0], y_ax[-1], t_ax[-1], t_ax[0]],
                       vmin=np.percentile(byz, 2), vmax=np.percentile(byz, 98))
        ax.set_title(f"YZ  at X={x_ax[xi]:.1f} mm", fontsize=9)
        ax.set_xlabel("Y (mm)"); ax.set_ylabel("Time (ps)")
        plt.colorbar(im, ax=ax, shrink=0.75)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "bscan.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # ── 3. Depth slices (C-scans at 8 evenly spaced depths) ──────────────────
    z_indices = np.linspace(skip, min(nt - 1, skip + int(5.0 / abs(t_ax[1]-t_ax[0]) * (2*N_PLA/C_MM_PS))),
                            8, dtype=int)
    z_indices = np.clip(z_indices, skip, nt - 1)

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), dpi=120)
    fig.suptitle(f"Sample {name}  —  Depth slices", fontsize=12)

    for i, zi in enumerate(z_indices):
        ax    = axes[i // 4, i % 4]
        slc   = env[:, :, zi]
        d_mm  = depth_mm(t_ax[zi], surface_ps)
        im = ax.imshow(slc, cmap="inferno", extent=extent_xy, origin="upper", aspect="equal",
                       vmin=np.percentile(slc, 2), vmax=np.percentile(slc, 98))
        ax.set_title(f"z ≈ {d_mm:.3f} mm  (t={t_ax[zi]:.2f} ps)", fontsize=8)
        ax.set_xlabel("X (mm)", fontsize=7); ax.set_ylabel("Y (mm)", fontsize=7)
        ax.tick_params(labelsize=6)
        plt.colorbar(im, ax=ax, shrink=0.75)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "depth_slices.png"), dpi=120, bbox_inches="tight")
    plt.close()

    print(f"  [{name}] saved to {out_dir}/")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# All sample names present in Measurements/Image Data
SAMPLE_NAMES = ["1", "2", "3", "4", "5", "5b", "6-9", "7", "8",
                "9-6.", "10", "11", "12", "13", "14", "15"]

print("=" * 60)
print(f"Manual inspection: {TPRJ}")
print("=" * 60)

with h5py.File(TPRJ, "r") as f:
    for sname in SAMPLE_NAMES:
        print(f"\nLoading sample '{sname}' ...", end=" ", flush=True)
        try:
            vol, dt, dx, dy, xmin, ymin, tax = load_sample(f, sname)
            print(f"shape={vol.shape}  dx={dx} dy={dy} mm")
            safe = sname.replace(".", "p").replace("-", "_")
            out_dir = os.path.join(OUT_ROOT, f"sample_{safe}")
            plot_sample(sname, vol, dx, dy, xmin, ymin, tax, out_dir)
        except Exception as e:
            print(f"ERROR: {e}")

print("\n" + "=" * 60)
print(f"Done. Results in {OUT_ROOT}/")
print("=" * 60)
