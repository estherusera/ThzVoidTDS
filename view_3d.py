"""
view_3d.py — interactive 3D viewer for the THz sample volumes
=============================================================
Each sample is stored as a (50, 100, 100) float32 volume in a slices dir:
    <slices_dir>/<sample>_slices.npy   normalized reflectivity, 0..1
    <slices_dir>/<sample>_depths.npy   depth (mm) for each of the 50 slices
The 100x100 in-plane grid covers the 20x20 mm ROI (0.2 mm/px).

Renders a rotatable volume in the browser with plotly. Voids show up as
low-reflectivity pockets, so by default we visualize the *signed deviation*
from each depth slice's background and highlight the strongest features.

Usage:
    thesis_env/bin/python view_3d.py 4                       # sample 4, slices_v2
    thesis_env/bin/python view_3d.py 4 --dir slices_v2_atlanta2
    thesis_env/bin/python view_3d.py 4 --mode raw            # raw reflectivity
    thesis_env/bin/python view_3d.py 4 --iso                 # isosurface instead of volume
    thesis_env/bin/python view_3d.py --list                 # list available samples
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go
except ImportError:
    sys.exit("plotly not installed. Run: thesis_env/bin/pip install plotly")

DX_MM = 0.2  # in-plane pixel size (20 mm / 100 px)


def list_samples(slices_dir: Path):
    names = sorted(p.name[: -len("_slices.npy")] for p in slices_dir.glob("*_slices.npy"))
    print(f"{len(names)} samples in {slices_dir}:")
    print("  " + "  ".join(names))


def load_volume(slices_dir: Path, sample: str):
    vol = np.load(slices_dir / f"{sample}_slices.npy").astype(np.float32)  # (Z, Y, X)
    depths = np.load(slices_dir / f"{sample}_depths.npy").astype(np.float32)  # (Z,) mm
    return vol, depths


def signed_deviation(vol: np.ndarray) -> np.ndarray:
    """Per-slice deviation from the slice median, normalized to [-1, 1].

    Negative = darker than background (void-like), positive = brighter.
    """
    med = np.median(vol, axis=(1, 2), keepdims=True)
    dev = vol - med
    scale = np.abs(dev).max()
    return dev / (scale + 1e-9)


def build_figure(vol, depths, sample, mode, iso):
    nz, ny, nx = vol.shape

    # physical coordinate grids (mm). depth on Z, in-plane on X/Y.
    xs = np.arange(nx) * DX_MM
    ys = np.arange(ny) * DX_MM
    zs = depths
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")  # (X, Y, Z) order for plotly

    if mode == "signed":
        data = signed_deviation(vol)
        cmin, cmax, cscale = -1.0, 1.0, "RdBu"
    else:  # raw
        data = vol
        cmin, cmax, cscale = float(vol.min()), float(vol.max()), "Viridis"

    # plotly wants flat arrays matching the meshgrid (X,Y,Z) ordering;
    # vol is (Z,Y,X) so transpose to (X,Y,Z).
    values = np.transpose(data, (2, 1, 0))

    common = dict(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
        value=values.flatten(),
        colorscale=cscale, cmin=cmin, cmax=cmax,
    )

    if iso:
        # show the strongest void-like (and bright) shells
        if mode == "signed":
            surfaces = dict(isomin=-0.6, isomax=0.6, surface_count=4)
        else:
            lo, hi = np.percentile(data, [5, 95])
            surfaces = dict(isomin=float(lo), isomax=float(hi), surface_count=4)
        trace = go.Isosurface(**common, **surfaces, opacity=0.5, caps=dict(
            x_show=False, y_show=False, z_show=False))
    else:
        trace = go.Volume(
            **common,
            opacity=0.1,
            surface_count=18,
            opacityscale="extremes" if mode == "signed" else "uniform",
        )

    fig = go.Figure(data=trace)
    fig.update_layout(
        title=f"Sample {sample} — {mode} ({'isosurface' if iso else 'volume'})",
        scene=dict(
            xaxis_title="x (mm)",
            yaxis_title="y (mm)",
            zaxis_title="depth (mm)",
            zaxis=dict(autorange="reversed"),  # surface (depth 0) on top
            aspectmode="data",
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sample", nargs="?", help="sample name, e.g. 4 or 5b")
    ap.add_argument("--dir", default="slices_v2", help="slices directory")
    ap.add_argument("--mode", choices=["signed", "raw"], default="signed",
                    help="signed deviation from background (default) or raw reflectivity")
    ap.add_argument("--iso", action="store_true", help="isosurface render instead of volume")
    ap.add_argument("--list", action="store_true", help="list available samples and exit")
    ap.add_argument("--out", help="write HTML here instead of opening a browser")
    args = ap.parse_args()

    slices_dir = Path(args.dir)
    if not slices_dir.is_dir():
        sys.exit(f"no such dir: {slices_dir}")

    if args.list or not args.sample:
        list_samples(slices_dir)
        if not args.sample:
            return

    vol, depths = load_volume(slices_dir, args.sample)
    print(f"loaded sample {args.sample}: {vol.shape} (Z,Y,X), depth {depths[0]:.2f}..{depths[-1]:.2f} mm")

    fig = build_figure(vol, depths, args.sample, args.mode, args.iso)

    if args.out:
        fig.write_html(args.out)
        print(f"wrote {args.out}")
    else:
        fig.show()


if __name__ == "__main__":
    main()
