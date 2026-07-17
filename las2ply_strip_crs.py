#!/usr/bin/env python3
"""
las2ply_strip_crs.py  v7
========================
Convert a LiDAR .las / .laz point cloud to a plain .ply file with NO
embedded coordinate reference system (CRS/datum). 

Why: Metashape Standard blocks "LiDAR data" and throws
     "Unsupported datum transformation" when a CRS is embedded.
     A plain .ply with XYZ + RGB has no CRS metadata and imports fine.

     NOTE: Even after stripping VLR records, Metashape may still infer
     a coordinate system from the coordinate values themselves (e.g. UTM
     coordinates are large numbers). Use --recenter to shift coordinates
     to near-zero, preventing datum inference entirely.

     Use --voxel SIZE to downsample via voxel grid (one point per cell),
     producing a sparse cloud that may bypass LiDAR detection entirely.

Requirements:
    pip install laspy[lazrs] numpy

Usage:
    python las2ply_strip_crs.py input.las                          # output: input_stripped.ply
    python las2ply_strip_crs.py input.las --recenter               # recenter to near-zero
    python las2ply_strip_crs.py input.las --voxel 0.05             # 5cm voxel downsampling
    python las2ply_strip_crs.py input.las --recenter --voxel 0.1   # both combined
    python las2ply_strip_crs.py input.las -o out.ply
    python las2ply_strip_crs.py input.laz --no-color
"""

import argparse
import struct
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Dependency check
# ─────────────────────────────────────────────────────────────────────────────

def check_deps():
    missing = []
    try:
        import laspy
    except ImportError:
        missing.append("laspy[lazrs]")
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    if missing:
        print(f"[X]  Missing packages: {', '.join(missing)}")
        print(f"    Install with:  pip install {' '.join(missing)}")
        sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# PLY writer (binary little-endian, no CRS)
# ─────────────────────────────────────────────────────────────────────────────

def write_ply(path, xyz, rgb=None):
    """Write a binary PLY file with XYZ (float64) and optional RGB (uint8)."""
    n = len(xyz)
    has_color = rgb is not None and len(rgb) == n

    header_lines = [
        "ply",
        "format binary_little_endian 1.0",
        "comment Converted by las2ply_strip_crs.py - no CRS",
        f"element vertex {n}",
        "property double x",
        "property double y",
        "property double z",
    ]
    if has_color:
        header_lines += [
            "property uchar red",
            "property uchar green",
            "property uchar blue",
        ]
    header_lines.append("end_header")
    header = "\n".join(header_lines) + "\n"

    with open(path, "wb") as f:
        f.write(header.encode("utf-8"))
        if has_color:
            # Pack XYZ (3×float64) + RGB (3×uint8) per vertex
            import numpy as np
            # Build structured array
            dt = np.dtype([
                ("x", "<f8"), ("y", "<f8"), ("z", "<f8"),
                ("r", "u1"),  ("g", "u1"),  ("b", "u1"),
            ])
            arr = np.empty(n, dtype=dt)
            arr["x"] = xyz[:, 0]
            arr["y"] = xyz[:, 1]
            arr["z"] = xyz[:, 2]
            arr["r"] = rgb[:, 0]
            arr["g"] = rgb[:, 1]
            arr["b"] = rgb[:, 2]
            f.write(arr.tobytes())
        else:
            import numpy as np
            dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8")])
            arr = np.empty(n, dtype=dt)
            arr["x"] = xyz[:, 0]
            arr["y"] = xyz[:, 1]
            arr["z"] = xyz[:, 2]
            f.write(arr.tobytes())

    print(f"[OK] Wrote: {path}  ({n:,} points)")


# ─────────────────────────────────────────────────────────────────────────────
# Voxel downsampling
# ─────────────────────────────────────────────────────────────────────────────

def voxel_downsample(xyz, rgb, voxel_size, verbose=True):
    """Downsample point cloud: keep one point per voxel cell.

    Uses the point closest to each voxel center for best quality.
    Falls back to first-point-per-voxel for speed on huge clouds.
    """
    import numpy as np

    if voxel_size <= 0:
        raise ValueError(f"Voxel size must be > 0, got {voxel_size}")

    n_in = len(xyz)
    if verbose:
        print(f"    Voxel downsampling: cell size = {voxel_size}m")

    # Compute voxel indices for each point
    voxel_idx = np.floor(xyz / voxel_size).astype(np.int64)

    # Encode 3D voxel index into a single 64-bit key for fast grouping
    # Shift each axis to ensure positive values
    shifts = voxel_idx.min(axis=0)
    voxel_idx_shifted = voxel_idx - shifts

    # Create a unique key per voxel: ix * max_ny * max_nz + iy * max_nz + iz
    dims = voxel_idx_shifted.max(axis=0) + 1
    keys = (voxel_idx_shifted[:, 0] * dims[1] * dims[2]
            + voxel_idx_shifted[:, 1] * dims[2]
            + voxel_idx_shifted[:, 2])

    # For each unique voxel key, keep the point closest to voxel center
    voxel_centers = (np.floor(xyz / voxel_size) + 0.5) * voxel_size
    dist_to_center = np.sum((xyz - voxel_centers) ** 2, axis=1)

    # Sort by key then by distance — groupby will pick the closest
    order = np.lexsort((dist_to_center, keys))
    keys_sorted = keys[order]

    # Find first occurrence of each unique key (= closest to center)
    _, unique_idx = np.unique(keys_sorted, return_index=True)

    # Map back to original indices
    selected = order[unique_idx]
    selected.sort()  # preserve original point order

    xyz_out = xyz[selected]
    rgb_out = rgb[selected] if rgb is not None else None

    n_out = len(xyz_out)
    if verbose:
        ratio = n_out / n_in * 100
        print(f"    Downsampled: {n_in:,} -> {n_out:,} points ({ratio:.1f}% kept)")

    return xyz_out, rgb_out


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert(input_path, output_path=None, no_color=False, recenter=False, voxel_size=None, verbose=True):
    import laspy
    import numpy as np

    input_path = Path(input_path)
    if output_path is None:
        # Insert _stripped before extension to avoid overwriting original .ply
        output_path = input_path.parent / (input_path.stem + "_stripped.ply")
    output_path = Path(output_path)

    if verbose:
        print(f"[*] Reading: {input_path}")

    las = laspy.read(str(input_path))
    n = len(las.points)

    if verbose:
        print(f"    Points: {n:,}")
        # Show CRS info if present
        crs_info = []
        for vlr in las.vlrs:
            if vlr.record_id in (2111, 2112, 34735, 34736, 34737):
                crs_info.append(f"VLR record_id={vlr.record_id}")
        if crs_info:
            print(f"    CRS VLRs found (will be stripped): {', '.join(crs_info)}")
        else:
            print(f"    No CRS VLRs detected.")

    # ── Extract XYZ ──────────────────────────────────────────────────────────
    xyz = np.column_stack([
        las.x.scaled_array() if hasattr(las.x, "scaled_array") else np.array(las.x),
        las.y.scaled_array() if hasattr(las.y, "scaled_array") else np.array(las.y),
        las.z.scaled_array() if hasattr(las.z, "scaled_array") else np.array(las.z),
    ])

    # ── Recenter coordinates (optional) ───────────────────────────────────────
    if recenter:
        origin = xyz.min(axis=0)
        xyz = xyz - origin
        if verbose:
            print(f"    Recentered: origin shifted by ({origin[0]:.2f}, {origin[1]:.2f}, {origin[2]:.2f})")
            print(f"    New range: X [{xyz[:,0].min():.2f} .. {xyz[:,0].max():.2f}]")
            print(f"                Y [{xyz[:,1].min():.2f} .. {xyz[:,1].max():.2f}]")
            print(f"                Z [{xyz[:,2].min():.2f} .. {xyz[:,2].max():.2f}]")

    # ── Extract RGB BEFORE voxel downsampling (so colors survive thinning) ────
    rgb = None
    if not no_color:
        try:
            r = np.array(las.red,   dtype=np.uint16)
            g = np.array(las.green, dtype=np.uint16)
            b = np.array(las.blue,  dtype=np.uint16)
            # LAS stores 16-bit color; scale to 8-bit
            if r.max() > 255:
                r = (r / 256).astype(np.uint8)
                g = (g / 256).astype(np.uint8)
                b = (b / 256).astype(np.uint8)
            else:
                r = r.astype(np.uint8)
                g = g.astype(np.uint8)
                b = b.astype(np.uint8)
            rgb = np.column_stack([r, g, b])
            if verbose:
                print(f"    Color: RGB found, included in output.")
        except Exception:
            if verbose:
                print(f"    Color: No RGB in source — writing XYZ only.")

    # ── Voxel downsampling (optional) ─────────────────────────────────────────
    if voxel_size is not None:
        xyz, rgb = voxel_downsample(xyz, rgb, voxel_size, verbose=verbose)

    # ── Write PLY ─────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_ply(output_path, xyz, rgb)

    return output_path


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    check_deps()

    p = argparse.ArgumentParser(
        description="Convert .las/.laz → plain .ply (strips CRS for Metashape Standard import).",
    )
    p.add_argument("input", help="Input .las or .laz file path.")
    p.add_argument("-o", "--output", default=None,
                   help="Output .ply path (default: same folder, same name, .ply extension).")
    p.add_argument("--no-color", action="store_true",
                   help="Skip RGB color - write XYZ only.")
    p.add_argument("--recenter", action="store_true",
                   help="Shift coordinates to near-zero (prevents Metashape datum inference).")
    p.add_argument("--voxel", type=float, default=None, metavar="SIZE",
                   help="Voxel downsampling cell size in meters (e.g. 0.05 = 5cm). "
                        "Keeps one point per voxel cell for a sparse cloud.")
    args = p.parse_args()

    convert(
        input_path=args.input,
        output_path=args.output,
        no_color=args.no_color,
        recenter=args.recenter,
        voxel_size=args.voxel,
    )

    print()
    input_p = Path(args.input)
    default_out = input_p.parent / (input_p.stem + "_stripped.ply")
    print("-> In Metashape Standard:")
    print("  File -> Import -> Import Point Cloud...")
    print(f"  Select: {args.output or default_out}")


if __name__ == "__main__":
    main()
