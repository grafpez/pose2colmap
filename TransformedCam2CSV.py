#!/usr/bin/env python3
"""
TransformedCam2CSV.py
=====================
Convert SHARE3DCAM PointCloud Studio TransformedCam.json camera poses
→ Metashape Reference CSV (filename, X, Y, Z, roll, pitch, yaw).

Output is written to the SAME folder as the source TransformedCam.json.

Usage (auto-discover):
    python TransformedCam2CSV.py --folder "H:/my_scan/Output/Undistort"

Usage (explicit path):
    python TransformedCam2CSV.py "H:/my_scan/Output/Undistort/TransformedCam.json"

Metashape Reference pane import CSV format:
    filename,X,Y,Z,roll,pitch,yaw
    DSC_0001.jpg,0.0,0.0,0.0,0.0,0.0,0.0
    DSC_0002.jpg,0.1,0.0,0.0,0.0,0.0,0.0
"""

import json
import math
import os
import sys
import glob
import argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Quaternion → Euler (ZYX = yaw, pitch, roll)
# ─────────────────────────────────────────────────────────────────────────────

def quat_to_euler(qw, qx, qy, qz):
    """Convert quaternion (w,x,y,z) to Euler angles (roll, pitch, yaw) in degrees.
    Uses ZYX cardan sequence: yaw(z) @ pitch(y) @ roll(x).
    Results are wrapped to [-180, +180] degrees.
    """
    # Roll (X-axis rotation)
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)  # gimbal lock
    else:
        pitch = math.asin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


def rotmat_to_euler_zyx(R):
    """Convert 3×3 rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
    Uses ZYX cardan sequence.
    """
    # Pitch (Y-axis rotation) — handle gimbal lock
    sy = math.sqrt(R[0][2] ** 2 + R[1][2] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll  = math.atan2( R[1][2],  R[0][2])
        pitch = math.atan2(-R[2][2],  sy)
        yaw   = math.atan2( R[2][1],  R[2][0])
    else:
        # Gimbal lock: pitch ≈ ±90°
        roll  = math.atan2(-R[1][0],  R[1][1])
        pitch = math.atan2(-R[2][2],  sy)
        yaw   = 0.0

    return (
        math.degrees(roll),
        math.degrees(pitch),
        math.degrees(yaw),
    )


# ─────────────────────────────────────────────────────────────────────────────
# TransformedCam helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_transformed_cam(path):
    """Load TransformedCam.json and return sorted frames list."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", data)
    return sorted(frames, key=lambda fr: fr.get("timestamp", 0))


def split_frames(frames):
    """Split mixed frames by file_path prefix → (left_frames, right_frames)."""
    left, right = [], []
    for fr in frames:
        fp = fr.get("file_path", "").replace("\\", "/").lower()
        if fp.startswith("right/") or fp.startswith("r/"):
            right.append(fr)
        else:
            left.append(fr)
    return left, right


def get_euler_from_frame(fr):
    """Extract Euler angles (roll, pitch, yaw) from a TransformedCam frame.
    
    Supports two formats:
      1. quaternion: {qw, qx, qy, qz}
      2. rotation matrix: transform as 4×4 [R|t] or [R|t|R_shift]
    
    Returns (roll, pitch, yaw) in degrees.
    """
    T = fr.get("transform")
    if T is not None:
        R = [[T[i][j] for j in range(3)] for i in range(3)]
        return rotmat_to_euler_zyx(R)

    # Try quaternion
    qw = fr.get("qw", 1.0)
    qx = fr.get("qx", 0.0)
    qy = fr.get("qy", 0.0)
    qz = fr.get("qz", 0.0)
    norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    if norm > 1e-9:
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
        return quat_to_euler(qw, qx, qy, qz)

    raise ValueError(f"Cannot extract orientation from frame: {fr}")


def get_position_from_frame(fr):
    """Extract world position (X, Y, Z) from a TransformedCam frame."""
    T = fr.get("transform")
    if T is not None:
        return T[0][3], T[1][3], T[2][3]

    # Fallback to separate fields
    x = fr.get("x", fr.get("position_x", 0.0))
    y = fr.get("y", fr.get("position_y", 0.0))
    z = fr.get("z", fr.get("position_z", 0.0))
    return x, y, z


def get_image_name(fr):
    """Extract a clean image filename from a TransformedCam frame."""
    fp = fr.get("file_path", fr.get("image_name", fr.get("name", "")))
    return Path(fp).name


# ─────────────────────────────────────────────────────────────────────────────
# Main conversion
# ─────────────────────────────────────────────────────────────────────────────

def convert(json_path, output_csv=None, left_right="both"):
    """
    Convert TransformedCam.json → Metashape Reference CSV.
    
    Parameters
    ----------
    json_path : str or Path
        Path to TransformedCam.json.
    output_csv : str or Path, optional
        Output CSV path. Defaults to <json_path>/../TransformedCam.csv.
    left_right : str
        "both" | "left" | "right" — which camera(s) to include.
    
    Returns
    -------
    output_csv : Path
        Path to the written CSV file.
    """
    json_path  = Path(json_path)
    frames_all = load_transformed_cam(json_path)

    if left_right == "both":
        frames = frames_all
    else:
        left_frames, right_frames = split_frames(frames_all)
        frames = left_frames if left_right == "left" else right_frames

    if not frames:
        print(f"⚠  No frames found for '{left_right}' — nothing to export.")
        sys.exit(0)

    if output_csv is None:
        output_csv = json_path.parent / "TransformedCam.csv"
    output_csv = Path(output_csv)

    rows = []
    for fr in frames:
        name = get_image_name(fr)
        x, y, z = get_position_from_frame(fr)
        roll, pitch, yaw = get_euler_from_frame(fr)

        rows.append({
            "filename": name,
            "X": x,
            "Y": y,
            "Z": z,
            "roll":  roll,
            "pitch": pitch,
            "yaw":   yaw,
        })

    # ── Write CSV ────────────────────────────────────────────────────────────
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        f.write("filename,X,Y,Z,roll,pitch,yaw\n")
        for r in rows:
            f.write(
                f"{r['filename']},"
                f"{r['X']:.6f},{r['Y']:.6f},{r['Z']:.6f},"
                f"{r['roll']:.4f},{r['pitch']:.4f},{r['yaw']:.4f}\n"
            )

    print(f"[✓] Wrote: {output_csv}  ({len(rows)} rows)")
    return output_csv


def auto_discover(folder):
    """Find TransformedCam.json in folder (recursive, most-deep first)."""
    folder = Path(folder)
    candidates = sorted(folder.rglob("TransformedCam.json"), key=lambda p: len(p.parts))
    if candidates:
        return candidates[0]  # deepest match
    return None


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

HELP_EPILOG = """
Examples:
  # Auto-discover TransformedCam.json in the undistort folder
  python TransformedCam2CSV.py --folder "H:/my_scan/Output/Undistort"

  # Explicit path
  python TransformedCam2CSV.py "H:/my_scan/Output/Undistort/TransformedCam.json"

  # Export left camera only
  python TransformedCam2CSV.py --folder "H:/my_scan/Output/Undistort" --left

  # Custom output path
  python TransformedCam2CSV.py --folder "H:/my_scan/Output/Undistort" -o "H:/my_scan/poses.csv"

Metashape import:
  1. File → Import → Import CSV... (Reference)
  2. Select the TransformedCam.csv
  3. Check 'Skip row with non-numeric X column' if needed
  4. Apply — images will be positioned and oriented roughly in Metashape
"""

def main():
    p = argparse.ArgumentParser(
        description="Convert SHARE3DCAM TransformedCam.json → Metashape Reference CSV.",
        epilog=HELP_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "input", nargs="?", default=None,
        help="Path to TransformedCam.json (overrides --folder auto-discovery)."
    )
    p.add_argument(
        "--folder", "-f",
        help="Scan this folder recursively for TransformedCam.json."
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output CSV path (default: <json_parent>/TransformedCam.csv)."
    )
    p.add_argument(
        "--left-right", "-l",
        choices=["both", "left", "right"], default="both",
        help="Which camera(s) to export: both, left-only, or right-only. [default: both]"
    )
    args = p.parse_args()

    # ── Resolve input path ──────────────────────────────────────────────────
    if args.input:
        json_path = Path(args.input)
        if not json_path.is_file():
            print(f"❌  File not found: {json_path}")
            sys.exit(1)
    elif args.folder:
        json_path = auto_discover(args.folder)
        if json_path is None:
            print(f"❌  TransformedCam.json not found in: {args.folder}")
            sys.exit(1)
        print(f"[*] Auto-discovered: {json_path}")
    else:
        # Try current directory
        json_path = Path("TransformedCam.json")
        if not json_path.is_file():
            json_path = auto_discover(".")
            if json_path is None:
                print("❌  No TransformedCam.json found. Provide --folder or a file path.")
                print("    Example: python TransformedCam2CSV.py --folder 'H:/my_scan/Output/Undistort'")
                sys.exit(1)

    # ── Convert ───────────────────────────────────────────────────────────────
    output_csv = convert(
        json_path=json_path,
        output_csv=args.output,
        left_right=args.left_right,
    )
    print(f"\n→ Import in Metashape: File → Import → Import CSV... (Reference)")
    print(f"  File: {output_csv}")


if __name__ == "__main__":
    main()
