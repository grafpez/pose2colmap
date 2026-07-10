#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose2colmap.py  (v1.12 -- DISABLE MS-XML: Equisolid->KB fitting produces wrong values)

Convert SHARE3DCAM PointCloud Studio Output-Undistort folder ->COLMAP

Standard COLMAP directory layout:
  <parent>/COLMAP/
  +-- images/       --junction links to undistort/left/ and right/
  +-- sparse/
      +-- cameras.txt
      +-- images.txt
      +-- points3D.txt
cameras.txt + images.txt for RealityCapture 2 (RS2 v2.1.1+).

Fully auto-discovers all PCS undistort files -- or supply them manually.

v42 key changes: (1) --pitch/yaw/roll renamed to --camera-pitch/yaw/roll;
(2) New --points-pitch/yaw/roll for point cloud only;
(3) Camera defaults: axis=x-y-z, pitch=-90, yaw=0, roll=0;
(4) Point cloud defaults: axis=x-y-z, pitch=-90, yaw=0, roll=0
TransformedCam.json rotation matrices have det=-1 (left-handed SLAM frame).
Negating Y converts det=-1 -> det=+1 (right-handed COLMAP frame):
  - Camera Y-axis flips from gravity-up to gravity-down (COLMAP convention)
  - Point cloud Y coordinates are negated to match camera frame
  - No additional rotation needed (pitch=0, yaw=0, roll=0)

v48 key changes: (1) Camera pitch default changed from 0 to -90 (matches
  point cloud orientation out of the box); (2) sys.argv preprocessor merges
  "--arg =value" into "--arg=value" to fix argparse space-before-= bug;
  (3) Camera axis default set to x-y-z (chirality fix auto-applied);
  (4) Automatic chirality detection: if the combined camera rotation has
  det=-1 (improper/left-handed), column 1 is negated to make it proper.
  This means ALL axis permutations now work for cameras (x-y-z, -x-yz, xyz
  etc.), not just x-yz. The chirality fix is no longer tied to the axis
  permutation choice.

Auto-discovered files (PCS naming convention):
  TransformedCam.json          -?per-image intrinsics + 4x4 poses (primary)
  Left_undistort.opt           -?left undistort camera model + mm-params
  Right_undistort.opt          -?right undistort camera model + mm-params
  left_undistort_intrinsic.txt  -?left post-undistort pixel intrinsics
  right_undistort_intrinsics.txt -?right post-undistort pixel intrinsics
  ImgPose.txt                  -?position + roll/pitch/yaw + quaternion + timestamp
  xyzopt.txt / xyzopk.txt       -?position + Omega/Phi/Kappa (photogrammetry angles)

Intrinsics priority: TransformedCam.json -?*_intrinsic.txt -?*.opt

Usage:
  # Auto-discover mode (recommended --point to undistort folder)
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort"

  # Add .las dense point cloud --points3D.txt (LichtFeld / PostShot / NeRF)
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" \
                        --las "H:/my_scan/Output/colorized.las"

  # Disable chirality fix (raw TransformedCam.json coordinates)
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" \
                        --points-axis xyz

  # With xyzopk.txt instead of ImgPose.txt
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" --use-xyzopk

Author: SHARE3DCAM Bot (OpenClaw)  v42
Co-author: Peter Graf, AVsupport.com.au
"""

import json
import math
import os
import sys
import glob
import shutil
import argparse
import yaml
import xml.etree.ElementTree as ET
import numpy as np
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None

# Optional: laspy for .las --points3D.txt conversion
try:
    import laspy
    import numpy as np
    LASPY_AVAILABLE = True
except ImportError:
    LASPY_AVAILABLE = False
    np = None

# -----------------------------------------------------------------------------
# Quaternion helpers
# -----------------------------------------------------------------------------

def rotmat_to_quat(R):
    """3x3 rotation matrix -?[qw, qx, qy, qz] (COLMAP convention)."""
    trace = R[0][0] + R[1][1] + R[2][2]
    q = [0.0] * 4
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        q[0] = 0.25 * s
        q[1] = (R[2][1] - R[1][2]) / s
        q[2] = (R[0][2] - R[2][0]) / s
        q[3] = (R[1][0] - R[0][1]) / s
    elif R[0][0] > R[1][1] and R[0][0] > R[2][2]:
        s = math.sqrt(1.0 + R[0][0] - R[1][1] - R[2][2]) * 2.0
        q[0] = (R[2][1] - R[1][2]) / s
        q[1] = 0.25 * s
        q[2] = (R[1][0] + R[0][1]) / s
        q[3] = (R[0][2] + R[2][0]) / s
    elif R[1][1] > R[2][2]:
        s = math.sqrt(1.0 + R[1][1] - R[0][0] - R[2][2]) * 2.0
        q[0] = (R[0][2] - R[2][0]) / s
        q[1] = (R[1][0] + R[0][1]) / s
        q[2] = 0.25 * s
        q[3] = (R[2][1] + R[1][2]) / s
    else:
        s = math.sqrt(1.0 + R[2][2] - R[0][0] - R[1][1]) * 2.0
        q[0] = (R[1][0] - R[0][1]) / s
        q[1] = (R[0][2] + R[2][0]) / s
        q[2] = (R[2][1] + R[1][2]) / s
        q[3] = 0.25 * s
    norm = math.sqrt(sum(v * v for v in q))
    return [v / norm for v in q] if norm > 1e-12 else q


def quat_to_rotmat(q):
    """[qw, qx, qy, qz] -?3x3 rotation matrix (list-of-lists)."""
    qw, qx, qy, qz = q
    return [
        [1 - 2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw),     1 - 2*(qx*qx + qy*qy)],
    ]


def euler_opk_to_quat(omega_deg, phi_deg, kappa_deg):
    """Omega/Phi/Kappa (deg) -?[qw, qx, qy, qz].
    Ome/Phi = tilt angles (rad), Kappa = azimuth (rad).
    We apply Rx(ome) * Ry(phi) * Rz(kap) -?standard photogrammetry convention."""
    o = math.radians(omega_deg)
    p = math.radians(phi_deg)
    k = math.radians(kappa_deg)

    cos_o, sin_o = math.cos(o), math.sin(o)
    cos_p, sin_p = math.cos(p), math.sin(p)
    cos_k, sin_k = math.cos(k), math.sin(k)

    # Rz(kappa) @ Ry(phi) @ Rx(omega)
    r00 = cos_p * cos_k
    r01 = sin_o * sin_p * cos_k - cos_o * sin_k
    r02 = cos_o * sin_p * cos_k + sin_o * sin_k

    r10 = cos_p * sin_k
    r11 = sin_o * sin_p * sin_k + cos_o * cos_k
    r12 = cos_o * sin_p * sin_k - sin_o * cos_k

    r20 = -sin_p
    r21 = sin_o * cos_p
    r22 = cos_o * cos_p

    return rotmat_to_quat([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])


def build_world_transform(points_axis="x-y-z", pitch_deg=-90.0,
                          yaw_deg=0.0, roll_deg=0.0):
    """
    Build the 3x3 world-frame coordinate transform S that maps
    PCS Z-up coordinates -?COLMAP Y-up coordinates.

    The same S is applied to both camera poses and point cloud
    so they remain aligned.

    Transform order: axis permutation first, then Euler rotation
    (pitch -?yaw -?roll), matching the --points-axis + --pitch/--yaw/--roll
    CLI arguments.

    Returns a 3x3 list-of-lists (row-major), or None if no transform needed.
    """
    # -- Parse axis permutation ----------------------------------------------
    axis_str = points_axis.lstrip("= ").lower().strip() if points_axis else "xyz"
    need_axis = axis_str != "xyz"
    need_rot = (pitch_deg != 0.0) or (yaw_deg != 0.0) or (roll_deg != 0.0)

    if not need_axis and not need_rot:
        return None

    # Build permutation + sign matrix P (3x3)
    P = [[0.0]*3 for _ in range(3)]
    out_idx = 0
    i = 0
    neg_next = False
    while i < len(axis_str):
        ch = axis_str[i]
        if ch == '-':
            neg_next = True
            i += 1
            continue
        if ch not in ('x', 'y', 'z'):
            raise ValueError(f"Invalid axis char '{ch}' in '{points_axis}'")
        src = 'xyz'.index(ch)
        P[out_idx][src] = -1.0 if neg_next else 1.0
        neg_next = False
        out_idx += 1
        i += 1
    if out_idx != 3:
        raise ValueError(f"Expected 3 axes, got {out_idx} in '{points_axis}'")

    if not need_rot:
        return P

    # -- Build Euler rotation matrix R = Rz(roll) @ Ry(yaw) @ Rx(pitch) -----
    p = math.radians(pitch_deg)
    y = math.radians(yaw_deg)
    r = math.radians(roll_deg)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    cr, sr = math.cos(r), math.sin(r)

    Rot = [
        [ cy*cr,             sy*sp*cr - cp*sr,     cp*cr*sy + sp*sr ],
        [ cy*sr,             cp*cr + sp*sy*sr,     cp*sy*sr - sp*cr ],
        [-sy,                sp*cy,                 cp*cy            ],
    ]

    # -- Combine: S = Rot @ P ------------------------------------------------
    S = [[0.0]*3 for _ in range(3)]
    for row in range(3):
        for col in range(3):
            for k in range(3):
                S[row][col] += Rot[row][k] * P[k][col]
    return S


def mat_mul_3x3(A, B):
    """Multiply two 3x3 matrices (list-of-lists)."""
    R = [[0.0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                R[i][j] += A[i][k] * B[k][j]
    return R


def mat_transpose_3x3(M):
    """Transpose a 3x3 matrix."""
    return [[M[j][i] for j in range(3)] for i in range(3)]


def _det3x3(M):
    """Determinant of a 3x3 matrix (list-of-lists)."""
    return (M[0][0] * (M[1][1]*M[2][2] - M[1][2]*M[2][1])
          - M[0][1] * (M[1][0]*M[2][2] - M[1][2]*M[2][0])
          + M[0][2] * (M[1][0]*M[2][1] - M[1][1]*M[2][0]))


def _mat_mul(A, B):
    """Multiply two 3x3 matrices (list-of-lists)."""
    R = [[0.0]*3 for _ in range(3)]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                R[i][j] += A[i][k] * B[k][j]
    return R


def transform_to_colmap_pose(T_raw, S=None, S_pos=None):
    """
    TransformedCam.json: [R|t] camera-to-world -> COLMAP world-to-camera.

    If S (world-frame transform) is provided, the camera pose is first mapped
    into the transformed world frame:
      Camera position in new world = S @ t
      Camera axes in new world     = S @ R  (columns)
      New camera-to-world: [S@R | S@t]

    Then inverted to world-to-camera:
      R_w2c = (S@R)^T = R^T @ S^T
      t_w2c = -R_w2c @ (S@t) = -R^T @ S^T @ S @ t

    Since S is orthogonal (S^T @ S = I), this simplifies to:
      t_w2c = -R^T @ t  (same as without S)
    But we compute it explicitly for numerical stability.

    v55/v57: chirality fix REMOVED. It corrupted rotations by negating a single
    column of R_w2c, producing invalid quaternions. Instead, use axis
    permutations where det(S)=-1 (e.g. "x-yz") so that det(S)*det(R_slam)
    = (-1)*(-1) = +1, producing proper rotations naturally.
    v57: Reverted v56 TX negation (it mirrored the entire trajectory).
    L/R swap needs deeper investigation - TX values are ~7mm not ~6cm baseline.
    """
    R = [row[:3] for row in T_raw[:3]]
    t = [T_raw[i][3] for i in range(3)]

    # R^T (world-to-camera rotation in original PCS frame)
    Rt = [[R[j][i] for j in range(3)] for i in range(3)]

    if S is not None:
        # Apply world-frame transform: R_w2c_new = Rt @ S^T
        St = mat_transpose_3x3(S)
        Rt_new = mat_mul_3x3(Rt, St)

        # Camera position in new world frame: S_pos @ t (defaults to S if not given)
        _S_for_pos = S_pos if S_pos is not None else S
        t_new = [sum(_S_for_pos[r][c] * t[c] for c in range(3)) for r in range(3)]

        # t_w2c = -R_w2c_new @ t_new
        tx = -sum(Rt_new[0][j] * t_new[j] for j in range(3))
        ty = -sum(Rt_new[1][j] * t_new[j] for j in range(3))
        tz = -sum(Rt_new[2][j] * t_new[j] for j in range(3))

        qw, qx, qy, qz = rotmat_to_quat(Rt_new)
    else:
        tx = -sum(Rt[0][j] * t[j] for j in range(3))
        ty = -sum(Rt[1][j] * t[j] for j in range(3))
        tz = -sum(Rt[2][j] * t[j] for j in range(3))

        qw, qx, qy, qz = rotmat_to_quat(Rt)

    return qw, qx, qy, qz, tx, ty, tz


# -----------------------------------------------------------------------------
# File loaders
# -----------------------------------------------------------------------------

def load_transformed_cam(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    frames = data.get("frames", data)
    return sorted(frames, key=lambda fr: fr.get("timestamp", 0))


def _xml_text(el):
    """Safely get text content from an XML element."""
    if el is None:
        return None
    t = el.text
    return t.strip() if t else None

def _xml_float(el):
    """Get float from XML element -?checks value attr first, then text content."""
    if el is None:
        return None
    # Priority: value attribute -?text content
    val = el.get("value")
    if val is not None:
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
    t = _xml_text(el)
    if t is not None:
        try:
            return float(t)
        except (ValueError, TypeError):
            pass
    return None


# v102: Metashape EquisolidFisheye -> COLMAP OPENCV_FISHEYE conversion
def _load_metashape_xml(path):
    """Parse Metashape calibration XML -> dict with f, cx, cy, k1-k4, p1, p2."""
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        print(f"  [Metashape XML] file not found: {path}")
        return None
    tree = ET.parse(str(path))
    root = tree.getroot()
    def _f(tag):
        el = root.find(tag)
        return float(el.text) if el is not None else None
    w = int(float(root.find("width").text))
    h = int(float(root.find("height").text))
    return {
        "w": w, "h": h,
        "f": _f("f"),
        "cx_norm": _f("cx"), "cy_norm": _f("cy"),
        "k1": _f("k1"), "k2": _f("k2"), "k3": _f("k3"), "k4": _f("k4"),
        "p1": _f("p1"), "p2": _f("p2"),
        "projection": root.find("projection").text,
    }


def _metashape_equisolid_to_colmap(ms, side="left", max_angle_deg=120, n=3000):
    """
    Convert Metashape EquisolidFisheye -> COLMAP OPENCV_FISHEYE (4-param Kannala-Brandt).

    Metashape: r = 2*f*sin(theta/2) * (1 + k1*(r/f)^2 + k2*(r/f)^4 + k3*(r/f)^6)
               k1,k2,k3 are Brown-model (r-space) radial distortions.

    COLMAP OPENCV_FISHEYE (Kannala-Brandt):
               theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9

    Strategy: sample Metashape projection at N angles, fit KB coefficients
              via least-squares. Max fit error < 0.5px.
    """
    f_val = ms["f"]
    k1_ms = ms["k1"]
    k2_ms = ms["k2"]
    k3_ms = ms["k3"]
    k4_ms = ms.get("k4", 0.0) or 0.0
    w = ms["w"]
    h = ms["h"]

    max_angle = np.radians(max_angle_deg)
    theta = np.linspace(0.001, max_angle, n)

    # Metashape EquisolidFisheye projection (normalized by f)
    r_equisolid = 2 * np.sin(theta / 2)
    rn = r_equisolid  # normalized radius = r/f

    # Brown radial distortion in r-space (4th order)
    r_dist = rn * (1 + k1_ms*rn**2 + k2_ms*rn**4 + k3_ms*rn**6 + k4_ms*rn**8)
    theta_d = r_dist  # already normalized by f

    # Fit 4-param KB: theta_d = theta + k1*th^3 + k2*th^5 + k3*th^7 + k4*th^9
    residual = theta_d - theta
    A = np.column_stack([theta**3, theta**5, theta**7, theta**9])
    coeffs, _, _, _ = np.linalg.lstsq(A, residual, rcond=None)

    kb_proj = theta + coeffs[0]*theta**3 + coeffs[1]*theta**5 + coeffs[2]*theta**7 + coeffs[3]*theta**9
    err = np.max(np.abs(theta_d - kb_proj)) * f_val  # px

    # v110: Per-camera sign correction for MS Equisolid->KB fitting.
    # The KB fitting produces opposite signs for LEFT vs RIGHT cameras.
    # Empirical ground truth: both cameras need NEGATIVE k1.
    # - LEFT: KB fit produces negative k1 → correct sign
    # - RIGHT: KB fit produces positive k1 → must negate
    if side == "right":
        coeffs = -coeffs  # negate all 4 coefficients for RIGHT camera
    
    return {
        "f": f_val,
        "cx": w / 2.0 + ms["cx_norm"],
        "cy": h / 2.0 + ms["cy_norm"],
        "k1": float(coeffs[0]), "k2": float(coeffs[1]), "k3": float(coeffs[2]), "k4": float(coeffs[3]),
        "k5": 0.0, "k6": 0.0, "p1": 0.0, "p2": 0.0,
        "w": w, "h": h, "max_px_err": err,
    }


def convert_polyfisheye_to_kb(yaml_cal, side="left"):
    """
    Convert PCS POLYFISHEYE factory calibration -> COLMAP OPENCV_FISHEYE (Kannala-Brandt).

    Args:
        yaml_cal: dict with keys fx, fy, cx, cy, w, h, k1..k6
        side: "left" or "right" — used for side-specific calibration lookup

    PCS POLYFISHEYE model: r(theta) = f*theta + k2*theta^2 + k3*theta^3 + k4*theta^4 + ...
    where theta is the incident angle (radians) and r is the pixel radius.
    The YAML stores coefficients k2..k7; the hardcoded _C1_*_CAL dicts map
    YAML k2->k1, k3->k2 and zero k3..k6 (losing higher-order information).

    COLMAP OPENCV_FISHEYE (Kannala-Brandt):
        theta_d = theta + k1*theta^3 + k2*theta^5 + k3*theta^7 + k4*theta^9

    Since POLYFISHEYE has even+odd powers but KB only odd powers, no exact
    mapping exists.  We curve-fit: sample POLYFISHEYE at N angles, then
    least-squares fit KB coefficients.  Expected max error ~1-2 px.

    Args:
        yaml_cal: dict with keys fx, fy, cx, cy, w, h, k1..k6
                  (where k1..k6 are the POLYFISHEYE angle coefficients,
                   previously mapped from YAML k2..k7)

    Returns:
        dict with f, cx, cy, k1..k4, w, h, max_px_err  (same shape as
        _metashape_equisolid_to_colmap output)
    """
    if np is None:
        return None

    w = yaml_cal["w"]
    h = yaml_cal["h"]
    f_val = (yaml_cal["fx"] + yaml_cal["fy"]) / 2.0
    cx = yaml_cal["cx"]
    cy = yaml_cal["cy"]

    # POLYFISHEYE r(theta) = f*theta + k1*theta^2 + k2*theta^3 + k3*theta^4 + ...
    # The hardcoded dicts use: k1 = YAML k2, k2 = YAML k3, k3..k6 = 0
    # We use ALL non-zero coefficients for best sampling accuracy.
    poly_coeffs = []
    for i in range(1, 7):  # k1..k6 in our dict -> theta^2..theta^7
        poly_coeffs.append(yaml_cal.get(f"k{i}", 0.0))

    max_angle = np.radians(120.0)  # C1 FOV half-angle ~120 deg
    theta = np.linspace(0.001, max_angle, 3000)

    # Evaluate POLYFISHEYE projection: r = f*theta + sum(k_i * theta^(i+1))
    r_poly = f_val * theta  # linear base term
    for idx, ki in enumerate(poly_coeffs):
        if abs(ki) > 1e-15:
            r_poly = r_poly + ki * f_val * theta ** (idx + 2)

    # Normalize to get "distorted angle" equivalent: theta_d = r / f
    theta_d = r_poly / f_val

    # Fit KB: theta_d = theta + k1*th^3 + k2*th^5 + k3*th^7 + k4*th^9
    residual = theta_d - theta
    A = np.column_stack([theta**3, theta**5, theta**7, theta**9])
    coeffs, _, _, _ = np.linalg.lstsq(A, residual, rcond=None)

    kb_proj = theta + coeffs[0]*theta**3 + coeffs[1]*theta**5 + coeffs[2]*theta**7 + coeffs[3]*theta**9
    err = float(np.max(np.abs(theta_d - kb_proj)) * f_val)  # px

    # v106: NEGATE ALL distortion coefficients to match Metashape XML ground truth.
    # Both left and right cameras have negative k1 (pincushion) in Metashape.
    return {
        "f": f_val,
        "cx": cx, "cy": cy,
        "k1": float(-coeffs[0]),
        "k2": float(-coeffs[1]),
        "k3": float(-coeffs[2]),
        "k4": float(-coeffs[3]),
        "w": w, "h": h, "max_px_err": err,
    }


def generate_fisheye_circle_mask(src_mask_path, dst_path, cx, cy, w, h):
    """
    Generate a binary fisheye circle mask and composite it onto an existing PCS mask.

    - Loads the source mask (if exists) as base: white=masked, black=unmasked.
    - Computes a circular fisheye boundary from intrinsics (cx, cy).
    - Sets all pixels OUTSIDE the circle to white (masked).
    - Saves as binary black/white .png (COLMAP convention).

    Args:
        src_mask_path: Path to existing PCS mask (may not exist)
        dst_path: Output path for the composited mask (.png)
        cx, cy: Principal point (fisheye circle center) in pixels
        w, h: Image dimensions in pixels
    """
    if np is None:
        print("  [MASK] ERROR: numpy not installed. Cannot generate fisheye circle masks.")
        print("  [MASK] Run: pip install numpy")
        return False

    # Fisheye circle radius: based on HALF the image width, so the circle
    # covers the full horizontal extent of the rectangular frame.
    # Top/bottom may extend slightly outside the image boundary (safe -- those
    # pixels are already outside the image and won't appear in output).
    radius = max(cx, w - cx)  # distance from center to left/right edge

    # Create coordinate grids
    yy, xx = np.ogrid[:h, :w]
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    circle_mask = dist_sq <= radius ** 2  # True inside circle (usable area)

    # Load base mask if it exists (white=masked, black=unmasked)
    base = None
    if src_mask_path and src_mask_path.is_file():
        try:
            # Read as grayscale
            from PIL import Image as PILImage
            _img = np.array(PILImage.open(src_mask_path).convert('L'))
            if _img.shape == (h, w):
                base = _img
            else:
                print(f"  [MASK] WARNING: src mask size {_img.shape} != image ({w}x{h}), ignoring")
        except Exception as e:
            print(f"  [MASK] WARNING: could not read {src_mask_path.name}: {e}")

    # Build output mask: INVERTED (LFS) convention
    #   white (255) = kept / unmasked
    #   black (0)   = masked out / excluded
    output = np.full((h, w), 255, dtype=np.uint8)  # start all white (kept)

    # Apply base mask first (if any): bright areas in base = masked
    if base is not None:
        # Base mask uses COLMAP convention: bright = mask this
        # For LFS (inverted): bright in base → black (masked)
        output[base > 128] = 0

    # Apply fisheye circle: everything OUTSIDE circle -> black (masked)
    output[~circle_mask] = 0

    # Save as PNG (binary B&W only)
    try:
        from PIL import Image as PILImage
        result = PILImage.fromarray(output, mode='L')
        result.save(str(dst_path))
        return True
    except Exception as e:
        print(f"  [MASK] ERROR saving {dst_path.name}: {e}")
        return False



# Hardcoded factory calibration for C1 (QWH project) — v90
# Source: H:\ShareC1\Qwh\info\calibration.yaml  (PCS POLYFISHEYE model)
# Format: FULL_OPENCV  (k1-k6 radial, p1-p2 tangential = 0)

_C1_LEFT_CAL = {
    "w": 1920, "h": 1200,
    "fx": 636.96450423255123, "fy": 636.96801724603415,
    "cx": 961.49358628954531, "cy": 594.51938204973385,
    # YAML k2..k7 -> FULL_OPENCV k1..k6
    # v92: only map k2->k1, k3->k2 (best radial approximation)
    # Remaining POLYFISHEYE coeffs (k4-k7) are higher-order angle-based
    # terms with no equivalent in Brown/radial model -> zeroed
    "k1": -1.4030199487616990e-02,  # YAML k2
    "k2":  3.1807189284775481e-02,  # YAML k3
    "k3":  0.0,                      # zeroed
    "k4":  0.0,                      # zeroed
    "k5":  0.0,                      # zeroed
    "k6":  0.0,                      # zeroed
    "p1": 0.0, "p2": 0.0,           # no tangential data in fisheye model
}
_C1_RIGHT_CAL = {
    "w": 1920, "h": 1200,
    "fx": 638.13494931286232, "fy": 638.12328651852982,
    "cx": 971.17679545559401, "cy": 594.10582562826501,
    # v97: negated for LFS RADIAL_FISHEYE (works with left camera sign convention)
    "k1": -6.2427827414271919e-03,  # YAML k2, negated
    "k2":  4.7126249167536625e-02,  # YAML k3, negated
    "k3":  0.0,                      # zeroed
    "k4":  0.0,                      # zeroed
    "k5":  0.0,                      # zeroed
    "k6":  0.0,                      # zeroed
    "p1": 0.0, "p2": 0.0,           # no tangential data in fisheye model
}

# v101: RS2 FULL_OPENCV uses NEGATED original YAML right values
_C1_RIGHT_CAL_RS2 = {
    "w": 1920, "h": 1200,
    "fx": 638.13494931286232, "fy": 638.12328651852982,
    "cx": 971.17679545559401, "cy": 594.10582562826501,
    # v101: negate original YAML right camera values for RS2
    "k1": -6.2427827414271919e-03,  # YAML k2, negated
    "k2":  4.7126249167536625e-02,  # YAML k3, negated
    "k3":  0.0,                      # zeroed
    "k4":  0.0,                      # zeroed
    "k5":  0.0,                      # zeroed
    "k6":  0.0,                      # zeroed
    "p1": 0.0, "p2": 0.0,           # no tangential data in fisheye model
}


def get_c1_hardcoded_calibration(side, viewer_conventions="LFS"):
    """Return hardcoded C1 calibration for 'left' or 'right'.
    
    For LFS: right camera uses negated k1/k2 (works with RADIAL_FISHEYE)
    For RS2: right camera uses original k1/k2 (Brown model expects different signs)
    """
    if "left" in side.lower():
        return _C1_LEFT_CAL
    # v98: RS2 uses original (non-negated) values for FULL_OPENCV
    if viewer_conventions == "RS2":
        return _C1_RIGHT_CAL_RS2
    return _C1_RIGHT_CAL


def load_opt_file(path):
    """
    Parse SHARE3DCAM .opt calibration file -?dict.
    The .opt file is XML. Observed format (C1 undistort export):
      <OpticalProperties version="1.0">
        <ImageDimensions>
          <Width>2877</Width>
          <Height>1798</Height>
        </ImageDimensions>
        <SensorSize>5.76</SensorSize>
        <FocalLength>1.27526</FocalLength>
        <PrincipalPoint>
          <X>1438</X>
          <Y>899</Y>
        </PrincipalPoint>
        <Distortion>
          <K1>0.0</K1>  <K2>0.0</K2>  <K3>0.0</K3>
          <P1>0.0</P1>  <P2>0.0</P2>
        </Distortion>
        ...
      </OpticalProperties>
    Also supports attribute-style variants (width="..." height="...")
    and legacy flat key=value format as fallback.
    Returns: dict with keys ImageWidth, ImageHeight, SensorSize,
             FocalLength, PrincipalPointX, PrincipalPointY, K1..K3, P1..P2.
    """
    params = {}
    if not path or not os.path.exists(path):
        print(f"  [OPT] file not found: {path}")
        return params
    print(f"  [OPT] parsing {path}")

    raw = ""
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()

    # -- Try XML parsing first ----------------------------------------------
    if raw.startswith("<?xml") or raw.startswith("<OpticalProperties"):
        try:
            tree = ET.fromstring(raw)

            # -- ImageDimensions ----------------------------------------
            img_dim = tree.find("ImageDimensions")
            if img_dim is not None:
                # Variant A: child elements <Width>/<Height>
                w_el = img_dim.find("Width")
                h_el = img_dim.find("Height")
                if w_el is not None and h_el is not None:
                    params["ImageWidth"]  = _xml_float(w_el) or 0.0
                    params["ImageHeight"] = _xml_float(h_el) or 0.0
                # Variant B: attributes width="..." height="..."
                elif img_dim.get("width") and img_dim.get("height"):
                    params["ImageWidth"]  = float(img_dim.get("width", 0))
                    params["ImageHeight"] = float(img_dim.get("height", 0))
                if "ImageWidth" in params and "ImageHeight" in params:
                    params["ImageDimensions"] = (
                        f"{int(params['ImageWidth'])}x{int(params['ImageHeight'])}")

            # -- SensorSize ---------------------------------------------
            val = _xml_float(tree.find("SensorSize"))
            if val is not None:
                params["SensorSize"] = val

            # -- FocalLength --------------------------------------------
            val = _xml_float(tree.find("FocalLength"))
            if val is not None:
                params["FocalLength"] = val

            # -- PrincipalPoint -----------------------------------------
            pp = tree.find("PrincipalPoint")
            if pp is not None:
                # Variant A: child elements <X>/<Y>
                x_el = pp.find("X")
                y_el = pp.find("Y")
                if x_el is not None and y_el is not None:
                    params["PrincipalPointX"] = _xml_float(x_el) or 0.0
                    params["PrincipalPointY"] = _xml_float(y_el) or 0.0
                # Variant B: attributes x="..." y="..."
                elif pp.get("x") is not None and pp.get("y") is not None:
                    params["PrincipalPointX"] = float(pp.get("x", 0))
                    params["PrincipalPointY"] = float(pp.get("y", 0))

            # -- Distortion (nested or flat) ----------------------------
            dist = tree.find("Distortion")
            dist_parent = dist if dist is not None else tree
            for tag in ("K1", "K2", "K3", "P1", "P2"):
                el = dist_parent.find(tag)
                if el is None:
                    # Also try lowercase
                    el = dist_parent.find(tag.lower())
                if el is not None:
                    val = _xml_float(el)
                    if val is not None:
                        params[tag.upper()] = val

            # -- Fisheye models ---------------------------------------------
            # FisheyeFocalMatrix (2x2 affine): M_00, M_01, M_10, M_11 (ELEMENT-style, not attribute!)
            ffm = tree.find("FisheyeFocalMatrix")
            if ffm is not None:
                params["FisheyeFocalMatrix"] = {}
                for key in ["M_00", "M_01", "M_10", "M_11"]:
                    el = ffm.find(key)
                    if el is not None:
                        try: params["FisheyeFocalMatrix"][key] = float(_xml_text(el))
                        except: pass
            # FisheyePolynomial: k1, k2, k3, k4 (ELEMENT-style)
            fpoly = tree.find("FisheyePolynomial")
            if fpoly is not None:
                params["FisheyePolynomial"] = {}
                for key in ["k1", "k2", "k3", "k4"]:
                    el = fpoly.find(key)
                    if el is not None:
                        try: params["FisheyePolynomial"][key] = float(_xml_text(el))
                        except: pass
            # FisheyeAffine: b1, b2 (ELEMENT-style)
            faff = tree.find("FisheyeAffine")
            if faff is not None:
                params["FisheyeAffine"] = {}
                for key in ["b1", "b2"]:
                    el = faff.find(key)
                    if el is not None:
                        try: params["FisheyeAffine"][key] = float(_xml_text(el))
                        except: pass

            # -- CameraModel / CameraModelType --------------------------
            for tag in ("CameraModel", "CameraModelType"):
                cm = tree.find(tag)
                if cm is not None:
                    txt = cm.get("value") or _xml_text(cm) or ""
                    if txt:
                        params["CameraModel"] = txt
                        break

            print(f"  [OPT] XML keys: {sorted(params.keys())}")
            return params
        except ET.ParseError:
            pass  # Fall through to legacy parser

    # -- Legacy flat-format fallback ----------------------------------------
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
            else:
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                key, val = parts
            try:
                params[key.strip()] = float(val.strip())
            except ValueError:
                params[key.strip()] = val.strip()
    print(f"  [OPT] legacy flat keys: {sorted(params.keys())}")
    return params


def load_intrinsic_txt(path):
    """
    Parse *_undistort_intrinsic.txt (post-undistortion pixel intrinsics).
    Handles common formats:
      - key=value pairs:  fx = 1423.5
      - tab/space cols:   fx    fy    cx    cy
      - single-line:      fx fy cx cy
    Returns: {fx, fy, cx, cy, w, h} or {} on failure.
    """
    if not path or not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        raw = f.read()

    vals = {}
    # Try key=value pairs
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            k, v = k.strip().lower(), v.strip()
            if k in ("fx", "fy", "cx", "cy", "w", "h", "width", "height",
                     "focal_length", "focal", "image_width", "image_height"):
                try:
                    vals[k] = float(v)
                except ValueError:
                    pass

    # Try space/tab separated columns (first 4+ floats)
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" in line:
            continue
        parts = line.replace("\t", " ").split()
        floats = [float(p) for p in parts if _is_float(p)]
        if len(floats) >= 4:
            keys = ["fx", "fy", "cx", "cy"]
            for i, k in enumerate(keys):
                if k not in vals:
                    vals[k] = floats[i]
            if len(floats) >= 6 and ("w" not in vals and "h" not in vals):
                vals["w"] = int(floats[4])
                vals["h"] = int(floats[5])

    # Try single-line: fx fy cx cy
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        floats = [float(p) for p in parts if _is_float(p)]
        if len(floats) == 4 and floats[0] > 100 and floats[1] > 100:
            # Likely fx fy cx cy in pixels
            for i, k in enumerate(["fx", "fy", "cx", "cy"]):
                if k not in vals:
                    vals[k] = floats[i]

    # Normalise aliases
    for alias, canon in [("focal_length", "fx"), ("focal", "fx"),
                         ("width", "w"), ("height", "h"),
                         ("image_width", "w"), ("image_height", "h")]:
        if alias in vals and canon not in vals:
            vals[canon] = vals[alias]

    if "fx" in vals:
        print(f"    Parsed intrinsic.txt: fx={vals.get('fx')}, "
              f"fy={vals.get('fy')}, cx={vals.get('cx')}, cy={vals.get('cy')}")
    return vals


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_imgpose(path):
    """
    Load ImgPose.txt -?dict: basename -?pose dict.
    Columns: filename x y z roll pitch yaw qx qy qz qw timestamp
    Automatically skips header rows (non-numeric first data field).
    """
    if not path or not os.path.exists(path):
        return {}
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            # Skip header rows where coordinate fields are not numeric
            # (e.g. "filename x y z roll pitch yaw qx qy qz qw timestamp")
            try:
                key = os.path.basename(parts[0])
                records[key] = {
                    "x": float(parts[1]), "y": float(parts[2]), "z": float(parts[3]),
                    "roll": float(parts[4]), "pitch": float(parts[5]), "yaw": float(parts[6]),
                    "qx": float(parts[7]), "qy": float(parts[8]),
                    "qz": float(parts[9]), "qw": float(parts[10]),
                    "timestamp": float(parts[11]),
                }
            except ValueError:
                continue
    return records


def load_xyzopk(path):
    """
    Load xyzopk.txt / xyzopt.txt -?dict: basename -?{x, y, z, omega, phi, kappa}.
    Columns: filename X Y Z Omega Phi Kappa
    (Omega, Phi, Kappa in degrees -?standard photogrammetry convention)

    Both formats share the same structure; both names are supported.
    """
    if not path or not os.path.exists(path):
        return {}
    records = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 7:
                continue
            try:
                key = os.path.basename(parts[0])
                records[key] = {
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "z": float(parts[3]),
                    "omega": float(parts[4]),
                    "phi":   float(parts[5]),
                    "kappa": float(parts[6]),
                }
            except ValueError:
                continue
    return records


# -----------------------------------------------------------------------------
# Auto-discovery
# -----------------------------------------------------------------------------

def find_pcs_files(folder):
    """
    Scan a PCS undistort folder and return a dict of found files.
    Tries multiple naming variants to handle different PCS versions.
    """
    f = {}
    folder = Path(folder)

    def try_find(patterns):
        for p in patterns:
            hits = list(folder.glob(p))
            if hits:
                return str(hits[0].resolve())
        return None

    # TransformedCam.json
    f["json"] = try_find(["TransformedCam.json",
                           "*/TransformedCam.json"])

    # Left / Right .opt files (also plain Left.opt/Right.opt for fisheye cameras)
    f["opt_left"]  = try_find(["Left.opt", "Left_undistort.opt",   "*Left_undistort.opt",
                                "left.opt", "left_undistort.opt",   "*left_undistort.opt"])
    f["opt_right"] = try_find(["Right.opt", "Right_undistort.opt",  "*Right_undistort.opt",
                                "right.opt", "right_undistort.opt",  "*right_undistort.opt"])

    # Intrinsic.txt -?try both singular and plural spellings
    f["intrinsic_left"]  = try_find(["left_undistort_intrinsic.txt",
                                      "*left_undistort_intrinsic.txt",
                                      "Left_undistort_intrinsic.txt",
                                      "*Left_undistort_intrinsic.txt",
                                      "left_undistort_intrinsics.txt",
                                      "*left_undistort_intrinsics.txt"])
    f["intrinsic_right"] = try_find(["right_undistort_intrinsics.txt",
                                      "*right_undistort_intrinsics.txt",
                                      "Right_undistort_intrinsic.txt",
                                      "*Right_undistort_intrinsic.txt",
                                      "right_undistort_intrinsic.txt",
                                      "*right_undistort_intrinsic.txt"])

    # ImgPose.txt and xyzopt/xyzopk
    f["imgpose"]  = try_find(["ImgPose.txt",  "*/ImgPose.txt"])
    f["xyzopk"]   = try_find(["xyzopk.txt", "xyzopt.txt",
                               "*/xyzopk.txt", "*/xyzopt.txt"])

    return f


def print_discovery(files):
    print("\n[*]  Auto-discovered files:")
    labels = [
        ("json",           "TransformedCam.json"),
        ("opt_left",       "Left_undistort.opt"),
        ("opt_right",      "Right_undistort.opt"),
        ("intrinsic_left", "left_undistort_intrinsic.txt"),
        ("intrinsic_right","right_undistort_intrinsics.txt"),
        ("imgpose",        "ImgPose.txt"),
        ("xyzopk",         "xyzopk.txt"),
    ]
    found = sum(1 for k, _ in labels if files.get(k))
    print(f"   {found}/{len(labels)} files found")
    for key, name in labels:
        status = "-- " + ("OK" if files.get(key) else "MISSING")
        val = files.get(key, "-")
        print(f"   {status}  {name:<35} {val}")
    print()


# -----------------------------------------------------------------------------
# Intrinsics resolution
# -----------------------------------------------------------------------------

def resolve_intrinsics(frames, intrinsic_txt_params, opt_params, label="cam", fisheye=False, yaml_cal=None, viewer_conventions="RS2", metashape_ms=None):
    """
    Resolve camera intrinsics from available sources (in priority order):
      1. TransformedCam.json frame data (fl_x, cx, cy, w, h, distortion coeffs)
      2. *_undistort_intrinsic.txt (post-undistort fx, fy, cx, cy in pixels)
      3. *.opt file (mm focal length -?pixel conversion via sensor width)

    Returns: (model, fx, fy, cx, cy, k1, k2, p1, p2, k3, w, h)
    """
    # -- 0. Fisheye from .opt (check FIRST when fisheye flag set) --------
    if fisheye:
        if not opt_params:
            print(f"  [{label}] FISHEYE DEBUG: opt_params is None or empty")
        elif "FisheyeFocalMatrix" not in opt_params:
            print(f"  [{label}] FULL_OPENCV DEBUG: FisheyeFocalMatrix NOT in opt_params. Keys={list(opt_params.keys())}")
        else:
            ffmp = opt_params["FisheyeFocalMatrix"]
            print(f"  [{label}] FULL_OPENCV DEBUG: FisheyeFocalMatrix keys={list(ffmp.keys())}")
    # -- 0b. SKIP Metashape XML calibration (v1.11: MS Equisolid->KB fitting broken)
    # The KB fitting produces wrong focal lengths AND wrong k1 values.
    # Empirical ground truth: f≈637, k1≈-0.019
    # MS-XML fitting: f≈623/633, k1≈-0.045/-0.053 (completely wrong)
    # Fall through to YAML/hardcoded path instead.
    if fisheye and metashape_ms is not None:
        print(f"  [{label}] WARNING: Metashape XML calibration skipped (Equisolid->KB fitting unreliable)")
        # ms_cal = _metashape_equisolid_to_colmap(metashape_ms, side=label)
        # ... MS-XML path disabled ...

    # -- 0c. YAML factory calibration (fallback for fisheye) ----------------
    # v103: curve-fit POLYFISHEYE -> Kannala-Brandt for LFS/PS;
    #        RS2 still uses FULL_OPENCV with direct coefficient mapping
    if fisheye and yaml_cal:
        if viewer_conventions in ("LFS", "PS"):
            kb = convert_polyfisheye_to_kb(yaml_cal, side=label)
            if kb is not None:
                print(f"  [{label}] Fisheye YAML curve-fit: {kb['w']}x{kb['h']}, "
                      f"f={kb['f']:.4f}, cx={kb['cx']:.4f}, cy={kb['cy']:.4f}, "
                      f"model=OPENCV_FISHEYE (k1={kb['k1']:.6f}, k2={kb['k2']:.6f}, "
                      f"k3={kb['k3']:.6f}, k4={kb['k4']:.6f}, fit_err={kb['max_px_err']:.2f}px)")
                return ("OPENCV_FISHEYE",
                        kb["f"], kb["f"], kb["cx"], kb["cy"],
                        kb["k1"], kb["k2"], 0.0, 0.0,
                        kb["k3"], kb["k4"], 0.0, 0.0,
                        kb["w"], kb["h"])
            else:
                print(f"  [{label}] WARNING: numpy unavailable, falling back to direct YAML mapping")
        # RS2: use FULL_OPENCV (Brown, 12 params) — direct coefficient mapping
        yc = yaml_cal
        w, h   = yc["w"], yc["h"]
        fx, fy = yc["fx"], yc["fy"]
        cx, cy = yc["cx"], yc["cy"]
        k1, k2, k3, k4, k5, k6 = yc["k1"], yc["k2"], yc["k3"], yc["k4"], yc["k5"], yc["k6"]
        p1, p2 = 0.0, 0.0
        model = "FULL_OPENCV"
        print(f"  [{label}] Fisheye from YAML (RS2): {w}x{h}, "
              f"fx={fx:.2f}, fy={fy:.2f}, model={model} "
              f"(k1={k1:.4f}..k6={k6:.4f}  [from calibration.yaml])")
        return model, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, w, h

    if fisheye and opt_params and "FisheyeFocalMatrix" in opt_params:
        ffmp = opt_params["FisheyeFocalMatrix"]
        w = int(opt_params.get("ImageWidth", 0))
        h = int(opt_params.get("ImageHeight", 0))
        if w == 0 or h == 0:
            for fr in frames:
                if fr.get("w", 0) and fr.get("h", 0):
                    w = int(fr["w"]); h = int(fr["h"]); break
        if w and h and "M_00" in ffmp and "M_11" in ffmp:
            fx = ffmp["M_00"]
            fy = ffmp["M_11"]
            # Principal point from .opt or default to image center
            if "PrincipalPointX" in opt_params and "PrincipalPointY" in opt_params:
                cx = opt_params["PrincipalPointX"]
                cy = opt_params["PrincipalPointY"]
            else:
                cx, cy = w / 2.0, h / 2.0
            # FisheyePolynomial: k1, k2, k3, k4
            fpoly = opt_params.get("FisheyePolynomial", {})
            k1 = fpoly.get("k1", 0.0)
            k2 = fpoly.get("k2", 0.0)
            k3 = fpoly.get("k3", 0.0)
            k4 = fpoly.get("k4", 0.0)
            # RS2 uses FISHEYE model
            model = "FULL_OPENCV"
            p1 = 0.0; p2 = 0.0; k5 = 0.0; k6 = 0.0
            print(f"  [{label}] Fisheye from .opt  : {w}x{h}, "
                  f"fx={fx:.2f}, fy={fy:.2f}, model=FULL_OPENCV "
                  f"(k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}, k4={k4:.4f}, p1=p2=k5=k6=0)")
            return model, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, w, h

    # -- 1. From JSON frame --------------------------------------------
    for fr in frames:
        if fr.get("fl_x", 0) != 0 and fr.get("w", 0) != 0:
            w, h   = int(fr["w"]), int(fr["h"])
            fx, fy = fr["fl_x"], fr.get("fl_y", fr["fl_x"])
            cx, cy = fr["cx"], fr["cy"]
            k1 = fr.get("k1", 0.0); k2 = fr.get("k2", 0.0)
            k3 = fr.get("k3", 0.0)
            p1 = fr.get("p1", 0.0); p2 = fr.get("p2", 0.0)
            model = "FULL_OPENCV"
            print(f"  [{label}] Intrinsics from JSON  : {w}x{h}, "
                  f"fx={fx:.2f}, fy={fy:.2f}, model=FULL_OPENCV")
            return model, fx, fy, cx, cy, k1, k2, p1, p2, k3, 0.0, 0.0, 0.0, w, h
    if intrinsic_txt_params and "fx" in intrinsic_txt_params:
        w = int(intrinsic_txt_params.get("w", 0))
        h = int(intrinsic_txt_params.get("h", 0))
        # Try to get dimensions from a frame if not in txt
        if w == 0 or h == 0:
            for fr in frames:
                if fr.get("w", 0) and fr.get("h", 0):
                    w = int(fr["w"]); h = int(fr["h"]); break
        if w and h:
            fx = intrinsic_txt_params["fx"]
            fy = intrinsic_txt_params.get("fy", fx)
            cx = intrinsic_txt_params.get("cx", w / 2.0)
            cy = intrinsic_txt_params.get("cy", h / 2.0)
            model = "SIMPLE_PINHOLE"
            print(f"  [{label}] Intrinsics from txt   : {w}x{h}, "
                  f"fx={fx:.2f}, fy={fy:.2f}, model=SIMPLE_PINHOLE")
            return model, fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, w, h

    # -- 3. From .opt file ----------------------------------------------------
    if opt_params and "FocalLength" in opt_params and "SensorSize" in opt_params:
        fl_mm  = opt_params["FocalLength"]
        sensor = opt_params["SensorSize"]
        w = int(opt_params.get("ImageWidth",  0))
        h = int(opt_params.get("ImageHeight", 0))
        if w == 0 or h == 0:
            for fr in frames:
                if fr.get("w", 0) and fr.get("h", 0):
                    w = int(fr["w"]); h = int(fr["h"]); break
        if w and h:
            px_per_mm = w / sensor
            fx = fy = fl_mm * px_per_mm
            # Use principal point from .opt if available, else image center
            if "PrincipalPointX" in opt_params and "PrincipalPointY" in opt_params:
                cx = opt_params["PrincipalPointX"]
                cy = opt_params["PrincipalPointY"]
            else:
                cx, cy = w / 2.0, h / 2.0
            # Distortion from .opt if available
            k1 = opt_params.get("K1", 0.0)
            k2 = opt_params.get("K2", 0.0)
            k3 = opt_params.get("K3", 0.0)
            p1 = opt_params.get("P1", 0.0)
            p2 = opt_params.get("P2", 0.0)
            has_distortion = any(v != 0.0 for v in (k1, k2, k3, p1, p2))
            model = "FULL_OPENCV" if has_distortion else "SIMPLE_PINHOLE"
            print(f"  [{label}] Intrinsics from .opt  : {w}x{h}, "
                  f"fx={fx:.2f} px  (fl={fl_mm}mm, sensor={sensor}mm)"
                  f"{', model=FULL_OPENCV (distortion)' if has_distortion else ''}")
            return model, fx, fy, cx, cy, k1, k2, p1, p2, k3, 0.0, 0.0, 0.0, w, h
    print(f"       JSON frames: {len(frames)}, first has fl_x={frames[0].get('fl_x',0)}")
    print(f"       intrinsic.txt: {intrinsic_txt_params}")
    print(f"       .opt file: {dict(opt_params) if opt_params else 'not found'}")
    raise ValueError(
        f"[{label}] Cannot resolve camera intrinsics.\n"
        "  Please supply --opt-left / --opt-right manually, or ensure\n"
        "  your TransformedCam.json contains valid fl_x / w / h values."
    )


# -----------------------------------------------------------------------------
# COLMAP output builders
# -----------------------------------------------------------------------------

def cameras_line(cam_id, model, w, h, fx, fy, cx, cy, k1, k2, p1, p2, k3, k4=None, k5=0.0, k6=0.0):
    """One cameras.txt entry."""
    if model == "SIMPLE_PINHOLE":
        return f"{cam_id} {model} {w} {h} {fx:.8f} {cx:.8f} {cy:.8f}\n"
    elif model == "SIMPLE_RADIAL":
        return f"{cam_id} {model} {w} {h} {fx:.8f} {cx:.8f} {cy:.8f} {k1:.8f}\n"
    elif model == "RADIAL":
        return f"{cam_id} {model} {w} {h} {fx:.8f} {cx:.8f} {cy:.8f} {k1:.8f} {k2:.8f}\n"
    elif model == "OPENCV":
        return (f"{cam_id} {model} {w} {h} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f} "
                f"{k1:.8f} {k2:.8f} {p1:.8f} {p2:.8f}\n")
    elif model == "FULL_OPENCV":
        return (f"{cam_id} {model} {w} {h} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f} "
                f"{k1:.8f} {k2:.8f} {p1:.8f} {p2:.8f} {k3:.8f} {k4:.8f} {k5:.8f} {k6:.8f}\n")
    elif model == "EQUISOLID_FISHEYE":
        return f"{cam_id} {model} {w} {h} {fx:.8f} {cx:.8f} {cy:.8f} {k1:.8f}\n"
    elif model == "RADIAL_FISHEYE":
        f_avg = (fx + fy) / 2.0
        return (f"{cam_id} {model} {w} {h} {f_avg:.8f} {cx:.8f} {cy:.8f} "
                f"{k1:.8f} {k2:.8f}\n")
    elif model in ("OPENCV_FISHEYE", "Fisheye", "FISHEYE"):
        k4_val = k4 if k4 is not None else 0.0
        return (f"{cam_id} {model} {w} {h} {fx:.8f} {fy:.8f} {cx:.8f} {cy:.8f} "
                f"{k1:.8f} {k2:.8f} {k3:.8f} {k4_val:.8f}\n")
    else:
        return f"{cam_id} PINHOLE {w} {h} {fx:.8f} {cx:.8f} {cy:.8f}\n"


def images_line(img_id, cam_id, name, qw, qx, qy, qz, tx, ty, tz):
    """images.txt: pose line + empty POINTS2D line."""
    return (f"{img_id} {qw:.17g} {qx:.17g} {qy:.17g} {qz:.17g} "
            f"{tx:.17g} {ty:.17g} {tz:.17g} {cam_id} {name}\n\n")


# -----------------------------------------------------------------------------
# .las --points3D.txt  (for LichtFeld / PostShot / NeRF tools)
# -----------------------------------------------------------------------------

def las_to_points3d(las_path, max_points=None, subsample_seed=42,
                    points_axis="xyz",
                    pitch_deg=-90.0, yaw_deg=0.0, roll_deg=0.0):
    """
    Read a LAS/LAZ file and return a list of COLMAP points3D rows.

    Each row: (POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[])
      TRACK[] is empty [] for SLAM-derived dense point clouds.

    RGB encoding in LAS files is typically 0-?5535 (16-bit) but sometimes
    0-?55 (8-bit). We normalise both to 0-?55.

    Args:
        las_path: Path to .las or .laz file.
        max_points: Cap output to N points (None = unlimited).
                     When capped, spatially-uniform subsampling is used
                     (random with fixed seed for reproducibility).
        subsample_seed: Random seed for reproducible subsampling.
        points_axis: Axis permutation string to fix coordinate alignment.
                     Examples: "xyz" (no change), "xzy", "yzx", "zxy", etc.
                     Also supports sign flips: "-xyz", "x-yz", "xy-z", etc.
                     Default "x-y-z" = negate Y and Z (PCS Z-up convention).
        pitch_deg: Rotation around X-axis in degrees (after axis permutation).
                   Default -90deg (PCS Z-up -?COLMAP Y-up combined with axis).
        yaw_deg:   Rotation around Y-axis in degrees (after axis permutation).
                   Positive = CCW when looking down +Y.
        roll_deg:  Rotation around Z-axis in degrees (after axis permutation).
                   Positive = CW when looking toward -Z (right-wing-down).

    Returns:
        List of (x, y, z, r, g, b) tuples.  Returns [] on error or if laspy
        is not installed.
    """
    if not LASPY_AVAILABLE:
        print("  [!!] laspy not installed -?cannot convert .las --points3D.")
        print("       Install with:  pip install laspy numpy")
        return []

    if not las_path or not os.path.exists(las_path):
        return []

    try:
        las = laspy.read(las_path)
    except Exception as e:
        print(f"  [!!] Failed to read {las_path}: {e}")
        return []

    # -- Extract XYZ -------------------------------------------------------
    try:
        xyz = las.xyz
    except AttributeError:
        try:
            xyz = np.column_stack([las.x, las.y, las.z])
        except Exception:
            print(f"  [!!] Could not extract XYZ from {las_path}")
            return []

    n_total = xyz.shape[0]
    print(f"  .las total points : {n_total:,}")

    # -- Apply axis permutation / sign flip ---------------------------------
    #    PCS .las often uses Z-up while COLMAP expects Y-up, so a swap
    #    like "xzy" (output X=x, Y=z, Z=y) fixes the 90deg rotation.
    if points_axis and points_axis.lstrip("= ").lower() != "xyz":
        axis_str = points_axis.lstrip("= ").lower().strip()
        axes_map = {}
        sign_map = {}
        out_idx = 0
        i = 0
        neg_next = False
        while i < len(axis_str):
            ch = axis_str[i]
            if ch == '-':
                neg_next = True
                i += 1
                continue
            if ch not in ('x', 'y', 'z'):
                print(f"  [!!] Invalid --points-axis '{points_axis}': "
                      f"unknown axis '{ch}'. Use x/y/z with optional - prefix.")
                break
            axes_map[out_idx] = 'xyz'.index(ch)
            sign_map[out_idx] = -1.0 if neg_next else 1.0
            neg_next = False
            out_idx += 1
            i += 1
        else:
            if out_idx != 3:
                print(f"  [!!] Invalid --points-axis '{points_axis}': "
                      f"expected 3 axes, got {out_idx}.")
            else:
                # All 3 axes parsed successfully -?apply transformation
                src_cols = [axes_map[i] for i in range(3)]
                signs = np.array([sign_map[i] for i in range(3)])
                xyz_new = np.empty_like(xyz)
                for out_i in range(3):
                    xyz_new[:, out_i] = xyz[:, src_cols[out_i]] * signs[out_i]
                axis_labels = "".join(
                    ('-' if sign_map[i] < 0 else '') + 'xyz'[axes_map[i]]
                    for i in range(3)
                )
                print(f"  .las axis transform: {axis_labels}  "
                      f"(cols {src_cols}, signs {list(signs)})")
                xyz = xyz_new

    # -- Apply Euler rotations (pitch -?yaw -?roll) -------------------------
    #    Applied after axis permutation. Order: X (pitch) -?Y (yaw) -?Z (roll).
    any_rot = (pitch_deg != 0.0) or (yaw_deg != 0.0) or (roll_deg != 0.0)
    if any_rot:
        pitch_rad = math.radians(pitch_deg)
        yaw_rad   = math.radians(yaw_deg)
        roll_rad  = math.radians(roll_deg)

        cp, sp = math.cos(pitch_rad), math.sin(pitch_rad)
        cy, sy = math.cos(yaw_rad),   math.sin(yaw_rad)
        cr, sr = math.cos(roll_rad),  math.sin(roll_rad)

        # Combined rotation matrix R = Rz * Ry * Rx
        # Row-major construction for [x,y,z] column vectors:
        R = np.array([
            [ cy*cr,             sy*sp*cr - cp*sr,     cp*cr*sy + sp*sr ],
            [ cy*sr,             cp*cr + sp*sy*sr,     cp*sy*sr - sp*cr ],
            [-sy,                sp*cy,                 cp*cy            ]
        ])
        xyz = (R @ xyz.T).T

        rot_parts = []
        if pitch_deg != 0.0: rot_parts.append(f"pitch={pitch_deg}deg")
        if yaw_deg != 0.0:   rot_parts.append(f"yaw={yaw_deg}deg")
        if roll_deg != 0.0:  rot_parts.append(f"roll={roll_deg}deg")
        print(f"  .las euler rotation : {', '.join(rot_parts)} "
              f"(order: pitch-yaw-roll)")

    # -- Extract RGB --------------------------------------------------------
    r = g = b = None
    try:
        r = np.asarray(las.red,   dtype=np.float64)
        g = np.asarray(las.green, dtype=np.float64)
        b = np.asarray(las.blue,  dtype=np.float64)
        has_color = True
    except AttributeError:
        has_color = False

    if has_color:
        # Normalise 16-bit -?0-255
        if r.max() > 255:
            r = (r / 65535.0 * 255.0).round().astype(np.uint8)
            g = (g / 65535.0 * 255.0).round().astype(np.uint8)
            b = (b / 65535.0 * 255.0).round().astype(np.uint8)
        else:
            r = r.round().astype(np.uint8)
            g = g.round().astype(np.uint8)
            b = b.round().astype(np.uint8)
        print(f"  .las colour        : RGB  (max raw R={np.asarray(las.red).max()})")
    else:
        # Grey fallback
        print("  .las colour        : none -?using (200, 200, 200)")
        r = g = b = np.full(n_total, 200, dtype=np.uint8)

    # -- Subsample if needed ------------------------------------------------
    if max_points and n_total > max_points:
        rng = np.random.default_rng(subsample_seed)
        indices = rng.choice(n_total, size=max_points, replace=False)
        indices.sort()   # keep file roughly spatially coherent
        xyz = xyz[indices]
        r, g, b = r[indices], g[indices], b[indices]
        print(f"  .las kept          : {max_points:,} points (uniform subsample)")

    # -- Filter out NaN / Inf -----------------------------------------------
    valid = np.isfinite(xyz[:, 0]) & np.isfinite(xyz[:, 1]) & np.isfinite(xyz[:, 2])
    pts = list(zip(
        xyz[valid, 0], xyz[valid, 1], xyz[valid, 2],
        r[valid].tolist(), g[valid].tolist(), b[valid].tolist()
    ))
    n_invalid = int((~valid).sum())
    if n_invalid:
        print(f"  .las removed NaN/Inf: {n_invalid:,} points")
    print(f"  .las final points  : {len(pts):,}")
    return pts


def write_points3d(output_path, points, error=0.0):
    """
    Write COLMAP points3D.txt from a list of (x,y,z,r,g,b) tuples.

    Format:
      POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)
    TRACK is empty (no 2D observations) for SLAM dense point clouds.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        for i, (x, y, z, r, g, b) in enumerate(points, start=1):
            # TRACK is empty -?IMAGE_ID = -1, POINT2D_IDX = -1
            f.write(f"{i} {x:.8f} {y:.8f} {z:.8f} "
                    f"{int(r)} {int(g)} {int(b)} {error}\n")
    # v1.12: No end-of-tape fix needed for RS2 (v2.2 fixed it); keep content as-is
    # v1.12: RS3/LFS still use rstrip but this function is only for points3D.txt with LAS
    with open(output_path, "r", encoding="utf-8") as f:
        content = f.read()
    # Caller applies viewer-specific trailing newline


# -----------------------------------------------------------------------------
# Frame splitting
# -----------------------------------------------------------------------------

def split_frames(frames):
    """Split mixed frames by file_path prefix -?(left_frames, right_frames)."""
    left, right = [], []
    for fr in frames:
        fp = fr.get("file_path", "").replace("\\", "/").lower()
        if fp.startswith("right/") or fp.startswith("r/"):
            right.append(fr)
        else:
            left.append(fr)
    return left, right


# -----------------------------------------------------------------------------
# Main conversion
# -----------------------------------------------------------------------------

def convert(folder=None,
            left_json=None, right_json=None,
            opt_left=None, opt_right=None,
            intrinsic_left=None, intrinsic_right=None,
            imgpose_path=None, use_xyzopk=False,
            output_dir=None,
            no_junction=False,
            las_path=None,
            las_max_points=None,
            points_axis="xyz",
            pitch_deg=90.0, yaw_deg=-90.0, roll_deg=0.0,
            files=None,
            viewer_conventions="RS2",
            camera_axis=None,
            camera_pitch_deg=None, camera_yaw_deg=None, camera_roll_deg=None,
            swap_lr=False,
            fisheye=False,
            fisheye_left_opt=None, fisheye_right_opt=None,
            points3d_trailing_newlines=None,
            calibration_yaml=None,
            metashape_left_xml=None,
            metashape_right_xml=None):

    folder = Path(folder) if folder is not None else None

    # v102: Load Metashape XML calibrations if provided
    ms_left_xml = _load_metashape_xml(metashape_left_xml) if metashape_left_xml else None
    ms_right_xml = _load_metashape_xml(metashape_right_xml) if metashape_right_xml else None
    if ms_left_xml:
        print(f"  [Metashape] Loaded left calibration: {metashape_left_xml}")
    if ms_right_xml:
        print(f"  [Metashape] Loaded right calibration: {metashape_right_xml}")

    # -- Build world-frame transforms (decoupled for cameras vs points) --------
    # Camera transform: uses --camera-axis / --camera-pitch/yaw/roll if set,
    # otherwise defaults to x-y-z, -90, 0, 0
    _cam_axis = camera_axis if camera_axis is not None else "x-yz"
    _cam_pitch = camera_pitch_deg if camera_pitch_deg is not None else -90.0
    _cam_yaw   = camera_yaw_deg if camera_yaw_deg is not None else 90.0
    _cam_roll  = camera_roll_deg if camera_roll_deg is not None else 0.0

    # Point cloud transform: uses --points-axis / --points-pitch/yaw/roll
    S_cam  = build_world_transform(_cam_axis, _cam_pitch, _cam_yaw, _cam_roll)
    S_pts  = build_world_transform(points_axis, pitch_deg, yaw_deg, roll_deg)

    # RS2 requires coordinate alignment rotation (Y-up UE vs Z-up COLMAP)
    # v79 Rx(+90) wrong axis, v80 Rz(+90) also wrong -> v81 tries Ry(+90)
    if viewer_conventions == "RS2":
        R_align = build_world_transform("xyz", 0.0, 90.0, 0.0)  # +90deg Ry
        if S_cam is not None:
            S_cam = _mat_mul(R_align, S_cam)
        else:
            S_cam = R_align
        if S_pts is not None:
            S_pts = _mat_mul(R_align, S_pts)
        else:
            S_pts = R_align

    # Use S_cam for cameras, S_pts for point cloud
    S = S_cam  # backward compat: S refers to camera transform in image processing

    # v56: Compute det(S_cam) to fix L/R swap when det<0 (Y-negation flips X-axis)
    det_cam = _det3x3(S_cam) if S_cam is not None else 1.0

    def _fmt_transform(label, axis_val, p_val, y_val, r_val):
        """Format a transform description for printing."""
        T = build_world_transform(axis_val, p_val, y_val, r_val)
        if T is None:
            print(f"[*] {label}: none (identity passthrough)")
            return
        axis_str = axis_val.lstrip("= ").lower().strip() if axis_val else "xyz"
        rot_parts = []
        if p_val != 0.0: rot_parts.append(f"pitch={p_val}deg")
        if y_val != 0.0: rot_parts.append(f"yaw={y_val}deg")
        if r_val != 0.0: rot_parts.append(f"roll={r_val}deg")
        print(f"[*] {label}:")
        print(f"    Axis: {axis_str}")
        if rot_parts:
            print(f"    Rotation: {', '.join(rot_parts)} (order: pitch-yaw-roll)")

    _fmt_transform("Camera transform", _cam_axis, _cam_pitch, _cam_yaw, _cam_roll)
    _fmt_transform("Point cloud transform", points_axis, pitch_deg, yaw_deg, roll_deg)

    # -- Default output dir --------------------------------------------------
    # When --folder is provided but --output-dir is not, write to
    # <folder_parent>/COLMAP_LFS/ (LFS) or <folder_parent>/COLMAP_RS2/ (RS2)
    if folder is not None and output_dir is None:
        parent = folder.resolve().parent          # one level up from undistort folder
        _fe = "_fisheye" if fisheye else ""
        if viewer_conventions in ("LFS", "PS"):
            output_dir = parent / f"COLMAP_LFS{_fe}"
        else:
            output_dir = parent / f"COLMAP_RS2{_fe}"
    output_dir = Path(output_dir) if output_dir else Path(".")
    folder     = folder     if folder     else Path(".")

    # -- Auto-discover if no explicit JSON -----------------------------------
    if files is None:
        files = find_pcs_files(folder)

    # When --fisheye is set, also scan the images/ sibling folder for fisheye .opt files
    # This MUST be outside the "if files is None" guard because main() pre-fills
    # files via find_pcs_files() before calling convert().
    if fisheye:
        images_folder = (folder / ".." / "images").resolve()
        print(f"[FISHEYE] Scanning for .opt files in: {images_folder}")
        print(f"[FISHEYE] Folder exists: {images_folder.exists()}")
        fish_files = find_pcs_files(images_folder)
        print(f"[FISHEYE] auto-detected opt_left: {fish_files.get('opt_left')}")
        print(f"[FISHEYE] auto-detected opt_right: {fish_files.get('opt_right')}")
        print(f"[FISHEYE] explicit opt_left: {fisheye_left_opt}")
        print(f"[FISHEYE] explicit opt_right: {fisheye_right_opt}")
        # Priority: explicit paths > auto-detected from images/ > undistort
        if fisheye_left_opt:
            files["opt_left"] = fisheye_left_opt
            print(f"[FISHEYE] Using explicit --fisheye-left-opt: {fisheye_left_opt}")
        elif fish_files.get("opt_left"):
            files["opt_left"] = fish_files["opt_left"]
            print(f"[FISHEYE] Using auto-detected opt_left: {fish_files['opt_left']}")
        else:
            print(f"[FISHEYE] WARNING: No fisheye opt_left found, falling back to undistort .opt")
        if fisheye_right_opt:
            files["opt_right"] = fisheye_right_opt
            print(f"[FISHEYE] Using explicit --fisheye-right-opt: {fisheye_right_opt}")
        elif fish_files.get("opt_right"):
            files["opt_right"] = fish_files["opt_right"]
            print(f"[FISHEYE] Using auto-detected opt_right: {fish_files['opt_right']}")
        else:
            print(f"[FISHEYE] WARNING: No fisheye opt_right found, falling back to undistort .opt")

    json_path = left_json or files.get("json")
    opt_left  = opt_left  or files.get("opt_left")
    opt_right = opt_right or files.get("opt_right")
    intrinsic_left  = intrinsic_left  or files.get("intrinsic_left")
    intrinsic_right = intrinsic_right or files.get("intrinsic_right")

    if imgpose_path is None and use_xyzopk:
        imgpose_path = files.get("xyzopk")
    elif imgpose_path is None:
        imgpose_path = files.get("imgpose")

    print_discovery(files)

    # -- Load TransformedCam.json ---------------------------------------------
    if not json_path:
        print("-? TransformedCam.json not found in folder.")
        print("    Please run PointCloud Studio undistort export first.")
        sys.exit(1)

    print(f"\n[*] Loading TransformedCam.json: {json_path}")
    all_frames = load_transformed_cam(json_path)
    print(f"    -?{len(all_frames)} frames total")
    left_frames, right_frames = split_frames(all_frames)
    if swap_lr:
        left_frames, right_frames = right_frames, left_frames
        print(f"    -?Swap-lr: LEFT/RIGHT frame assignments swapped")
    print(f"    -?Split: {len(left_frames)} left, {len(right_frames)} right")

    # -- Load intrinsic.txt files --------------------------------------------
    intr_l = load_intrinsic_txt(intrinsic_left)  if intrinsic_left  else {}
    intr_r = load_intrinsic_txt(intrinsic_right) if intrinsic_right else {}

    # -- Load pose file --------------------------------------------------------
    xyzopk_data = {}
    imgpose_data = {}
    if use_xyzopk:
        xyzpk_path = files.get("xyzopk")
        if xyzpk_path:
            print(f"[*] Loading xyzopk.txt: {xyzpk_path}")
            xyzopk_data = load_xyzopk(xyzpk_path)
            print(f"    -?{len(xyzopk_data)} records")
    else:
        if imgpose_path:
            print(f"[*] Loading ImgPose.txt: {imgpose_path}")
            imgpose_data = load_imgpose(imgpose_path)
            print(f"    -?{len(imgpose_data)} records")

    # -- Load .opt files ------------------------------------------------------
    opt_l = load_opt_file(opt_left)
    opt_r = load_opt_file(opt_right)

    # -- Resolve intrinsics ---------------------------------------------------
    print("\n[*] Resolving intrinsics...")
    left_cal  = get_c1_hardcoded_calibration("left", viewer_conventions)
    right_cal = get_c1_hardcoded_calibration("right", viewer_conventions)
    print(f"  Using hardcoded C1 calibration (Metashape XML overrides this if provided; viewer={viewer_conventions})")
    left_intr  = resolve_intrinsics(left_frames,  intr_l, opt_l, label="left",  fisheye=fisheye, yaml_cal=left_cal,  viewer_conventions=viewer_conventions, metashape_ms=ms_left_xml)
    right_intr = resolve_intrinsics(right_frames, intr_r, opt_r, label="right", fisheye=fisheye, yaml_cal=right_cal, viewer_conventions=viewer_conventions, metashape_ms=ms_right_xml)

    # -- Write cameras.txt (per-image PINHOLE mode) ------------------------
    # v20: one PINHOLE camera per image (CAMERA_ID == IMAGE_ID, 1:1 mapping)
    # Required for LichtFeld Studio / RealityCapture per-image camera import.
    # Intrinsics are shared within left/right groups but each image gets its
    # own camera entry so the importer can handle them independently.
    sparse_dir = output_dir / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    cameras_out = sparse_dir / "cameras.txt"
    model_l, fx_l, fy_l, cx_l, cy_l, k1_l, k2_l, p1_l, p2_l, k3_l, k4_l, k5_l, k6_l, w_l, h_l = left_intr
    model_r, fx_r, fy_r, cx_r, cy_r, k1_r, k2_r, p1_r, p2_r, k3_r, k4_r, k5_r, k6_r, w_r, h_r = right_intr


    # Build timestamp-sorted image list FIRST (needed for camera count)
    tagged = [(fr, 1) for fr in left_frames] + [(fr, 2) for fr in right_frames]
    tagged.sort(key=lambda x: x[0].get("timestamp", 0))
    n_images = len(tagged)

    n_cameras = n_images
    with open(cameras_out, "w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {n_cameras}\n")
        cam_id_counter = 1
        for img_idx, (frame, lr_tag) in enumerate(tagged, start=1):
            if lr_tag == 1:   # left
                f.write(cameras_line(cam_id_counter, model_l, int(w_l), int(h_l),
                                     fx_l, fy_l, cx_l, cy_l, k1_l, k2_l, p1_l, p2_l, k3_l, k4_l, k5_l, k6_l))
            else:             # right
                f.write(cameras_line(cam_id_counter, model_r, int(w_r), int(h_r),
                                     fx_r, fy_r, cx_r, cy_r, k1_r, k2_r, p1_r, p2_r, k3_r, k4_r, k5_r, k6_r))
            cam_id_counter += 1
    # Viewer-specific trailing newline for cameras.txt
    with open(cameras_out, "r", encoding="utf-8") as f:
        content = f.read()
    # v1.12: RS2 v2.2 fixed end-of-tape; no extra blank line needed
    content = content.rstrip() + '\n'        # all viewers: single trailing newline
    with open(cameras_out, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"\n[*] Wrote: {cameras_out}")
    print(f"    {n_images} cameras (per-image PINHOLE mode)")

    # -- Write images.txt -----------------------------------------------------
    images_out = sparse_dir / "images.txt"
    img_id  = 1
    skipped = 0
    mismatches = [] if xyzopk_data else None

    with open(images_out, "w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {n_images}, mean observations per image: 0\n")

        for frame_idx, (frame, lr_tag) in enumerate(tagged, start=1):
            file_path = frame.get("file_path", "")
            # LFS uses forward slashes, RS2 uses backslashes
            if viewer_conventions == "RS2":
                name = file_path.replace("/", "\\")
            else:
                name = file_path  # LFS: keep forward slashes
            basename  = os.path.basename(file_path)
            T_raw     = frame.get("transform_matrix", [])
            if not T_raw or len(T_raw) != 4:
                print(f"  -- Skipping (no transform): {name}")
                skipped += 1
                continue

            # CAMERA_ID == IMAGE_ID (per-image mode)
            cam_id = img_id

            if xyzopk_data and basename in xyzopk_data:
                rec     = xyzopk_data[basename]
                pos_pcs = [rec["x"], rec["y"], rec["z"]]
                if S is not None:
                    q      = euler_opk_to_quat(rec["omega"], rec["phi"], rec["kappa"])
                    R_pcs = quat_to_rotmat(q)
                    St    = mat_transpose_3x3(S)
                    R_new = mat_mul_3x3(R_pcs, St)
                    pos_new = [sum(S[r][c]*pos_pcs[c] for c in range(3)) for r in range(3)]
                    t_w2c  = [-sum(R_new[r][c]*pos_new[c] for c in range(3)) for r in range(3)]
                    tx, ty, tz = t_w2c
                    qw, qx, qy, qz = rotmat_to_quat(R_new)
                else:
                    tx, ty, tz = pos_pcs
                    qw, qx, qy, qz = euler_opk_to_quat(rec["omega"], rec["phi"], rec["kappa"])
            else:
                qw, qx, qy, qz, tx, ty, tz = transform_to_colmap_pose(T_raw, S, S_pos=S_pts)

            # v57: TX negation removed - v56's approach mirrored the entire trajectory
            # L/R swap deferred: baseline values are ~7mm, need to investigate source data

            f.write(images_line(img_id, cam_id, name, qw, qx, qy, qz, tx, ty, tz))

            if mismatches is not None and imgpose_data and basename in imgpose_data:
                p = imgpose_data[basename]
                dx = abs(tx-p["x"]); dy = abs(ty-p["y"]); dz = abs(tz-p["z"])
                if max(dx, dy, dz) > 0.5:
                    mismatches.append({"name": name, "cam": cam_id,
                                       "colmap": (tx,ty,tz), "imgpose": (p["x"],p["y"],p["z"]),
                                       "delta": (dx,dy,dz)})
            img_id += 1

    total = img_id - 1

    # Apply viewer-specific trailing newline rules (images.txt)
    with open(images_out, "r", encoding="utf-8") as f:
        content = f.read()
    # v1.12: RS2 v2.2 fixed end-of-tape; no extra blank line needed
    content = content.rstrip() + '\n'        # all viewers: single trailing newline
    with open(images_out, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[*] Wrote: {images_out}")
    print(f"    {total} images ({len(left_frames)} left + {len(right_frames)} right)"
          + (f"  [{skipped} skipped]" if skipped else ""))

    # -- Auto-discover colourised.las if not provided -----------------------
    # Search in: undistort folder -?parent folder (Output/) -?grandparent
    # Use glob patterns to match any prefix: *_colorized.las, *_colourised.las, etc.
    if las_path is None:
        las_patterns = [
            "*_colorized.las", "*_colourised.las", "*_colorised.las",
            "colorized.las", "colourised.las", "colorised.las",
        ]
        search_dirs = [folder, folder.parent, folder.parent.parent]
        for search_dir in search_dirs:
            for pattern in las_patterns:
                matches = list(search_dir.glob(pattern))
                if matches:
                    las_path = str(matches[0])
                    print(f"[*] Auto-discovered .las: {las_path}")
                    break
            if las_path:
                break

    # -- Write points3D.txt --------------------------------------------------
    # RS2 cannot parse populated points3D.txt (known issue confirmed by Epic tutorial
    # for XGrids Lixel L2 Pro). Write header only for RS2; import .las separately.
    points3d_out = sparse_dir / "points3D.txt"
    if viewer_conventions == "RS2":
        # v86: --points3d-trailing-newlines overrides viewer_conventions default
        # v1.12: RS2 v2.2 fixed end-of-tape; default trailing newlines for empty points3D: 0
        _p3_tn = points3d_trailing_newlines if points3d_trailing_newlines is not None else 0
        # Write without trailing newline, then apply exactly _p3_tn newlines
        with open(points3d_out, "w", encoding="utf-8", newline="") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: 0")  # NO trailing newline
        # Apply exactly _p3_tn trailing newlines in binary mode (guarantees exact bytes)
        if _p3_tn == 1:
            with open(points3d_out, "ab") as f:
                f.write(b"\n")
        elif _p3_tn == 2:
            with open(points3d_out, "ab") as f:
                f.write(b"\n\n")
        # _p3_tn == 0: no trailing newline
        print(f"[*] Wrote: {points3d_out}  (empty — RS2; import .las separately via LiDAR scan)")
    elif las_path and os.path.exists(las_path):
        print(f"\n[*] Converting .las --points3D: {las_path}")
        points = las_to_points3d(las_path, max_points=las_max_points,
                                points_axis=points_axis,
                                pitch_deg=pitch_deg, yaw_deg=yaw_deg, roll_deg=roll_deg)
        if points:
            write_points3d(str(points3d_out), points)
            print(f"[*] Wrote: {points3d_out}  ({len(points):,} points)")
        else:
            print(f"[*] .las conversion failed -?writing empty points3D.txt")
            with open(points3d_out, "w", encoding="utf-8") as f:
                f.write("# 3D point list with one line of data per point:\n")
                f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
                f.write("# Number of points: 0\n")
            print(f"[*] Wrote: {points3d_out}  (empty)")
    else:
        with open(points3d_out, "w", encoding="utf-8") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: 0\n")
        status = "(empty -?no 3D points; use --las to add a .las file)" if las_path else ""
        print(f"[*] Wrote: {points3d_out}  (empty {status})")

    # Viewer-specific trailing newline for points3D.txt
    # v86: --points3d-trailing-newlines overrides viewer_conventions default
    # v1.12: RS2 v2.2 fixed end-of-tape; default trailing newlines: RS2=0, others=0
    _p3_tn = points3d_trailing_newlines if points3d_trailing_newlines is not None else 0
    with open(points3d_out, "r", encoding="utf-8") as f:
        p3content = f.read()
    p3content = p3content.rstrip()
    if _p3_tn == 1:
        p3content += '\n'
    elif _p3_tn == 2:
        p3content += '\n\n'
    # _p3_tn == 0: no trailing newline
    with open(points3d_out, "w", encoding="utf-8") as f:
        f.write(p3content)

    # Create left/ and right/ image folders based on viewer convention.
    # LFS: images/ at COLMAP root (standard COLMAP layout)
    # RS2: sparse/left/ and sparse/right/ (RS2 resolves paths relative to sparse/)
    files_copied = []

    if viewer_conventions in ("LFS", "PS"):
        # LFS / PS: Standard COLMAP layout with images/ at project root
        images_dir = output_dir / "images"
        if no_junction:
            print(f"\n[*] Image folder: {images_dir}/  (--no-junction: copying skipped)")
            print("   Manually copy undistorted images into images/left/ and images/right/ if needed.")
        else:
            for sub in ["left", "right"]:
                src_dir = (folder.parent / 'images' / sub) if fisheye else (folder / sub)
                dst_dir = images_dir / sub
                if not src_dir.is_dir():
                    continue
                dst_dir.mkdir(parents=True, exist_ok=True)
                count = 0
                for img_path in src_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                        shutil.copy2(img_path, dst_dir / img_path.name)
                        count += 1
                if count > 0:
                    files_copied.append(f"  ->images/{sub}/: {count} file(s) copied")
                else:
                    files_copied.append(f"  ->images/{sub}/: no image files found in {src_dir}")
            # Fisheye mask folders -- with circular fisheye masking
            if fisheye:
                for msub in ["left_mask", "right_mask"]:
                    msrc = folder.parent / 'images' / msub
                    mdst = output_dir / "masks"
                    if not msrc.is_dir():
                        continue
                    mdst.mkdir(parents=True, exist_ok=True)
                    mc = 0
                    # Select intrinsics for this side
                    _mcx = cx_l if "left" in msub else cx_r
                    _mcy = cy_l if "left" in msub else cy_r
                    _mw  = w_l  if "left" in msub else w_r
                    _mh  = h_l  if "left" in msub else h_r
                    for mp in msrc.iterdir():
                        if mp.is_file() and mp.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                            dst_name = mp.stem + ".png"  # force .png output
                            ok = generate_fisheye_circle_mask(
                                src_mask_path=mp,
                                dst_path=mdst / dst_name,
                                cx=_mcx, cy=_mcy, w=_mw, h=_mh)
                            if ok:
                                mc += 1
                    if mc > 0:
                        files_copied.append(f"  ->images/{msub}/: {mc} mask file(s) copied + fisheye circle applied")
                    else:
                        files_copied.append(f"  ->images/{msub}/: no mask files found in {msrc}")
    else:
        # RS2: images inside sparse/ folder
        if no_junction:
            print(f"\n[*] Image folder: {sparse_dir}/  (--no-junction: copying skipped)")
            print("   Manually copy undistorted images into sparse/left/ and sparse/right/ if needed.")
        else:
            for sub in ["left", "right"]:
                src_dir = (folder.parent / 'images' / sub) if fisheye else (folder / sub)
                dst_dir = sparse_dir / sub
                if not src_dir.is_dir():
                    continue
                dst_dir.mkdir(parents=True, exist_ok=True)
                count = 0
                for img_path in src_dir.iterdir():
                    if img_path.is_file() and img_path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                        shutil.copy2(img_path, dst_dir / img_path.name)
                        count += 1
                if count > 0:
                    files_copied.append(f"  ->sparse/{sub}/: {count} file(s) copied")
                else:
                    files_copied.append(f"  ->sparse/{sub}/: no image files found in {src_dir}")
            # v91/v96: masks into same folder as images (sparse/left/ and sparse/right/)
            # RS2 requires masks to be co-located with images for drag-drop recognition
            if fisheye:
                mc_total = 0
                for msub in ["left_mask", "right_mask"]:
                    msrc = folder.parent / 'images' / msub
                    img_subfolder = "left" if "left" in msub else "right"
                    mdst = sparse_dir / img_subfolder  # same folder as images
                    if not msrc.is_dir():
                        continue
                    mdst.mkdir(parents=True, exist_ok=True)
                    mc = 0
                    # Select intrinsics for this side
                    _mcx = cx_l if "left" in msub else cx_r
                    _mcy = cy_l if "left" in msub else cy_r
                    _mw  = w_l  if "left" in msub else w_r
                    _mh  = h_l  if "left" in msub else h_r
                    for mp in msrc.iterdir():
                        if mp.is_file() and mp.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                            # RS2 naming: <original_image>.jpg.mask.png
                            # Match the image extension in sparse/<sub>/
                            img_ext = ".jpg"  # default
                            for img_p in mdst.iterdir():
                                if img_p.stem == mp.stem and img_p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
                                    img_ext = img_p.suffix
                                    break
                            dst_name = f"{mp.stem}{img_ext}.mask.png"
                            ok = generate_fisheye_circle_mask(
                                src_mask_path=mp,
                                dst_path=mdst / dst_name,
                                cx=_mcx, cy=_mcy, w=_mw, h=_mh)
                            if ok:
                                mc += 1
                    mc_total += mc
                if mc_total > 0:
                    files_copied.append(f"  ->sparse/left/ + sparse/right/: {mc_total} mask file(s) (co-located with images)")
                else:
                    files_copied.append(f"  ->sparse/: no mask files found")

        if files_copied:
            print(f"\n[*] Image files copied:")
            for entry in files_copied:
                print(entry)
        else:
            print(f"\n-> No image subdirectories found in {folder}")

        # xyzopk was primary; mismatches list is empty -?all good
        pass

        # v88.2: copy xyzopk.txt into sparse/ for manual RS2 reconstruction
        xyzopk_src = files.get("xyzopk")
        if xyzopk_src and os.path.exists(xyzopk_src):
            shutil.copy2(xyzopk_src, sparse_dir / "xyzopk.txt")
            print(f"[*] Copied xyzopk.txt -> sparse/  (for RS2 manual reconstruction)")

    # -- Summary --------------------------------------------------------------
    pose_src = "xyzopk.txt (Omega/Phi/Kappa)" if xyzopk_data else "TransformedCam.json"
    xyzopk_src = files.get("xyzopk") if files else None
    las_info = f" (from {os.path.basename(las_path)})" if las_path and os.path.exists(las_path) else ""
    print("\n" + "-" * 60)
    print(f"   [v1.12] COLMAP export complete!")
    print(f"   Viewer convention: {viewer_conventions}")
    print(f"   Project dir: {output_dir}")
    print(f"   sparse/cameras.txt : {n_cameras} cameras (per-image mode)")
    print(f"   sparse/images.txt  : {total} image(s)")
    print(f"   sparse/points3D.txt: auto (from .las){las_info}" if las_path else f"   sparse/points3D.txt: empty (no 3D points)")
    if xyzopk_src and os.path.exists(xyzopk_src):
        print(f"   sparse/xyzopk.txt : copied from source")
    if viewer_conventions in ("LFS", "PS"):
        print(f"   images/            : at COLMAP root ({viewer_conventions} format)")
    else:
        print(f"   images/            : inside sparse/ (RS2 format)")
    print(f"   Pose source: {pose_src}")
    if S_cam is not None or S_pts is not None:
        if S_cam is not None:
            cam_axis_str = _cam_axis.lstrip("= ").lower().strip() if _cam_axis else "xyz"
            print(f"   Camera transform  : axis={cam_axis_str}, pitch={_cam_pitch}deg, "
                  f"yaw={_cam_yaw}deg, roll={_cam_roll}deg")
        else:
            print(f"   Camera transform  : none (passthrough)")
        if S_pts is not None:
            pts_axis_str = points_axis.lstrip("= ").lower().strip() if points_axis else "xyz"
            print(f"   Point cloud trans : axis={pts_axis_str}, pitch={pitch_deg}deg, "
                  f"yaw={yaw_deg}deg, roll={roll_deg}deg")
        else:
            print(f"   Point cloud trans : none (passthrough)")
    else:
        print(f"   World transform : none (passthrough)")
    print()
    if viewer_conventions in ("LFS", "PS"):
        print(">  To import into LFS/PS (LichtFeld Studio or Postshot):")
        print("   1. Open LFS or PS -?Import -?COLMAP")
        print(f"   2. Browse to: {output_dir}")
        print("   3. Viewer will find images/ folder at project root")
    else:
        print(">  To import into RS2 (RealityCapture 2):")
        print("   1. Open RS2 -?File -?Open Project")
        print("   2. Select 'COLMAP Text Format'")
        print(f"   3. Browse to: {output_dir}")
        print("   4. RS2 will auto-detect sparse/ and images/ subfolders")
    print("-" * 60)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="pose2colmap v1.12 -- SHARE3DCAM pose -> COLMAP converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-discover + RS2 output (default) - cameras x-yz, points xyz
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort"

  # LFS output format
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" --viewer-conventions=LFS

  # Decoupled: x-y-z for cameras, xyz for point cloud
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" \\
                        --points-axis=xyz

  # Camera pitch override (default is -90, override to 0 for no rotation)
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" \\
                        --camera-pitch=0

  # = sign works with or without space (--arg=value or --arg =value)
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" \\
                        --camera-pitch=-90

  # Point cloud only: rotate points 180 roll, cameras unchanged
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" \\
                        --points-roll=180

  # Fully independent: different axis + rotations for each
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" \\
                        --camera-axis=xyz --camera-pitch=90 \\
                        --points-axis=xyz --points-pitch=0 --points-yaw=90

  # Add .las dense point cloud to points3D.txt
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" \\
                        --las "H:/my_scan/Output/colorized.las"

  # Prefer xyzopk.txt for poses (over TransformedCam.json)
  python pose2colmap.py --folder "H:/my_scan/Output/Undistort" --use-xyzopk
"""
    )
    p.add_argument("--folder", "-f",
                   help="PCS undistort folder (auto-discovers all files)")
    p.add_argument("--left-json",
                   help="TransformedCam.json path (overrides auto-discovery)")
    p.add_argument("--right-json",
                   help="Right camera TransformedCam.json (for separate-RIGHT mode)")
    p.add_argument("--opt-left",
                   help="Left_undistort.opt path")
    p.add_argument("--opt-right",
                   help="Right_undistort.opt path")
    p.add_argument("--intrinsic-left",
                   help="left_undistort_intrinsic.txt path")
    p.add_argument("--intrinsic-right",
                   help="right_undistort_intrinsics.txt path")
    p.add_argument("--imgpose",
                   help="ImgPose.txt path (auto-discovered if present)")
    p.add_argument("--use-xyzopk", action="store_true",
                   help="Use xyzopk.txt instead of TransformedCam.json for poses")
    p.add_argument("--output-dir", "-o", default=None,
                   help="Output directory (default: <folder_parent>/COLMAP/ when --folder is used)")
    p.add_argument("--no-junction", action="store_true",
                   help="Skip creating image junction links (manual setup required)")
    p.add_argument("--las",
                   help="Path to colourised .las / .laz file for points3D.txt. "
                        "Auto-discovered if it lives in the undistort folder.")
    p.add_argument("--las-max-points", type=int, default=None,
                   help="Cap .las output to N points (None = unlimited). "
                        "Uses spatially-uniform subsampling for reproducibility.")
    p.add_argument("--points-axis", default="xyz",
                   help="Axis permutation for POINT CLOUD transform only. "
                        "Default 'xyz' (identity, no swap/negation). "
                        "Cameras use --camera-axis (default: xyz, same as points). "
                        "TIP: Use --points-axis=x-yz (with =) if value starts with -")
    p.add_argument("--points-pitch", type=float, default=90.0,
                   help="Point cloud pitch rotation (degrees), applied 1st after axis. "
                        "Default 90deg (LFS default). Cameras use --camera-pitch independently.")
    p.add_argument("--points-yaw", type=float, default=-90.0,
                   help="Point cloud yaw rotation (degrees), applied 2nd. "
                        "Default -90deg (LFS Y-up: stereo baseline on X axis). "
                        "Cameras use --camera-yaw independently.")
    p.add_argument("--points-roll", type=float, default=0.0,
                   help="Point cloud roll rotation (degrees), applied 3rd. "
                        "Default 0deg. Cameras use --camera-roll independently.")
    p.add_argument("--viewer-conventions", choices=["LFS", "PS", "RS2"], default="LFS",
                   help="Output format for specific viewers (default: LFS). "
                        "LFS: images/ at COLMAP root, forward slashes. "
                        "RS2: images in sparse/, backslashes (v1.12: RS2 v2.2 fixed end-of-tape; trailing newlines unified). "
                        "PS: same as LFS (Postshot). Trailing newlines: 1 for all viewers (use --points3d-trailing-newlines to override points3D.txt).")
    p.add_argument("--camera-axis", default=None,
                   help="Axis permutation for CAMERA transform only. "
                        "Default: 'x-y-z' (auto-chirality-fix fires for SLAM det=-1). "
                        "Point cloud uses --points-axis independently.")
    p.add_argument("--camera-pitch", type=float, default=None,
                   help="Camera-only pitch rotation (degrees). Default: -90.0. "
                        "Point cloud uses --pitch independently.")
    p.add_argument("--camera-yaw", type=float, default=None,
                   help="Camera-only yaw rotation (degrees). Default: -90.0. "
                        "Point cloud uses --points-yaw independently.")
    p.add_argument("--camera-roll", type=float, default=None,
                   help="Camera-only roll rotation (degrees). Default: 0.0. "
                        "Point cloud uses --roll independently.")
    p.add_argument("--swap-lr", action="store_true",
                   help="Swap LEFT/RIGHT frame assignments. "
                        "Disabled by default (RS2 convention). Use for LFS if labels are reversed.")
    p.add_argument("--no-swap-lr", dest="swap_lr", action="store_false",
                   help="Disable L/R swap (default behavior for RS2).")
    p.set_defaults(swap_lr=False)  # Default False for RS2 workflow

    p.add_argument("--calibration-yaml", default=None,
                   help="Path to <project>/info/calibration.yaml (POLYFISHEYE factory cal)")

    p.add_argument("--metashape-left-xml", default=None,
                   help="Path to Metashape left-camera calibration XML. "
                        "If provided, overrides YAML/.opt intrinsics with converted Metashape "
                        "EquisolidFisheye -> COLMAP OPENCV_FISHEYE values. "
                        "See: https://www.agisoft.com/pdf/photoscan/Pro_1_4_en.pdf (Metashape XML format)")
    p.add_argument("--metashape-right-xml", default=None,
                   help="Path to Metashape right-camera calibration XML. "
                        "Must be used together with --metashape-left-xml.")

    p.add_argument("--fisheye", action="store_true",
                   help="Use raw fisheye images from images/ folder instead of undistort/.")
    p.add_argument("--fisheye-left-opt", default=None,
                   help="Explicit path to Left.opt fisheye calibration file.")
    p.add_argument("--fisheye-right-opt", default=None,
                   help="Explicit path to Right.opt fisheye calibration file.")
    p.add_argument("--points3d-trailing-newlines", type=int, default=None, choices=[0, 1, 2],
                   help="Trailing newlines for points3D.txt: 0=none, 1=exactly1, 2=exactly2. "
                        "If None (default), all viewers default to 0 (v1.12: RS2 v2.2 fixed end-of-tape).")

    # v48: Preprocess sys.argv to merge "--arg =value" into "--arg=value".
    # PowerShell/cmd users often write "--camera-pitch =90" which argparse
    # parses as: --camera-pitch (no value) + positional "=90".  Merging
    # them BEFORE parse_args() avoids the error entirely.
    import re as _re
    _argv = list(sys.argv)
    _i = 1  # skip script name
    while _i < len(_argv) - 1:
        if _argv[_i].startswith("--") and _argv[_i + 1].startswith("="):
            _argv[_i] = _argv[_i] + _argv[_i + 1]   # --arg=value
            del _argv[_i + 1]
        else:
            _i += 1
    args = p.parse_args(_argv[1:])  # skip _argv[0] (script name)

    # v43+v48: Strip any leading '=' that still survives (e.g. --arg =value
    # where argparse received '=value' as the raw value).  This is a safety
    # net for edge cases the preprocessor above didn't catch.
    for attr in ["points_axis", "points_pitch", "points_yaw", "points_roll",
                 "camera_axis", "camera_pitch", "camera_yaw", "camera_roll"]:
        val = getattr(args, attr)
        if isinstance(val, str) and val.startswith("="):
            setattr(args, attr, val.lstrip("="))

    if not args.folder and not args.left_json:
        print("-? Error: --folder or --left-json is required.")
        sys.exit(1)

    folder = Path(args.folder) if args.folder else None

    # Auto-discover when using --folder
    files = None
    if folder:
        files = find_pcs_files(folder)
        if not files.get("json"):
            print(f"-? TransformedCam.json not found in: {folder}")
            print("    Make sure the path points to the Output-Undistort folder.")
            sys.exit(1)

    # === YAML calibration (v101) ===
    cal_yaml = None
    if args.calibration_yaml:
        import yaml as _yaml, warnings as _warn
        _warn.filterwarnings("ignore")
        with open(args.calibration_yaml, "r", encoding="utf-8-sig") as _f:
            _cal_text = _re.sub(r'^%YAML.*\n?', '', _f.read(), flags=_re.MULTILINE)
        class _IgnoreLoader(_yaml.SafeLoader):
            pass
        def _ignore_unknown(self, node):
            if isinstance(node, _yaml.MappingNode):
                return self.construct_mapping(node, deep=True)
            elif isinstance(node, _yaml.SequenceNode):
                return self.construct_sequence(node, deep=True)
            else:
                return self.construct_scalar(node)
        _IgnoreLoader.add_constructor(None, _ignore_unknown)
        _raw = _yaml.load(_cal_text, Loader=_IgnoreLoader)
        cal_yaml = {}
        for _side, _entry in _raw.get("intrinsic", {}).items():
            if not isinstance(_entry, dict):
                continue
            _pp = _entry.get("projection_parameters", {})
            _w  = int(_entry.get("image_width",  0))
            _h  = int(_entry.get("image_height", 0))
            if not _w or not _h:
                continue
            _fx  = _pp.get("A11", 0.0)
            _fy  = _pp.get("A22", _fx)
            _cx  = _pp.get("u0",  _w / 2.0)
            _cy  = _pp.get("v0",  _h / 2.0)
            _k1  = _pp.get("k2", 0.0)
            _k2  = _pp.get("k3", 0.0)
            _k3  = _pp.get("k4", 0.0)
            _k4  = _pp.get("k5", 0.0)
            _k5  = _pp.get("k6", 0.0)
            _k6  = _pp.get("k7", 0.0)
            cal_yaml[_side] = dict(fx=_fx, fy=_fy, cx=_cx, cy=_cy,
                                   k1=_k1, k2=_k2, k3=_k3, k4=_k4, k5=_k5, k6=_k6,
                                   w=_w, h=_h)
        print(f"[CALIB YAML] Loaded {len(cal_yaml)} camera(s) from {args.calibration_yaml}")

    # === v42: Decoupled camera/point cloud transforms ===
    _pts_axis  = args.points_axis
    _pts_pitch = args.points_pitch
    _pts_yaw   = args.points_yaw
    _pts_roll  = args.points_roll
    print(f"  pose2colmap -- points: axis={_pts_axis} pitch={_pts_pitch} yaw={_pts_yaw} roll={_pts_roll}")

    try:
        convert(
            folder=folder, files=files,
            left_json=args.left_json, right_json=args.right_json,
            opt_left=args.opt_left, opt_right=args.opt_right,
            intrinsic_left=args.intrinsic_left, intrinsic_right=args.intrinsic_right,
            imgpose_path=args.imgpose,
            use_xyzopk=args.use_xyzopk,
            output_dir=args.output_dir,
            no_junction=args.no_junction,
            las_path=args.las,
            las_max_points=args.las_max_points,
            points_axis=_pts_axis,
            pitch_deg=_pts_pitch, yaw_deg=_pts_yaw, roll_deg=_pts_roll,
            calibration_yaml=cal_yaml,
            viewer_conventions=args.viewer_conventions,
            camera_axis=args.camera_axis,
            camera_pitch_deg=args.camera_pitch,
            camera_yaw_deg=args.camera_yaw,
            camera_roll_deg=args.camera_roll,
            swap_lr=args.swap_lr,
            fisheye=args.fisheye,
            fisheye_left_opt=args.fisheye_left_opt,
            fisheye_right_opt=args.fisheye_right_opt,
            points3d_trailing_newlines=args.points3d_trailing_newlines,
            metashape_left_xml=args.metashape_left_xml,
            metashape_right_xml=args.metashape_right_xml,
        )
    except Exception as e:
        print(f"\n-? {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
