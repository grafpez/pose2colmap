# pose2colmap
(this is a personal project with the help of SHARE3DCAM bot [openclaw] provided)

Creates COLMAP datasets from SHARE C1 LiDAR scanner data for use with RS2 or PS/LFS.

Purpose of this script is to provide pathways for the datasets created by the LiDAR scanner to help processing in other softwares (RealityScan2; Postshot& LichtFeld Studio) via a COLMAP dataset format bridge, in order to aid 3D reconstruction using hybrid datasets. It also serves as a dataset-test bed for fisheye workflows between the different methods. 

# Dependencies:

Third-party (need pip install):
numpy — pip install numpy
pyyaml — pip install pyyaml (imported as import yaml)
pillow — pip install pillow (imported as PIL)
laspy — pip install laspy (optional, only needed if you process .las → points3D.txt)
-
PySide6 (for GUI, ~100MB download)

Standard library (already in Python, no install):
pathlib, argparse, glob, json, math, os, re, shutil, sys, traceback, xml.etree.ElementTree

So the essential install is:
pip install numpy pyyaml pillow


Add laspy only if you're working with .las point clouds. For your QW_Ramp run (undistort images), you don't need laspy.

# User Manual
please see attached user manual txt file for details.

# Additional Files:
batch_resize_undistort.py
=========================
Batch resize/crop undistorted fisheye images

Two modes:
  Default:  2877x1798 -> 1920x1200  (horizontal crop, 16:10, ~55% pixel reduction)
  --vertical: 2877x1798 -> 1920x1798  (vertical-preserving crop, ~33% pixel reduction)

Workflow:
  1. Backs up the Undistort folder to Undistort_original (skips if already exists)
  2. Resizes/crops all JPG/PNG files from Undistort_original (recursive, preserves subfolders)
  3. Overwrites originals in the Undistort folder, keeping identical paths

las.2ply_strip_crs.py
=====================
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


     

