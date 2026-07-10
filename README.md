# pose2colmap
Creates COLMAP datasets from SHARE C1 LiDAR scanner data for use with RS2 or PS/LFS

# Dependencies:

Third-party (need pip install):
numpy — pip install numpy
pyyaml — pip install pyyaml (imported as import yaml)
pillow — pip install pillow (imported as PIL)
laspy — pip install laspy (optional, only needed if you process .las → points3D.txt)

Standard library (already in Python, no install):
pathlib, argparse, glob, json, math, os, re, shutil, sys, traceback, xml.etree.ElementTree

So the essential install is:
pip install numpy pyyaml pillow


Add laspy only if you're working with .las point clouds. For your QW_Ramp run (undistort images), you don't need laspy.

# User Manual
please see attached user manual txt file for details.
