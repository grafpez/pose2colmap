#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose2colmap_gui.py -- Dark Fusion GUI wrapper for pose2colmap.py

This is a standalone front-end. It does NOT modify or import pose2colmap.py;
it simply builds the command-line argument list and runs the script as a
subprocess (via QProcess), streaming stdout/stderr into a live log pane.

Requirements (GUI only):
    pip install PySide6

The script's own dependencies (numpy, pyyaml, laspy, ...) must be available
to whichever Python interpreter you point the "Python" field at.

Run:
    python pose2colmap_gui.py
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer
from PySide6.QtGui import QColor, QFont, QPalette, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QScrollArea, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QDoubleSpinBox, QPlainTextEdit, QFileDialog, QMessageBox,
    QSplitter,
)


# --------------------------------------------------------------------------- #
# Dark Fusion palette
# --------------------------------------------------------------------------- #
def apply_dark_fusion(app: QApplication) -> None:
    app.setStyle("Fusion")
    app.setFont(QFont("Arial", 11))

    p = QPalette()
    base = QColor(37, 37, 38)
    panel = QColor(53, 53, 53)
    text = QColor(220, 220, 220)
    disabled = QColor(127, 127, 127)
    highlight = QColor(42, 130, 218)

    p.setColor(QPalette.Window, panel)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, panel)
    p.setColor(QPalette.ToolTipBase, panel)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, panel)
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.BrightText, Qt.red)
    p.setColor(QPalette.Link, highlight)
    p.setColor(QPalette.Highlight, highlight)
    p.setColor(QPalette.HighlightedText, Qt.black)

    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, disabled)
    p.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
    p.setColor(QPalette.Disabled, QPalette.HighlightedText, disabled)

    app.setPalette(p)


# --------------------------------------------------------------------------- #
# Small helpers for building rows
# --------------------------------------------------------------------------- #
def path_row(placeholder: str, pick_dir: bool = False, file_filter: str = "All files (*.*)"):
    """Return (widget, line_edit). widget = [QLineEdit][Browse]."""
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    edit = QLineEdit()
    edit.setPlaceholderText(placeholder)
    btn = QPushButton("Browse…")
    btn.setFixedWidth(90)

    def browse():
        if pick_dir:
            chosen = QFileDialog.getExistingDirectory(w, "Select folder", edit.text() or "")
        else:
            chosen, _ = QFileDialog.getOpenFileName(w, "Select file", edit.text() or "", file_filter)
        if chosen:
            edit.setText(chosen)

    btn.clicked.connect(browse)
    lay.addWidget(edit, 1)
    lay.addWidget(btn)
    return w, edit


def angle_spin(default: float) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(-360.0, 360.0)
    s.setDecimals(3)
    s.setSingleStep(1.0)
    s.setValue(default)
    s.setFixedWidth(120)
    return s


# --------------------------------------------------------------------------- #
# Recenter to origin (RTK / georeferenced scans)
# --------------------------------------------------------------------------- #
def _is_number(s: str) -> bool:
    try:
        float(s); return True
    except ValueError:
        return False


def _quat_to_R(qw, qx, qy, qz):
    """world->camera rotation from a COLMAP quaternion (normalised)."""
    n = math.sqrt(qw*qw + qx*qx + qy*qy + qz*qz) or 1.0
    qw, qx, qy, qz = qw/n, qx/n, qy/n, qz/n
    return (
        (1-2*(qy*qy+qz*qz), 2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)),
        (2*(qx*qy+qz*qw),   1-2*(qx*qx+qz*qz), 2*(qy*qz-qx*qw)),
        (2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw),   1-2*(qx*qx+qy*qy)),
    )


def recenter_colmap(sparse_dir: Path, mode: str = "cameras") -> dict:
    """Rigidly shift cameras + points so the scene sits near origin, without
    changing how anything projects (X' = X - O for points; t' = t + R·O for
    cameras). Writes recenter_offset.txt/.json so the shift can be undone.

    mode: 'cameras' (mean camera centre), 'points' (mean point), 'first_camera'.
    Streams points3D.txt so multi-GB clouds don't blow up memory.
    """
    sparse_dir = Path(sparse_dir)
    images = sparse_dir / "images.txt"
    points = sparse_dir / "points3D.txt"
    if not images.exists():
        raise FileNotFoundError(f"images.txt not found in {sparse_dir}")

    img_lines = images.read_text(encoding="utf-8").splitlines(keepends=False)
    poses, centers = [], []
    for idx, line in enumerate(img_lines):
        if not line or line.startswith("#"):
            continue
        p = line.split()
        if len(p) >= 10 and not _is_number(p[-1]):   # pose line (name non-numeric)
            q = [float(p[1]), float(p[2]), float(p[3]), float(p[4])]
            t = [float(p[5]), float(p[6]), float(p[7])]
            R = _quat_to_R(*q)
            C = [-(R[0][i]*t[0] + R[1][i]*t[1] + R[2][i]*t[2]) for i in range(3)]
            centers.append(C)
            poses.append((idx, p, R, t))
    if not poses:
        raise ValueError("No camera pose lines found in images.txt")

    n_scanned = 0
    if mode == "points":
        sx = sy = sz = 0.0
        if points.exists():
            with open(points, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip() or line.startswith("#"):
                        continue
                    q = line.split()
                    if len(q) >= 4:
                        try:
                            sx += float(q[1]); sy += float(q[2]); sz += float(q[3])
                        except ValueError:
                            continue
                        n_scanned += 1
        if n_scanned:
            O = [sx/n_scanned, sy/n_scanned, sz/n_scanned]
        else:
            mode = "cameras"   # no points -> fall back
    if mode == "first_camera":
        O = centers[0]
    if mode == "cameras":
        O = [sum(c[i] for c in centers)/len(centers) for i in range(3)]

    # rewrite images.txt: t' = t + R·O
    for (idx, p, R, t) in poses:
        RO = [R[i][0]*O[0] + R[i][1]*O[1] + R[i][2]*O[2] for i in range(3)]
        tp = [t[0]+RO[0], t[1]+RO[1], t[2]+RO[2]]
        name = " ".join(p[9:])
        img_lines[idx] = (f"{p[0]} {p[1]} {p[2]} {p[3]} {p[4]} "
                          f"{tp[0]:.17g} {tp[1]:.17g} {tp[2]:.17g} {p[8]} {name}")
    tmp = images.with_suffix(".txt.tmp")
    tmp.write_text("\n".join(img_lines) + "\n", encoding="utf-8")
    os.replace(tmp, images)

    # rewrite points3D.txt: X' = X - O  (streamed)
    n_points = 0
    if points.exists():
        tmpp = points.with_suffix(".txt.tmp")
        with open(points, "r", encoding="utf-8") as fin, \
             open(tmpp, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip() or line.startswith("#"):
                    fout.write(line); continue
                q = line.split()
                if len(q) < 4:
                    fout.write(line); continue
                try:
                    x = float(q[1]) - O[0]; y = float(q[2]) - O[1]; z = float(q[3]) - O[2]
                except ValueError:
                    fout.write(line); continue
                rest = (" " + " ".join(q[4:])) if len(q) > 4 else ""
                fout.write(f"{q[0]} {x:.8f} {y:.8f} {z:.8f}{rest}\n")
                n_points += 1
        os.replace(tmpp, points)

    info = {"mode": mode, "offset": O,
            "note": "Add offset back to coordinates to restore the original CRS."}
    (sparse_dir / "recenter_offset.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (sparse_dir / "recenter_offset.txt").write_text(
        "# pose2colmap_gui recenter offset (subtracted from all coordinates).\n"
        "# Add these values back to georeference to the original CRS.\n"
        f"mode: {mode}\n"
        f"offset_x: {O[0]!r}\noffset_y: {O[1]!r}\noffset_z: {O[2]!r}\n",
        encoding="utf-8")

    return {"offset": O, "mode": mode, "n_cameras": len(poses), "n_points": n_points}


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("pose2colmap — GUI")
        self.resize(920, 860)
        self.proc: QProcess | None = None
        self._proc_mode: str | None = None
        self._auto: dict[int, str] = {}   # id(edit) -> last auto-filled value
        self._disc: dict[str, str] = {}   # last scan's imgpose / xyzopk paths

        # ---- default script / interpreter paths -------------------------- #
        here = Path(__file__).resolve().parent
        guess = here / "pose2colmap.py"

        # ---------------- form (scrollable) ------------------------------- #
        form = QWidget()
        fl = QVBoxLayout(form)
        fl.setSpacing(10)

        # Project ----------------------------------------------------------
        g_proj = QGroupBox("Project  (Point Cloud Studio output folder)")
        proj_l = QGridLayout(g_proj)
        proj_l.addWidget(QLabel("Project folder"), 0, 0)
        prow = QWidget(); prl = QHBoxLayout(prow); prl.setContentsMargins(0, 0, 0, 0)
        self.project_edit = QLineEdit()
        self.project_edit.setPlaceholderText("Set Project Folder")
        proj_browse = QPushButton("Browse…"); proj_browse.setFixedWidth(90)
        proj_rescan = QPushButton("Re-scan"); proj_rescan.setFixedWidth(90)
        proj_browse.clicked.connect(self._browse_project)
        proj_rescan.clicked.connect(lambda: self._reload_project(force=True))
        self.project_edit.editingFinished.connect(self._reload_project)
        prl.addWidget(self.project_edit, 1)
        prl.addWidget(proj_browse)
        prl.addWidget(proj_rescan)
        proj_l.addWidget(prow, 0, 1, 1, 2)
        self.auto_pop = QCheckBox("Auto-populate fields from project folder")
        self.auto_pop.setChecked(True)
        proj_l.addWidget(self.auto_pop, 1, 1, 1, 2)
        self.project_status = QLabel("Pick a project folder to auto-fill the fields below.")
        self.project_status.setWordWrap(True)
        self.project_status.setStyleSheet("color: #9aa0a6;")
        proj_l.addWidget(self.project_status, 2, 1, 1, 2)
        fl.addWidget(g_proj)

        # Run config -------------------------------------------------------
        g_run = QGroupBox("Run configuration")
        gl = QGridLayout(g_run)
        self.py_edit = QLineEdit(sys.executable)
        py_w, self.py_edit = self._labeled_path(gl, 0, "Python interpreter",
                                                 self.py_edit.text(), pick_dir=False,
                                                 file_filter="Python (python* python.exe *);;All files (*.*)")
        scr_w, self.script_edit = self._labeled_path(
            gl, 1, "pose2colmap.py",
            str(guess) if guess.exists() else "",
            pick_dir=False, file_filter="Python (*.py);;All files (*.*)")
        fl.addWidget(g_run)

        # Input ------------------------------------------------------------
        g_in = QGroupBox("Input  (use --folder for auto-discovery, or set files manually)")
        il = QGridLayout(g_in)
        _, self.folder_edit = self._labeled_path(il, 0, "Undistort folder (--folder)", "", pick_dir=True)
        _, self.leftjson_edit = self._labeled_path(il, 1, "--left-json", "", file_filter="JSON (*.json);;All files (*.*)")
        _, self.rightjson_edit = self._labeled_path(il, 2, "--right-json", "", file_filter="JSON (*.json);;All files (*.*)")
        _, self.optleft_edit = self._labeled_path(il, 3, "--opt-left", "", file_filter="OPT (*.opt);;All files (*.*)")
        _, self.optright_edit = self._labeled_path(il, 4, "--opt-right", "", file_filter="OPT (*.opt);;All files (*.*)")
        _, self.intrleft_edit = self._labeled_path(il, 5, "--intrinsic-left", "", file_filter="Text (*.txt);;All files (*.*)")
        _, self.intrright_edit = self._labeled_path(il, 6, "--intrinsic-right", "", file_filter="Text (*.txt);;All files (*.*)")
        _, self.imgpose_edit = self._labeled_path(il, 7, "--imgpose", "", file_filter="Text (*.txt);;All files (*.*)")
        self.use_xyzopk = QCheckBox("Use xyzopk.txt for poses (--use-xyzopk)")
        il.addWidget(self.use_xyzopk, 8, 1, 1, 2)
        self.use_xyzopk.toggled.connect(self._on_xyzopk_toggled)
        self.rightjson_edit.setToolTip(
            "Only for separate-RIGHT mode. A normal PCS project has a single "
            "TransformedCam.json (mapped to --left-json), so leave this empty.")
        fl.addWidget(g_in)

        # Output -----------------------------------------------------------
        g_out = QGroupBox("Output")
        ol = QGridLayout(g_out)
        _, self.outdir_edit = self._labeled_path(ol, 0, "--output-dir (default: <parent>/COLMAP/)", "", pick_dir=True)
        ol.addWidget(QLabel("Viewer conventions"), 1, 0)
        self.viewer_combo = QComboBox()
        self.viewer_combo.addItems(["LFS", "PS", "RS2"])
        ol.addWidget(self.viewer_combo, 1, 1)
        self.no_junction = QCheckBox("Skip image junction links (--no-junction)")
        ol.addWidget(self.no_junction, 2, 1, 1, 2)
        fl.addWidget(g_out)

        # Recenter ---------------------------------------------------------
        g_rc = QGroupBox("Recenter to origin  (RTK / georeferenced scans)")
        rcl = QGridLayout(g_rc)
        self.recenter_chk = QCheckBox(
            "Recenter scene to origin after conversion (fixes far-from-origin RTK coords)")
        rcl.addWidget(self.recenter_chk, 0, 0, 1, 3)
        rcl.addWidget(QLabel("Center on"), 1, 0)
        self.recenter_mode = QComboBox()
        self.recenter_mode.addItems(["Camera centroid", "Point cloud centroid", "First camera"])
        self.recenter_mode.setFixedWidth(200)
        self.recenter_mode.setEnabled(False)
        self.recenter_chk.toggled.connect(self.recenter_mode.setEnabled)
        rcl.addWidget(self.recenter_mode, 1, 1, Qt.AlignLeft)
        note = QLabel("Shifts cameras and points by one common offset so they stay aligned; "
                      "saves recenter_offset.txt in sparse/ so it can be undone.")
        note.setWordWrap(True); note.setStyleSheet("color: #9aa0a6;")
        rcl.addWidget(note, 2, 1, 1, 2)
        fl.addWidget(g_rc)

        # Point cloud / LAS -----------------------------------------------
        g_pts = QGroupBox("Point cloud / LAS")
        pl = QGridLayout(g_pts)
        _, self.las_edit = self._labeled_path(pl, 0, "--las (.las / .laz)", "",
                                              file_filter="LAS/LAZ (*.las *.laz);;All files (*.*)")
        pl.addWidget(QLabel("--las-max-points"), 1, 0)
        self.lasmax_edit = QLineEdit()
        self.lasmax_edit.setPlaceholderText("empty = unlimited")
        self.lasmax_edit.setFixedWidth(160)
        pl.addWidget(self.lasmax_edit, 1, 1, Qt.AlignLeft)

        pl.addWidget(QLabel("--points-axis"), 2, 0)
        self.pts_axis = QLineEdit("xyz")
        self.pts_axis.setFixedWidth(160)
        pl.addWidget(self.pts_axis, 2, 1, Qt.AlignLeft)

        row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("pitch")); self.pts_pitch = angle_spin(90.0); rl.addWidget(self.pts_pitch)
        rl.addWidget(QLabel("yaw"));   self.pts_yaw = angle_spin(-90.0); rl.addWidget(self.pts_yaw)
        rl.addWidget(QLabel("roll"));  self.pts_roll = angle_spin(0.0);  rl.addWidget(self.pts_roll)
        rl.addStretch(1)
        pl.addWidget(QLabel("points pitch/yaw/roll"), 3, 0)
        pl.addWidget(row, 3, 1, 1, 2)

        pl.addWidget(QLabel("--points3d-trailing-newlines"), 4, 0)
        self.trailing_combo = QComboBox()
        self.trailing_combo.addItems(["default", "0", "1", "2"])
        self.trailing_combo.setFixedWidth(160)
        pl.addWidget(self.trailing_combo, 4, 1, Qt.AlignLeft)
        fl.addWidget(g_pts)

        # Camera transform -------------------------------------------------
        g_cam = QGroupBox("Camera transform  (leave 'override' off to use script defaults)")
        cl = QGridLayout(g_cam)
        self.cam_axis_chk = QCheckBox("override --camera-axis")
        self.cam_axis_edit = QLineEdit("x-y-z"); self.cam_axis_edit.setFixedWidth(160)
        self.cam_axis_edit.setEnabled(False)
        self.cam_axis_chk.toggled.connect(self.cam_axis_edit.setEnabled)
        cl.addWidget(self.cam_axis_chk, 0, 0)
        cl.addWidget(self.cam_axis_edit, 0, 1, Qt.AlignLeft)

        self.cam_ang_chk = QCheckBox("override --camera-pitch/yaw/roll")
        cl.addWidget(self.cam_ang_chk, 1, 0)
        crow = QWidget(); crl = QHBoxLayout(crow); crl.setContentsMargins(0, 0, 0, 0)
        crl.addWidget(QLabel("pitch")); self.cam_pitch = angle_spin(-90.0); crl.addWidget(self.cam_pitch)
        crl.addWidget(QLabel("yaw"));   self.cam_yaw = angle_spin(-90.0);   crl.addWidget(self.cam_yaw)
        crl.addWidget(QLabel("roll"));  self.cam_roll = angle_spin(0.0);    crl.addWidget(self.cam_roll)
        crl.addStretch(1)
        for s in (self.cam_pitch, self.cam_yaw, self.cam_roll):
            s.setEnabled(False)
        self.cam_ang_chk.toggled.connect(
            lambda on: [s.setEnabled(on) for s in (self.cam_pitch, self.cam_yaw, self.cam_roll)])
        cl.addWidget(crow, 1, 1, 1, 2)

        self.swap_lr = QCheckBox("Swap LEFT/RIGHT (--swap-lr)")
        cl.addWidget(self.swap_lr, 2, 1, 1, 2)
        fl.addWidget(g_cam)

        # Calibration ------------------------------------------------------
        g_cal = QGroupBox("Calibration (optional)")
        call = QGridLayout(g_cal)
        _, self.calyaml_edit = self._labeled_path(call, 0, "--calibration-yaml", "",
                                                  file_filter="YAML (*.yaml *.yml);;All files (*.*)")
        _, self.metaleft_edit = self._labeled_path(call, 1, "--metashape-left-xml", "",
                                                   file_filter="XML (*.xml);;All files (*.*)")
        _, self.metaright_edit = self._labeled_path(call, 2, "--metashape-right-xml", "",
                                                    file_filter="XML (*.xml);;All files (*.*)")
        fl.addWidget(g_cal)

        # Fisheye ----------------------------------------------------------
        g_fish = QGroupBox("Fisheye (optional)")
        fll = QGridLayout(g_fish)
        self.fisheye = QCheckBox("Use raw fisheye images (--fisheye)")
        fll.addWidget(self.fisheye, 0, 1, 1, 2)
        _, self.fishleft_edit = self._labeled_path(fll, 1, "--fisheye-left-opt", "",
                                                   file_filter="OPT (*.opt);;All files (*.*)")
        _, self.fishright_edit = self._labeled_path(fll, 2, "--fisheye-right-opt", "",
                                                    file_filter="OPT (*.opt);;All files (*.*)")
        fl.addWidget(g_fish)

        fl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form)

        # ---------------- output console --------------------------------- #
        console_box = QWidget()
        cvl = QVBoxLayout(console_box)
        cvl.setContentsMargins(0, 0, 0, 0)
        self.cmd_preview = QLineEdit()
        self.cmd_preview.setReadOnly(True)
        self.cmd_preview.setPlaceholderText("Assembled command appears here")
        self.cmd_preview.setFont(QFont("Consolas", 10))
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setFont(QFont("Consolas", 10))
        cvl.addWidget(QLabel("Command:"))
        cvl.addWidget(self.cmd_preview)
        cvl.addWidget(QLabel("Output:"))
        cvl.addWidget(self.log, 1)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(scroll)
        splitter.addWidget(console_box)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        # ---------------- buttons ---------------------------------------- #
        btn_bar = QWidget()
        bl = QHBoxLayout(btn_bar)
        self.preview_btn = QPushButton("Preview command")
        self.copy_btn = QPushButton("Copy command")
        self.run_btn = QPushButton("Run")
        self.stop_btn = QPushButton("Stop")
        self.clear_btn = QPushButton("Clear log")
        self.stop_btn.setEnabled(False)
        self.run_btn.setDefault(True)
        for b in (self.preview_btn, self.copy_btn, self.clear_btn):
            bl.addWidget(b)
        bl.addStretch(1)
        bl.addWidget(self.stop_btn)
        bl.addWidget(self.run_btn)

        self.preview_btn.clicked.connect(self.on_preview)
        self.copy_btn.clicked.connect(self.on_copy)
        self.run_btn.clicked.connect(self.on_run)
        self.stop_btn.clicked.connect(self.on_stop)
        self.clear_btn.clicked.connect(lambda: self.log.clear())

        # Fields that project auto-population manages (for reset-on-change).
        self._managed = [
            self.folder_edit, self.leftjson_edit,
            self.optleft_edit, self.optright_edit,
            self.intrleft_edit, self.intrright_edit,
            self.imgpose_edit, self.las_edit,
            self.fishleft_edit, self.fishright_edit,
            self.calyaml_edit,
        ]
        self._last_project = ""

        # ---------------- dependency banner (red, near top) -------------- #
        self.dep_banner = QPushButton()
        self.dep_banner.setVisible(False)
        self.dep_banner.setCursor(Qt.PointingHandCursor)
        self.dep_banner.setStyleSheet(
            "QPushButton {"
            "  background-color: #7a1f1f; color: #ffffff;"
            "  border: 1px solid #ff5555; border-radius: 4px;"
            "  padding: 8px; text-align: left; font-weight: bold; }"
            "QPushButton:hover { background-color: #8f2626; }"
            "QPushButton:disabled { background-color: #4a2020; color: #cbb; }")
        self.dep_banner.clicked.connect(self.install_dependencies)

        central = QWidget()
        cl2 = QVBoxLayout(central)
        cl2.addWidget(self.dep_banner)
        cl2.addWidget(splitter, 1)
        cl2.addWidget(btn_bar)
        self.setCentralWidget(central)

        # Re-check packages when the interpreter or the .las selection changes.
        self.py_edit.editingFinished.connect(self.check_dependencies)
        self.las_edit.editingFinished.connect(self.check_dependencies)
        # First check after the window is shown (probe is a quick subprocess).
        QTimer.singleShot(0, self.check_dependencies)

    # ---- helper to add a labeled path row into a grid -------------------- #
    def _labeled_path(self, grid: QGridLayout, r: int, label: str, initial: str,
                      pick_dir: bool = False, file_filter: str = "All files (*.*)"):
        grid.addWidget(QLabel(label), r, 0)
        w, edit = path_row(label, pick_dir=pick_dir, file_filter=file_filter)
        if initial:
            edit.setText(initial)
        grid.addWidget(w, r, 1, 1, 2)
        return w, edit

    # ---------------------------------------------------------------------- #
    # Project auto-population
    # ---------------------------------------------------------------------- #
    def _browse_project(self):
        chosen = QFileDialog.getExistingDirectory(
            self, "Select Point Cloud Studio project (output) folder",
            self.project_edit.text() or "")
        if chosen:
            self.project_edit.setText(chosen)
            self._reload_project()

    def _reload_project(self, force: bool = False):
        """Called whenever the project folder changes. Re-populates from the
        new folder, resetting previously auto-filled fields first so no stale
        paths from a different project survive."""
        new = self.project_edit.text().strip()
        if not force:
            if not self.auto_pop.isChecked():
                return
            if new == self._last_project:
                return  # nothing actually changed
        self._last_project = new
        self.scan_project(reset=True, announce=True)

    def _reset_auto_fields(self):
        """Clear fields that still hold an auto-filled value; leave manual edits."""
        for edit in self._managed:
            prev = self._auto.get(id(edit))
            if prev is not None and edit.text().strip() == prev:
                edit.clear()
                self._auto.pop(id(edit), None)

    def _auto_fill(self, edit: QLineEdit, value: str) -> bool:
        """Fill only if the field is empty or still holds a prior auto value.
        Never overwrites something the user typed manually."""
        if not value:
            return False
        prev = self._auto.get(id(edit))
        cur = edit.text().strip()
        if cur == "" or cur == prev:
            edit.setText(value)
            self._auto[id(edit)] = value
            return True
        return False

    @staticmethod
    def _pick_las(root: Path):
        files = sorted(list(root.glob("*.las")) + list(root.glob("*.laz")))
        if not files:
            return None
        # Prefer a colourised cloud (but not the "uncolorized" one).
        col = [f for f in files
               if "colorized" in f.name.lower() and "uncolorized" not in f.name.lower()]
        return col[0] if col else files[0]

    @staticmethod
    def _discover_in_folder(folder: Path) -> dict:
        """Mirror pose2colmap.find_pcs_files() so the fields show exactly the
        paths the script would auto-discover from --folder."""
        def try_find(patterns):
            for pat in patterns:
                hits = sorted(folder.glob(pat))
                if hits:
                    return str(hits[0].resolve())
            return None

        return {
            "json": try_find(["TransformedCam.json", "*/TransformedCam.json"]),
            "opt_left": try_find(["Left.opt", "Left_undistort.opt", "*Left_undistort.opt",
                                  "left.opt", "left_undistort.opt", "*left_undistort.opt"]),
            "opt_right": try_find(["Right.opt", "Right_undistort.opt", "*Right_undistort.opt",
                                   "right.opt", "right_undistort.opt", "*right_undistort.opt"]),
            "intrinsic_left": try_find(["left_undistort_intrinsic.txt", "*left_undistort_intrinsic.txt",
                                        "Left_undistort_intrinsic.txt", "*Left_undistort_intrinsic.txt",
                                        "left_undistort_intrinsics.txt", "*left_undistort_intrinsics.txt"]),
            "intrinsic_right": try_find(["right_undistort_intrinsics.txt", "*right_undistort_intrinsics.txt",
                                         "Right_undistort_intrinsic.txt", "*Right_undistort_intrinsic.txt",
                                         "right_undistort_intrinsic.txt", "*right_undistort_intrinsic.txt"]),
            "imgpose": try_find(["ImgPose.txt", "*/ImgPose.txt"]),
            "xyzopk": try_find(["xyzopk.txt", "xyzopt.txt", "*/xyzopk.txt", "*/xyzopt.txt"]),
        }

    def _on_xyzopk_toggled(self, _checked: bool):
        """Keep the --imgpose field pointing at the right pose file: xyzopk.txt
        when 'use xyzopk' is on, ImgPose.txt otherwise (auto fields only)."""
        target = self._disc.get("xyzopk") if self.use_xyzopk.isChecked() else self._disc.get("imgpose")
        if target:
            self._auto_fill(self.imgpose_edit, target)

    def scan_project(self, announce: bool = True, reset: bool = False):
        text = self.project_edit.text().strip()
        if not text:
            if announce:
                self.project_status.setText("No project folder set.")
            return
        root = Path(text)
        if not root.is_dir():
            self.project_status.setText(f"Not a folder: {root}")
            return

        # Wipe stale auto-filled paths from a previous project (keeps manual edits).
        if reset:
            self._reset_auto_fields()

        found, missing = [], []

        # 1) undistort folder (the one containing TransformedCam.json) -> --folder
        undistort = None
        cand = root / "undistort"
        if (cand / "TransformedCam.json").exists():
            undistort = cand
        elif (root / "TransformedCam.json").exists():
            undistort = root
        else:
            for sub in sorted(p for p in root.iterdir() if p.is_dir()):
                if (sub / "TransformedCam.json").exists():
                    undistort = sub
                    break
        if undistort:
            if self._auto_fill(self.folder_edit, str(undistort)):
                found.append(f"folder → {undistort.name}/")
        else:
            missing.append("undistort/TransformedCam.json")

        # 1b) files inside the undistort folder that pose2colmap auto-discovers.
        #     Discover from whatever --folder will actually be (manual or auto)
        #     so the shown paths always match what the script would use.
        base_text = self.folder_edit.text().strip()
        base = Path(base_text) if base_text and Path(base_text).is_dir() else undistort
        if base:
            self._disc = self._discover_in_folder(base)
            d = self._disc
            if d.get("json") and self._auto_fill(self.leftjson_edit, d["json"]):
                found.append("left-json")
            if d.get("opt_left") and self._auto_fill(self.optleft_edit, d["opt_left"]):
                found.append("opt-left")
            if d.get("opt_right") and self._auto_fill(self.optright_edit, d["opt_right"]):
                found.append("opt-right")
            if d.get("intrinsic_left") and self._auto_fill(self.intrleft_edit, d["intrinsic_left"]):
                found.append("intrinsic-left")
            if d.get("intrinsic_right") and self._auto_fill(self.intrright_edit, d["intrinsic_right"]):
                found.append("intrinsic-right")
            pose = d.get("xyzopk") if self.use_xyzopk.isChecked() else d.get("imgpose")
            if pose and self._auto_fill(self.imgpose_edit, pose):
                found.append("imgpose")

        # 2) LAS point cloud -> --las
        las = self._pick_las(root)
        if las:
            if self._auto_fill(self.las_edit, str(las)):
                found.append(f"las → {las.name}")
        else:
            missing.append("*.las")

        # 3) fisheye .opt files (images/Left.opt, images/Right.opt)
        img = root / "images"
        lo, ro = img / "Left.opt", img / "Right.opt"
        if lo.exists() and self._auto_fill(self.fishleft_edit, str(lo)):
            found.append("fisheye-left-opt")
        if ro.exists() and self._auto_fill(self.fishright_edit, str(ro)):
            found.append("fisheye-right-opt")

        # 4) POLYFISHEYE factory calibration (info/calibration.yaml)
        cyaml = None
        for c in [root / "info" / "calibration.yaml", *root.rglob("calibration.yaml")]:
            if c.exists():
                cyaml = c
                break
        if cyaml and self._auto_fill(self.calyaml_edit, str(cyaml)):
            found.append("calibration.yaml")

        msg = ("Filled: " + ", ".join(found)) if found else "Nothing new to fill."
        if missing:
            msg += "   |   Not found: " + ", ".join(missing)
        self.project_status.setText(msg)

        # a --las value may now be set, which makes laspy required -> re-check
        self.check_dependencies()

    # ---------------------------------------------------------------------- #
    # Argument assembly
    # ---------------------------------------------------------------------- #
    def build_args(self) -> list[str]:
        """Return argv (excluding python + script). Value args use --k=v tokens
        so negative values / leading-dash axes never get parsed as options."""
        a: list[str] = []

        def kv(flag: str, value: str):
            a.append(f"{flag}={value}")

        t = lambda e: e.text().strip()

        if t(self.folder_edit):     kv("--folder", t(self.folder_edit))
        if t(self.leftjson_edit):   kv("--left-json", t(self.leftjson_edit))
        if t(self.rightjson_edit):  kv("--right-json", t(self.rightjson_edit))
        if t(self.optleft_edit):    kv("--opt-left", t(self.optleft_edit))
        if t(self.optright_edit):   kv("--opt-right", t(self.optright_edit))
        if t(self.intrleft_edit):   kv("--intrinsic-left", t(self.intrleft_edit))
        if t(self.intrright_edit):  kv("--intrinsic-right", t(self.intrright_edit))
        if t(self.imgpose_edit):    kv("--imgpose", t(self.imgpose_edit))
        if self.use_xyzopk.isChecked(): a.append("--use-xyzopk")

        if t(self.outdir_edit):     kv("--output-dir", t(self.outdir_edit))
        kv("--viewer-conventions", self.viewer_combo.currentText())
        if self.no_junction.isChecked(): a.append("--no-junction")

        if t(self.las_edit):        kv("--las", t(self.las_edit))
        if t(self.lasmax_edit):     kv("--las-max-points", t(self.lasmax_edit))
        kv("--points-axis", self.pts_axis.text().strip() or "xyz")
        kv("--points-pitch", self._fmt(self.pts_pitch.value()))
        kv("--points-yaw", self._fmt(self.pts_yaw.value()))
        kv("--points-roll", self._fmt(self.pts_roll.value()))
        if self.trailing_combo.currentText() != "default":
            kv("--points3d-trailing-newlines", self.trailing_combo.currentText())

        if self.cam_axis_chk.isChecked():
            kv("--camera-axis", self.cam_axis_edit.text().strip() or "x-y-z")
        if self.cam_ang_chk.isChecked():
            kv("--camera-pitch", self._fmt(self.cam_pitch.value()))
            kv("--camera-yaw", self._fmt(self.cam_yaw.value()))
            kv("--camera-roll", self._fmt(self.cam_roll.value()))
        if self.swap_lr.isChecked(): a.append("--swap-lr")

        if t(self.calyaml_edit):    kv("--calibration-yaml", t(self.calyaml_edit))
        if t(self.metaleft_edit):   kv("--metashape-left-xml", t(self.metaleft_edit))
        if t(self.metaright_edit):  kv("--metashape-right-xml", t(self.metaright_edit))

        if self.fisheye.isChecked(): a.append("--fisheye")
        if t(self.fishleft_edit):   kv("--fisheye-left-opt", t(self.fishleft_edit))
        if t(self.fishright_edit):  kv("--fisheye-right-opt", t(self.fishright_edit))

        return a

    @staticmethod
    def _fmt(v: float) -> str:
        # integer-valued floats -> no trailing ".0"
        return str(int(v)) if v == int(v) else repr(v)

    def _quote(self, s: str) -> str:
        return f'"{s}"' if (" " in s or not s) else s

    def full_command(self):
        py = self.py_edit.text().strip() or sys.executable
        script = self.script_edit.text().strip()
        return py, script, self.build_args()

    # ---------------------------------------------------------------------- #
    # Actions
    # ---------------------------------------------------------------------- #
    def on_preview(self):
        py, script, args = self.full_command()
        parts = [self._quote(py), self._quote(script)] + [self._quote(x) for x in args]
        self.cmd_preview.setText(" ".join(parts))

    def on_copy(self):
        self.on_preview()
        QApplication.clipboard().setText(self.cmd_preview.text())
        self.statusBar().showMessage("Command copied to clipboard", 2500)

    def on_run(self):
        py, script, args = self.full_command()
        if not script or not Path(script).exists():
            QMessageBox.warning(self, "Missing script",
                                "Set a valid path to pose2colmap.py in the Run configuration section.")
            return
        if not self.folder_edit.text().strip() and not self.leftjson_edit.text().strip():
            QMessageBox.warning(self, "Missing input",
                                "The script requires either --folder or --left-json.")
            return
        if self.check_dependencies():
            QMessageBox.warning(self, "Missing packages",
                                "Required Python packages are missing. Use the red banner "
                                "at the top to install them, then run again.")
            return

        self.on_preview()
        self.log.appendPlainText(f"$ {self.cmd_preview.text()}\n")
        cwd = str(Path(script).resolve().parent)
        self._launch(py, ["-u", script] + args, mode="run", cwd=cwd)

    # ---- shared process launcher ---------------------------------------- #
    def _launch(self, program: str, args: list[str], mode: str, cwd: str | None = None):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            QMessageBox.information(self, "Busy", "A process is already running.")
            return
        self.proc = QProcess(self)
        self._proc_mode = mode
        self.proc.setProcessChannelMode(QProcess.MergedChannels)
        if cwd:
            self.proc.setWorkingDirectory(cwd)
        self.proc.readyReadStandardOutput.connect(self._on_output)
        self.proc.finished.connect(self._on_finished)
        self.proc.errorOccurred.connect(self._on_error)
        self.run_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.dep_banner.setEnabled(False)
        self.proc.start(program, args)

    def on_stop(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            self.proc.kill()
            self.log.appendPlainText("\n[stopped by user]\n")

    # ---------------------------------------------------------------------- #
    # Dependency check / one-click install
    # ---------------------------------------------------------------------- #
    def _required_packages(self) -> dict:
        """import-name -> pip-name. yaml & numpy are always needed; laspy only
        when a --las cloud is set (its import is optional in the script)."""
        req = {"yaml": "pyyaml", "numpy": "numpy"}
        if self.las_edit.text().strip():
            req["laspy"] = "laspy"
        return req

    def _missing_packages(self) -> list[str]:
        """Return the pip names of missing packages for the selected interpreter."""
        py = self.py_edit.text().strip() or sys.executable
        req = self._required_packages()
        probe = ("import importlib.util as u;"
                 f"mods={list(req.keys())!r};"
                 "print('\\n'.join(m for m in mods if u.find_spec(m) is None))")
        kwargs = {}
        if os.name == "nt":
            kwargs["creationflags"] = 0x08000000  # CREATE_NO_WINDOW (no console flash)
        try:
            out = subprocess.run([py, "-c", probe], capture_output=True,
                                 text=True, timeout=20, **kwargs)
        except Exception:
            return []  # can't probe (bad interpreter path) -> don't nag
        missing_imports = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return [req[m] for m in missing_imports if m in req]

    def check_dependencies(self) -> list[str]:
        """Update the red banner. Returns the list of missing pip names."""
        missing = self._missing_packages()
        if missing:
            names = ", ".join(missing)
            self.dep_banner.setText(
                f"⚠  The {names} package(s) are missing. pose2colmap cannot run "
                "before they are installed.  Click to install now.")
            self.dep_banner.setVisible(True)
        else:
            self.dep_banner.setVisible(False)
        return missing

    def install_dependencies(self):
        if self.proc and self.proc.state() != QProcess.NotRunning:
            return
        missing = self._missing_packages()
        if not missing:
            self.check_dependencies()
            return
        py = self.py_edit.text().strip() or sys.executable
        cmd = [py, "-m", "pip", "install", *missing]
        self.log.appendPlainText("$ " + " ".join(self._quote(c) for c in cmd) + "\n")
        self._launch(py, ["-m", "pip", "install", *missing], mode="install")

    def _on_output(self):
        if not self.proc:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        self.log.moveCursor(QTextCursor.End)
        self.log.insertPlainText(data)
        self.log.moveCursor(QTextCursor.End)

    def _on_finished(self, code, _status):
        mode = getattr(self, "_proc_mode", "run")
        self.log.appendPlainText(f"\n[process exited with code {code}]\n")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.dep_banner.setEnabled(True)
        self.proc = None
        self._proc_mode = None
        if mode == "install":
            if code == 0:
                self.log.appendPlainText("[install finished — re-checking packages]\n")
            self.check_dependencies()
        elif mode == "run" and code == 0 and self.recenter_chk.isChecked():
            self._do_recenter()

    def _resolve_output_dir(self) -> Path | None:
        """Mirror pose2colmap's output-dir logic so we can find sparse/ after a run."""
        out = self.outdir_edit.text().strip()
        if out:
            return Path(out)
        folder = self.folder_edit.text().strip()
        if not folder:
            return None
        parent = Path(folder).resolve().parent
        fe = "_fisheye" if self.fisheye.isChecked() else ""
        viewer = self.viewer_combo.currentText()
        name = f"COLMAP_LFS{fe}" if viewer in ("LFS", "PS") else f"COLMAP_RS2{fe}"
        return parent / name

    def _do_recenter(self):
        out = self._resolve_output_dir()
        if not out:
            self.log.appendPlainText("[recenter] could not resolve output dir — skipped.\n")
            return
        sparse = out / "sparse"
        mode_map = {"Camera centroid": "cameras",
                    "Point cloud centroid": "points",
                    "First camera": "first_camera"}
        mode = mode_map.get(self.recenter_mode.currentText(), "cameras")
        self.log.appendPlainText(f"\n[recenter] centering on {mode} in {sparse} …\n")
        try:
            res = recenter_colmap(sparse, mode=mode)
        except Exception as e:
            self.log.appendPlainText(f"[recenter] FAILED: {e}\n")
            return
        ox, oy, oz = res["offset"]
        self.log.appendPlainText(
            f"[recenter] done — shifted {res['n_cameras']} cameras and "
            f"{res['n_points']} points.\n"
            f"[recenter] offset subtracted: ({ox:.3f}, {oy:.3f}, {oz:.3f})\n"
            f"[recenter] saved recenter_offset.txt / .json in sparse/ (add back to georeference).\n")

    def _on_error(self, err):
        self.log.appendPlainText(f"\n[QProcess error: {err}]\n")
        self.run_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.dep_banner.setEnabled(True)


def main():
    app = QApplication(sys.argv)
    apply_dark_fusion(app)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
