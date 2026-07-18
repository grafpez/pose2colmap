#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pose2colmap_gui.py -- Dark Fusion GUI wrapper for pose2colmap.py

v1.1 - recursive auto-discovery fix
"""

import os, subprocess, sys
from pathlib import Path

from PySide6.QtCore import Qt, QProcess, QTimer
from PySide6.QtGui import QColor, QFont, QPalette, QTextCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QScrollArea, QVBoxLayout, QHBoxLayout,
    QGridLayout, QGroupBox, QLabel, QLineEdit, QPushButton, QCheckBox,
    QComboBox, QDoubleSpinBox, QPlainTextEdit, QFileDialog, QMessageBox,
    QSplitter,
)

_VERSION = 'v1.1'

def apply_dark_fusion(app: QApplication) -> None:
	app.setStyle('Fusion')
	app.setFont(QFont('Arial', 11))
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

def path_row(placeholder, pick_dir=False, file_filter='All files (*.*)'):
	w = QWidget()
	lay = QHBoxLayout(w)
	lay.setContentsMargins(0, 0, 0, 0)
	edit = QLineEdit()
	edit.setPlaceholderText(placeholder)
	btn = QPushButton('Browse...')
	btn.setFixedWidth(90)
	def browse():
		if pick_dir:
			chosen = QFileDialog.getExistingDirectory(w, 'Select folder', edit.text() or '')
		else:
			chosen, _ = QFileDialog.getOpenFileName(w, 'Select file', edit.text() or '', file_filter)
		if chosen:
			edit.setText(chosen)
	btn.clicked.connect(browse)
	lay.addWidget(edit, 1)
	lay.addWidget(btn)
	return w, edit

def angle_spin(default):
	s = QDoubleSpinBox()
	s.setRange(-360.0, 360.0)
	s.setDecimals(3)
	s.setSingleStep(1.0)
	s.setValue(default)
	s.setFixedWidth(120)
	return s

class MainWindow(QMainWindow):
	def __init__(self):
		super().__init__()
		self.setWindowTitle('pose2colmap \u2014 GUI ' + _VERSION)
		self.resize(920, 860)
		self.proc = None
		self._proc_mode = None
		self._auto = {}
		self._disc = {}
		here = Path(__file__).resolve().parent
		guess = here / 'pose2colmap.py'

		form = QWidget()
		fl = QVBoxLayout(form)
		fl.setSpacing(10)

		# Project
		g_proj = QGroupBox('Project (Point Cloud Studio output folder)')
		proj_l = QGridLayout(g_proj)
		proj_l.addWidget(QLabel('Project folder'), 0, 0)
		prow = QWidget(); prl = QHBoxLayout(prow); prl.setContentsMargins(0, 0, 0, 0)
		self.project_edit = QLineEdit()
		self.project_edit.setPlaceholderText('Set Project Folder')
		proj_browse = QPushButton('Browse...'); proj_browse.setFixedWidth(90)
		proj_rescan = QPushButton('Re-scan'); proj_rescan.setFixedWidth(90)
		proj_browse.clicked.connect(self._browse_project)
		proj_rescan.clicked.connect(lambda: self._reload_project(force=True))
		self.project_edit.editingFinished.connect(self._reload_project)
		prl.addWidget(self.project_edit, 1)
		prl.addWidget(proj_browse)
		prl.addWidget(proj_rescan)
		proj_l.addWidget(prow, 0, 1, 1, 2)
		self.auto_pop = QCheckBox('Auto-populate fields from project folder')
		self.auto_pop.setChecked(True)
		proj_l.addWidget(self.auto_pop, 1, 1, 1, 2)
		self.project_status = QLabel('Pick a project folder to auto-fill the fields below.')
		self.project_status.setWordWrap(True)
		self.project_status.setStyleSheet('color: #9aa0a6;')
		proj_l.addWidget(self.project_status, 2, 1, 1, 2)
		fl.addWidget(g_proj)

		# Run config
		g_run = QGroupBox('Run configuration')
		gl = QGridLayout(g_run)
		self.py_edit = QLineEdit(sys.executable)
		py_w, self.py_edit = self._labeled_path(gl, 0, 'Python interpreter',
			self.py_edit.text(), pick_dir=False,
			file_filter='Python (python* python.exe *);;All files (*.*)')
		scr_w, self.script_edit = self._labeled_path(
			gl, 1, 'pose2colmap.py',
			str(guess) if guess.exists() else '',
			pick_dir=False, file_filter='Python (*.py);;All files (*.*)')
		fl.addWidget(g_run)

		# Input
		g_in = QGroupBox('Input (use --folder for auto-discovery, or set files manually)')
		il = QGridLayout(g_in)
		_, self.folder_edit = self._labeled_path(il, 0, 'Undistort folder (--folder)', '', pick_dir=True)
		_, self.leftjson_edit = self._labeled_path(il, 1, '--left-json', '', file_filter='JSON (*.json);;All files (*.*)')
		_, self.rightjson_edit = self._labeled_path(il, 2, '--right-json', '', file_filter='JSON (*.json);;All files (*.*)')
		_, self.optleft_edit = self._labeled_path(il, 3, '--opt-left', '', file_filter='OPT (*.opt);;All files (*.*)')
		_, self.optright_edit = self._labeled_path(il, 4, '--opt-right', '', file_filter='OPT (*.opt);;All files (*.*)')
		_, self.intrleft_edit = self._labeled_path(il, 5, '--intrinsic-left', '', file_filter='Text (*.txt);;All files (*.*)')
		_, self.intrright_edit = self._labeled_path(il, 6, '--intrinsic-right', '', file_filter='Text (*.txt);;All files (*.*)')
		_, self.imgpose_edit = self._labeled_path(il, 7, '--imgpose', '', file_filter='Text (*.txt);;All files (*.*)')
		self.use_xyzopk = QCheckBox('Use xyzopk.txt for poses (--use-xyzopk)')
		il.addWidget(self.use_xyzopk, 8, 1, 1, 2)
		self.use_xyzopk.toggled.connect(self._on_xyzopk_toggled)
		self.rightjson_edit.setToolTip(
			'Only for separate-RIGHT mode. A normal PCS project has a single '
			"TransformedCam.json (mapped to --left-json), so leave this empty.")
		fl.addWidget(g_in)

		# Output
		g_out = QGroupBox('Output')
		ol = QGridLayout(g_out)
		_, self.outdir_edit = self._labeled_path(ol, 0, '--output-dir (default: <parent>/COLMAP/)', '', pick_dir=True)
		ol.addWidget(QLabel('Viewer conventions'), 1, 0)
		self.viewer_combo = QComboBox()
		self.viewer_combo.addItems(['LFS', 'PS', 'RS2'])
		ol.addWidget(self.viewer_combo, 1, 1)
		self.no_junction = QCheckBox('Skip image junction links (--no-junction)')
		ol.addWidget(self.no_junction, 2, 1, 1, 2)
		fl.addWidget(g_out)

		# Point cloud / LAS
		g_pts = QGroupBox('Point cloud / LAS')
		pl = QGridLayout(g_pts)
		_, self.las_edit = self._labeled_path(pl, 0, '--las (.las / .laz)', '',
			file_filter='LAS/LAZ (*.las *.laz);;All files (*.*)')
		pl.addWidget(QLabel('--las-max-points'), 1, 0)
		self.lasmax_edit = QLineEdit()
		self.lasmax_edit.setPlaceholderText('empty = unlimited')
		self.lasmax_edit.setFixedWidth(160)
		pl.addWidget(self.lasmax_edit, 1, 1, Qt.AlignLeft)
		pl.addWidget(QLabel('--points-axis'), 2, 0)
		self.pts_axis = QLineEdit('xyz')
		self.pts_axis.setFixedWidth(160)
		pl.addWidget(self.pts_axis, 2, 1, Qt.AlignLeft)
		row = QWidget(); rl = QHBoxLayout(row); rl.setContentsMargins(0, 0, 0, 0)
		rl.addWidget(QLabel('pitch')); self.pts_pitch = angle_spin(90.0); rl.addWidget(self.pts_pitch)
		rl.addWidget(QLabel('yaw')); self.pts_yaw = angle_spin(-90.0); rl.addWidget(self.pts_yaw)
		rl.addWidget(QLabel('roll')); self.pts_roll = angle_spin(0.0); rl.addWidget(self.pts_roll)
		rl.addStretch(1)
		pl.addWidget(QLabel('points pitch/yaw/roll'), 3, 0)
		pl.addWidget(row, 3, 1, 1, 2)
		pl.addWidget(QLabel('--points3d-trailing-newlines'), 4, 0)
		self.trailing_combo = QComboBox()
		self.trailing_combo.addItems(['default', '0', '1', '2'])
		self.trailing_combo.setFixedWidth(160)
		pl.addWidget(self.trailing_combo, 4, 1, Qt.AlignLeft)
		fl.addWidget(g_pts)

		# Camera transform
		g_cam = QGroupBox('Camera transform (leave override off to use script defaults)')
		cl = QGridLayout(g_cam)
		self.cam_axis_chk = QCheckBox('override --camera-axis')
		self.cam_axis_edit = QLineEdit('x-y-z'); self.cam_axis_edit.setFixedWidth(160)
		self.cam_axis_edit.setEnabled(False)
		self.cam_axis_chk.toggled.connect(self.cam_axis_edit.setEnabled)
		cl.addWidget(self.cam_axis_chk, 0, 0)
		cl.addWidget(self.cam_axis_edit, 0, 1, Qt.AlignLeft)
		self.cam_ang_chk = QCheckBox('override --camera-pitch/yaw/roll')
		cl.addWidget(self.cam_ang_chk, 1, 0)
		crow = QWidget(); crl = QHBoxLayout(crow); crl.setContentsMargins(0, 0, 0, 0)
		crl.addWidget(QLabel('pitch')); self.cam_pitch = angle_spin(-90.0); crl.addWidget(self.cam_pitch)
		crl.addWidget(QLabel('yaw')); self.cam_yaw = angle_spin(-90.0); crl.addWidget(self.cam_yaw)
		crl.addWidget(QLabel('roll')); self.cam_roll = angle_spin(0.0); crl.addWidget(self.cam_roll)
		crl.addStretch(1)
		for s in (self.cam_pitch, self.cam_yaw, self.cam_roll): s.setEnabled(False)
		self.cam_ang_chk.toggled.connect(lambda on: [s.setEnabled(on) for s in (self.cam_pitch, self.cam_yaw, self.cam_roll)])
		cl.addWidget(crow, 1, 1, 1, 2)
		self.swap_lr = QCheckBox('Swap LEFT/RIGHT (--swap-lr)')
		cl.addWidget(self.swap_lr, 2, 1, 1, 2)
		fl.addWidget(g_cam)

		# Calibration
		g_cal = QGroupBox('Calibration (optional)')
		call = QGridLayout(g_cal)
		_, self.calyaml_edit = self._labeled_path(call, 0, '--calibration-yaml', '',
			file_filter='YAML (*.yaml *.yml);;All files (*.*)')
		_, self.metaleft_edit = self._labeled_path(call, 1, '--metashape-left-xml', '',
			file_filter='XML (*.xml);;All files (*.*)')
		_, self.metaright_edit = self._labeled_path(call, 2, '--metashape-right-xml', '',
			file_filter='XML (*.xml);;All files (*.*)')
		fl.addWidget(g_cal)

		# Fisheye
		g_fish = QGroupBox('Fisheye (optional)')
		fll = QGridLayout(g_fish)
		self.fisheye = QCheckBox('Use raw fisheye images (--fisheye)')
		fll.addWidget(self.fisheye, 0, 1, 1, 2)
		_, self.fishleft_edit = self._labeled_path(fll, 1, '--fisheye-left-opt', '',
			file_filter='OPT (*.opt);;All files (*.*)')
		_, self.fishright_edit = self._labeled_path(fll, 2, '--fisheye-right-opt', '',
			file_filter='OPT (*.opt);;All files (*.*)')
		fl.addWidget(g_fish)
		fl.addStretch(1)

		scroll = QScrollArea()
		scroll.setWidgetResizable(True)
		scroll.setWidget(form)

		# Output console
		console_box = QWidget()
		cvl = QVBoxLayout(console_box)
		cvl.setContentsMargins(0, 0, 0, 0)
		self.cmd_preview = QLineEdit()
		self.cmd_preview.setReadOnly(True)
		self.cmd_preview.setPlaceholderText('Assembled command appears here')
		self.cmd_preview.setFont(QFont('Consolas', 10))
		self.log = QPlainTextEdit()
		self.log.setReadOnly(True)
		self.log.setFont(QFont('Consolas', 10))
		cvl.addWidget(QLabel('Command:'))
		cvl.addWidget(self.cmd_preview)
		cvl.addWidget(QLabel('Output:'))
		cvl.addWidget(self.log, 1)
		splitter = QSplitter(Qt.Vertical)
		splitter.addWidget(scroll)
		splitter.addWidget(console_box)
		splitter.setStretchFactor(0, 3)
		splitter.setStretchFactor(1, 2)

		# Buttons
		btn_bar = QWidget()
		bl = QHBoxLayout(btn_bar)
		self.preview_btn = QPushButton('Preview command')
		self.copy_btn = QPushButton('Copy command')
		self.run_btn = QPushButton('Run')
		self.stop_btn = QPushButton('Stop')
		self.clear_btn = QPushButton('Clear log')
		self.open_out_btn = QPushButton('Open Output Folder')
		self.stop_btn.setEnabled(False)
		self.run_btn.setDefault(True)
		for b in (self.preview_btn, self.copy_btn, self.clear_btn, self.open_out_btn): bl.addWidget(b)
		bl.addStretch(1)
		bl.addWidget(self.stop_btn)
		bl.addWidget(self.run_btn)
		self.preview_btn.clicked.connect(self.on_preview)
		self.copy_btn.clicked.connect(self.on_copy)
		self.run_btn.clicked.connect(self.on_run)
		self.stop_btn.clicked.connect(self.on_stop)
		self.clear_btn.clicked.connect(lambda: self.log.clear())
		self.open_out_btn.clicked.connect(self.on_open_output)

		self._managed = [
			self.folder_edit, self.leftjson_edit,
			self.optleft_edit, self.optright_edit,
			self.intrleft_edit, self.intrright_edit,
			self.imgpose_edit, self.las_edit,
			self.fishleft_edit, self.fishright_edit,
			self.calyaml_edit,
		]
		self._last_project = ''

		# Dependency banner
		self.dep_banner = QPushButton()
		self.dep_banner.setVisible(False)
		self.dep_banner.setCursor(Qt.PointingHandCursor)
		self.dep_banner.setStyleSheet(
			'QPushButton { background-color: #7a1f1f; color: #fff; border: 1px solid #f55; '
			'border-radius: 4px; padding: 8px; text-align: left; font-weight: bold; }'
			'QPushButton:hover { background-color: #8f2626; }'
			'QPushButton:disabled { background-color: #4a2020; color: #cbb; }')
		self.dep_banner.clicked.connect(self.install_dependencies)

		central = QWidget()
		cl2 = QVBoxLayout(central)
		cl2.addWidget(self.dep_banner)
		cl2.addWidget(splitter, 1)
		cl2.addWidget(btn_bar)
		self.setCentralWidget(central)

		self.py_edit.editingFinished.connect(self.check_dependencies)
		self.las_edit.editingFinished.connect(self.check_dependencies)
		QTimer.singleShot(0, self.check_dependencies)

	def _labeled_path(self, grid, r, label, initial, pick_dir=False, file_filter='All files (*.*)'):
		grid.addWidget(QLabel(label), r, 0)
		w, edit = path_row(label, pick_dir=pick_dir, file_filter=file_filter)
		if initial: edit.setText(initial)
		grid.addWidget(w, r, 1, 1, 2)
		return w, edit

	def _browse_project(self):
		chosen = QFileDialog.getExistingDirectory(
			self, 'Select Point Cloud Studio project (output) folder',
			self.project_edit.text() or '')
		if chosen:
			self.project_edit.setText(chosen)
			self._reload_project()

	def _reload_project(self, force=False):
		new = self.project_edit.text().strip()
		if not force:
			if not self.auto_pop.isChecked(): return
		if new == self._last_project: return
		self._last_project = new
		self.scan_project(reset=True, announce=True)

	def _reset_auto_fields(self):
		for edit in self._managed:
			prev = self._auto.get(id(edit))
			if prev is not None and edit.text().strip() == prev:
				edit.clear()
				self._auto.pop(id(edit), None)

	def _auto_fill(self, edit, value):
		if not value: return False
		prev = self._auto.get(id(edit))
		cur = edit.text().strip()
		if cur == '' or cur == prev:
			edit.setText(value)
			self._auto[id(edit)] = value
			return True
		return False

	@staticmethod
	def _pick_las(root):
		files = sorted(list(root.rglob('*.las')) + list(root.rglob('*.laz')))
		if not files: return None
		col = [f for f in files if 'colorized' in f.name.lower() and 'uncolorized' not in f.name.lower()]
		return col[0] if col else files[0]

	@staticmethod
	def _discover_in_folder(folder):
		def try_find(patterns):
			for pat in patterns:
				deep = sorted(folder.glob('**/' + pat))
				if deep: return str(deep[0].resolve())
				flat = sorted(folder.glob(pat))
				if flat: return str(flat[0].resolve())
			return None
		return {
			'json': try_find(['TransformedCam.json']),
			'opt_left': try_find(['Left.opt','Left_undistort.opt','*Left_undistort.opt','left.opt','left_undistort.opt','*left_undistort.opt']),
			'opt_right': try_find(['Right.opt','Right_undistort.opt','*Right_undistort.opt','right.opt','right_undistort.opt','*right_undistort.opt']),
			'intrinsic_left': try_find(['left_undistort_intrinsic.txt','*left_undistort_intrinsic.txt','Left_undistort_intrinsic.txt','*Left_undistort_intrinsic.txt','left_undistort_intrinsics.txt','*left_undistort_intrinsics.txt']),
			'intrinsic_right': try_find(['right_undistort_intrinsic.txt','*right_undistort_intrinsic.txt','Right_undistort_intrinsic.txt','*Right_undistort_intrinsic.txt','right_undistort_intrinsics.txt','*right_undistort_intrinsics.txt']),
			'imgpose': try_find(['ImgPose.txt']),
			'xyzopk': try_find(['xyzopk.txt','xyzopt.txt']),
		}

	def _on_xyzopk_toggled(self, _checked):
		target = self._disc.get('xyzopk') if self.use_xyzopk.isChecked() else self._disc.get('imgpose')
		if target: self._auto_fill(self.imgpose_edit, target)

	def scan_project(self, announce=True, reset=False):
		text = self.project_edit.text().strip()
		if not text:
			if announce: self.project_status.setText('No project folder set.')
			return
		root = Path(text)
		if not root.is_dir():
			self.project_status.setText('Not a folder: ' + str(root))
			return
		if reset: self._reset_auto_fields()
		found, missing = [], []

		undistort = None
		deep_json = sorted(root.rglob('TransformedCam.json'))
		if deep_json: undistort = deep_json[0].parent
		if undistort:
			if self._auto_fill(self.folder_edit, str(undistort)):
				found.append('folder -> ' + undistort.name + '/')
		else:
			missing.append('TransformedCam.json')

		base_text = self.folder_edit.text().strip()
		base = Path(base_text) if base_text and Path(base_text).is_dir() else undistort
		if base:
			self._disc = self._discover_in_folder(base)
			d = self._disc
			if d.get('json') and self._auto_fill(self.leftjson_edit, d['json']): found.append('left-json')
			if d.get('opt_left') and self._auto_fill(self.optleft_edit, d['opt_left']): found.append('opt-left')
			if d.get('opt_right') and self._auto_fill(self.optright_edit, d['opt_right']): found.append('opt-right')
			if d.get('intrinsic_left') and self._auto_fill(self.intrleft_edit, d['intrinsic_left']): found.append('intrinsic-left')
			if d.get('intrinsic_right') and self._auto_fill(self.intrright_edit, d['intrinsic_right']): found.append('intrinsic-right')
			pose = d.get('xyzopk') if self.use_xyzopk.isChecked() else d.get('imgpose')
			if pose and self._auto_fill(self.imgpose_edit, pose): found.append('imgpose')

		las = self._pick_las(root)
		if las:
			if self._auto_fill(self.las_edit, str(las)): found.append('las -> ' + las.name)
		else:
			missing.append('*.las')

		lo = sorted(root.rglob('Left.opt'))
		if lo and self._auto_fill(self.fishleft_edit, str(lo[0])): found.append('fisheye-left-opt')
		ro = sorted(root.rglob('Right.opt'))
		if ro and self._auto_fill(self.fishright_edit, str(ro[0])): found.append('fisheye-right-opt')

		cyaml = sorted(root.rglob('calibration.yaml'))
		if cyaml and self._auto_fill(self.calyaml_edit, str(cyaml[0])): found.append('calibration.yaml')

		msg = ('Filled: ' + ', '.join(found)) if found else 'Nothing new to fill.'
		if missing: msg += ' | Not found: ' + ', '.join(missing)
		self.project_status.setText(msg)
		self.check_dependencies()

	def check_dependencies(self):
		missing = []
		for mod in ['numpy', 'yaml', 'PIL']:
			try:
				__import__(mod)
			except ImportError:
				missing.append(mod)
		mod_names = {'numpy': 'numpy', 'yaml': 'pyyaml', 'PIL': 'pillow'}
		if missing:
			names = ', '.join(mod_names[m] for m in missing)
			self.dep_banner.setText('⚠ Missing: ' + names + ' — click to install')
			self.dep_banner.setVisible(True)
			self._dep_missing = missing
			return True
		else:
			self.dep_banner.setVisible(False)
			self._dep_missing = []
			return False

	def install_dependencies(self):
		missing = getattr(self, '_dep_missing', [])
		if not missing:
			self.check_dependencies()
			return
		mod_names = {'numpy': 'numpy', 'yaml': 'pyyaml', 'PIL': 'pillow'}
		pkgs = [mod_names[m] for m in missing]
		self.dep_banner.setText('Installing ' + ', '.join(pkgs) + '…')
		self.dep_banner.setEnabled(False)
		py = self.py_edit.text().strip() or sys.executable
		self._launch(py, ['-m', 'pip', 'install', '--quiet', '--user'] + pkgs + ['--break-system-packages'],
			mode='install')

	def build_args(self):
		a = []
		def kv(flag, value): a.append(flag + '=' + value)
		t = lambda e: e.text().strip()
		if t(self.folder_edit): kv('--folder', t(self.folder_edit))
		if t(self.leftjson_edit): kv('--left-json', t(self.leftjson_edit))
		if t(self.rightjson_edit): kv('--right-json', t(self.rightjson_edit))
		if t(self.optleft_edit): kv('--opt-left', t(self.optleft_edit))
		if t(self.optright_edit): kv('--opt-right', t(self.optright_edit))
		if t(self.intrleft_edit): kv('--intrinsic-left', t(self.intrleft_edit))
		if t(self.intrright_edit): kv('--intrinsic-right', t(self.intrright_edit))
		if t(self.imgpose_edit): kv('--imgpose', t(self.imgpose_edit))
		if self.use_xyzopk.isChecked(): a.append('--use-xyzopk')
		if t(self.outdir_edit): kv('--output-dir', t(self.outdir_edit))
		kv('--viewer-conventions', self.viewer_combo.currentText())
		if self.no_junction.isChecked(): a.append('--no-junction')
		if t(self.las_edit): kv('--las', t(self.las_edit))
		if t(self.lasmax_edit): kv('--las-max-points', t(self.lasmax_edit))
		kv('--points-axis', self.pts_axis.text().strip() or 'xyz')
		kv('--points-pitch', self._fmt(self.pts_pitch.value()))
		kv('--points-yaw', self._fmt(self.pts_yaw.value()))
		kv('--points-roll', self._fmt(self.pts_roll.value()))
		if self.trailing_combo.currentText() != 'default':
			kv('--points3d-trailing-newlines', self.trailing_combo.currentText())
		if self.cam_axis_chk.isChecked():
			kv('--camera-axis', self.cam_axis_edit.text().strip() or 'x-y-z')
		if self.cam_ang_chk.isChecked():
			kv('--camera-pitch', self._fmt(self.cam_pitch.value()))
			kv('--camera-yaw', self._fmt(self.cam_yaw.value()))
			kv('--camera-roll', self._fmt(self.cam_roll.value()))
		if self.swap_lr.isChecked(): a.append('--swap-lr')
		if t(self.calyaml_edit): kv('--calibration-yaml', t(self.calyaml_edit))
		if t(self.metaleft_edit): kv('--metashape-left-xml', t(self.metaleft_edit))
		if t(self.metaright_edit): kv('--metashape-right-xml', t(self.metaright_edit))
		if self.fisheye.isChecked(): a.append('--fisheye')
		if t(self.fishleft_edit): kv('--fisheye-left-opt', t(self.fishleft_edit))
		if t(self.fishright_edit): kv('--fisheye-right-opt', t(self.fishright_edit))
		return a

	@staticmethod
	def _fmt(v):
		return str(int(v)) if v == int(v) else repr(v)

	def _quote(self, s):
		return '"' + s + '"' if (' ' in s or not s) else s

	def full_command(self):
		py = self.py_edit.text().strip() or sys.executable
		script = self.script_edit.text().strip()
		return py, script, self.build_args()

	def on_preview(self):
		py, script, args = self.full_command()
		parts = [self._quote(py), self._quote(script)] + [self._quote(x) for x in args]
		self.cmd_preview.setText(' '.join(parts))

	def on_copy(self):
		self.on_preview()
		QApplication.clipboard().setText(self.cmd_preview.text())
		self.statusBar().showMessage('Command copied to clipboard', 2500)

	def on_run(self):
		py, script, args = self.full_command()
		if not script or not Path(script).exists():
			QMessageBox.warning(self, 'Missing script', 'Set a valid path to pose2colmap.py.')
			return
		if not self.folder_edit.text().strip() and not self.leftjson_edit.text().strip():
			QMessageBox.warning(self, 'Missing input', 'The script requires either --folder or --left-json.')
			return
		if self.check_dependencies():
			QMessageBox.warning(self, 'Missing packages',
				'Required Python packages are missing. Use the red banner at the top to install them.')
			return
		self.on_preview()
		self.log.appendPlainText('$ ' + self.cmd_preview.text() + '\n')
		cwd = str(Path(script).resolve().parent)
		self._launch(py, ['-u', script] + args, mode='run', cwd=cwd)

	def _launch(self, program, args, mode, cwd=None):
		if self.proc and self.proc.state() != QProcess.NotRunning:
			QMessageBox.information(self, 'Busy', 'A process is already running.')
			return
		self.proc = QProcess(self)
		self._proc_mode = mode
		self.proc.setProcessChannelMode(QProcess.MergedChannels)
		if cwd: self.proc.setWorkingDirectory(cwd)
		self.proc.readyReadStandardOutput.connect(self._on_output)
		self.proc.finished.connect(self._on_finished)
		self.proc.errorOccurred.connect(self._on_error)
		self.run_btn.setEnabled(False)
		self.stop_btn.setEnabled(True)
		self.dep_banner.setEnabled(False)
		self.proc.start(program, args)

	def _on_output(self):
		data = self.proc.readAllStandardOutput().data().decode('utf-8', errors='replace')
		self.log.appendPlainText(data)

	def _on_finished(self, exit_code, exit_status):
		mode = getattr(self, '_proc_mode', None)
		if mode == 'install':
			self.check_dependencies()
			self.dep_banner.setEnabled(True)
		else:
			if exit_code == 0:
				self.log.appendPlainText('[Done — exit code 0]\n')
			else:
				self.log.appendPlainText(f'[Finished — exit code {exit_code}]\n')
		self.run_btn.setEnabled(True)
		self.stop_btn.setEnabled(False)
		self.dep_banner.setEnabled(True)

	def _on_error(self, error):
		self.log.appendPlainText(f'[Process error: {error}]\n')
		self.run_btn.setEnabled(True)
		self.stop_btn.setEnabled(False)
		self.dep_banner.setEnabled(True)

	def on_stop(self):
		if self.proc and self.proc.state() != QProcess.NotRunning:
			self.proc.kill()
			self.stop_btn.setEnabled(False)
			self.run_btn.setEnabled(True)
			self.dep_banner.setEnabled(True)
			self.log.appendPlainText('[Stopped]\n')

	def on_open_output(self):
		# Viewer convention -> folder name suffix
		viewer = self.viewer_combo.currentText()
		fe = '_fisheye' if self.fisheye.isChecked() else ''
		out = self.outdir_edit.text().strip()
		if not out:
			folder = self.folder_edit.text().strip()
			if not folder:
				QMessageBox.information(self, "No output folder",
				    "Set an input folder or an output folder first.")
				return
			parent = Path(folder).resolve().parent
			if viewer in ('LFS', 'PS'):
				out = str(parent / f'COLMAP_LFS{fe}')
			else:
				out = str(parent / f'COLMAP_RS2{fe}')
		else:
			if not Path(out).is_absolute():
				out = str(Path.cwd() / out)
		out_path = Path(out)
		if out_path.is_dir():
			os.startfile(out)
		elif out_path.exists():
			os.startfile(str(out_path.parent))
		else:
			folder = self.folder_edit.text().strip()
			parent = Path(folder).resolve().parent if folder else out_path.parent
			candidates = [str(parent / f'COLMAP_LFS{fe}'),
				str(parent / f'COLMAP_RS2{fe}'),
				str(parent / 'COLMAP_LFS'),
				str(parent / 'COLMAP_RS2'),
				str(parent / 'COLMAP')]
			found = next((c for c in candidates if Path(c).is_dir()), None)
			if found:
				os.startfile(found)
			else:
				try:
					os.startfile(str(parent))
				except FileNotFoundError:
					import subprocess as _sub
					_sub.run(['explorer.exe', str(parent)], check=False)
				QMessageBox.information(self, "Folder not yet created",
					"Expected output folder:\n" + out + "\n\n"
					"Does not exist yet. Its parent has been opened instead.\n"
					"The folder will be created when pose2colmap.py runs successfully.")

def main():
	app = QApplication(sys.argv)
	apply_dark_fusion(app)
	win = MainWindow()
	win.show()
	sys.exit(app.exec())

if __name__ == '__main__':
	main()
