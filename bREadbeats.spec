# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# Include data files (presets.json, icon, and splash screen)
datas = [
    ('presets.json', '.'),
    ('bREadbeats.ico', '.'),
    ('splash_screen.png', '.')
]
binaries = []

# Hidden imports - all libraries that might not be auto-detected
hiddenimports = [
    'sounddevice', 'aubio', 'pyaudiowpatch', 'numpy',
    'PyQt6', 'dateutil', 'six', 'fonttools', 'comtypes', 'pillow',
    'pyqtgraph',
    'scipy', 'scipy.signal', 'scipy.signal._sosfilt', 'scipy.signal._lfilter',
    'PyQt6.QtCore', 'PyQt6.QtGui', 'PyQt6.QtWidgets',
    'queue', 'threading', 'json', 'pathlib'
]

# Collect all data and binaries for PyQt6
tmp_ret = collect_all('PyQt6')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]
# scipy and matplotlib: Only specific functions used, no need to collect_all


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='bREadbeats',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Console window for debugging (set False for release)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='bREadbeats.ico',  # Icon for Windows Explorer AND taskbar
)
