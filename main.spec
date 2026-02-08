# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

# Collect PyQt6 properly
pyqt_datas, pyqt_binaries, pyqt_hiddenimports = collect_all('PyQt6')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=pyqt_binaries,
    datas=[('bREadbeats.ico', '.'), ('splash_screen.png', '.'), ('C:\\Users\\andre\\.bREadbeats\\presets.json', '.bREadbeats')] + pyqt_datas,
    hiddenimports=['scipy', 'scipy.signal', 'scipy.signal._sosfilt', 'scipy.signal._lfilter', 'pyqtgraph', 'fonttools', 'comtypes', 'pillow'] + pyqt_hiddenimports,
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
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['bREadbeats.ico'],
)
