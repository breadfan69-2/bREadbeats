# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files


LEARNING_DATA_FILES = [
    (str(path), 'defaults/learning')
    for path in sorted(Path('defaults/learning').glob('*.json'))
]

IMAGEIO_FFMPEG_DATA = collect_data_files('imageio_ffmpeg')


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('bREadbeats.ico', '.'),
        ('splash_screen.png', '.'),
        # Bundled learning assets (materialized to local files at frozen startup)
        ('learned_profile_slots.json', '.'),
        *LEARNING_DATA_FILES,
        ('datasets/rule_fit.json', 'datasets'),
        *IMAGEIO_FFMPEG_DATA,
    ],
    hiddenimports=[
        'config_persistence',
        'imageio_ffmpeg',
        'orbital_replay',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'librosa',
        'aubio',
        'matplotlib',
        'sklearn',
        'numba',
        'llvmlite',
        'imageio',
        'PIL',
        'pictex',
        'setuptools',
        'pkg_resources',
        # Keep scipy.signal only; exclude bulky scipy namespaces not used at runtime.
        'scipy.cluster',
        'scipy.constants',
        'scipy.datasets',
        'scipy.fft',
        'scipy.fftpack',
        'scipy.integrate',
        'scipy.interpolate',
        'scipy.io',
        'scipy.odr',
        'scipy.optimize',
        'scipy.sparse',
        'scipy.spatial',
        'scipy.stats',
        'scipy.tests',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name='bREadbeats',
    exclude_binaries=False,
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
