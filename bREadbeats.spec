# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('bREadbeats.ico', '.'),
        # Bundled learning assets (materialized to local files at frozen startup)
        ('learned_profile_slots.json', '.'),
        ('defaults/learning/profile.refresh_3h_single.json', 'defaults/learning'),
        ('defaults/learning/rule_fit.refresh_3h_single_v3.json', 'defaults/learning'),
        ('datasets/rule_fit.json', 'datasets'),
    ],
    hiddenimports=[
        'config_persistence',
    ],
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
