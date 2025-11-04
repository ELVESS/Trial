# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SAM Browser ONNX App
Build with: pyinstaller sam_browser_onnx.spec
"""

import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all segment_anything modules
sam_hiddenimports = collect_submodules('segment_anything')

# Collect static files
static_files = [
    ('static', 'static'),
]

# Collect ONNX models (optional - can be large)
onnx_files = [
    ('models/onnx', 'models/onnx'),
]

# Collect utils
utils_files = [
    ('utils', 'utils'),
]

a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=static_files + onnx_files + utils_files,
    hiddenimports=[
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
    ] + sam_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='sam_browser_onnx',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='sam_browser_onnx',
)
