# PyInstaller spec for building a single-file executable.
import os
from pathlib import Path
from PyInstaller.utils.hooks import copy_metadata
from PyInstaller.utils.hooks import collect_submodules

block_cipher = None

base = Path(__file__).parent

a = Analysis(
    ['process_folder.py'],
    pathex=[str(base)],
    binaries=[],
    datas=[],
    hiddenimports=collect_submodules('paddleocr') + collect_submodules('paddleocr.ppocr'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

datas = copy_metadata('paddleocr')
a.datas += datas

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='process_folder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
