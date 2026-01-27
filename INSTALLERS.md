# Installer and Runtime Versions (Pinned)

These installers are kept locally to ensure deterministic GPU OCR setup.

## Included installers (local files, not versioned in GitHub due to size limits)
- `cuda_11.8.0_522.06_windows.exe` — SHA256 `B70F38F27321C0A53993438A91970A2E3C426F46DA4C42ECEFF1EEEA031A6555`
- `cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip` — SHA256 `78B4E5C455C4E8303B5D6C5401916FB0D731EA5DA72B040CFA81E0A340040AE3`

## Notes
- CUDA toolkit path used at runtime: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`
- cuDNN version used: 8.6 for CUDA 11.x (files placed in the v11.8 bin/include/lib folders).
- Paddle build in `.venv`: `paddlepaddle-gpu==2.6.1` (CUDA 11.8 + cuDNN 8.6).

If these installers are removed, re-download matching versions and verify SHA256 before use.

## Setup steps
1. Verify installer hashes:
   ```powershell
   Get-FileHash cuda_11.8.0_522.06_windows.exe
   Get-FileHash cudnn-windows-x86_64-8.6.0.163_cuda11-archive.zip
   ```
   Compare to the SHA256 above.
2. Install CUDA 11.8 using the `.exe` (default path `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8`).
3. Extract cuDNN zip and copy contents into the CUDA 11.8 toolkit:
   - `bin\*` → `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin\`
   - `include\*` → `...\v11.8\include\`
   - `lib\x64\*` → `...\v11.8\lib\x64\`
4. PATH: `run.bat` and `process_folder.py` prepend `v11.8` bin/lib (and Packet Tracer zlib). If running manually, prepend:
   ```
   set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64;%PATH%
   ```
5. Verify runtime in the venv:
   ```powershell
   python - <<'PY'
   import paddle
   print(paddle.__version__)
   print('CUDA', paddle.version.cuda(), 'cuDNN', paddle.version.cudnn())
   PY
   # Expect: 2.6.1, CUDA 11.8, cuDNN 8.6.x
   ```
