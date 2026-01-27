@echo off
REM Activate venv and run the OCR pipeline.
pushd %~dp0
REM Ensure CUDA 11.8 runtime (cublas64_11.dll, cuDNN 8.6) is first on PATH
set "PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\lib\x64;%PATH%"
REM Add zlibwapi.dll location (Cisco Packet Tracer)
set "PATH=C:\Program Files\Cisco Packet Tracer 9.0.0\bin;%PATH%"
if not exist ".venv\Scripts\activate.bat" (
  echo Virtual environment not found. Please create it first: python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt
  pause
  exit /b 1
)
call .\.venv\Scripts\activate
python process_folder.py
pause
popd
