@echo off
REM Activate venv and run the OCR pipeline.
pushd %~dp0
if not exist ".venv\Scripts\activate.bat" (
  echo Virtual environment not found. Please create it first: python -m venv .venv && .\.venv\Scripts\activate && pip install -r requirements.txt
  pause
  exit /b 1
)
call .\.venv\Scripts\activate
python process_folder.py
pause
popd
