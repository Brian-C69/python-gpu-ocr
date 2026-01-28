@echo off
REM Delete all files in output_txt, input, and working (keeps the folders themselves).
pushd "%~dp0"
for %%D in ("output_txt" "input" "working") do (
  if exist "%%~fD" (
    echo Cleaning %%~fD ...
    del /f /q "%%~fD\*" 2>nul
    for /d %%S in ("%%~fD\*") do rd /s /q "%%~fS"
  ) else (
    echo Skipped %%~fD (not found).
  )
)
echo Done.
popd
