@echo off
setlocal
REM Delete and recreate the contents of output_txt, input, and working (keeps the folders).
pushd "%~dp0"
for %%D in ("output_txt" "input" "working") do (
  if exist "%%~fD" (
    echo Cleaning %%~fD ...
    rd /s /q "%%~fD" 2>nul
  ) else (
    echo Creating %%~fD ...
  )
  mkdir "%%~fD" 2>nul
)
echo Done.
echo.
pause
popd
endlocal
