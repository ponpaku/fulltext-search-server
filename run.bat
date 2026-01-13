@echo off
setlocal
rem System Version: 1.1.11
rem File Version: 1.0.1

rem Root dir of this project
set "PROJECT_ROOT=%~dp0"
pushd "%PROJECT_ROOT%"
set "VENV_DIR=%PROJECT_ROOT%venv"
set "ACTIVATE_BAT=%VENV_DIR%\Scripts\activate.bat"

if not exist "%ACTIVATE_BAT%" (
    python -m venv "%VENV_DIR%"
)
call "%ACTIVATE_BAT%"
pip install --upgrade pip
if exist "%PROJECT_ROOT%requirements.txt" (
    pip install -r "%PROJECT_ROOT%requirements.txt"
)

python "%PROJECT_ROOT%app.py"
popd
