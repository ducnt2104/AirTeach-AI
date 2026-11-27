@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title AirTeach AI - Khoi Chay Nhanh

echo ============================================================================
echo   AIRTEACH AI (DNT - Victer) - KIEM TRA NHANH VA KHOI CHAY UNG DUNG
echo ============================================================================

rem 1. Kiem tra nhanh VC++ Runtime DLL
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" (
    if not exist "%~dp0_internal\vcruntime140_1.dll" (
        if not exist "%~dp0dist\AirTeachAI\_internal\vcruntime140_1.dll" (
            echo [CANH BAO] Chua tim thay vcruntime140_1.dll. Hay chay auto_setup_and_build.bat de tu dong cai!
        )
    )
)

rem 2. Uu tien khoi chay file .EXE doc lap ngay trong thu muc hien tai hoac dist\AirTeachAI
set "TARGET_EXE="
if exist "%~dp0AirTeachAI.exe" set "TARGET_EXE=%~dp0AirTeachAI.exe"
if not defined TARGET_EXE if exist "%~dp0dist\AirTeachAI\AirTeachAI.exe" set "TARGET_EXE=%~dp0dist\AirTeachAI\AirTeachAI.exe"

if defined TARGET_EXE (
    echo [OK] Dang mo "%TARGET_EXE%" ...
    if /i "%~1"=="--smoke-test" (
        "%TARGET_EXE%" %*
        exit /b %ERRORLEVEL%
    )
    start "" "%TARGET_EXE%" %*
    exit /b 0
)

rem 3. Che do phat trien: khoi chay bang moi truong ao .venv
if exist "%~dp0.venv\Scripts\python.exe" (
    echo [OK] Dang mo AirTeach AI tu moi truong ao .venv ...
    "%~dp0.venv\Scripts\python.exe" "%~dp0main.py" %*
    exit /b %ERRORLEVEL%
)

echo [LOI] Chua tim thay AirTeachAI.exe hoac moi truong ao .venv!
echo Hay click dup vao "auto_setup_and_build.bat" de tu dong cai dat va dong goi .exe.
if not defined AIRTEACH_NO_PAUSE pause
exit /b 1
