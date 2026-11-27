@echo off
setlocal
cd /d "%~dp0"

echo [1/3] Running PyInstaller build (AirTeachAI.spec)...
python build_exe.py
if errorlevel 1 (
    echo [ERROR] PyInstaller build failed!
    exit /b 1
)

echo [2/3] Creating Portable ZIP archive...
powershell -NoProfile -Command "Compress-Archive -Path 'dist\AirTeachAI\*' -DestinationPath 'dist\AirTeachAI_Portable.zip' -Force"

echo [3/3] Checking Inno Setup Compiler (ISCC)...
set "ISCC_CMD="
where iscc >nul 2>&1
if %errorlevel%==0 set "ISCC_CMD=iscc"
if not defined ISCC_CMD if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" set "ISCC_CMD=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not defined ISCC_CMD if exist "C:\Program Files\Inno Setup 6\ISCC.exe" set "ISCC_CMD=C:\Program Files\Inno Setup 6\ISCC.exe"
if not defined ISCC_CMD if exist "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" set "ISCC_CMD=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"

if defined ISCC_CMD (
    "%ISCC_CMD%" installer.iss
    if errorlevel 1 (
        echo [ERROR] Inno Setup compilation failed!
        exit /b 1
    )
    echo [OK] Installer created: dist\AirTeachAI_Setup.exe
) else (
    echo [INFO] Inno Setup ^(ISCC.exe^) not found on this PC.
    echo [OK] Standalone EXE ready: dist\AirTeachAI\AirTeachAI.exe
    echo [OK] Portable ZIP ready: dist\AirTeachAI_Portable.zip
)

echo Done!
endlocal
