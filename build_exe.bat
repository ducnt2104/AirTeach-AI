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
where iscc >nul 2>&1
if %errorlevel%==0 (
    iscc installer.iss
    echo [OK] Installer created in dist\AirTeachAI_Setup.exe
) else (
    echo [INFO] Inno Setup (iscc.exe) not found in PATH. Portable bundle ready at dist\AirTeachAI\AirTeachAI.exe and dist\AirTeachAI_Portable.zip
)

echo Done!
endlocal
