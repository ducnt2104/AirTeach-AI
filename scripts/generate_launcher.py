bat_content = """@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

title AirTeach AI Launcher

cd /d "%~dp0"

echo ===============================================================================
echo        AIRTEACH AI - TOUCHLESS GESTURE TEACHING PLATFORM
echo        He thong day hoc tuong tac khong cham bang AI (Windows Desktop)
echo ===============================================================================
echo.

if not exist "logs" mkdir "logs"
if not exist "data" mkdir "data"
if not exist "data\\workspaces" mkdir "data\\workspaces"
if not exist "models" mkdir "models"

set "LAUNCH_LOG=logs\\launcher.log"
echo [%date% %time%] AirTeach AI Launcher starting... > "%LAUNCH_LOG%"

set "PYTHON_EXE="

if exist ".venv\\Scripts\\python.exe" (
    set "PYTHON_EXE=.venv\\Scripts\\python.exe"
    echo [INFO] Su dung moi truong ao: .venv\\Scripts\\python.exe
    echo [INFO] Virtual environment detected: .venv >> "%LAUNCH_LOG%"
    goto :PYTHON_FOUND
)

if exist "..\\.venv\\Scripts\\python.exe" (
    set "PYTHON_EXE=..\\.venv\\Scripts\\python.exe"
    echo [INFO] Su dung moi truong ao: ..\\.venv\\Scripts\\python.exe
    echo [INFO] Virtual environment detected: ..\\.venv >> "%LAUNCH_LOG%"
    goto :PYTHON_FOUND
)

where python >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON_EXE=python"
    echo [INFO] Su dung Python he thong (PATH)
    echo [INFO] Using system Python >> "%LAUNCH_LOG%"
    goto :PYTHON_FOUND
)

echo [ERROR] Khong tim thay Python 3.10+ tren may tinh!
echo Vui long cai dat Python tu https://www.python.org hoac tao moi truong ao .venv.
echo [ERROR] Python not found in PATH or .venv >> "%LAUNCH_LOG%"
pause
exit /b 1

:PYTHON_FOUND
echo [INFO] Kiem tra phien ban Python...
"%PYTHON_EXE%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >> "%LAUNCH_LOG%" 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Python phien ban hien tai thap hon 3.10. Khuyen nghi Python 3.10+.
    echo [WARNING] Python version lower than 3.10 >> "%LAUNCH_LOG%"
)

echo [INFO] Kiem tra thu vien cot loi (PySide6, OpenCV, MediaPipe, NumPy, Cryptography, Dotenv)...
"%PYTHON_EXE%" -c "import PySide6, cv2, mediapipe, numpy, cryptography, dotenv" >> "%LAUNCH_LOG%" 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Phat hien thieu thu vien. Dang cai dat tu requirements.txt...
    echo [INFO] Installing requirements.txt >> "%LAUNCH_LOG%"
    "%PYTHON_EXE%" -m pip install -r requirements.txt >> "%LAUNCH_LOG%" 2>&1
    if %errorlevel% neq 0 (
        echo [ERROR] Khong the cai dat thu vien. Xem chi tiet tai: %LAUNCH_LOG%
        pause
        exit /b 1
    )
)

echo [SUCCESS] Khoi chay ung dung AirTeach AI...
echo [INFO] Launching main.py at %time% >> "%LAUNCH_LOG%"
echo.

"%PYTHON_EXE%" main.py %*
set "APP_EXIT_CODE=%errorlevel%"

echo.
echo [INFO] Ung dung ket thuc voi ma: %APP_EXIT_CODE%
echo [%date% %time%] Application exited with code %APP_EXIT_CODE% >> "%LAUNCH_LOG%"

if %APP_EXIT_CODE% neq 0 (
    echo [WARNING] Phat hien ma thoat bat thuong: %APP_EXIT_CODE%. Xem chi tiet tai %LAUNCH_LOG%
)

endlocal
exit /b %APP_EXIT_CODE%
"""

with open("Run_AirTeach_AI.bat", "wb") as f:
    f.write(bat_content.replace("\r\n", "\n").replace("\n", "\r\n").encode("utf-8"))
print("Wrote Run_AirTeach_AI.bat with CRLF line endings.")
