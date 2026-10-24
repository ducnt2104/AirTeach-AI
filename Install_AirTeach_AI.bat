@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul 2>&1

title AirTeach AI - Trình Cài Đặt Tự Động & Khởi Chạy (Windows)

cd /d "%~dp0"

echo ===============================================================================
echo        AIRTEACH AI - TOUCHLESS GESTURE TEACHING PLATFORM
echo        He thong day hoc tuong tac khong cham bang AI (Windows Desktop)
echo ===============================================================================
echo.
echo [1/6] Khoi tao thu muc du an va ghi nhat ky...

if not exist "logs" mkdir "logs"
if not exist "data" mkdir "data"
if not exist "data\workspaces" mkdir "data\workspaces"
if not exist "models" mkdir "models"
if not exist "assets\icons" mkdir "assets\icons"

set "INSTALL_LOG=logs\installer.log"
echo [%date% %time%] Bat dau cai dat AirTeach AI... > "%INSTALL_LOG%"

echo [2/6] Kiem tra moi truong Python tren may tinh...
set "PYTHON_CMD="

if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    echo   ✓ Phat hien moi truong ao noi bo (.venv)
    goto :PYTHON_READY
)

where python >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('python -c "import sys; print(sys.version_info >= (3, 10))"') do set "PY_OK=%%i"
    if "!PY_OK!"=="True" (
        set "PYTHON_CMD=python"
        echo   ✓ Phat hien Python he thong (>= 3.10)
        goto :CREATE_VENV
    )
)

where py >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%i in ('py -3 -c "import sys; print(sys.version_info >= (3, 10))"') do set "PY_OK=%%i"
    if "!PY_OK!"=="True" (
        set "PYTHON_CMD=py -3"
        echo   ✓ Phat hien Python Launcher py -3 (>= 3.10)
        goto :CREATE_VENV
    )
)

echo [CANH BAO] Khong tim thay Python 3.10+ tren may tinh!
echo   Dang tu dong tai va cai dat Python 3.11 chinh thuc tu python.org...
set "PY_INSTALLER_URL=https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe"
set "PY_INSTALLER_FILE=%TEMP%\python-3.11.9-amd64.exe"

powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; (New-Object System.Net.WebClient).DownloadFile('%PY_INSTALLER_URL%', '%PY_INSTALLER_FILE%')" >> "%INSTALL_LOG%" 2>&1

if exist "%PY_INSTALLER_FILE%" (
    echo   Dang cai dat Python 3.11 (vui long doi trong giay lat)...
    "%PY_INSTALLER_FILE%" /quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_test=0 >> "%INSTALL_LOG%" 2>&1
    del /f /q "%PY_INSTALLER_FILE%" >nul 2>&1
    set "PYTHON_CMD=python"
) else (
    echo [LOI] Khong the tu dong tai Python. Vui long cai dat Python 3.10+ tu https://www.python.org va tick 'Add Python to PATH'.
    pause
    exit /b 1
)

:CREATE_VENV
echo [3/6] Thiet lap moi truong ao co lap (.venv)...
if not exist ".venv\Scripts\python.exe" (
    echo   Dang khoi tao .venv...
    %PYTHON_CMD% -m venv .venv >> "%INSTALL_LOG%" 2>&1
)
if exist ".venv\Scripts\python.exe" (
    set "PYTHON_CMD=.venv\Scripts\python.exe"
    echo   ✓ Moi truong ao .venv da san sang.
)

:PYTHON_READY
echo [4/6] Kiem tra va cai dat cac thu vien can thiet (requirements.txt)...
echo   (Qua trinh nay co the mat 1-3 phut trong lan dau tien...)
"%PYTHON_CMD%" -m pip install --upgrade pip >> "%INSTALL_LOG%" 2>&1
"%PYTHON_CMD%" -m pip install -r requirements.txt >> "%INSTALL_LOG%" 2>&1

if %errorlevel% neq 0 (
    echo [LOI] Co loi khi cai dat thu vien. Vui long kiem tra file: %INSTALL_LOG%
    pause
    exit /b 1
)
echo   ✓ Tat ca thu vien (PySide6, OpenCV, MediaPipe, NumPy, SciPy, Keyring...) da san sang!

echo [5/6] Kiem tra va tai cac mo hinh AI ma nguon mo (YuNet, SFace, HandLandmarker)...
"%PYTHON_CMD%" main.py --check >> "%INSTALL_LOG%" 2>&1
if %errorlevel% equ 0 (
    echo   ✓ Toan bo mo hinh AI va he co so du lieu da duoc kiem tra hoan tat.
) else (
    echo   [CANH BAO] Kiem tra mo hinh co canh bao, xem chi tiet tai: %INSTALL_LOG%
)

echo [6/6] Tao Shortcut bieu tuong AirTeach AI tren man hinh Desktop...
"%PYTHON_CMD%" scripts\create_desktop_shortcut.py

echo.
echo ===============================================================================
echo   CHUC MUNG! CAI DAT HOAN TAT 100%!
echo   Icon 'AirTeach AI' da duoc tao tren man hinh Desktop cua ban.
echo   Tu bay gio, ban chi can nhap dup chuot vao Icon tren Desktop de mo phan mem!
echo ===============================================================================
echo.
echo Dang khoi chay AirTeach AI...
start "" "%PYTHON_CMD%" main.py

exit /b 0
