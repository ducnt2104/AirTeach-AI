@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title AirTeach AI - Cai Dat Moi Truong 1-Click (setup_all)

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
set "LOG_FILE=%REPO_DIR%\setup_all.log"
set "VENV_DIR=%REPO_DIR%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_EXE="

echo ============================================================================ > "%LOG_FILE%"
echo   AIRTEACH AI - SETUP ALL ENVIRONMENT LOG (%DATE% %TIME%) >> "%LOG_FILE%"
echo ============================================================================ >> "%LOG_FILE%"

call :log "============================================================================"
call :log "  AIRTEACH AI (DNT - Victer) - CAI DAT TOAN BO MOI TRUONG 1-CLICK"
call :log "============================================================================"
call :log "Thu muc du an : %REPO_DIR%"
call :log "File log      : %LOG_FILE%"
echo.

rem ============================================================================
rem [1/6] KIEM TRA VA TU DONG CAI DAT MICROSOFT VISUAL C++ 2015-2022 (x64)
rem ============================================================================
call :log "[1/6] Kiem tra Microsoft Visual C++ 2015-2022 Redistributable (x64)..."
set "NEED_VCR=0"
if not exist "%SystemRoot%\System32\vcruntime140.dll" set "NEED_VCR=1"
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" set "NEED_VCR=1"
if not exist "%SystemRoot%\System32\msvcp140.dll" set "NEED_VCR=1"

if "!NEED_VCR!"=="0" (
    call :log "  [OK] Microsoft Visual C++ 2015-2022 Redistributable x64 da co san."
    goto :step_python
)

call :log "  [CAI DAT] Dang tu dong cai dat Visual C++ Redistributable x64..."
where winget >nul 2>&1
if not errorlevel 1 (
    winget install -e --id Microsoft.VCRedist.2015+.x64 --silent --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
)
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" (
    call :log "  [FALLBACK] Tai truc tiep vc_redist.x64.exe bang PowerShell..."
    set "AIRTEACH_VCR_TMP=%TEMP%\vc_redist.x64.exe"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $env:AIRTEACH_VCR_TMP -UseBasicParsing" >> "%LOG_FILE%" 2>&1
    if not errorlevel 1 if exist "%TEMP%\vc_redist.x64.exe" (
        powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -FilePath $env:AIRTEACH_VCR_TMP -ArgumentList '/install','/quiet','/norestart' -Verb RunAs -PassThru -Wait; exit $p.ExitCode" >> "%LOG_FILE%" 2>&1
    )
)
if exist "%SystemRoot%\System32\vcruntime140_1.dll" (
    call :log "  [OK] Da cai dat xong Microsoft Visual C++ Redistributable x64."
) else (
    call :log "  [CANH BAO] Chua xac nhan duoc vcruntime140_1.dll trong System32; tiep tuc."
)

rem ============================================================================
rem [2/6] KIEM TRA VA TU DONG CAI DAT PYTHON 3.11 / 3.12 (64-BIT)
rem ============================================================================
:step_python
echo.
call :log "[2/6] Kiem tra Python 3.11 / 3.12 (64-bit)..."
call :find_python
if defined PYTHON_EXE goto :python_ready

call :log "  [CAI DAT] Chua tim thay Python 3.11/3.12 x64. Dang tu dong cai dat..."
where winget >nul 2>&1
if not errorlevel 1 (
    winget install -e --id Python.Python.3.11 --architecture x64 --scope user --silent --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
    call :refresh_path
    call :find_python
)
if defined PYTHON_EXE goto :python_ready

call :log "  [FALLBACK] Tai Python 3.11.9 64-bit truc tiep tu python.org..."
set "AIRTEACH_PY_INST=%TEMP%\python-3.11.9-amd64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile $env:AIRTEACH_PY_INST -UseBasicParsing" >> "%LOG_FILE%" 2>&1
if not errorlevel 1 if exist "%TEMP%\python-3.11.9-amd64.exe" (
    "%TEMP%\python-3.11.9-amd64.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=1 >> "%LOG_FILE%" 2>&1
    call :refresh_path
    call :find_python
)
if not defined PYTHON_EXE (
    call :log "[LOI NGHIEM TRONG] Khong tim thay va khong the cai Python 3.11/3.12 64-bit."
    goto :fail
)

:python_ready
set "PY_VER="
for /f "tokens=2 delims= " %%V in ('"%PYTHON_EXE%" --version 2^>^&1') do set "PY_VER=%%V"
call :log "  [OK] Python !PY_VER! 64-bit: %PYTHON_EXE%"

rem ============================================================================
rem [3/6] KIEM TRA VA TU DONG CAI DAT RUST VA CARGO (RUSTUP)
rem ============================================================================
echo.
call :log "[3/6] Kiem tra Rust va Cargo (rustup) cho nhan tang toc airteach_core_rs..."
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

where cargo >nul 2>&1
if errorlevel 1 (
    call :log "  [CAI DAT] Chua co Rust/Cargo. Dang tu dong cai dat Rustup..."
    where winget >nul 2>&1
    if not errorlevel 1 (
        winget install -e --id Rustlang.Rustup --silent --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
        call :refresh_path
    )
)
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

where cargo >nul 2>&1
if errorlevel 1 (
    call :log "  [FALLBACK] Tai rustup-init.exe truc tiep tu win.rustup.rs..."
    set "AIRTEACH_RUSTUP=%TEMP%\rustup-init.exe"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://win.rustup.rs/x86_64' -OutFile $env:AIRTEACH_RUSTUP -UseBasicParsing" >> "%LOG_FILE%" 2>&1
    if not errorlevel 1 if exist "%TEMP%\rustup-init.exe" (
        "%TEMP%\rustup-init.exe" -y --default-toolchain stable --profile minimal >> "%LOG_FILE%" 2>&1
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    )
)

where cargo >nul 2>&1
if errorlevel 1 (
    call :log "  [CANH BAO] Chua cai duoc Rust/Cargo. Ung dung van chay an toan voi Python Fallback."
) else (
    for /f "delims=" %%R in ('cargo --version 2^>nul') do call :log "  [OK] %%R"
    for /f "delims=" %%C in ('rustc --version 2^>nul') do call :log "  [OK] %%C"
)

rem ============================================================================
rem [4/6] TAO / KICH HOAT .venv VA CAI DAT requirements.txt (3x RETRY)
rem ============================================================================
echo.
call :log "[4/6] Thiet lap moi truong ao .venv va cai dat thu vien tu requirements.txt..."
if not exist "%VENV_PY%" (
    call :log "  Dang tao moi truong ao .venv..."
    "%PYTHON_EXE%" -m venv "%VENV_DIR%" >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log "[LOI] Khong the tao moi truong ao .venv."
        goto :fail
    )
)
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    call :log "[LOI] Khong the kich hoat moi truong ao .venv."
    goto :fail
)
call :log "  [OK] Da kich hoat .venv: %VENV_DIR%"

call :pip_retry install --upgrade pip setuptools wheel
if errorlevel 1 (
    call :log "[LOI] Nang cap pip/setuptools/wheel that bai."
    goto :fail
)

call :pip_retry install --prefer-binary -r "%REPO_DIR%\requirements.txt"
if errorlevel 1 (
    call :log "[LOI] Cai dat requirements.txt that bai sau 3 lan thu."
    goto :fail
)

"%VENV_PY%" -c "import PySide6, mediapipe, cv2, fitz, docx, pptx, PIL, numpy, google.genai, onnxruntime, requests, PyInstaller, maturin; print('CORE_IMPORTS_OK')" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "[LOI] Mot so thu vien Python cot loi khong the import."
    goto :fail
)
call :log "  [OK] Da cai dat va kiem tra thanh cong toan bo thu vien Python cot loi."

rem ============================================================================
rem [5/6] TAI / KIEM TRA MEDIAPIPE MODELS, NOTOSANS FONTS, ICONS VA LOCALES
rem ============================================================================
echo.
call :log "[5/6] Chay scripts\fetch_assets.py va kiem tra locales (vi.json, en.json)..."
"%VENV_PY%" "%REPO_DIR%\scripts\fetch_assets.py" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "[LOI] scripts\fetch_assets.py bao loi. Xem chi tiet trong setup_all.log."
    goto :fail
)

"%VENV_PY%" -c "import json, pathlib, sys; root = pathlib.Path(sys.argv[1]); [json.loads((root / 'locales' / f).read_text(encoding='utf-8')) for f in ('vi.json', 'en.json')]; print('LOCALES_OK')" "%REPO_DIR%" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "[LOI] Thieu hoac loi cu phap JSON trong locales\vi.json hoac locales\en.json."
    goto :fail
)
call :log "  [OK] MediaPipe models, NotoSans fonts, icons va locales\vi.json, locales\en.json hop le."

rem ============================================================================
rem [6/6] KIEM TRA LIBREOFFICE (soffice) VA MICROSOFT OFFICE COM
rem ============================================================================
echo.
call :log "[6/6] Kiem tra cong cu chuyen doi tai lieu (LibreOffice va Microsoft Office)..."
set "SOFFICE_BIN="
for /f "delims=" %%S in ('where soffice 2^>nul') do (
    if not defined SOFFICE_BIN set "SOFFICE_BIN=%%~S"
)
if not defined SOFFICE_BIN if exist "%ProgramFiles%\LibreOffice\program\soffice.exe" set "SOFFICE_BIN=%ProgramFiles%\LibreOffice\program\soffice.exe"
if not defined SOFFICE_BIN if exist "%ProgramFiles(x86)%\LibreOffice\program\soffice.exe" set "SOFFICE_BIN=%ProgramFiles(x86)%\LibreOffice\program\soffice.exe"

if defined SOFFICE_BIN (
    call :log "  [OK] Phat hien LibreOffice: !SOFFICE_BIN!"
) else (
    call :log "  [THONG TIN] Chua cai LibreOffice [soffice]."
)

set "HAS_MS_WORD=0"
set "HAS_MS_PPT=0"
reg query "HKCR\Word.Application" >nul 2>&1
if not errorlevel 1 set "HAS_MS_WORD=1"
reg query "HKCR\PowerPoint.Application" >nul 2>&1
if not errorlevel 1 set "HAS_MS_PPT=1"

if "!HAS_MS_WORD!"=="1" (
    call :log "  [OK] Phat hien Microsoft Word COM."
) else (
    call :log "  [THONG TIN] Chua cai Microsoft Word COM."
)
if "!HAS_MS_PPT!"=="1" (
    call :log "  [OK] Phat hien Microsoft PowerPoint COM."
) else (
    call :log "  [THONG TIN] Chua cai Microsoft PowerPoint COM."
)

call :log "  [OK] AirTeach AI da tich hop san bo render DOCX va PPTX thuan Python lam du phong offline."

rem Khoi tao truoc thu muc du lieu nguoi dung tai %APPDATA%\AirTeachAI
set "APPDATA_ROOT=%APPDATA%\AirTeachAI"
for %%D in ("" "\logs" "\doc_cache" "\face_embeddings" "\recordings" "\models") do (
    if not exist "%APPDATA_ROOT%%%~D" mkdir "%APPDATA_ROOT%%%~D" >nul 2>&1
)

echo.
call :log "============================================================================"
call :log "  [HOAN TAT] MOI TRUONG AIRTEACH AI DA SAN SANG."
call :log "  - Buoc tiep theo : Chay build_exe.bat de bien dich Rust va dong goi .exe"
call :log "  - Hoac chay ngay : run_airteach.bat"
call :log "  - File log       : %LOG_FILE%"
call :log "============================================================================"
if not defined AIRTEACH_NO_PAUSE if not defined CI (
    echo.
    pause
)
exit /b 0

rem ============================================================================
rem SUBROUTINES
rem ============================================================================
:log
set "LOG_MSG=%~1"
echo(!LOG_MSG!
echo([%TIME%] !LOG_MSG!>> "%LOG_FILE%"
exit /b 0

:pip_retry
set "PIP_ATTEMPT=0"
:pip_retry_loop
set /a "PIP_ATTEMPT+=1"
echo   [pip !PIP_ATTEMPT!/3] pip %*
echo [%TIME%]   [pip !PIP_ATTEMPT!/3] pip %* >> "%LOG_FILE%"
"%VENV_PY%" -m pip %* >> "%LOG_FILE%" 2>&1
if not errorlevel 1 exit /b 0
if !PIP_ATTEMPT! geq 3 exit /b 1
call :log "  [CANH BAO] Loi mang tam thoi, thu lai sau 5 giay..."
timeout /t 5 /nobreak >nul
goto :pip_retry_loop

:refresh_path
set "AIRTEACH_PATH_TMP=%TEMP%\airteach_refresh_path.cmd"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$q = [char]34; $pct = [char]37;" ^
    "$m = [Environment]::GetEnvironmentVariable('PATH','Machine');" ^
    "$u = [Environment]::GetEnvironmentVariable('PATH','User');" ^
    "$joined = (@($u, $m) | Where-Object { $_ -and $_.Trim() }) -join ';';" ^
    "$joined = $joined.Replace([string]$q, '').Replace([string]$pct, ([string]$pct + [string]$pct));" ^
    "$line = 'set ' + $q + 'PATH=' + $joined + $q + [Environment]::NewLine;" ^
    "[System.IO.File]::WriteAllText($env:AIRTEACH_PATH_TMP, $line, [System.Text.UTF8Encoding]::new($false))"
if exist "%AIRTEACH_PATH_TMP%" call "%AIRTEACH_PATH_TMP%"
set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%USERPROFILE%\.cargo\bin;%PATH%"
exit /b 0

:find_python
set "PYTHON_EXE="
where py >nul 2>&1
if not errorlevel 1 (
    for %%V in (3.11-64 3.12-64) do (
        for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)" 2^>nul') do call :test_python_311_312 "%%~P"
        if defined PYTHON_EXE exit /b 0
    )
)
for %%C in (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python312\python.exe"
) do (
    if exist "%%~C" call :test_python_311_312 "%%~C"
    if defined PYTHON_EXE exit /b 0
)
for /f "delims=" %%P in ('where python 2^>nul') do (
    call :test_python_311_312 "%%~P"
    if defined PYTHON_EXE exit /b 0
)
exit /b 0

:test_python_311_312
if not exist "%~1" exit /b 1
echo "%~1" | findstr /i "WindowsApps" >nul
if not errorlevel 1 exit /b 1
"%~1" -c "import sys, struct; raise SystemExit(0 if (sys.version_info[:2] in ((3, 11), (3, 12)) and struct.calcsize('P') * 8 == 64) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=%~1"
exit /b 0

:fail
echo.
call :log "============================================================================"
call :log "  [THAT BAI] Vui long kiem tra chi tiet loi trong file: %LOG_FILE%"
call :log "============================================================================"
if not defined AIRTEACH_NO_PAUSE if not defined CI (
    echo.
    pause
)
exit /b 1
