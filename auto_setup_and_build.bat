@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title AirTeach AI - Auto Setup, Rust Build va Dong Goi EXE

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
set "LOG_FILE=%REPO_DIR%\setup_build.log"
set "VENV_DIR=%REPO_DIR%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_EXE="

echo ============================================================================ > "%LOG_FILE%"
echo   AIRTEACH AI - AUTO SETUP AND BUILD LOG (%DATE% %TIME%) >> "%LOG_FILE%"
echo ============================================================================ >> "%LOG_FILE%"

call :log "============================================================================"
call :log "  AIRTEACH AI - TU DONG CAI DAT MOI TRUONG, BIEN DICH RUST VA DONG GOI .EXE"
call :log "============================================================================"
call :log "Thu muc du an: %REPO_DIR%"
call :log "File log     : %LOG_FILE%"
echo.

rem ============================================================================
rem [1/7] KIEM TRA VA CAI DAT VISUAL C++ REDISTRIBUTABLE x64
rem ============================================================================
call :log "[1/7] Kiem tra Visual C++ 2015-2022 Redistributable (x64)..."
set "NEED_VCR=0"
if not exist "%SystemRoot%\System32\vcruntime140.dll" set "NEED_VCR=1"
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" set "NEED_VCR=1"
if not exist "%SystemRoot%\System32\msvcp140.dll" set "NEED_VCR=1"

if "!NEED_VCR!"=="0" (
    call :log "  [OK] Visual C++ Redistributable x64 da co san."
    goto :step_python
)

call :log "  [CAI DAT] Dang cai dat Visual C++ Redistributable x64..."
where winget >nul 2>&1
if not errorlevel 1 (
    winget install -e --id Microsoft.VCRedist.2015+.x64 --silent --accept-package-agreements --accept-source-agreements >> "%LOG_FILE%" 2>&1
)
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" (
    call :log "  [FALLBACK] Tai truc tiep vc_redist.x64.exe bang PowerShell..."
    set "AIRTEACH_VCR_TMP=%TEMP%\vc_redist.x64.exe"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $env:AIRTEACH_VCR_TMP -UseBasicParsing" >> "%LOG_FILE%" 2>&1
    if not errorlevel 1 (
        powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -FilePath $env:AIRTEACH_VCR_TMP -ArgumentList '/install','/quiet','/norestart' -Verb RunAs -PassThru -Wait; exit $p.ExitCode" >> "%LOG_FILE%" 2>&1
    )
)
if exist "%SystemRoot%\System32\vcruntime140_1.dll" (
    call :log "  [OK] Da cai dat xong Visual C++ Redistributable x64."
) else (
    call :log "  [CANH BAO] Chua xac nhan duoc vcruntime140_1.dll; tiep tuc kiem tra."
)

rem ============================================================================
rem [2/7] KIEM TRA VA CAI DAT PYTHON 3.11 / 3.12 (64-BIT)
rem ============================================================================
:step_python
echo.
call :log "[2/7] Kiem tra Python 3.11 / 3.12 (64-bit)..."
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
if not errorlevel 1 if exist "%AIRTEACH_PY_INST%" (
    "%AIRTEACH_PY_INST%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=1 >> "%LOG_FILE%" 2>&1
    call :refresh_path
    call :find_python
)
if not defined PYTHON_EXE (
    call :log "[LOI NGHIEM TRONG] Khong tim thay va khong the cai Python 3.11/3.12 x64!"
    goto :fail
)

:python_ready
set "PY_VER="
for /f "tokens=2 delims= " %%V in ('"%PYTHON_EXE%" --version 2^>^&1') do set "PY_VER=%%V"
call :log "  [OK] Python !PY_VER! (64-bit): %PYTHON_EXE%"

rem ============================================================================
rem [3/7] TAO MOI TRUONG AO (.venv) VA CAI DAT THU VIEN (requirements.txt)
rem ============================================================================
echo.
call :log "[3/7] Thiet lap moi truong ao .venv va cai dat thu vien tu requirements.txt..."
if not exist "%VENV_PY%" (
    "%PYTHON_EXE%" -m venv "%VENV_DIR%" >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log "[LOI] Khong the tao moi truong ao .venv!"
        goto :fail
    )
)
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    call :log "[LOI] Khong the kich hoat .venv!"
    goto :fail
)
call :log "  [OK] Da kich hoat .venv: %VENV_DIR%"

call :pip_retry install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

call :pip_retry install --prefer-binary -r "%REPO_DIR%\requirements.txt"
if errorlevel 1 (
    call :log "[LOI] Cai dat requirements.txt that bai!"
    goto :fail
)

"%VENV_PY%" -c "import PySide6, mediapipe, cv2, fitz, docx, pptx, PIL, numpy, google.genai, onnxruntime, requests; print('CORE_IMPORTS_OK')" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "[LOI] Mot so thu vien Python cot loi khong the import!"
    goto :fail
)
call :log "  [OK] Da cai dat va kiem tra thanh cong toan bo thu vien Python cot loi."

rem ============================================================================
rem [4/7] KIEM TRA / CAI DAT RUST VA BIEN DICH LOI RUST (airteach_core_rs)
rem ============================================================================
echo.
call :log "[4/7] Kiem tra Rust Toolchain (rustc / cargo) va bien dich airteach_core_rs..."
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
where cargo >nul 2>&1
if errorlevel 1 (
    call :log "  [FALLBACK] Tai rustup-init.exe truc tiep tu win.rustup.rs..."
    set "AIRTEACH_RUSTUP=%TEMP%\rustup-init.exe"
    powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://win.rustup.rs/x86_64' -OutFile $env:AIRTEACH_RUSTUP -UseBasicParsing" >> "%LOG_FILE%" 2>&1
    if not errorlevel 1 if exist "%AIRTEACH_RUSTUP%" (
        "%AIRTEACH_RUSTUP%" -y --default-toolchain stable --profile minimal >> "%LOG_FILE%" 2>&1
        set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
    )
)

where cargo >nul 2>&1
if errorlevel 1 (
    call :log "  [CANH BAO] Khong cai duoc Rust/Cargo. Ung dung van chay an toan bang Python Fallback."
    goto :step_assets
)

for /f "delims=" %%R in ('cargo --version 2^>nul') do call :log "  [OK] %%R"
if exist "%REPO_DIR%\rust_core\Cargo.toml" (
    call :log "  Dang bien dich airteach_core_rs (maturin develop --release)..."
    "%VENV_DIR%\Scripts\maturin.exe" develop --release --manifest-path "%REPO_DIR%\rust_core\Cargo.toml" >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log "  [CANH BAO] Bien dich Rust gap loi (xem setup_build.log). Tiep tuc voi Python Fallback."
        cmd /c exit /b 0
    ) else (
        "%VENV_PY%" -c "import airteach_core_rs; print('  [OK] Rust Core:', airteach_core_rs.version(), '-', airteach_core_rs.backend_info())"
        "%VENV_PY%" -c "import airteach_core_rs; print('  [OK] Rust Core:', airteach_core_rs.version(), '-', airteach_core_rs.backend_info())" >> "%LOG_FILE%" 2>&1
    )
)

rem ============================================================================
rem [5/7] KIEM TRA VA TAI MODELS, FONTS, ICONS, LOCALES
rem ============================================================================
:step_assets
echo.
call :log "[5/7] Kiem tra tai nguyen (MediaPipe Models, NotoSans Fonts, Icons, Locales)..."
"%VENV_PY%" "%REPO_DIR%\scripts\fetch_assets.py" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "[LOI] scripts\fetch_assets.py bao loi (xem setup_build.log)!"
    goto :fail
)
if not exist "%REPO_DIR%\locales\vi.json" (
    call :log "  [CANH BAO] Khong tim thay locales\vi.json."
) else (
    call :log "  [OK] Da kiem tra day du assets, models, fonts va locales."
)

rem ============================================================================
rem [6/7] DONG GOI DOC LAP AirTeachAI.exe BANG PYINSTALLER
rem ============================================================================
echo.
call :log "[6/7] Dang dong goi AirTeachAI.exe (PyInstaller AirTeachAI.spec --clean --noconfirm)..."
"%VENV_DIR%\Scripts\pyinstaller.exe" --clean --noconfirm "%REPO_DIR%\AirTeachAI.spec" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "[LOI] PyInstaller dong goi that bai! Xem chi tiet trong setup_build.log."
    goto :fail
)
if not exist "%REPO_DIR%\dist\AirTeachAI\AirTeachAI.exe" (
    call :log "[LOI] Khong tim thay file ket qua dist\AirTeachAI\AirTeachAI.exe!"
    goto :fail
)
copy /y "%REPO_DIR%\run_airteach.bat" "%REPO_DIR%\dist\AirTeachAI\run_airteach.bat" >nul 2>&1
copy /y "%REPO_DIR%\Chay_AirTeachAI_AutoFix.bat" "%REPO_DIR%\dist\AirTeachAI\Chay_AirTeachAI_AutoFix.bat" >nul 2>&1
call :log "  [OK] Da tao thanh cong dist\AirTeachAI\AirTeachAI.exe"

rem ============================================================================
rem [7/7] CHAY SMOKE TEST KIEM TRA FILE .EXE DOC LAP
rem ============================================================================
echo.
call :log "[7/7] Chay kiem thu doc lap dist\AirTeachAI\AirTeachAI.exe --smoke-test..."
"%REPO_DIR%\dist\AirTeachAI\AirTeachAI.exe" --smoke-test >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "[LOI] AirTeachAI.exe --smoke-test that bai! Xem chi tiet trong setup_build.log."
    goto :fail
)
call :log "  [OK] Smoke test AirTeachAI.exe thanh cong (khong thieu DLL / module)!"

echo.
call :log "============================================================================"
call :log "  [HOAN TAT 100%%] AIRTEACH AI DA SAN SANG CHAY NGAY!"
call :log "  - File EXE       : %REPO_DIR%\dist\AirTeachAI\AirTeachAI.exe"
call :log "  - Launcher Nhanh : %REPO_DIR%\run_airteach.bat"
call :log "  - File Log       : %LOG_FILE%"
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
echo %~1
echo [%TIME%] %~1 >> "%LOG_FILE%"
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
