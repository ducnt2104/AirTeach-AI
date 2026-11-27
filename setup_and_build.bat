@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title AirTeach AI — Tu dong Cai dat va Dong goi EXE (All-in-One)

echo ============================================================================
echo   AIRTEACH AI — TU DONG CAI DAT MOI TRUONG VA DONG GOI FILE .EXE CHUAN KHKT
echo ============================================================================
echo   Thu muc du an: "%~dp0"
echo.

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
set "VENV_DIR=%REPO_DIR%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "PYTHON_EXE="

rem ============================================================================
rem [1/6] KIEM TRA & CAI DAT CORE OS RUNTIMES (VC++ x64, PYTHON 3.11, GIT)
rem ============================================================================
echo [1/6] Kiem tra Core OS Runtimes va Ngon ngu lap trinh...

rem --- 1A. Microsoft Visual C++ 2015-2022 Redistributable (x64) ---
set "NEED_VCR=0"
if not exist "%SystemRoot%\System32\vcruntime140.dll" set "NEED_VCR=1"
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" set "NEED_VCR=1"
if not exist "%SystemRoot%\System32\msvcp140.dll" set "NEED_VCR=1"

if "!NEED_VCR!"=="0" (
    echo   [OK] Visual C++ 2015-2022 Redistributable x64 da co san.
    goto :check_python
)

echo   [CAI DAT] Dang cai Microsoft Visual C++ 2015-2022 Redistributable x64...
where winget >nul 2>&1
if not errorlevel 1 (
    winget install -e --id Microsoft.VCRedist.2015+.x64 --silent --accept-package-agreements --accept-source-agreements >nul 2>&1
)

if exist "%SystemRoot%\System32\vcruntime140_1.dll" (
    echo   [OK] Da cai dat Visual C++ Redistributable x64 qua winget.
    goto :check_python
)

echo   [FALLBACK] Tai truc tiep vc_redist.x64.exe bang PowerShell...
set "AIRTEACH_VCR_TMP=%TEMP%\vc_redist.x64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $env:AIRTEACH_VCR_TMP -UseBasicParsing"
if not errorlevel 1 (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -FilePath $env:AIRTEACH_VCR_TMP -ArgumentList '/install','/quiet','/norestart' -Verb RunAs -PassThru -Wait; exit $p.ExitCode"
)
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" (
    echo   [CANH BAO] Khong the xac nhan vcruntime140_1.dll trong System32. Tiep tuc voi quy trinh kiem tra...
) else (
    echo   [OK] Da cai dat xong Visual C++ Redistributable x64!
)

:check_python
rem --- 1B. Python 3.11 (64-bit) ---
call :find_python311
if defined PYTHON_EXE goto :python_ready

echo   [CAI DAT] Chua co Python 3.11 64-bit. Dang tu dong cai dat...
where winget >nul 2>&1
if not errorlevel 1 (
    winget install -e --id Python.Python.3.11 --architecture x64 --scope user --silent --accept-package-agreements --accept-source-agreements
    call :refresh_path
    call :find_python311
)
if defined PYTHON_EXE goto :python_ready

echo   [FALLBACK] Dang tai Python 3.11.9 64-bit truc tiep tu python.org...
set "AIRTEACH_PY_INST=%TEMP%\python-3.11.9-amd64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile $env:AIRTEACH_PY_INST -UseBasicParsing"
if not errorlevel 1 if exist "%AIRTEACH_PY_INST%" (
    echo   Dang cai dat Python 3.11.9 che do im lang...
    "%AIRTEACH_PY_INST%" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_launcher=1
    call :refresh_path
    call :find_python311
)
if defined PYTHON_EXE goto :python_ready

rem Fallback cuoi neu may hoan toan offline nhung co Python >= 3.10 64-bit san
call :find_python_fallback
if defined PYTHON_EXE (
    echo   [THONG BAO] Khong cai duoc Python 3.11; tam su dung Python 64-bit hien co: "%PYTHON_EXE%"
    goto :python_ready
)

echo [LOI NGHIEM TRONG] Khong tim thay va khong the cai dat Python 3.11 64-bit!
echo Vui long kiem tra ket noi mang hoac cai dat thu cong Python 3.11 x64 tu python.org.
goto :fail

:python_ready
set "PY_VER="
for /f "tokens=2 delims= " %%V in ('"%PYTHON_EXE%" --version 2^>^&1') do set "PY_VER=%%V"
echo   [OK] Su dung Python !PY_VER! 64-bit tai: "%PYTHON_EXE%"

rem --- 1C. Git & Toan ven Ma nguon ---
where git >nul 2>&1
if errorlevel 1 (
    echo   [THONG BAO] Khong co Git CLI trong PATH ^(khong bat buoc neu da tai ma nguon^).
) else (
    for /f "delims=" %%G in ('git --version 2^>nul') do echo   [OK] %%G
)

for %%F in ("main.py" "requirements.txt" "AirTeachAI.spec" "config\vision_config.json" "scripts\fetch_assets.py") do (
    if not exist "%REPO_DIR%\%%~F" (
        echo [LOI] Thieu tep ma nguon bat buoc: "%%~F"
        goto :fail
    )
)
echo   [OK] Toan bo tep cau hinh va ma nguon goc hop le.

rem ============================================================================
rem [2/6] KHOI TAO MOI TRUONG AO CO LAP (.venv) & CONG CU DONG GOI
rem ============================================================================
echo.
echo [2/6] Thiet lap moi truong ao Python co lap (.venv)...

set "RECREATE_VENV=0"
if not exist "%VENV_PY%" set "RECREATE_VENV=1"
if "!RECREATE_VENV!"=="0" (
    set "TARGET_RAW="
    set "VENV_RAW="
    for /f "tokens=2 delims= " %%A in ('"%PYTHON_EXE%" --version 2^>^&1') do set "TARGET_RAW=%%A"
    for /f "tokens=2 delims= " %%B in ('"%VENV_PY%" --version 2^>^&1') do set "VENV_RAW=%%B"
    for /f "tokens=1,2 delims=." %%M in ("!TARGET_RAW!") do set "TARGET_MM=%%M.%%N"
    for /f "tokens=1,2 delims=." %%M in ("!VENV_RAW!") do set "VENV_MM=%%M.%%N"
    if not "!VENV_MM!"=="!TARGET_MM!" (
        echo   [THONG BAO] .venv cu dung Python !VENV_MM! ^(can !TARGET_MM!^). Dang tao lai .venv sach...
        rmdir /s /q "%VENV_DIR%"
        set "RECREATE_VENV=1"
    )
)

if "!RECREATE_VENV!"=="1" (
    "%PYTHON_EXE%" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo [LOI] Khong the tao moi truong ao .venv!
        goto :fail
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo [LOI] Khong the kich hoat .venv\Scripts\activate.bat!
    goto :fail
)
echo   [OK] Da kich hoat moi truong ao: "%VENV_DIR%"

echo   Dang cap nhat pip, setuptools, wheel, pyinstaller va maturin...
call :pip_retry install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail
call :pip_retry install --prefer-binary --upgrade "pyinstaller>=6.10.0,<7.0.0" "maturin>=1.5.0,<2.0"
if errorlevel 1 goto :fail

rem ============================================================================
rem [3/6] CAI DAT DAY DU THU VIEN TU requirements.txt
rem ============================================================================
echo.
echo [3/6] Cai dat thu vien Python tu requirements.txt ^(prefer-binary + auto-retry^)...
call :pip_retry install --prefer-binary -r "%REPO_DIR%\requirements.txt"
if errorlevel 1 (
    echo [LOI] Cai dat requirements.txt that bai sau 3 lan thu!
    goto :fail
)

echo   Kiem tra import cac module loi...
"%VENV_PY%" -c "import PySide6, mediapipe, cv2, fitz, docx, pptx, PIL, numpy, google.genai, win32com.client, requests; print('  [OK] Import thanh cong tat ca thu vien cot loi!')"
if errorlevel 1 (
    echo [LOI] Mot so thu vien cot loi khong import duoc trong .venv!
    goto :fail
)

rem ============================================================================
rem [4/6] TU DONG KIEM TRA & BIEN DICH NHAN RUST (airteach_core_rs)
rem ============================================================================
echo.
echo [4/6] Kiem tra Rust Toolchain (cargo / maturin)...
where cargo >nul 2>&1
if errorlevel 1 (
    if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"
)

where cargo >nul 2>&1
if errorlevel 1 (
    echo   [THONG BAO] Khong tim thay Rust/Cargo tren may nay.
    echo   Bo qua bien dich rust_core - he thong tu dong chay che do Python toi uu!
    goto :after_rust
)

for /f "delims=" %%R in ('cargo --version 2^>nul') do echo   [OK] Phat hien Rust: %%R
if not exist "%REPO_DIR%\rust_core\Cargo.toml" (
    echo   [THONG BAO] Khong co rust_core\Cargo.toml; giu nguyen backend Python.
    goto :after_rust
)

echo   Dang bien dich rust_core (airteach_core_rs) bang maturin develop --release...
"%VENV_DIR%\Scripts\maturin.exe" develop --release --manifest-path "%REPO_DIR%\rust_core\Cargo.toml"
if errorlevel 1 goto :rust_fallback
"%VENV_PY%" -c "import airteach_core_rs; print('  [OK]', airteach_core_rs.backend_info())"
goto :after_rust

:rust_fallback
echo   [CANH BAO] Bien dich rust_core khong thanh cong ^(co the thieu C linker hoac mang crates.io^).
echo   He thong tu dong chuyen sang Python Fallback muot ma va khong dung build.
cmd /c exit /b 0

:after_rust

rem ============================================================================
rem [5/6] TAI & KIEM DINH AI MODELS, FONT TIENG VIET NOTOSANS VA ICON
rem ============================================================================
echo.
echo [5/6] Kiem tra va tai Model AI, Font tieng Viet NotoSans va Icon da kich thuoc...
"%VENV_PY%" "%REPO_DIR%\scripts\fetch_assets.py"
if errorlevel 1 (
    echo [LOI] Tai hoac kiem tra tai nguyen that bai!
    goto :fail
)

rem ============================================================================
rem [6/6] DONG GOI DOC LAP AirTeachAI.exe & KIEM THU SMOKE TEST
rem ============================================================================
echo.
echo [6/6] Dang dong goi AirTeachAI.exe bang PyInstaller (--clean --noconfirm)...
"%VENV_DIR%\Scripts\pyinstaller.exe" --clean --noconfirm "%REPO_DIR%\AirTeachAI.spec"
if errorlevel 1 (
    echo [LOI] PyInstaller dong goi that bai!
    goto :fail
)

if not exist "%REPO_DIR%\dist\AirTeachAI\AirTeachAI.exe" (
    echo [LOI] Khong tim thay file ket qua dist\AirTeachAI\AirTeachAI.exe!
    goto :fail
)

copy /y "%REPO_DIR%\Chay_AirTeachAI_AutoFix.bat" "%REPO_DIR%\dist\AirTeachAI\Chay_AirTeachAI_AutoFix.bat" >nul
if errorlevel 1 (
    echo [LOI] Khong the sao chep Chay_AirTeachAI_AutoFix.bat vao dist\AirTeachAI\
    goto :fail
)

echo.
echo   Dang chay Smoke Test kiem tra EXE doc lap ^(khong thieu DLL/module^)...
"%REPO_DIR%\dist\AirTeachAI\AirTeachAI.exe" --smoke-test
if errorlevel 1 (
    echo [LOI] Smoke test AirTeachAI.exe --smoke-test that bai!
    goto :fail
)

echo.
echo ============================================================================
echo   [HOAN TAT 100%%] DA SAN SANG BAN CHINH THUC CHO MAY THI!
echo   - File EXE        : "%REPO_DIR%\dist\AirTeachAI\AirTeachAI.exe"
echo   - Launcher May Thi: "%REPO_DIR%\dist\AirTeachAI\Chay_AirTeachAI_AutoFix.bat"
echo   Copy nguyen thu muc "dist\AirTeachAI" sang USB/May thi va chay
echo   "Chay_AirTeachAI_AutoFix.bat" de mo ngay lap tuc!
echo ============================================================================
if not defined AIRTEACH_NO_PAUSE if not defined CI (
    echo.
    pause
)
exit /b 0

rem ============================================================================
rem CAC HAM CON HO TRO (BATCH SUBROUTINES)
rem ============================================================================
:pip_retry
set "PIP_ATTEMPT=0"
:pip_retry_loop
set /a "PIP_ATTEMPT+=1"
echo   [pip !PIP_ATTEMPT!/3] "%VENV_PY%" -m pip %*
"%VENV_PY%" -m pip %*
if not errorlevel 1 exit /b 0
if !PIP_ATTEMPT! geq 3 exit /b 1
echo   [CANH BAO] Loi mang tam thoi. Cho 5 giay truoc khi thu lai...
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
set "PATH=%LOCALAPPDATA%\Programs\Python\Python311;%LOCALAPPDATA%\Programs\Python\Python311\Scripts;%ProgramFiles%\Python311;%ProgramFiles%\Python311\Scripts;%USERPROFILE%\.cargo\bin;%PATH%"
exit /b 0

:find_python311
set "PYTHON_EXE="
where py >nul 2>&1
if not errorlevel 1 (
    for /f "delims=" %%P in ('py -3.11-64 -c "import sys; print(sys.executable)" 2^>nul') do call :test_python311 "%%~P"
    if defined PYTHON_EXE exit /b 0
)
for %%C in (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "C:\Python311\python.exe"
) do (
    if exist "%%~C" call :test_python311 "%%~C"
    if defined PYTHON_EXE exit /b 0
)
for /f "delims=" %%P in ('where python 2^>nul') do (
    call :test_python311 "%%~P"
    if defined PYTHON_EXE exit /b 0
)
exit /b 0

:test_python311
if not exist "%~1" exit /b 1
echo "%~1" | findstr /i "WindowsApps" >nul
if not errorlevel 1 exit /b 1
"%~1" -c "import sys, struct; raise SystemExit(0 if (sys.version_info[:2] == (3, 11) and struct.calcsize('P') * 8 == 64) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=%~1"
exit /b 0

:find_python_fallback
set "PYTHON_EXE="
for /f "delims=" %%P in ('where python 2^>nul') do (
    call :test_python_ge310 "%%~P"
    if defined PYTHON_EXE exit /b 0
)
exit /b 0

:test_python_ge310
if not exist "%~1" exit /b 1
echo "%~1" | findstr /i "WindowsApps" >nul
if not errorlevel 1 exit /b 1
"%~1" -c "import sys, struct; raise SystemExit(0 if (sys.version_info >= (3, 10) and struct.calcsize('P') * 8 == 64) else 1)" >nul 2>&1
if not errorlevel 1 set "PYTHON_EXE=%~1"
exit /b 0

:fail
echo.
echo ============================================================================
echo   [THAT BAI] Qua trinh cai dat hoac build gap loi. Xem chi tiet o tren.
echo ============================================================================
if not defined AIRTEACH_NO_PAUSE if not defined CI (
    echo.
    pause
)
exit /b 1
