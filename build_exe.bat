@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title AirTeach AI - Bien Dich Rust va Dong Goi EXE (build_exe)

set "REPO_DIR=%~dp0"
if "%REPO_DIR:~-1%"=="\" set "REPO_DIR=%REPO_DIR:~0,-1%"
set "LOG_FILE=%REPO_DIR%\build_exe.log"
set "VENV_DIR=%REPO_DIR%\.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "MATURIN_EXE=%VENV_DIR%\Scripts\maturin.exe"
set "PYINSTALLER_EXE=%VENV_DIR%\Scripts\pyinstaller.exe"

echo ============================================================================ > "%LOG_FILE%"
echo   AIRTEACH AI - RUST BUILD AND EXE PACKAGING LOG (%DATE% %TIME%) >> "%LOG_FILE%"
echo ============================================================================ >> "%LOG_FILE%"

call :log "============================================================================"
call :log "  AIRTEACH AI (DNT - Victer) - BIEN DICH RUST VA DONG GOI .EXE DOC LAP"
call :log "============================================================================"
call :log "Thu muc du an : %REPO_DIR%"
call :log "File log      : %LOG_FILE%"
echo.

rem ============================================================================
rem [1/5] KIEM TRA VA KICH HOAT MOI TRUONG AO .venv
rem ============================================================================
call :log "[1/5] Kiem tra va kich hoat moi truong ao .venv..."
if not exist "%VENV_PY%" (
    call :log "  [THONG TIN] Chua thay .venv. Tu dong chay setup_all.bat truoc..."
    if not exist "%REPO_DIR%\setup_all.bat" (
        call :log "[LOI] Khong tim thay setup_all.bat de khoi tao .venv."
        goto :fail
    )
    set "PREV_NO_PAUSE=%AIRTEACH_NO_PAUSE%"
    set "AIRTEACH_NO_PAUSE=1"
    call "%REPO_DIR%\setup_all.bat"
    set "SETUP_RC=!ERRORLEVEL!"
    set "AIRTEACH_NO_PAUSE=!PREV_NO_PAUSE!"
    set "LOG_FILE=%REPO_DIR%\build_exe.log"
    if not "!SETUP_RC!"=="0" (
        call :log "[LOI] setup_all.bat that bai. Xem setup_all.log de biet chi tiet."
        goto :fail
    )
)

call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    call :log "[LOI] Khong the kich hoat moi truong ao .venv."
    goto :fail
)
set "PY_VER="
for /f "tokens=2 delims= " %%V in ('"%VENV_PY%" --version 2^>^&1') do set "PY_VER=%%V"
call :log "  [OK] Da kich hoat .venv [Python !PY_VER!]: %VENV_DIR%"

rem ============================================================================
rem [2/5] BIEN DICH LOI RUST (airteach_core_rs) VA KIEM TRA IMPORT
rem ============================================================================
echo.
call :log "[2/5] Bien dich loi tang toc Rust airteach_core_rs (maturin develop --release)..."
if exist "%USERPROFILE%\.cargo\bin\cargo.exe" set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

if not exist "%REPO_DIR%\rust_core\Cargo.toml" (
    call :log "  [CANH BAO] Khong tim thay rust_core\Cargo.toml. Bo qua bien dich Rust."
    goto :step_assets
)

where cargo >nul 2>&1
if errorlevel 1 (
    call :log "  [CANH BAO] Khong tim thay cargo trong PATH. Hay chay setup_all.bat de cai Rust."
    "%VENV_PY%" -c "import airteach_core_rs; print('  [OK] Rust Core co san:', airteach_core_rs.version(), '-', airteach_core_rs.backend_info())" >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log "  [THONG TIN] Tiep tuc dong goi voi che do Python Fallback."
        cmd /c exit /b 0
    ) else (
        "%VENV_PY%" -c "import airteach_core_rs; print('  [OK] Rust Core co san:', airteach_core_rs.version(), '-', airteach_core_rs.backend_info())"
    )
    goto :step_assets
)

if not exist "%MATURIN_EXE%" (
    call :log "[LOI] Khong tim thay .venv\Scripts\maturin.exe. Hay chay setup_all.bat truoc."
    goto :fail
)

for /f "delims=" %%R in ('cargo --version 2^>nul') do call :log "  [OK] %%R"
"%MATURIN_EXE%" develop --release --manifest-path "%REPO_DIR%\rust_core\Cargo.toml" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "[LOI] Bien dich rust_core bang maturin that bai. Xem chi tiet trong build_exe.log."
    goto :fail
)

"%VENV_PY%" -c "import airteach_core_rs; print('  [OK] Rust Core:', airteach_core_rs.version(), '-', airteach_core_rs.backend_info())"
if errorlevel 1 (
    call :log "[LOI] Khong the import airteach_core_rs sau khi bien dich."
    goto :fail
)
"%VENV_PY%" -c "import airteach_core_rs; print('  [OK] Rust Core:', airteach_core_rs.version(), '-', airteach_core_rs.backend_info())" >> "%LOG_FILE%" 2>&1

rem ============================================================================
rem [3/5] KIEM TRA TAI NGUYEN (ASSETS / MODELS / LOCALES / CONFIG)
rem ============================================================================
:step_assets
echo.
call :log "[3/5] Kiem tra tai nguyen dong goi (models, fonts, icons, locales, config)..."
if exist "%REPO_DIR%\scripts\fetch_assets.py" (
    "%VENV_PY%" "%REPO_DIR%\scripts\fetch_assets.py" >> "%LOG_FILE%" 2>&1
    if errorlevel 1 (
        call :log "[LOI] scripts\fetch_assets.py bao loi. Xem chi tiet trong build_exe.log."
        goto :fail
    )
)
if not exist "%REPO_DIR%\locales\vi.json" (
    call :log "[LOI] Thieu locales\vi.json."
    goto :fail
)
if not exist "%REPO_DIR%\locales\en.json" (
    call :log "[LOI] Thieu locales\en.json."
    goto :fail
)
call :log "  [OK] Da xac nhan day du models, assets, locales va config."

rem ============================================================================
rem [4/5] DONG GOI STANDALONE EXE BANG PYINSTALLER (AirTeachAI.spec)
rem ============================================================================
echo.
call :log "[4/5] Dang dong goi dist\AirTeachAI\AirTeachAI.exe bang PyInstaller..."
if not exist "%PYINSTALLER_EXE%" (
    call :log "[LOI] Khong tim thay .venv\Scripts\pyinstaller.exe. Hay chay setup_all.bat truoc."
    goto :fail
)
if not exist "%REPO_DIR%\AirTeachAI.spec" (
    call :log "[LOI] Khong tim thay AirTeachAI.spec."
    goto :fail
)

"%PYINSTALLER_EXE%" --clean --noconfirm "%REPO_DIR%\AirTeachAI.spec" >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "[LOI] PyInstaller dong goi that bai. Xem chi tiet trong build_exe.log."
    goto :fail
)
if not exist "%REPO_DIR%\dist\AirTeachAI\AirTeachAI.exe" (
    call :log "[LOI] Khong tim thay file ket qua dist\AirTeachAI\AirTeachAI.exe."
    goto :fail
)

if exist "%REPO_DIR%\run_airteach.bat" copy /y "%REPO_DIR%\run_airteach.bat" "%REPO_DIR%\dist\AirTeachAI\run_airteach.bat" >nul 2>&1
if exist "%REPO_DIR%\Chay_AirTeachAI_AutoFix.bat" copy /y "%REPO_DIR%\Chay_AirTeachAI_AutoFix.bat" "%REPO_DIR%\dist\AirTeachAI\Chay_AirTeachAI_AutoFix.bat" >nul 2>&1
call :log "  [OK] Da tao thanh cong dist\AirTeachAI\AirTeachAI.exe va copy launchers."

rem ============================================================================
rem [5/5] CHAY SMOKE TEST KIEM TRA ZERO MISSING DLL / MODULE
rem ============================================================================
echo.
call :log "[5/5] Chay kiem thu doc lap dist\AirTeachAI\AirTeachAI.exe --smoke-test..."
"%REPO_DIR%\dist\AirTeachAI\AirTeachAI.exe" --smoke-test >> "%LOG_FILE%" 2>&1
if errorlevel 1 (
    call :log "[LOI] AirTeachAI.exe --smoke-test that bai. Xem chi tiet trong build_exe.log."
    goto :fail
)
call :log "  [OK] Smoke test AirTeachAI.exe thanh cong [khong thieu DLL hay module]."

echo.
call :log "============================================================================"
call :log "  [HOAN TAT 100%%] DONG GOI AIRTEACH AI THANH CONG."
call :log "  - File EXE        : %REPO_DIR%\dist\AirTeachAI\AirTeachAI.exe"
call :log "  - Launcher Nhanh  : %REPO_DIR%\dist\AirTeachAI\run_airteach.bat"
call :log "  - Launcher MayThi : %REPO_DIR%\dist\AirTeachAI\Chay_AirTeachAI_AutoFix.bat"
call :log "  - File Log        : %LOG_FILE%"
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
