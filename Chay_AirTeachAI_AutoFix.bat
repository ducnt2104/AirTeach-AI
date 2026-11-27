@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"
title AirTeach AI — Khoi chay Tu dong Sua loi May Thi

echo ============================================================================
echo   AIRTEACH AI — TRINH KHOI CHAY AN TOAN CHO MAY THI KHKT (AUTO-FIX)
echo ============================================================================
echo.

rem --- [1/4] Kiem tra Visual C++ 2015-2022 Redistributable (x64) ---
echo [1/4] Kiem tra Microsoft Visual C++ 2015-2022 Redistributable x64...
set "VCR_OK=1"
if not exist "%SystemRoot%\System32\vcruntime140_1.dll" set "VCR_OK=0"
if not exist "%SystemRoot%\System32\vcruntime140.dll" set "VCR_OK=0"
if not exist "%SystemRoot%\System32\msvcp140.dll" set "VCR_OK=0"

if "!VCR_OK!"=="1" goto :vcredist_ready

echo   [CANH BAO] Thieu thu vien Visual C++ x64 trong System32. Dang tu dong tai va cai dat...
set "AIRTEACH_VCR_EXE=%TEMP%\vc_redist.x64.exe"
powershell -NoProfile -ExecutionPolicy Bypass -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://aka.ms/vs/17/release/vc_redist.x64.exe' -OutFile $env:AIRTEACH_VCR_EXE -UseBasicParsing"
if errorlevel 1 goto :vcredist_warn

echo   Dang cai dat vc_redist.x64.exe che do im lang (/install /quiet /norestart)...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$p = Start-Process -FilePath $env:AIRTEACH_VCR_EXE -ArgumentList '/install','/quiet','/norestart' -Verb RunAs -PassThru -Wait; exit $p.ExitCode"
if errorlevel 1 goto :vcredist_warn

if exist "%SystemRoot%\System32\vcruntime140_1.dll" (
    echo   [OK] Da cai dat xong Microsoft Visual C++ 2015-2022 Redistributable x64!
    goto :vcredist_ready
)

:vcredist_warn
echo   [CANH BAO] Khong the tu cai VC++ he thong ^(co the thieu mang hoac quyen Admin^).
echo   Chuong trinh se thu su dung ban DLL dong goi san trong thu muc _internal.

:vcredist_ready

rem --- [2/4] Khoi tao thu muc du lieu AppData & kiem tra quyen Camera ---
echo [2/4] Khoi tao thu muc du lieu nguoi dung tai %%APPDATA%%\AirTeachAI...
set "APPDATA_ROOT=%APPDATA%\AirTeachAI"
for %%D in ("" "\logs" "\doc_cache" "\face_embeddings" "\recordings" "\models") do (
    if not exist "%APPDATA_ROOT%%%~D" mkdir "%APPDATA_ROOT%%%~D" >nul 2>&1
)
echo   [OK] Thu muc du lieu san sang: "%APPDATA_ROOT%"

echo   Kiem tra quyen truy cap Camera Windows...
set "CAM_DENIED=0"
set "CAM_SYSTEM_DENIED=0"

reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam" /v Value 2>nul | findstr /i "Deny" >nul
if not errorlevel 1 set "CAM_DENIED=1"

reg query "HKCU\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam\NonPackaged" /v Value 2>nul | findstr /i "Deny" >nul
if not errorlevel 1 set "CAM_DENIED=1"

reg query "HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam" /v Value 2>nul | findstr /i "Deny" >nul
if not errorlevel 1 set "CAM_SYSTEM_DENIED=1"

if "!CAM_DENIED!"=="1" (
    echo   [CANH BAO] Quyen Camera desktop dang bi tat. Dang tu dong bat lai Allow...
    reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam\NonPackaged" /v Value /t REG_SZ /d Allow /f >nul 2>&1
    reg add "HKCU\Software\Microsoft\Windows\CurrentVersion\CapabilityAccessManager\ConsentStore\webcam" /v Value /t REG_SZ /d Allow /f >nul 2>&1
)
if "!CAM_SYSTEM_DENIED!"=="1" (
    echo   [LUU Y] Windows Privacy dang tat quyen Camera toan he thong!
    echo   Vui long bat "Camera access" trong Settings / Privacy and security / Camera.
) else (
    echo   [OK] Quyen Camera da san sang.
)

rem --- [3/4] Tim va khoi chay AirTeachAI.exe ---
echo [3/4] Dang tim file thuc thi AirTeachAI.exe...
set "TARGET_EXE="
if exist "%~dp0AirTeachAI.exe" set "TARGET_EXE=%~dp0AirTeachAI.exe"
if not defined TARGET_EXE if exist "%~dp0dist\AirTeachAI\AirTeachAI.exe" set "TARGET_EXE=%~dp0dist\AirTeachAI\AirTeachAI.exe"

if not defined TARGET_EXE (
    echo.
    echo [LOI] Khong tim thay AirTeachAI.exe ben canh launcher hoac trong dist\AirTeachAI\!
    echo Vui long chay "setup_and_build.bat" de dong goi ung dung truoc.
    set "EXIT_CODE=2"
    goto :end_pause
)

echo   [OK] Khoi chay: "%TARGET_EXE%"
echo   [THONG TIN] Neu ung dung dong dot ngot, chi tiet se duoc luu vao crash_log.txt.
echo.

"%TARGET_EXE%" %*
set "EXIT_CODE=!ERRORLEVEL!"

if "!EXIT_CODE!"=="0" (
    echo.
    echo [4/4] AirTeach AI da thoat an toan ^(Ma thoat: 0^).
    goto :finish
)

rem --- [4/4] Ghi bao cao su co chi tiet khi thoat voi ma loi khac 0 ---
echo.
echo [4/4] PHAT HIEN SU CO! AirTeachAI.exe da dung voi ma loi: !EXIT_CODE!
set "AIRTEACH_CRASH_EXE=%TARGET_EXE%"
set "AIRTEACH_CRASH_LOG=%~dp0crash_log.txt"
set "AIRTEACH_CRASH_CODE=!EXIT_CODE!"
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "$ErrorActionPreference='Continue';" ^
    "$exitCode = [int]$env:AIRTEACH_CRASH_CODE;" ^
    "$u32 = ([BitConverter]::ToUInt32([BitConverter]::GetBytes($exitCode), 0));" ^
    "$hex = ('0x{0:X8}' -f $u32);" ^
    "$reason = switch ($hex) {" ^
    "  '0xC0000135' { 'Thieu thu vien DLL he thong (thuong do thieu Microsoft Visual C++ 2015-2022 Redistributable x64).' }" ^
    "  '0xC000007B' { 'Xung dot kien truc DLL 32-bit va 64-bit (STATUS_INVALID_IMAGE_FORMAT).' }" ^
    "  '0xC0000005' { 'Loi truy cap bo nho (Access Violation) - co the do driver GPU/Camera hoac OpenGL.' }" ^
    "  '0xC0000409' { 'Stack buffer overrun / loi Abort tu thu vien C/C++ native.' }" ^
    "  '0x00000001' { 'Ung dung bao loi khoi tao (xem log chi tiet ben duoi).' }" ^
    "  default { 'Loi thuc thi khong xac dinh - xem dong cuoi trong app.log.' }" ^
    "};" ^
    "$sys32 = Join-Path $env:SystemRoot 'System32';" ^
    "$vc1 = Test-Path (Join-Path $sys32 'vcruntime140.dll');" ^
    "$vc2 = Test-Path (Join-Path $sys32 'vcruntime140_1.dll');" ^
    "$vc3 = Test-Path (Join-Path $sys32 'msvcp140.dll');" ^
    "$appRoot = Join-Path $env:APPDATA 'AirTeachAI';" ^
    "$appLog = Join-Path $appRoot 'logs\app.log';" ^
    "$lines = @(" ^
    "  '================ AIRTEACH AI - BAO CAO SU CO ================'," ^
    "  ('Thoi gian       : ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))," ^
    "  ('May tinh        : ' + $env:COMPUTERNAME)," ^
    "  ('He dieu hanh    : ' + [System.Environment]::OSVersion.VersionString)," ^
    "  ('File thuc thi   : ' + $env:AIRTEACH_CRASH_EXE)," ^
    "  ('Ma loi (Exit)   : ' + $exitCode + ' (' + $hex + ')')," ^
    "  ('Nguyen nhan     : ' + $reason)," ^
    "  ('VC++ DLLs       : vcruntime140=' + $vc1 + ', vcruntime140_1=' + $vc2 + ', msvcp140=' + $vc3)," ^
    "  ('AppData Dir     : ' + $appRoot)" ^
    ");" ^
    "if (Test-Path $appLog) {" ^
    "  $lines += '';" ^
    "  $lines += '--- 60 dong log gan nhat (app.log) ---';" ^
    "  $lines += (Get-Content -Path $appLog -Tail 60 -Encoding UTF8);" ^
    "} else {" ^
    "  $lines += '--- Chua co file app.log (ung dung co the dung truoc khi vao Python main) ---';" ^
    "}" ^
    "$outPath = $env:AIRTEACH_CRASH_LOG;" ^
    "try { [System.IO.File]::WriteAllLines($outPath, $lines, [System.Text.UTF8Encoding]::new($false)); }" ^
    "catch { $outPath = Join-Path $appRoot 'logs\crash_log.txt'; [System.IO.File]::WriteAllLines($outPath, $lines, [System.Text.UTF8Encoding]::new($false)); }" ^
    "Write-Host '';" ^
    "Write-Host ('Da ghi bao cao su co tai: ' + $outPath) -ForegroundColor Yellow;" ^
    "$lines | ForEach-Object { Write-Host $_ }"

:end_pause
if not defined AIRTEACH_NO_PAUSE if not defined CI (
    echo.
    pause
)

:finish
exit /b !EXIT_CODE!
