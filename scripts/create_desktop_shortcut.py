"""
AirTeach AI - Cross-Platform Desktop Shortcut Generator
=======================================================
Creates a native desktop shortcut with the official AirTeach AI icon on:
  - Windows: %USERPROFILE%\\Desktop\\AirTeach AI.lnk (using WScript.Shell and airteach.ico)
  - Linux:   ~/Desktop/airteach.desktop and ~/.local/share/applications/airteach.desktop (using airteach.png)
  - macOS:   ~/Desktop/AirTeach AI.app (native macOS application bundle with airteach.icns)
"""
from __future__ import annotations

import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = ROOT / "assets" / "icons"


def get_desktop_path() -> Path:
    """Return the current user's Desktop path across platforms."""
    if sys.platform == "win32":
        user_profile = os.environ.get("USERPROFILE")
        if user_profile:
            desktop = Path(user_profile) / "Desktop"
            if desktop.exists():
                return desktop
            onedrive_desktop = Path(user_profile) / "OneDrive" / "Desktop"
            if onedrive_desktop.exists():
                return onedrive_desktop
        return Path(os.path.expanduser("~/Desktop"))

    # Linux (handles localized Desktop folders e.g. Bàn làm việc)
    desktop_en = Path(os.path.expanduser("~/Desktop"))
    if desktop_en.exists():
        return desktop_en
    desktop_vi = Path(os.path.expanduser("~/Bàn làm việc"))
    if desktop_vi.exists():
        return desktop_vi
    return desktop_en


def create_windows_shortcut() -> bool:
    desktop = get_desktop_path()
    shortcut_path = desktop / "AirTeach AI.lnk"
    icon_path = ASSETS_DIR / "airteach.ico"

    dist_exe = ROOT / "dist" / "AirTeachAI" / "AirTeachAI.exe"
    vbs_launcher = ROOT / "AirTeach_AI.vbs"
    bat_launcher = ROOT / "Run_AirTeach_AI.bat"

    if dist_exe.exists():
        target_path = dist_exe
        arguments = ""
    elif vbs_launcher.exists():
        target_path = Path("wscript.exe")
        arguments = f'"{vbs_launcher}"'
    else:
        target_path = bat_launcher
        arguments = ""

    description = "AirTeach AI - Touchless Gesture Teaching Workstation"

    print(f"[INFO] Creating Windows shortcut at: {shortcut_path}")
    print(f"  Target: {target_path} {arguments}")
    print(f"  Icon:   {icon_path}")

    ps_script = f"""
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut('{str(shortcut_path)}')
$Shortcut.TargetPath = '{str(target_path)}'
$Shortcut.Arguments = '{arguments}'
$Shortcut.WorkingDirectory = '{str(ROOT)}'
$Shortcut.IconLocation = '{str(icon_path)}, 0'
$Shortcut.Description = '{description}'
$Shortcut.Save()
"""
    try:
        subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True,
            check=True,
        )
        if shortcut_path.exists():
            print(f"[SUCCESS] Windows desktop shortcut created: {shortcut_path}")
            return True
    except Exception as e:
        print(f"[ERROR] Failed to create Windows shortcut: {e}")
    return False


def create_linux_shortcut() -> bool:
    desktop = get_desktop_path()
    icon_path = ASSETS_DIR / "airteach.png"
    if not icon_path.exists():
        icon_path = ASSETS_DIR / "logo.svg"

    launcher_path = ROOT / "run_linux.sh"
    if launcher_path.exists():
        launcher_path.chmod(launcher_path.stat().st_mode | stat.S_IEXEC)

    desktop_content = f"""[Desktop Entry]
Version=1.0
Type=Application
Name=AirTeach AI
Comment=Nền tảng dạy học tương tác không chạm bằng AI
Exec={str(launcher_path)}
Icon={str(icon_path)}
Path={str(ROOT)}
Terminal=false
StartupNotify=true
Categories=Education;Science;AudioVideo;
Keywords=teaching;gesture;smartboard;ai;education;
"""
    success = False

    # 1. System Applications Menu
    app_menu_dir = Path(os.path.expanduser("~/.local/share/applications"))
    try:
        app_menu_dir.mkdir(parents=True, exist_ok=True)
        menu_entry = app_menu_dir / "airteach.desktop"
        menu_entry.write_text(desktop_content, encoding="utf-8")
        menu_entry.chmod(menu_entry.stat().st_mode | stat.S_IEXEC)
        print(f"[SUCCESS] Added to Linux Application Menu: {menu_entry}")
        success = True
    except Exception as e:
        print(f"[WARNING] Could not create app menu entry: {e}")

    # 2. Desktop Shortcut
    if desktop.exists():
        shortcut = desktop / "airteach.desktop"
        try:
            shortcut.write_text(desktop_content, encoding="utf-8")
            shortcut.chmod(shortcut.stat().st_mode | stat.S_IEXEC)
            subprocess.run(
                ["gio", "set", str(shortcut), "metadata::trusted", "true"],
                capture_output=True,
            )
            print(f"[SUCCESS] Created Linux Desktop shortcut: {shortcut}")
            success = True
        except Exception as e:
            print(f"[WARNING] Could not create Desktop shortcut: {e}")

    return success


def create_macos_shortcut() -> bool:
    desktop = get_desktop_path()
    icon_path = ASSETS_DIR / "airteach.icns"
    app_bundle = ROOT / "AirTeach AI.app"
    macos_dir = app_bundle / "Contents" / "MacOS"
    resources_dir = app_bundle / "Contents" / "Resources"

    try:
        macos_dir.mkdir(parents=True, exist_ok=True)
        resources_dir.mkdir(parents=True, exist_ok=True)

        if icon_path.exists():
            shutil.copy2(str(icon_path), str(resources_dir / "airteach.icns"))

        plist_content = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleExecutable</key>
    <string>AirTeachAI</string>
    <key>CFBundleIconFile</key>
    <string>airteach.icns</string>
    <key>CFBundleIdentifier</key>
    <string>com.airteach.ai</string>
    <key>CFBundleName</key>
    <string>AirTeach AI</string>
    <key>CFBundlePackageType</key>
    <string>APPL</string>
    <key>CFBundleShortVersionString</key>
    <string>1.0.0</string>
    <key>NSHighResolutionCapable</key>
    <true/>
    <key>NSCameraUsageDescription</key>
    <string>AirTeach AI uses the camera for touchless hand gesture tracking.</string>
</dict>
</plist>
"""
        (app_bundle / "Contents" / "Info.plist").write_text(plist_content, encoding="utf-8")

        launcher_script = f"""#!/usr/bin/env bash
cd "{ROOT}"
if [ -f "{ROOT}/.venv/bin/activate" ]; then
    source "{ROOT}/.venv/bin/activate"
fi
exec python3 "{ROOT}/main.py" "$@"
"""
        exec_file = macos_dir / "AirTeachAI"
        exec_file.write_text(launcher_script, encoding="utf-8")
        exec_file.chmod(exec_file.stat().st_mode | stat.S_IEXEC)

        if desktop.exists():
            dest_app = desktop / "AirTeach AI.app"
            if dest_app.exists():
                shutil.rmtree(str(dest_app), ignore_errors=True)
            shutil.copytree(str(app_bundle), str(dest_app))
            print(f"[SUCCESS] Created macOS Desktop App bundle: {dest_app}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to create macOS App bundle: {e}")
        return False


def create_shortcut() -> bool:
    if sys.platform == "win32":
        return create_windows_shortcut()
    elif sys.platform.startswith("linux"):
        return create_linux_shortcut()
    elif sys.platform == "darwin":
        return create_macos_shortcut()
    else:
        print(f"[WARNING] Unsupported OS for desktop shortcut: {sys.platform}")
        return False


if __name__ == "__main__":
    ok = create_shortcut()
    sys.exit(0 if ok else 1)
