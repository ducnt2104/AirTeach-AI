"""
AirTeach AI - Desktop Shortcut Generator
========================================
Creates a native Windows Desktop shortcut (.lnk) with the official AirTeach AI icon.
Supports pointing to:
  1. Compiled standalone `dist/AirTeachAI/AirTeachAI.exe` (if built)
  2. Silent Windows VBScript launcher `AirTeach_AI.vbs` (no console window)
  3. Batch launcher `Run_AirTeach_AI.bat`
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def get_desktop_path() -> Path:
    """Return the current user's Desktop path reliably on Windows."""
    # Priority 1: Environment variable USERPROFILE
    user_profile = os.environ.get("USERPROFILE")
    if user_profile:
        desktop = Path(user_profile) / "Desktop"
        if desktop.exists():
            return desktop

    # Priority 2: OneDrive desktop fallback (common in Windows 10/11)
    if user_profile:
        onedrive_desktop = Path(user_profile) / "OneDrive" / "Desktop"
        if onedrive_desktop.exists():
            return onedrive_desktop

    # Priority 3: os.path.expanduser
    return Path(os.path.expanduser("~/Desktop"))


def create_shortcut() -> bool:
    desktop = get_desktop_path()
    shortcut_path = desktop / "AirTeach AI.lnk"
    icon_path = ROOT / "assets" / "icons" / "airteach.ico"

    if not icon_path.exists():
        print(f"[WARNING] Icon not found at {icon_path}, fallback to default icon.")

    # Determine best target executable / launcher
    dist_exe = ROOT / "dist" / "AirTeachAI" / "AirTeachAI.exe"
    vbs_launcher = ROOT / "AirTeach_AI.vbs"
    bat_launcher = ROOT / "Run_AirTeach_AI.bat"

    if dist_exe.exists():
        target_path = dist_exe
        arguments = ""
        description = "AirTeach AI - Touchless Gesture Teaching Workstation"
    elif vbs_launcher.exists():
        target_path = Path("wscript.exe")
        arguments = f'"{vbs_launcher}"'
        description = "AirTeach AI - Touchless Gesture Teaching Workstation"
    else:
        target_path = bat_launcher
        arguments = ""
        description = "AirTeach AI - Touchless Gesture Teaching Workstation"

    print(f"[INFO] Creating desktop shortcut at: {shortcut_path}")
    print(f"  Target: {target_path} {arguments}")
    print(f"  Icon:   {icon_path}")
    print(f"  WorkDir:{ROOT}")

    # Use PowerShell COM object WScript.Shell to create the shortcut
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
        res = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
            capture_output=True,
            text=True,
            check=True
        )
        if shortcut_path.exists():
            print(f"[SUCCESS] Desktop shortcut created successfully: {shortcut_path}")
            return True
        else:
            print(f"[ERROR] Shortcut file not found after creation: {shortcut_path}")
            return False
    except Exception as e:
        print(f"[ERROR] Failed to create shortcut via PowerShell: {e}")
        return False


if __name__ == "__main__":
    success = create_shortcut()
    sys.exit(0 if success else 1)
