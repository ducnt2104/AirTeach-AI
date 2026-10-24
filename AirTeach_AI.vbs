' ==============================================================================
' AirTeach AI - Silent Windows Launcher (Zero Console Window Popup)
' ==============================================================================
Option Explicit

Dim oShell, sCurDir, sVenvPython, sSysPython, sCmd, fso

Set oShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

' Get current directory where this VBS script resides
sCurDir = fso.GetParentFolderName(WScript.ScriptFullName)
oShell.CurrentDirectory = sCurDir

sVenvPython = sCurDir & "\.venv\Scripts\pythonw.exe"
sSysPython = "pythonw.exe"

' If virtual environment with pythonw.exe exists, launch directly without console
If fso.FileExists(sVenvPython) Then
    sCmd = """" & sVenvPython & """ """ & sCurDir & "\main.py"""
    oShell.Run sCmd, 0, False
Else
    ' Fallback to Run_AirTeach_AI.bat hidden
    sCmd = """" & sCurDir & "\Run_AirTeach_AI.bat"""
    oShell.Run sCmd, 0, False
End If

Set oShell = Nothing
Set fso = Nothing
