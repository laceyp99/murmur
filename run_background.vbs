Set WinScriptHost = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

Dim pythonPath
Dim scriptPath

' Get absolute path to the current directory
strPath = fso.GetParentFolderName(WScript.ScriptFullName)
scriptPath = Chr(34) & strPath & "\run.py" & Chr(34)

' Check for venv, otherwise use system pythonw
If fso.FileExists(strPath & "\venv\Scripts\pythonw.exe") Then
    pythonPath = Chr(34) & strPath & "\venv\Scripts\pythonw.exe" & Chr(34)
Else
    pythonPath = "pythonw.exe"
End If

WinScriptHost.Run pythonPath & " " & scriptPath, 0
Set WinScriptHost = Nothing