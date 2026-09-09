Set WinScriptHost = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")

Dim pythonPath
Dim scriptPath
Dim packagedPath

' Get absolute path to the current directory
strPath = fso.GetParentFolderName(WScript.ScriptFullName)
scriptPath = Chr(34) & strPath & "\run.py" & Chr(34)
packagedPath = strPath & "\dist\Murmur\murmur.exe"

' Prefer source mode for development, then fall back to a packaged release
If fso.FileExists(strPath & "\venv\Scripts\pythonw.exe") Then
    pythonPath = Chr(34) & strPath & "\venv\Scripts\pythonw.exe" & Chr(34)
    WinScriptHost.Run pythonPath & " " & scriptPath, 0
ElseIf fso.FileExists(packagedPath) Then
    WinScriptHost.Run Chr(34) & packagedPath & Chr(34), 0
Else
    pythonPath = "pythonw.exe"
    WinScriptHost.Run pythonPath & " " & scriptPath, 0
End If

Set WinScriptHost = Nothing
