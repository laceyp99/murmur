Set WinScriptHost = CreateObject("WScript.Shell")
WinScriptHost.Run Chr(34) & ".\venv\Scripts\pythonw.exe" & Chr(34) & " run.py", 0
Set WinScriptHost = Nothing