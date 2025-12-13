@echo off
echo Stopping Kiro AI System...
taskkill /F /IM python.exe
echo Stopped Python.
echo.
echo NOTE: Ollama is left running as it is a system service. 
echo To stop Ollama, close its window or use Task Manager.
echo.
pause
