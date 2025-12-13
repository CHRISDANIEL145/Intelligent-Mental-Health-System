@echo off
echo Stopping DC Well Being...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq DC Well Being App*"
taskkill /F /IM python.exe
taskkill /F /IM ollama.exe
echo Done.
pause
