@echo off
REM Double-click to check the 72h-bias logger is alive. Window stays open.
cd /d "%~dp0"
"C:\Users\zacha\miniconda3\envs\env1\python.exe" check_logging.py
echo.
pause
