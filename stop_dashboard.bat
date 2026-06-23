@echo off
REM One-click: stop the BTC dashboard server to free its ~500 MB.
REM Targets ONLY the streamlit dashboard - leaves the logger task and Jupyter alone.
powershell -NoProfile -Command ^
  "$p = Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like '*streamlit*btc_analysis_app*' };" ^
  "if ($p) { $p | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }; Write-Host ('Dashboard stopped - freed ~{0} MB.' -f [math]::Round((($p | Measure-Object WorkingSetSize -Sum).Sum)/1MB)) }" ^
  "else { Write-Host 'No dashboard server running. Nothing to free.' }"
echo.
pause
