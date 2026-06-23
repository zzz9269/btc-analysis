@echo off
title BTC Analysis Dashboard
cd /d "C:\Users\zacha\OneDrive\Projects\BTC"
echo Starting BTC Analysis Dashboard...

REM Find the first free port at/above 8501, so the shortcut still works even
REM when something else (another Streamlit app, etc.) already holds 8501.
for /f %%p in ('powershell -NoProfile -Command "$p=8501; while(Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue){$p++}; $p"') do set PORT=%%p

echo Using port %PORT% .  A browser tab will open automatically in a few seconds.
echo When done viewing: press Ctrl+C here, or run stop_dashboard.bat to free ~500 MB.
echo.
REM Open the dashboard in your default browser ~7s after start (once the server
REM is up), on the SAME port we actually launched on, in the background so it
REM doesn't block the server.
start "" /b powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 7; Start-Process 'http://localhost:%PORT%'"
REM --server.port overrides the 8501 in .streamlit/config.toml; the rest of that
REM file (headless, no file-watcher) still applies so the viewer uses less memory.
streamlit run btc_analysis_app.py --server.port %PORT%
pause
