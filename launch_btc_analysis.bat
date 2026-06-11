@echo off
title BTC Analysis Dashboard
cd /d "C:\Users\zacha\OneDrive\Projects\BTC"
echo Starting BTC Analysis Dashboard...
echo.
streamlit run btc_analysis_app.py --server.runOnSave=true
pause
