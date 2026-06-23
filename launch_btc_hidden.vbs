' Launches the BTC Analysis dashboard headless, in the background, with no
' visible console window. Settings (headless / no file-watcher) come from
' .streamlit\config.toml. Open http://localhost:8501 in a browser to view.
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "C:\Users\zacha\OneDrive\Projects\BTC"
sh.Run """C:\Users\zacha\miniconda3\envs\env1\python.exe"" -m streamlit run btc_analysis_app.py", 0, False
