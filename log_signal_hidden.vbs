' Silent BTC signal logger.
' Runs log_signal.py with NO console window and NO popup (the "0" = hidden,
' "False" = fire-and-forget). Invoked by the "BTC Signal Logger" scheduled
' task at logon and every 5 minutes. Each run is a short-lived python process
' that POSTs one row to Supabase and exits, so idle memory use is ~zero.
Dim sh
Set sh = CreateObject("WScript.Shell")
sh.CurrentDirectory = "C:\Users\zacha\OneDrive\Projects\BTC"
sh.Run """C:\Users\zacha\miniconda3\envs\env1\python.exe"" log_signal.py", 0, False
