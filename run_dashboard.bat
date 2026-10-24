@echo off
rem Launch the gold forecast dashboard (Streamlit) in the default browser.
rem Picks the next free port if 8501 is taken. Close this window to stop it.
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo .venv not found in %~dp0 - create it first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m streamlit run src\goldcast\dashboard\app.py --browser.gatherUsageStats false
pause
