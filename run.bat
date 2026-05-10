@echo off
chcp 65001 > nul
cd /d "%~dp0"

echo ============================================
echo   MIDAS v01  -  Low-k Dielectric Platform
echo   Materials Informatics for Design and
echo   Automated Screening
echo ============================================
echo.
echo [INFO] Starting Streamlit...
echo [INFO] Browser: http://localhost:8502
echo [INFO] Press Ctrl+C to stop.
echo.

python -m streamlit run app.py --server.port 8502 --browser.gatherUsageStats false

pause
