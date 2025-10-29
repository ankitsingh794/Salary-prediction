@echo off
REM Launch Salary Prediction Web Application
REM Professional Streamlit Frontend

echo ========================================================================
echo                  SALARY PREDICTION WEB APPLICATION
echo ========================================================================
echo.

REM Check if Streamlit is installed
python -c "import streamlit" >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing Streamlit and Plotly...
    python -m pip install streamlit plotly
    echo.
)

echo [INFO] Starting web application...
echo.
echo The app will open in your default browser.
echo Press Ctrl+C to stop the server.
echo.
echo ========================================================================
echo.

REM Launch Streamlit app
streamlit run app.py

pause
