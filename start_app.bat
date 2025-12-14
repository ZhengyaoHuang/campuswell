@echo off
REM =========================================
REM Start CampusWell Depression Risk Predictor
REM =========================================

echo.
echo ========================================
echo   CampusWell Depression Risk Predictor
echo ========================================
echo.

cd /d "%~dp0"

REM Check and activate virtual environment
echo [1/3] Checking virtual environment...
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
    echo       Virtual environment: OK
) else (
    echo.
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv venv
    echo.
    pause
    exit /b
)

echo.
echo [2/3] Setting up environment...
set PYTHONPATH=src
echo       PYTHONPATH: %PYTHONPATH%

echo.
echo [3/3] Starting FastAPI server...
echo.
echo ========================================
echo   Server will start at:
echo   http://127.0.0.1:8000
echo ========================================
echo.
echo Opening browser in 3 seconds...
echo Press Ctrl+C to stop the server
echo.

REM Wait 3 seconds then open browser
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8000

REM Start FastAPI server (this will show uvicorn's output)
python -m uvicorn app.main:app --reload

echo.
echo Server stopped.
pause