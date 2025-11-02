@echo off
REM =========================================
REM Start Depression Predictor Web App
REM =========================================

echo Starting the Depression Risk Predictor...
cd /d "%~dp0"

REM Activate virtual environment
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
) else (
    echo [ERROR] Virtual environment not found. Please create it first.
    pause
    exit /b
)

REM Set Python path for FastAPI imports
set PYTHONPATH=src

REM Start FastAPI server
echo Launching FastAPI on http://127.0.0.1:8000 ...
python -m uvicorn app.main:app --reload

echo.
pause
