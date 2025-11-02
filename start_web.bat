@echo off
echo 🚀 Starting CampusWell Web App...
set PYTHONPATH=src
call venv\Scripts\activate
python -m uvicorn app.main:app --reload
pause
