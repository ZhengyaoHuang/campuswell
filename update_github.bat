@echo off
echo ================================
echo  🚀 Updating Project on GitHub
echo ================================

:: Move to your project folder
cd /d "C:\Users\yao\Downloads\campuswell_pro_webapp"

:: Check if .git exists
if not exist ".git" (
    echo ❌ This folder is not a Git repository!
    pause
    exit /b
)

:: Add all changes
echo 🧩 Adding modified files...
git add .

:: Commit changes with a timestamp
setlocal enabledelayedexpansion
for /f "tokens=1-4 delims=/ " %%a in ("%date%") do (
    set today=%%a-%%b-%%c
)
git commit -m "Auto-update on !today!"

:: Push to GitHub
echo ☁️ Pushing changes to GitHub...
git push origin main

echo ✅ Update complete! Changes successfully pushed.
pause
