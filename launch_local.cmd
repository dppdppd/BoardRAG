@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM Move to project root (directory of this script)
set "ROOT=%~dp0"
cd /d "%ROOT%"

echo.
echo === BoardRAG Local Dev Launcher ===

taskkill /F /IM python.exe
taskkill /F /IM node.exe 

call venv\Scripts\activate

REM Ensure Python venv exists
if not exist "venv\Scripts\activate.bat" (
  echo [ERROR] Python virtual environment not found at venv\
  echo Create it with:
  echo     python -m venv venv
  echo     venv\Scripts\python -m pip install -r requirements.txt
  pause
  exit /b 1
)

REM Start FastAPI backend (uvicorn) in a new terminal
start "Board Game Jippity API" cmd /k pushd "%ROOT%" ^&^& call venv\Scripts\activate.bat ^&^& python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

REM Start Next.js frontend in a new terminal
if exist "web" (
  start "Board Game Jippity Web" cmd /k pushd "%ROOT%web" ^&^& npm install ^&^& npm run dev
) else (
  echo [WARN] web directory not found, skipping frontend.
)

echo.
echo Backend: http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:3000
echo Two terminals have been opened for API and Web. Press Ctrl+C in those windows to stop.
echo.

endlocal

