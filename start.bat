@echo off
echo Starting Enhanced Math Quest LaTeX Pipeline...
echo.

echo Starting Python Backend...
start "Backend" cmd /c "venv\Scripts\python server\app.py"

echo Waiting for backend to initialize...
timeout /t 5 /nobreak > nul

echo Starting Frontend...
start "Frontend" cmd /c "npm run dev"

echo.
echo Both servers started! Access the app at: http://localhost:5173
echo Backend API available at: http://localhost:5000
echo.
pause
