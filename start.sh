#!/bin/bash
echo "Starting Enhanced Math Quest LaTeX Pipeline..."
echo

echo "Starting Python Backend..."
source venv/bin/activate
python server/app.py &
BACKEND_PID=$!

echo "Waiting for backend to initialize..."
sleep 5

echo "Starting Frontend..."
npm run dev &
FRONTEND_PID=$!

echo
echo "Both servers started! Access the app at: http://localhost:5173"
echo "Backend API available at: http://localhost:5000"
echo "Backend PID: $BACKEND_PID"
echo "Frontend PID: $FRONTEND_PID"
echo

# Trap to kill both processes on script exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT

wait
