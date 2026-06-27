#!/bin/bash

# Define cleanup function to kill background processes when stopping the script
cleanup() {
    echo ""
    echo "Stopping LLM2Seq..."
    kill $(jobs -p) 2>/dev/null
    exit
}

# Catch Ctrl+C (SIGINT) and terminal termination
trap cleanup SIGINT SIGTERM

echo "Starting LLM2Seq Backend..."
cd App/backend || exit 1
# Assuming uvicorn is available in the current environment
uvicorn main:app --host 0.0.0.0 --port 8000 &
BE_PID=$!

echo "Starting LLM2Seq Frontend..."
cd ../frontend || exit 1
npm run dev &
FE_PID=$!

echo ""
echo "Both Frontend and Backend are running!"
echo "Backend API:  http://localhost:8000"
echo "Frontend Web: http://localhost:5173"
echo "Press Ctrl+C to stop both servers."
echo ""

wait
