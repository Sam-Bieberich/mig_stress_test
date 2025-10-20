#!/bin/bash

##############################################################################
# Multi-Process Mayhem Test - Background Runner
# 
# Assumes MIG partitions are already created.
# Tests multi-process handling by launching multiple Python processes
# per MIG instance, each doing different operations simultaneously.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PID_FILE="${LOG_DIR}/multiproc_test_${TIMESTAMP}.pid"
BACKGROUND_LOG="${LOG_DIR}/background_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "======================================"
echo "Multi-Process Mayhem Test"
echo "======================================"
echo ""

# Check for MIG devices
MIG_COUNT=$(nvidia-smi -L | grep "MIG" | wc -l)
if [ "$MIG_COUNT" -eq 0 ]; then
    echo "ERROR: No MIG devices found!"
    echo "Please create MIG partitions first using:"
    echo "  cd .. && ./mig_easy_setup.sh"
    exit 1
fi

echo "Found $MIG_COUNT MIG device(s)"
echo ""

# Check for PyTorch
if ! python3 -c "import torch" 2>/dev/null; then
    echo "WARNING: PyTorch not found. Attempting to install..."
    pip3 install torch --index-url https://download.pytorch.org/whl/cu118 >> "$BACKGROUND_LOG" 2>&1
    
    if ! python3 -c "import torch" 2>/dev/null; then
        echo "ERROR: Failed to install PyTorch. Please install manually:"
        echo "  pip3 install torch"
        exit 1
    fi
    echo "PyTorch installed successfully."
fi
echo ""

# Run the multi-process test in background
echo "Starting multi-process test in background..."
echo ""

nohup bash "${SCRIPT_DIR}/multiproc_stress_test.sh" > "$BACKGROUND_LOG" 2>&1 &
STRESS_PID=$!

# Save PID
echo "$STRESS_PID" > "$PID_FILE"

echo "Multi-process test started successfully!"
echo ""
echo "Process ID: $STRESS_PID"
echo "PID File: $PID_FILE"
echo "Background Log: $BACKGROUND_LOG"
echo "Test Logs: ${LOG_DIR}/"
echo ""
echo "======================================"
echo "Monitoring Commands:"
echo "======================================"
echo ""
echo "Check if process is running:"
echo "  ps -p $STRESS_PID"
echo ""
echo "View live log output:"
echo "  tail -f $BACKGROUND_LOG"
echo ""
echo "View test progress:"
echo "  tail -f ${LOG_DIR}/multiproc_test_*.log"
echo ""
echo "Monitor GPU status:"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "View all Python processes:"
echo "  ps aux | grep multiproc_worker"
echo ""
echo "Stop the test:"
echo "  kill $STRESS_PID"
echo "  pkill -f multiproc_worker.py"
echo ""
