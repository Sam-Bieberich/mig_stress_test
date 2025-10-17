#!/bin/bash

##############################################################################
# CUDA API Edge Cases Test - Background Runner
# 
# Assumes MIG partitions are already created.
# Runs CUDA API edge cases test (unusual but valid CUDA operations) in background.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PID_FILE="${LOG_DIR}/cuda_test_${TIMESTAMP}.pid"
BACKGROUND_LOG="${LOG_DIR}/background_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "======================================"
echo "CUDA API Edge Cases Test"
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

# Run the CUDA API test in background
echo "Starting CUDA API edge cases test in background..."
echo ""

nohup bash "${SCRIPT_DIR}/cuda_stress_test.sh" > "$BACKGROUND_LOG" 2>&1 &
STRESS_PID=$!

# Save PID
echo "$STRESS_PID" > "$PID_FILE"

echo "CUDA API test started successfully!"
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
echo "  tail -f ${LOG_DIR}/cuda_test_*.log"
echo ""
echo "Monitor GPU status:"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "Stop the test:"
echo "  kill $STRESS_PID"
echo ""
echo "======================================"

# Wait and verify
sleep 2
if ps -p $STRESS_PID > /dev/null; then
    echo "✓ CUDA API test is running (PID: $STRESS_PID)"
    echo ""
    echo "This test exercises unusual but valid CUDA API patterns"
    echo "including edge cases, limits, and stress scenarios."
else
    echo "✗ ERROR: Test failed to start. Check $BACKGROUND_LOG for details."
    exit 1
fi
