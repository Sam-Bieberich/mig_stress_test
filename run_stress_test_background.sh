#!/bin/bash

##############################################################################
# Run MIG Stress Test in Background
# 
# This script launches the MIG stress test in the background and provides
# commands to monitor progress and check logs.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRESS_TEST_SCRIPT="${SCRIPT_DIR}/mig_stress_test.sh"
LOG_DIR="${SCRIPT_DIR}/stress_test_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PID_FILE="${LOG_DIR}/stress_test_${TIMESTAMP}.pid"
BACKGROUND_LOG="${LOG_DIR}/stress_test_background_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if stress test script exists
if [ ! -f "$STRESS_TEST_SCRIPT" ]; then
    echo "ERROR: Stress test script not found at $STRESS_TEST_SCRIPT"
    exit 1
fi

# Make sure the script is executable
chmod +x "$STRESS_TEST_SCRIPT"

echo "======================================"
echo "Starting MIG Stress Test in Background"
echo "======================================"
echo ""

# Check if PyTorch is installed
if ! python3 -c "import torch" 2>/dev/null; then
    echo "WARNING: PyTorch not found. Attempting to install..."
    echo "This may take a few minutes..."
    pip3 install torch --index-url https://download.pytorch.org/whl/cu118 >> "$BACKGROUND_LOG" 2>&1
    
    if ! python3 -c "import torch" 2>/dev/null; then
        echo "ERROR: Failed to install PyTorch. Please install manually:"
        echo "  pip3 install torch"
        exit 1
    fi
    echo "PyTorch installed successfully."
    echo ""
fi

# Start the stress test in background
echo "Launching stress test..."
nohup bash "$STRESS_TEST_SCRIPT" > "$BACKGROUND_LOG" 2>&1 &
STRESS_PID=$!

# Save PID
echo "$STRESS_PID" > "$PID_FILE"

echo "Stress test started successfully!"
echo ""
echo "Process ID: $STRESS_PID"
echo "PID File: $PID_FILE"
echo "Background Log: $BACKGROUND_LOG"
echo "Log Directory: $LOG_DIR"
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
echo "View main stress test log (once created):"
echo "  tail -f ${LOG_DIR}/stress_test_*.log"
echo ""
echo "View errors only:"
echo "  tail -f ${LOG_DIR}/stress_test_errors_*.log"
echo ""
echo "Stop the stress test:"
echo "  kill $STRESS_PID"
echo ""
echo "Check GPU status:"
echo "  watch -n 5 nvidia-smi"
echo ""
echo "======================================"

# Wait a moment and check if the process started successfully
sleep 2
if ps -p $STRESS_PID > /dev/null; then
    echo "✓ Stress test is running (PID: $STRESS_PID)"
    echo ""
    echo "The test will run for approximately 30 minutes per MIG device."
    echo "Check logs for progress and any abnormalities."
else
    echo "✗ ERROR: Stress test failed to start. Check $BACKGROUND_LOG for details."
    exit 1
fi
