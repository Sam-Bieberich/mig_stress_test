#!/bin/bash

##############################################################################
# Run MIG Intense Stress Test in Background
# 
# This script launches the intense MIG stress test in the background.
# Tests one MIG slice at 95% memory while all others run at 75% simultaneously.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STRESS_TEST_SCRIPT="${SCRIPT_DIR}/intense_stress.sh"
LOG_DIR="${SCRIPT_DIR}/stress_test_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PID_FILE="${LOG_DIR}/intense_stress_${TIMESTAMP}.pid"
BACKGROUND_LOG="${LOG_DIR}/intense_stress_background_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

# Check if stress test script exists
if [ ! -f "$STRESS_TEST_SCRIPT" ]; then
    echo "ERROR: Intense stress test script not found at $STRESS_TEST_SCRIPT"
    exit 1
fi

# Make sure the script is executable
chmod +x "$STRESS_TEST_SCRIPT"

echo "======================================"
echo "Starting INTENSE MIG Stress Test"
echo "======================================"
echo ""
echo "Test Configuration:"
echo "  - Primary device: 95% memory allocation"
echo "  - Background devices: 75% memory each"
echo "  - Duration: 30 minutes per primary device"
echo "  - All devices stressed simultaneously"
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

# Count MIG devices to estimate time
MIG_COUNT=$(nvidia-smi -L | grep "MIG" | wc -l)
TOTAL_MINUTES=$((MIG_COUNT * 30))
TOTAL_HOURS=$((TOTAL_MINUTES / 60))
REMAINING_MINUTES=$((TOTAL_MINUTES % 60))

echo "Detected $MIG_COUNT MIG devices"
echo "Estimated total test time: ${TOTAL_HOURS}h ${REMAINING_MINUTES}m"
echo ""

# Start the stress test in background
echo "Launching intense stress test in background..."
nohup bash "$STRESS_TEST_SCRIPT" > "$BACKGROUND_LOG" 2>&1 &
STRESS_PID=$!

# Save PID
echo "$STRESS_PID" > "$PID_FILE"

echo "Intense stress test started successfully!"
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
echo "  tail -f ${LOG_DIR}/intense_stress_*.log"
echo ""
echo "View Python worker logs:"
echo "  tail -f ${LOG_DIR}/intense_stress_python.log"
echo ""
echo "View errors only:"
echo "  tail -f ${LOG_DIR}/intense_stress_errors_*.log"
echo ""
echo "Stop the stress test:"
echo "  kill $STRESS_PID"
echo "  # Or force kill all related processes:"
echo "  pkill -P $STRESS_PID"
echo ""
echo "Monitor ALL GPUs in real-time:"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "Monitor MIG devices specifically:"
echo "  watch -n 2 'nvidia-smi -L && nvidia-smi --query-gpu=memory.used,memory.total --format=csv'"
echo ""
echo "======================================"

# Wait a moment and check if the process started successfully
sleep 2
if ps -p $STRESS_PID > /dev/null; then
    echo "✓ Intense stress test is running (PID: $STRESS_PID)"
    echo ""
    echo "⚠️  WARNING: This test is VERY intensive!"
    echo "   - Multiple MIG slices running simultaneously"
    echo "   - High memory usage across all devices"
    echo "   - Test duration: ~${TOTAL_HOURS}h ${REMAINING_MINUTES}m total"
    echo ""
    echo "Check logs regularly for any abnormalities."
else
    echo "✗ ERROR: Intense stress test failed to start."
    echo "Check $BACKGROUND_LOG for details."
    exit 1
fi
