#!/bin/bash

##############################################################################
# Background Launcher for Sequential Test Runner
# 
# Runs all MIG tests sequentially in the background with nohup
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG_DIR="${SCRIPT_DIR}/master_logs"
BACKGROUND_LOG="${MASTER_LOG_DIR}/background_run_${TIMESTAMP}.log"
PID_FILE="${MASTER_LOG_DIR}/sequential_run.pid"

# Create master log directory
mkdir -p "$MASTER_LOG_DIR"

echo "================================================================================"
echo "        MIG Stress Test Suite - Background Sequential Execution"
echo "================================================================================"
echo ""
echo "Starting all tests in background at $(date)"
echo ""

# Make the main script executable
chmod +x "${SCRIPT_DIR}/run_all_tests_sequential.sh"

# Check if another instance is running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "ERROR: Another sequential test run is already running (PID: $OLD_PID)"
        echo "Wait for it to complete or kill it with: kill $OLD_PID"
        exit 1
    else
        echo "Removing stale PID file..."
        rm "$PID_FILE"
    fi
fi

# Run in background with nohup
nohup bash "${SCRIPT_DIR}/run_all_tests_sequential.sh" > "$BACKGROUND_LOG" 2>&1 &
PROCESS_PID=$!

# Save PID
echo $PROCESS_PID > "$PID_FILE"

echo "Sequential test runner launched successfully!"
echo ""
echo "Process ID (PID): $PROCESS_PID"
echo "Background log:   $BACKGROUND_LOG"
echo "PID file:         $PID_FILE"
echo ""
echo "Master logs will be created in: $MASTER_LOG_DIR/"
echo ""
echo "================================================================================"
echo "                            MONITORING COMMANDS"
echo "================================================================================"
echo ""
echo "# Check if process is running:"
echo "  ps -p $PROCESS_PID"
echo ""
echo "# View live background log:"
echo "  tail -f $BACKGROUND_LOG"
echo ""
echo "# View master log (created after tests start):"
echo "  tail -f ${MASTER_LOG_DIR}/sequential_run_*.log"
echo ""
echo "# Check all master logs:"
echo "  ls -lh ${MASTER_LOG_DIR}/"
echo ""
echo "# Monitor GPU status:"
echo "  watch -n 2 nvidia-smi"
echo ""
echo "# Stop the sequential run:"
echo "  kill $PROCESS_PID"
echo ""
echo "================================================================================"
echo ""
echo "Estimated total runtime: ~4-5 hours for 8 tests"
echo "(30 min per test + 32 min pause between tests)"
echo ""
echo "The process will continue running even if you disconnect from SSH."
echo ""
