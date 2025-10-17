#!/bin/bash

##############################################################################
# Memory Thrashing Test - Complete Setup and Run Script
# 
# This script:
# 1. Sets up MIG partitions (7 instances)
# 2. Runs memory thrashing test (rapid alloc/free cycles)
# 3. Tests memory allocator and fragmentation handling
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
PID_FILE="${LOG_DIR}/thrashing_test_${TIMESTAMP}.pid"
BACKGROUND_LOG="${LOG_DIR}/background_${TIMESTAMP}.log"

# Create log directory
mkdir -p "$LOG_DIR"

echo "======================================"
echo "Memory Thrashing Test"
echo "======================================"
echo ""

# Step 1: Setup MIG partitions
echo "Step 1: Setting up MIG partitions..."
echo ""

# Check if MIG is enabled
if ! sudo nvidia-smi -i 0 -mig 1 2>&1 | grep -q "Enabled"; then
    echo "Enabling MIG mode on GPU 0..."
    sudo nvidia-smi -i 0 -mig 1
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to enable MIG mode. Exiting."
        exit 1
    fi
fi

# Delete old instances
echo "Deleting any old MIG instances..."
sudo nvidia-smi mig -dci -i 0 2>/dev/null
sudo nvidia-smi mig -dgi -i 0 2>/dev/null

# Create 7 partitions
echo "Creating 7 MIG partitions with profile 19..."
sudo nvidia-smi mig -cgi 19,19,19,19,19,19,19 -C
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create MIG partitions. Exiting."
    exit 1
fi

echo ""
echo "MIG partitions created successfully:"
nvidia-smi -L | grep MIG
echo ""

# Step 2: Check for PyTorch
echo "Step 2: Checking dependencies..."
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

# Step 3: Run the thrashing test
echo "Step 3: Starting memory thrashing test in background..."
echo ""

nohup bash "${SCRIPT_DIR}/thrashing_stress_test.sh" > "$BACKGROUND_LOG" 2>&1 &
STRESS_PID=$!

# Save PID
echo "$STRESS_PID" > "$PID_FILE"

echo "Memory thrashing test started successfully!"
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
echo "  tail -f ${LOG_DIR}/thrashing_test_*.log"
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
    echo "✓ Memory thrashing test is running (PID: $STRESS_PID)"
    echo ""
    echo "This test rapidly allocates and frees memory to stress"
    echo "the GPU memory allocator and test fragmentation handling."
else
    echo "✗ ERROR: Test failed to start. Check $BACKGROUND_LOG for details."
    exit 1
fi
