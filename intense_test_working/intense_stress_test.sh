#!/bin/bash

##############################################################################
# Intense MIG Stress Test Script
# 
# This script performs intensive stress testing on NVIDIA MIG slices by:
# - Testing one MIG slice at 95% memory (primary target)
# - Running all other MIG slices at 75% memory simultaneously (background load)
# - Rotating through each slice as the primary target
# - Running for 30 minutes per primary target
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
TEST_DURATION=1800  # 30 minutes in seconds
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/intense_stress_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/intense_stress_errors_${TIMESTAMP}.log"

# Use the main intense_stress.sh script from the parent directory
# This avoids code duplication
MAIN_INTENSE_SCRIPT="${SCRIPT_DIR}/../../intense_stress.sh"

if [ -f "$MAIN_INTENSE_SCRIPT" ]; then
    echo "Using main intense stress script from parent directory..."
    # Update log paths in the parent script and run it
    cd "${SCRIPT_DIR}"
    export LOG_DIR="${LOG_DIR}"
    bash "$MAIN_INTENSE_SCRIPT"
else
    echo "ERROR: Main intense stress script not found at $MAIN_INTENSE_SCRIPT"
    echo "Creating standalone version..."
    
    # If the main script doesn't exist, we'll create a minimal version here
    # For now, let's just reference that this test combines all MIG devices
    echo "Intense stress test requires multiple device coordination."
    echo "Please ensure the parent intense_stress.sh script is available."
    exit 1
fi
