#!/bin/bash

##############################################################################
# Master Sequential Test Runner
# 
# This script runs all MIG stress tests one after another with:
# - Automatic MIG setup
# - Module loading for PyTorch
# - 32-minute pause between tests (30-min test + 2-min buffer)
# - Failure handling and flagging
# - Comprehensive master logging
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MASTER_LOG_DIR="${SCRIPT_DIR}/master_logs"
MASTER_LOG="${MASTER_LOG_DIR}/sequential_run_${TIMESTAMP}.log"
FAILED_TESTS_LOG="${MASTER_LOG_DIR}/failed_tests_${TIMESTAMP}.log"

# Configuration
TEST_DURATION=1800  # 30 minutes in seconds
PAUSE_BETWEEN_TESTS=1920  # 32 minutes in seconds (30-min test + 2-min buffer)

# Create master log directory
mkdir -p "$MASTER_LOG_DIR"

# Initialize master log
cat > "$MASTER_LOG" << 'LOG_HEADER'
================================================================================
                    MIG STRESS TEST SUITE - SEQUENTIAL RUN
================================================================================
LOG_HEADER

echo "Start Time: $(date)" >> "$MASTER_LOG"
echo "Script Directory: $SCRIPT_DIR" >> "$MASTER_LOG"
echo "Test Duration: $TEST_DURATION seconds (30 minutes)" >> "$MASTER_LOG"
echo "Pause Between Tests: $PAUSE_BETWEEN_TESTS seconds (32 minutes)" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"
echo "================================================================================\n" >> "$MASTER_LOG"

# Initialize failed tests log
touch "$FAILED_TESTS_LOG"

# Function to log messages
log_master() {
    local level="$1"
    local message="$2"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$level] $timestamp - $message" | tee -a "$MASTER_LOG"
}

log_section() {
    local message="$1"
    echo "" | tee -a "$MASTER_LOG"
    echo "========================================" | tee -a "$MASTER_LOG"
    echo "$message" | tee -a "$MASTER_LOG"
    echo "========================================" | tee -a "$MASTER_LOG"
}

# Track test statistics
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
SKIPPED_TESTS=0
declare -a FAILED_TEST_NAMES=()

# Step 1: Check/Setup MIG
log_section "STEP 1: MIG SETUP"

# First check if MIG devices already exist
MIG_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c "MIG" || echo "0")

if [ "$MIG_COUNT" -gt 0 ]; then
    log_master "INFO" "MIG already configured with $MIG_COUNT devices - skipping setup"
    nvidia-smi -L | grep "MIG" >> "$MASTER_LOG"
else
    log_master "INFO" "No MIG devices found - attempting MIG setup..."
    
    if [ -f "${SCRIPT_DIR}/mig_easy_setup.sh" ]; then
        bash "${SCRIPT_DIR}/mig_easy_setup.sh" >> "$MASTER_LOG" 2>&1
        if [ $? -eq 0 ]; then
            log_master "SUCCESS" "MIG setup completed successfully"
        else
            log_master "ERROR" "MIG setup failed, but continuing with tests..."
        fi
    else
        log_master "WARNING" "mig_easy_setup.sh not found, skipping MIG setup"
    fi
    
    # Re-check MIG devices after setup
    MIG_COUNT=$(nvidia-smi -L 2>/dev/null | grep -c "MIG" || echo "0")
fi

log_master "INFO" "Detected $MIG_COUNT MIG devices"
nvidia-smi -L | grep "MIG" >> "$MASTER_LOG"
echo "" >> "$MASTER_LOG"

# Step 2: Load Required Modules
log_section "STEP 2: MODULE LOADING"
log_master "INFO" "Loading required modules for PyTorch..."

# Load GCC and CUDA
if command -v module &> /dev/null; then
    log_master "INFO" "Loading gcc module..."
    module load gcc >> "$MASTER_LOG" 2>&1
    
    log_master "INFO" "Loading cuda module..."
    module load cuda >> "$MASTER_LOG" 2>&1
    
    log_master "INFO" "Loading python3 module..."
    module load python3 >> "$MASTER_LOG" 2>&1
    
    log_master "SUCCESS" "Modules loaded successfully"
    
    # Show loaded modules
    log_master "INFO" "Currently loaded modules:"
    module list 2>&1 | tee -a "$MASTER_LOG"
else
    log_master "WARNING" "Module command not found - skipping module loading"
fi

# Verify Python and PyTorch
log_master "INFO" "Verifying Python installation..."
which python3 >> "$MASTER_LOG" 2>&1
python3 --version >> "$MASTER_LOG" 2>&1

log_master "INFO" "Checking PyTorch availability..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" >> "$MASTER_LOG" 2>&1 || log_master "WARNING" "PyTorch check failed - tests will attempt installation"

echo "" >> "$MASTER_LOG"

# Step 3: Discover Test Directories
log_section "STEP 3: TEST DISCOVERY"
log_master "INFO" "Discovering test directories..."

# Find all directories ending with '_test' in alphabetical order
TEST_DIRS=()
while IFS= read -r dir; do
    TEST_DIRS+=("$dir")
done < <(find "$SCRIPT_DIR" -maxdepth 1 -type d -name '*_test' | sort)

if [ ${#TEST_DIRS[@]} -eq 0 ]; then
    log_master "ERROR" "No test directories found!"
    exit 1
fi

log_master "INFO" "Found ${#TEST_DIRS[@]} test directories:"
for test_dir in "${TEST_DIRS[@]}"; do
    test_name=$(basename "$test_dir")
    log_master "INFO" "  - $test_name"
done

echo "" >> "$MASTER_LOG"

# Step 4: Run Tests Sequentially
log_section "STEP 4: SEQUENTIAL TEST EXECUTION"
OVERALL_START_TIME=$(date +%s)

for test_dir in "${TEST_DIRS[@]}"; do
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    test_name=$(basename "$test_dir")
    run_script="${test_dir}/run_test.sh"
    
    log_section "TEST $TOTAL_TESTS: $test_name"
    
    # Check if run_test.sh exists
    if [ ! -f "$run_script" ]; then
        log_master "ERROR" "run_test.sh not found in $test_name - SKIPPING"
        SKIPPED_TESTS=$((SKIPPED_TESTS + 1))
        continue
    fi
    
    # Make script executable
    chmod +x "$run_script"
    
    # Run the test
    TEST_START_TIME=$(date +%s)
    TEST_START_DATE=$(date '+%Y-%m-%d %H:%M:%S')
    log_master "INFO" "Starting $test_name at $TEST_START_DATE"
    log_master "INFO" "Expected duration: $TEST_DURATION seconds (30 minutes)"
    log_master "INFO" "Running: bash $run_script"
    
    # Execute test and capture exit code
    cd "$test_dir" || {
        log_master "ERROR" "Failed to change directory to $test_dir"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILED_TEST_NAMES+=("$test_name")
        echo "$test_name - Failed to change directory" >> "$FAILED_TESTS_LOG"
        continue
    }
    
    bash run_test.sh >> "$MASTER_LOG" 2>&1
    TEST_EXIT_CODE=$?
    
    cd "$SCRIPT_DIR" || exit 1
    
    TEST_END_TIME=$(date +%s)
    TEST_END_DATE=$(date '+%Y-%m-%d %H:%M:%S')
    TEST_ELAPSED=$((TEST_END_TIME - TEST_START_TIME))
    TEST_ELAPSED_MIN=$((TEST_ELAPSED / 60))
    
    # Check test result
    if [ $TEST_EXIT_CODE -eq 0 ]; then
        log_master "SUCCESS" "$test_name completed successfully"
        log_master "INFO" "End time: $TEST_END_DATE"
        log_master "INFO" "Duration: ${TEST_ELAPSED_MIN} minutes ($TEST_ELAPSED seconds)"
        PASSED_TESTS=$((PASSED_TESTS + 1))
    else
        log_master "ERROR" "$test_name FAILED with exit code $TEST_EXIT_CODE"
        log_master "INFO" "End time: $TEST_END_DATE"
        log_master "INFO" "Duration: ${TEST_ELAPSED_MIN} minutes ($TEST_ELAPSED seconds)"
        log_master "WARNING" "Flagging $test_name as failed and continuing..."
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILED_TEST_NAMES+=("$test_name")
        echo "$test_name - Exit code $TEST_EXIT_CODE at $TEST_END_DATE" >> "$FAILED_TESTS_LOG"
    fi
    
    # Pause between tests (unless this is the last test)
    if [ $TOTAL_TESTS -lt ${#TEST_DIRS[@]} ]; then
        PAUSE_MIN=$((PAUSE_BETWEEN_TESTS / 60))
        log_master "INFO" "Pausing for ${PAUSE_MIN} minutes ($PAUSE_BETWEEN_TESTS seconds) before next test..."
        
        PAUSE_START=$(date +%s)
        PAUSE_END=$((PAUSE_START + PAUSE_BETWEEN_TESTS))
        
        while [ $(date +%s) -lt $PAUSE_END ]; do
            REMAINING=$((PAUSE_END - $(date +%s)))
            REMAINING_MIN=$((REMAINING / 60))
            
            # Log every 5 minutes during pause
            if [ $((REMAINING % 300)) -eq 0 ] && [ $REMAINING -gt 0 ]; then
                log_master "INFO" "Pause in progress... ${REMAINING_MIN} minutes remaining"
            fi
            
            sleep 30
        done
        
        log_master "INFO" "Pause complete. Proceeding to next test."
    fi
    
    echo "" >> "$MASTER_LOG"
done

# Step 5: Final Summary
OVERALL_END_TIME=$(date +%s)
OVERALL_ELAPSED=$((OVERALL_END_TIME - OVERALL_START_TIME))
OVERALL_HOURS=$((OVERALL_ELAPSED / 3600))
OVERALL_MINUTES=$(((OVERALL_ELAPSED % 3600) / 60))

log_section "FINAL SUMMARY"
log_master "INFO" "All tests completed!"
log_master "INFO" "End time: $(date '+%Y-%m-%d %H:%M:%S')"
log_master "INFO" "Total runtime: ${OVERALL_HOURS}h ${OVERALL_MINUTES}m (${OVERALL_ELAPSED} seconds)"
echo "" >> "$MASTER_LOG"

log_master "INFO" "Test Statistics:"
log_master "INFO" "  Total tests discovered: $TOTAL_TESTS"
log_master "INFO" "  Passed: $PASSED_TESTS"
log_master "INFO" "  Failed: $FAILED_TESTS"
log_master "INFO" "  Skipped: $SKIPPED_TESTS"
echo "" >> "$MASTER_LOG"

if [ $FAILED_TESTS -gt 0 ]; then
    log_master "WARNING" "Failed tests:"
    for failed_test in "${FAILED_TEST_NAMES[@]}"; do
        log_master "WARNING" "  - $failed_test"
    done
    echo "" >> "$MASTER_LOG"
    log_master "INFO" "Failed test details saved to: $FAILED_TESTS_LOG"
fi

log_master "INFO" "Master log saved to: $MASTER_LOG"

echo "" >> "$MASTER_LOG"
echo "================================================================================" >> "$MASTER_LOG"
echo "                    SEQUENTIAL TEST RUN COMPLETED" >> "$MASTER_LOG"
echo "================================================================================" >> "$MASTER_LOG"

# Exit with error if any tests failed
if [ $FAILED_TESTS -gt 0 ]; then
    log_master "WARNING" "Exiting with error code due to failed tests"
    exit 1
else
    log_master "SUCCESS" "All tests passed successfully!"
    exit 0
fi
