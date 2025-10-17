#!/bin/bash

##############################################################################
# Standard MIG Stress Test Script
# 
# This script stress tests NVIDIA MIG slices one at a time by allocating
# and using all available memory for 30 minutes per slice.
# Logs all activities and captures any abnormalities (memory issues, crashes).
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
TEST_DURATION=1800  # 30 minutes in seconds
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/stress_test_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/stress_test_errors_${TIMESTAMP}.log"
PYTHON_SCRIPT="${SCRIPT_DIR}/standard_stress.py"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize logs
echo "==================================" | tee -a "$MAIN_LOG"
echo "Standard MIG Stress Test Started: $(date)" | tee -a "$MAIN_LOG"
echo "==================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Function to log messages
log_info() {
    echo "[INFO] $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG"
}

log_error() {
    echo "[ERROR] $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG" | tee -a "$ERROR_LOG"
}

log_warning() {
    echo "[WARNING] $(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$MAIN_LOG" | tee -a "$ERROR_LOG"
}

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Exiting."
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    log_error "python3 not found. Exiting."
    exit 1
fi

# Get list of MIG devices
log_info "Detecting MIG devices..."
MIG_DEVICES=$(nvidia-smi -L | grep "MIG" | awk '{print $6}' | tr -d '()')

if [ -z "$MIG_DEVICES" ]; then
    log_error "No MIG devices found. Make sure MIG is enabled and partitions are created."
    exit 1
fi

# Count MIG devices
MIG_COUNT=$(echo "$MIG_DEVICES" | wc -l)
log_info "Found $MIG_COUNT MIG device(s)"
echo "$MIG_DEVICES" | while read -r device; do
    log_info "  - $device"
done
echo "" | tee -a "$MAIN_LOG"

# Create the Python stress test script
log_info "Creating Python stress test script..."
cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
import torch
import sys
import time
import os
from datetime import datetime

def log_message(msg, log_file):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"[PYTHON] {timestamp} - {msg}"
    print(message, flush=True)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def stress_test_mig_device(device_uuid, duration, log_file):
    try:
        # Set CUDA device
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        log_message(f"Starting stress test on device: {device_uuid}", log_file)
        log_message(f"Test duration: {duration} seconds", log_file)
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            log_message(f"ERROR: CUDA not available for device {device_uuid}", log_file)
            return False
        
        device = torch.device('cuda:0')
        
        # Get device properties
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        log_message(f"Device name: {device_name}", log_file)
        log_message(f"Total memory: {total_memory / (1024**3):.2f} GB", log_file)
        
        # Calculate memory to allocate (95% to leave some headroom)
        target_memory = int(total_memory * 0.95)
        log_message(f"Target memory allocation: {target_memory / (1024**3):.2f} GB", log_file)
        
        # Allocate memory in chunks
        tensors = []
        chunk_size = 100 * 1024 * 1024  # 100 MB chunks
        allocated = 0
        
        log_message("Allocating memory...", log_file)
        while allocated < target_memory:
            try:
                remaining = target_memory - allocated
                current_chunk = min(chunk_size, remaining)
                num_elements = current_chunk // 4  # 4 bytes per float32
                
                tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
                tensors.append(tensor)
                allocated += current_chunk
                
                if len(tensors) % 10 == 0:  # Log every 1GB
                    log_message(f"Allocated: {allocated / (1024**3):.2f} GB", log_file)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log_message(f"Reached memory limit at {allocated / (1024**3):.2f} GB", log_file)
                    break
                else:
                    raise
        
        torch.cuda.synchronize()
        actual_allocated = torch.cuda.memory_allocated(0)
        log_message(f"Final allocated memory: {actual_allocated / (1024**3):.2f} GB", log_file)
        
        # Stress test: perform operations on the memory
        log_message("Starting memory stress operations...", log_file)
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            try:
                # Perform operations on tensors
                for i, tensor in enumerate(tensors):
                    # Simple arithmetic operations
                    tensor.mul_(1.0001)
                    tensor.add_(0.0001)
                    
                    if i % 100 == 0:  # Sync periodically
                        torch.cuda.synchronize()
                
                iteration += 1
                elapsed = time.time() - start_time
                
                # Log progress every 5 minutes
                if iteration % 10 == 0:
                    remaining = duration - elapsed
                    memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    log_message(f"Progress: {elapsed:.0f}s / {duration}s | Memory: {memory_used:.2f} GB | Iteration: {iteration}", log_file)
                
                # Small delay to prevent spinning too fast
                time.sleep(0.1)
                
            except RuntimeError as e:
                log_message(f"ERROR during stress test: {str(e)}", log_file)
                return False
        
        log_message(f"Stress test completed successfully. Total iterations: {iteration}", log_file)
        
        # Cleanup
        log_message("Cleaning up memory...", log_file)
        del tensors
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}", log_file)
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", log_file)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 standard_stress.py <device_uuid> <duration_seconds> <log_file>")
        sys.exit(1)
    
    device_uuid = sys.argv[1]
    duration = int(sys.argv[2])
    log_file = sys.argv[3]
    
    success = stress_test_mig_device(device_uuid, duration, log_file)
    sys.exit(0 if success else 1)
PYTHON_EOF

chmod +x "$PYTHON_SCRIPT"
log_info "Python stress test script created."
echo "" | tee -a "$MAIN_LOG"

# Run stress test on each MIG device
device_num=0
echo "$MIG_DEVICES" | while read -r device_uuid; do
    device_num=$((device_num + 1))
    
    log_info "========================================="
    log_info "Testing MIG Device $device_num of $MIG_COUNT"
    log_info "Device UUID: $device_uuid"
    log_info "========================================="
    
    # Get initial device status
    log_info "Initial device status:"
    nvidia-smi --id="$device_uuid" --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader >> "$MAIN_LOG" 2>&1
    
    # Run the stress test
    log_info "Starting $TEST_DURATION second stress test..."
    
    if python3 "$PYTHON_SCRIPT" "$device_uuid" "$TEST_DURATION" "$MAIN_LOG" 2>&1 | tee -a "$MAIN_LOG"; then
        log_info "Stress test completed successfully for device $device_uuid"
    else
        exit_code=$?
        log_error "Stress test FAILED for device $device_uuid (exit code: $exit_code)"
    fi
    
    # Get final device status
    log_info "Final device status:"
    nvidia-smi --id="$device_uuid" --query-gpu=name,memory.total,memory.used,memory.free --format=csv,noheader >> "$MAIN_LOG" 2>&1
    
    # Check for any errors in dmesg
    log_info "Checking system logs for GPU-related errors..."
    if dmesg | tail -100 | grep -i "gpu\|nvidia\|cuda" | grep -i "error\|fail\|crash" >> "$ERROR_LOG" 2>&1; then
        log_warning "GPU-related errors found in system logs. See $ERROR_LOG for details."
    fi
    
    echo "" | tee -a "$MAIN_LOG"
    
    # Small pause between devices
    if [ $device_num -lt $MIG_COUNT ]; then
        log_info "Pausing 10 seconds before next device..."
        sleep 10
    fi
done

# Final summary
log_info "========================================="
log_info "All MIG device stress tests completed!"
log_info "========================================="
log_info "Main log: $MAIN_LOG"
log_info "Error log: $ERROR_LOG"
log_info "Test completed: $(date)"

# Check if there were any errors
if [ -s "$ERROR_LOG" ]; then
    log_warning "Abnormalities detected during testing. Check $ERROR_LOG for details."
    exit 1
else
    log_info "No abnormalities detected. All tests passed successfully!"
    exit 0
fi
