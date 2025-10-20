#!/bin/bash

##############################################################################
# PCIe Bandwidth Saturation Test Script
# 
# Tests PCIe bandwidth fairness and contention by:
# - Simultaneous H2D and D2H transfers on all MIG instances
# - Using pinned memory for maximum bandwidth
# - Mixing transfers with compute operations
# - Measuring bandwidth fairness across devices
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
TEST_DURATION=180  # 3 minutes in seconds
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/pcie_test_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/pcie_test_errors_${TIMESTAMP}.log"
PYTHON_SCRIPT="${SCRIPT_DIR}/pcie_stress.py"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize logs
echo "==================================" | tee -a "$MAIN_LOG"
echo "PCIe Bandwidth Test Started: $(date)" | tee -a "$MAIN_LOG"
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

# Get list of MIG devices
log_info "Detecting MIG devices..."
MIG_DEVICES=$(nvidia-smi -L | grep "MIG" | awk '{print $6}' | tr -d '()')

if [ -z "$MIG_DEVICES" ]; then
    log_error "No MIG devices found. Make sure MIG is enabled and partitions are created."
    exit 1
fi

MIG_COUNT=$(echo "$MIG_DEVICES" | wc -l)
log_info "Found $MIG_COUNT MIG device(s)"
echo "$MIG_DEVICES" | while read -r device; do
    log_info "  - $device"
done
echo "" | tee -a "$MAIN_LOG"

# Create the Python PCIe bandwidth test script
log_info "Creating Python PCIe bandwidth script..."
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

def pcie_bandwidth_test(device_uuid, duration, log_file):
    """
    PCIe bandwidth saturation test:
    - Continuous H2D and D2H transfers using pinned memory
    - Mix with compute to stress PCIe + GPU simultaneously
    - Measure actual bandwidth achieved
    """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        log_message(f"Starting PCIe bandwidth test on device: {device_uuid}", log_file)
        log_message(f"Test duration: {duration} seconds", log_file)
        
        if not torch.cuda.is_available():
            log_message(f"ERROR: CUDA not available for device {device_uuid}", log_file)
            return False
        
        device = torch.device('cuda:0')
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        log_message(f"Device: {device_name}", log_file)
        log_message(f"Total memory: {total_memory / (1024**3):.2f} GB", log_file)
        
        # Allocate pinned memory on host for maximum transfer speed
        # Use 40% of GPU memory for transfers
        transfer_size = int(total_memory * 0.4)
        num_elements = transfer_size // 4  # float32
        
        log_message(f"Transfer size: {transfer_size / (1024**3):.2f} GB", log_file)
        log_message("Allocating pinned host memory...", log_file)
        
        # Create pinned memory on host
        host_tensor = torch.randn(num_elements, dtype=torch.float32, pin_memory=True)
        
        # Allocate device memory
        device_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        
        # Additional tensors for compute mixing
        compute_tensor_a = torch.randn(2048, 2048, dtype=torch.float32, device=device)
        compute_tensor_b = torch.randn(2048, 2048, dtype=torch.float32, device=device)
        
        log_message("Starting PCIe bandwidth saturation...", log_file)
        
        start_time = time.time()
        iteration = 0
        h2d_bytes = 0
        d2h_bytes = 0
        compute_ops = 0
        
        while time.time() - start_time < duration:
            iteration += 1
            
            # Phase 1: Host-to-Device transfer
            device_tensor.copy_(host_tensor, non_blocking=True)
            h2d_bytes += transfer_size
            
            # Phase 2: Compute operation (mixed with transfer)
            if iteration % 2 == 0:
                result = torch.mm(compute_tensor_a, compute_tensor_b)
                compute_ops += 1
            
            # Phase 3: Device-to-Host transfer
            host_tensor.copy_(device_tensor, non_blocking=False)  # Blocking to pace iterations
            d2h_bytes += transfer_size
            
            # Progress logging
            if iteration % 20 == 0:
                elapsed = time.time() - start_time
                h2d_bandwidth = (h2d_bytes / elapsed) / (1024**3)  # GB/s
                d2h_bandwidth = (d2h_bytes / elapsed) / (1024**3)  # GB/s
                total_bandwidth = (h2d_bytes + d2h_bytes) / elapsed / (1024**3)
                
                log_message(
                    f"Iter {iteration} | {elapsed:.0f}s/{duration}s | "
                    f"H2D: {h2d_bandwidth:.2f} GB/s | D2H: {d2h_bandwidth:.2f} GB/s | "
                    f"Total: {total_bandwidth:.2f} GB/s | Compute ops: {compute_ops}",
                    log_file
                )
        
        # Final statistics
        elapsed = time.time() - start_time
        total_data = (h2d_bytes + d2h_bytes) / (1024**3)  # GB
        avg_bandwidth = total_data / elapsed  # GB/s
        h2d_bandwidth = (h2d_bytes / elapsed) / (1024**3)
        d2h_bandwidth = (d2h_bytes / elapsed) / (1024**3)
        
        log_message("=" * 50, log_file)
        log_message(f"PCIe bandwidth test completed!", log_file)
        log_message(f"Total iterations: {iteration}", log_file)
        log_message(f"Total data transferred: {total_data:.2f} GB", log_file)
        log_message(f"H2D bandwidth: {h2d_bandwidth:.2f} GB/s", log_file)
        log_message(f"D2H bandwidth: {d2h_bandwidth:.2f} GB/s", log_file)
        log_message(f"Combined bandwidth: {avg_bandwidth:.2f} GB/s", log_file)
        log_message(f"Compute operations: {compute_ops}", log_file)
        log_message(f"Duration: {elapsed:.0f} seconds", log_file)
        log_message("=" * 50, log_file)
        
        # Cleanup
        del host_tensor
        del device_tensor
        del compute_tensor_a
        del compute_tensor_b
        if 'result' in locals():
            del result
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}", log_file)
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", log_file)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 pcie_stress.py <device_uuid> <duration_seconds> <log_file>")
        sys.exit(1)
    
    device_uuid = sys.argv[1]
    duration = int(sys.argv[2])
    log_file = sys.argv[3]
    
    success = pcie_bandwidth_test(device_uuid, duration, log_file)
    sys.exit(0 if success else 1)
PYTHON_EOF

chmod +x "$PYTHON_SCRIPT"
log_info "Python PCIe bandwidth script created."
echo "" | tee -a "$MAIN_LOG"

# Run PCIe test on ALL MIG devices SIMULTANEOUSLY
log_info "========================================="
log_info "SIMULTANEOUS PCIE BANDWIDTH TEST"
log_info "Starting on ALL $MIG_COUNT MIG devices in parallel"
log_info "========================================="
echo "" | tee -a "$MAIN_LOG"

# Ensure Python is available
PYTHON_CMD=$(which python3)
if [ -z "$PYTHON_CMD" ]; then
    log_error "python3 command not found in PATH"
    exit 1
fi
log_info "Using Python: $PYTHON_CMD"

# Array to store background process PIDs
declare -a WORKER_PIDS=()
device_num=0

# Launch PCIe test on each device in the background
while IFS= read -r device_uuid; do
    device_num=$((device_num + 1))
    
    DEVICE_LOG="${LOG_DIR}/pcie_device_${device_num}_${TIMESTAMP}.log"
    
    log_info "Launching worker for MIG Device $device_num/$MIG_COUNT (UUID: $device_uuid)"
    
    # Start the test in background with explicit python path
    "$PYTHON_CMD" "$PYTHON_SCRIPT" "$device_uuid" "$TEST_DURATION" "$DEVICE_LOG" >> "$DEVICE_LOG" 2>&1 &
    worker_pid=$!
    WORKER_PIDS+=($worker_pid)
    
    log_info "  -> Worker PID: $worker_pid"
    log_info "  -> Device log: $DEVICE_LOG"
done < <(echo "$MIG_DEVICES")

echo "" | tee -a "$MAIN_LOG"
log_info "All PCIe workers launched! Now monitoring..."
log_info "Test duration: $TEST_DURATION seconds"
log_info "Worker PIDs: ${WORKER_PIDS[*]}"
echo "" | tee -a "$MAIN_LOG"

# Wait for all background processes to complete
log_info "Waiting for all workers to complete..."
FAILED_COUNT=0
SUCCESS_COUNT=0

for worker_pid in "${WORKER_PIDS[@]}"; do
    if wait $worker_pid; then
        log_info "Worker PID $worker_pid completed successfully"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        exit_code=$?
        log_error "Worker PID $worker_pid FAILED (exit code: $exit_code)"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo "" | tee -a "$MAIN_LOG"

# Consolidate all device logs into main log
log_info "Consolidating device logs..."
for device_log in "${LOG_DIR}"/pcie_device_*_${TIMESTAMP}.log; do
    if [ -f "$device_log" ]; then
        echo "" >> "$MAIN_LOG"
        echo "=========================================" >> "$MAIN_LOG"
        echo "Device Log: $(basename $device_log)" >> "$MAIN_LOG"
        echo "=========================================" >> "$MAIN_LOG"
        cat "$device_log" >> "$MAIN_LOG"
    fi
done

echo "" | tee -a "$MAIN_LOG"

# Check for errors in dmesg
if dmesg | tail -100 | grep -i "gpu\|nvidia\|cuda\|pcie" | grep -i "error\|fail\|crash" >> "$ERROR_LOG" 2>&1; then
    log_warning "GPU/PCIe-related errors found in system logs."
fi

# Final summary
log_info "========================================="
log_info "PCIE BANDWIDTH TEST COMPLETED"
log_info "========================================="
log_info "Total MIG devices tested: $MIG_COUNT"
log_info "Successful workers: $SUCCESS_COUNT"
log_info "Failed workers: $FAILED_COUNT"
log_info "Test duration: $TEST_DURATION seconds"
log_info "Main log: $MAIN_LOG"
log_info "Error log: $ERROR_LOG"
log_info "Device logs: ${LOG_DIR}/pcie_device_*_${TIMESTAMP}.log"

if [ $FAILED_COUNT -gt 0 ] || [ -s "$ERROR_LOG" ]; then
    log_warning "Some tests failed or abnormalities detected. Check logs for details."
    exit 1
else
    log_info "All PCIe bandwidth tests passed successfully!"
    exit 0
fi
