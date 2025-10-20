#!/bin/bash

##############################################################################
# Multi-Process Mayhem Test Script
# 
# Tests multi-process handling on MIG instances by:
# - Launching 3-5 Python processes per MIG device
# - Each process does different operations (compute, memory, transfers)
# - Tests resource arbitration and multi-process isolation
# - All MIG devices tested simultaneously
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
TEST_DURATION=1800  # 3 minutes in seconds
PROCESSES_PER_MIG=4  # Number of processes per MIG device
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/multiproc_test_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/multiproc_test_errors_${TIMESTAMP}.log"
PYTHON_SCRIPT="${SCRIPT_DIR}/multiproc_worker.py"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize logs
echo "==================================" | tee -a "$MAIN_LOG"
echo "Multi-Process Test Started: $(date)" | tee -a "$MAIN_LOG"
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

# Create the Python worker script
log_info "Creating Python multi-process worker script..."
cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
import torch
import sys
import time
import os
import random
from datetime import datetime

def log_message(msg, log_file):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"[WORKER-{os.getpid()}] {timestamp} - {msg}"
    print(message, flush=True)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def worker_compute(device_uuid, duration, worker_id, log_file):
    """Worker that does heavy compute operations"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        log_message(f"Compute worker {worker_id} started", log_file)
        
        device = torch.device('cuda:0')
        
        # Allocate matrices for computation
        size = 1024
        a = torch.randn(size, size, dtype=torch.float32, device=device)
        b = torch.randn(size, size, dtype=torch.float32, device=device)
        
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < duration:
            # Matrix multiplication
            c = torch.mm(a, b)
            a = c
            iterations += 1
            
            if iterations % 500 == 0:
                elapsed = time.time() - start_time
                log_message(f"Compute worker {worker_id}: {iterations} iterations, {elapsed:.0f}s", log_file)
        
        log_message(f"Compute worker {worker_id} completed: {iterations} iterations", log_file)
        return True
    except Exception as e:
        log_message(f"Compute worker {worker_id} ERROR: {str(e)}", log_file)
        return False

def worker_memory(device_uuid, duration, worker_id, log_file):
    """Worker that does memory allocation/deallocation"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        log_message(f"Memory worker {worker_id} started", log_file)
        
        device = torch.device('cuda:0')
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        chunk_size = int(total_memory * 0.1)  # 10% chunks
        
        start_time = time.time()
        cycles = 0
        
        while time.time() - start_time < duration:
            tensors = []
            
            # Allocate multiple chunks
            for i in range(3):
                try:
                    num_elements = chunk_size // 4
                    t = torch.randn(num_elements, dtype=torch.float32, device=device)
                    tensors.append(t)
                except RuntimeError:
                    torch.cuda.empty_cache()
                    break
            
            # Do some work
            for t in tensors:
                t.mul_(1.01)
            
            torch.cuda.synchronize()
            
            # Free memory
            del tensors
            torch.cuda.empty_cache()
            
            cycles += 1
            
            if cycles % 100 == 0:
                elapsed = time.time() - start_time
                log_message(f"Memory worker {worker_id}: {cycles} cycles, {elapsed:.0f}s", log_file)
            
            time.sleep(0.01)
        
        log_message(f"Memory worker {worker_id} completed: {cycles} cycles", log_file)
        return True
    except Exception as e:
        log_message(f"Memory worker {worker_id} ERROR: {str(e)}", log_file)
        return False

def worker_transfer(device_uuid, duration, worker_id, log_file):
    """Worker that does host-device transfers"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        log_message(f"Transfer worker {worker_id} started", log_file)
        
        device = torch.device('cuda:0')
        
        # Allocate transfer buffers
        size = 50 * 1024 * 1024  # 50 MB
        num_elements = size // 4
        
        host_tensor = torch.randn(num_elements, dtype=torch.float32, pin_memory=True)
        device_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
        
        start_time = time.time()
        transfers = 0
        
        while time.time() - start_time < duration:
            # H2D transfer
            device_tensor.copy_(host_tensor)
            
            # D2H transfer
            host_tensor.copy_(device_tensor)
            
            transfers += 1
            
            if transfers % 200 == 0:
                elapsed = time.time() - start_time
                log_message(f"Transfer worker {worker_id}: {transfers} transfers, {elapsed:.0f}s", log_file)
            
            time.sleep(0.005)
        
        log_message(f"Transfer worker {worker_id} completed: {transfers} transfers", log_file)
        return True
    except Exception as e:
        log_message(f"Transfer worker {worker_id} ERROR: {str(e)}", log_file)
        return False

def worker_streams(device_uuid, duration, worker_id, log_file):
    """Worker that uses multiple CUDA streams"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        log_message(f"Streams worker {worker_id} started", log_file)
        
        device = torch.device('cuda:0')
        
        # Create multiple streams
        num_streams = 32
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        
        # Small tensors for stream operations
        tensors = [torch.randn(512, 512, dtype=torch.float32, device=device) for _ in range(num_streams)]
        
        start_time = time.time()
        iterations = 0
        
        while time.time() - start_time < duration:
            # Launch operations on different streams
            for i, (stream, tensor) in enumerate(zip(streams, tensors)):
                with torch.cuda.stream(stream):
                    tensor.mul_(1.001)
                    tensor.add_(0.001)
            
            # Synchronize all streams
            for stream in streams:
                stream.synchronize()
            
            iterations += 1
            
            if iterations % 200 == 0:
                elapsed = time.time() - start_time
                log_message(f"Streams worker {worker_id}: {iterations} iterations, {elapsed:.0f}s", log_file)
        
        log_message(f"Streams worker {worker_id} completed: {iterations} iterations", log_file)
        return True
    except Exception as e:
        log_message(f"Streams worker {worker_id} ERROR: {str(e)}", log_file)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 multiproc_worker.py <device_uuid> <duration> <worker_type> <log_file>")
        sys.exit(1)
    
    device_uuid = sys.argv[1]
    duration = int(sys.argv[2])
    worker_type = sys.argv[3]
    log_file = sys.argv[4]
    
    worker_id = os.getpid()
    
    if worker_type == "compute":
        success = worker_compute(device_uuid, duration, worker_id, log_file)
    elif worker_type == "memory":
        success = worker_memory(device_uuid, duration, worker_id, log_file)
    elif worker_type == "transfer":
        success = worker_transfer(device_uuid, duration, worker_id, log_file)
    elif worker_type == "streams":
        success = worker_streams(device_uuid, duration, worker_id, log_file)
    else:
        print(f"Unknown worker type: {worker_type}")
        sys.exit(1)
    
    sys.exit(0 if success else 1)
PYTHON_EOF

chmod +x "$PYTHON_SCRIPT"
log_info "Python multi-process worker script created."
echo "" | tee -a "$MAIN_LOG"

# Run multi-process test on ALL MIG devices SIMULTANEOUSLY
log_info "========================================="
log_info "SIMULTANEOUS MULTI-PROCESS TEST"
log_info "Launching $PROCESSES_PER_MIG processes per MIG device"
log_info "Testing $MIG_COUNT MIG devices in parallel"
log_info "Total processes: $((MIG_COUNT * PROCESSES_PER_MIG))"
log_info "========================================="
echo "" | tee -a "$MAIN_LOG"

# Ensure Python is available
PYTHON_CMD=$(which python3)
if [ -z "$PYTHON_CMD" ]; then
    log_error "python3 command not found in PATH"
    exit 1
fi
log_info "Using Python: $PYTHON_CMD"

# Array to store all worker PIDs
declare -a ALL_WORKER_PIDS=()

# Worker types to rotate through
WORKER_TYPES=("compute" "memory" "transfer" "streams")

device_num=0

# Launch multiple processes for each MIG device
while IFS= read -r device_uuid; do
    device_num=$((device_num + 1))
    
    DEVICE_LOG="${LOG_DIR}/multiproc_device_${device_num}_${TIMESTAMP}.log"
    
    log_info "========================================="
    log_info "MIG Device $device_num/$MIG_COUNT (UUID: $device_uuid)"
    log_info "Launching $PROCESSES_PER_MIG worker processes..."
    log_info "========================================="
    
    # Launch multiple workers for this device
    for proc_num in $(seq 1 $PROCESSES_PER_MIG); do
        # Rotate through worker types
        worker_type_idx=$(((proc_num - 1) % ${#WORKER_TYPES[@]}))
        worker_type="${WORKER_TYPES[$worker_type_idx]}"
        
        log_info "  Process $proc_num: $worker_type worker"
        
        # Launch worker in background
        "$PYTHON_CMD" "$PYTHON_SCRIPT" "$device_uuid" "$TEST_DURATION" "$worker_type" "$DEVICE_LOG" >> "$DEVICE_LOG" 2>&1 &
        worker_pid=$!
        ALL_WORKER_PIDS+=($worker_pid)
        
        log_info "    -> PID: $worker_pid"
    done
    
    echo "" | tee -a "$MAIN_LOG"
done < <(echo "$MIG_DEVICES")

TOTAL_WORKERS=${#ALL_WORKER_PIDS[@]}

log_info "========================================="
log_info "All $TOTAL_WORKERS workers launched!"
log_info "Test duration: $TEST_DURATION seconds"
log_info "========================================="
echo "" | tee -a "$MAIN_LOG"

# Wait for all workers to complete
log_info "Waiting for all workers to complete..."
FAILED_COUNT=0
SUCCESS_COUNT=0

for worker_pid in "${ALL_WORKER_PIDS[@]}"; do
    if wait $worker_pid 2>/dev/null; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        exit_code=$?
        if [ $exit_code -ne 0 ]; then
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    fi
done

echo "" | tee -a "$MAIN_LOG"

# Consolidate all device logs into main log
log_info "Consolidating device logs..."
for device_log in "${LOG_DIR}"/multiproc_device_*_${TIMESTAMP}.log; do
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
if dmesg | tail -100 | grep -i "gpu\|nvidia\|cuda" | grep -i "error\|fail\|crash" >> "$ERROR_LOG" 2>&1; then
    log_warning "GPU-related errors found in system logs."
fi

# Final summary
log_info "========================================="
log_info "MULTI-PROCESS TEST COMPLETED"
log_info "========================================="
log_info "Total MIG devices tested: $MIG_COUNT"
log_info "Processes per device: $PROCESSES_PER_MIG"
log_info "Total workers launched: $TOTAL_WORKERS"
log_info "Successful workers: $SUCCESS_COUNT"
log_info "Failed workers: $FAILED_COUNT"
log_info "Test duration: $TEST_DURATION seconds"
log_info "Main log: $MAIN_LOG"
log_info "Error log: $ERROR_LOG"
log_info "Device logs: ${LOG_DIR}/multiproc_device_*_${TIMESTAMP}.log"

if [ $FAILED_COUNT -gt 0 ] || [ -s "$ERROR_LOG" ]; then
    log_warning "Some workers failed or abnormalities detected. Check logs for details."
    exit 1
else
    log_info "All multi-process tests passed successfully!"
    exit 0
fi
