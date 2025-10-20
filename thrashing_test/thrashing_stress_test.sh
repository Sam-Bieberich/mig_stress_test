#!/bin/bash

##############################################################################
# Memory Thrashing Stress Test Script
# 
# This script performs aggressive memory allocation/deallocation cycles
# to stress test the GPU memory allocator and fragmentation handling.
# Tests each MIG slice with rapid alloc/free patterns for 3 minutes.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
TEST_DURATION=1800  # 3 minutes in seconds
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/thrashing_test_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/thrashing_test_errors_${TIMESTAMP}.log"
PYTHON_SCRIPT="${SCRIPT_DIR}/thrashing_stress.py"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize logs
echo "==================================" | tee -a "$MAIN_LOG"
echo "Memory Thrashing Test Started: $(date)" | tee -a "$MAIN_LOG"
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

# Create the Python thrashing test script
log_info "Creating Python memory thrashing script..."
cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
import torch
import sys
import time
import os
import random
from datetime import datetime

def log_message(msg, log_file):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"[PYTHON] {timestamp} - {msg}"
    print(message, flush=True)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def memory_thrashing_test(device_uuid, duration, log_file):
    """
    Performs aggressive memory allocation/deallocation cycles:
    - Rapid alloc/free cycles
    - Variable chunk sizes
    - Random allocation patterns
    - Fragmentation stress
    """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        log_message(f"Starting memory thrashing test on device: {device_uuid}", log_file)
        log_message(f"Test duration: {duration} seconds", log_file)
        
        if not torch.cuda.is_available():
            log_message(f"ERROR: CUDA not available for device {device_uuid}", log_file)
            return False
        
        device = torch.device('cuda:0')
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        log_message(f"Device: {device_name}", log_file)
        log_message(f"Total memory: {total_memory / (1024**3):.2f} GB", log_file)
        
        start_time = time.time()
        cycle_count = 0
        allocation_count = 0
        deallocation_count = 0
        oom_count = 0
        
        # Define allocation patterns - MORE INTENSIVE
        chunk_sizes = [
            10 * 1024 * 1024,    # 10 MB
            50 * 1024 * 1024,    # 50 MB
            100 * 1024 * 1024,   # 100 MB
            200 * 1024 * 1024,   # 200 MB
            500 * 1024 * 1024,   # 500 MB
            750 * 1024 * 1024,   # 750 MB
            1024 * 1024 * 1024,  # 1 GB
        ]
        
        log_message("Starting aggressive memory thrashing cycles...", log_file)
        
        while time.time() - start_time < duration:
            try:
                cycle_count += 1
                tensors = []
                
                # Phase 1: Rapid allocation with varying sizes - MORE INTENSIVE
                num_allocs = random.randint(10, 40)
                for i in range(num_allocs):
                    try:
                        chunk_size = random.choice(chunk_sizes)
                        num_elements = chunk_size // 4
                        tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
                        tensors.append(tensor)
                        allocation_count += 1
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            oom_count += 1
                            torch.cuda.empty_cache()
                            break
                        else:
                            raise
                
                torch.cuda.synchronize()
                
                # Phase 2: Aggressive compute operations on tensors - MORE INTENSIVE
                if tensors:
                    # More operations on more tensors
                    num_ops = min(10, len(tensors))
                    for tensor in random.sample(tensors, num_ops):
                        # Mix of operations to stress compute + memory
                        tensor.mul_(1.01)
                        tensor.add_(0.01)
                        tensor.pow_(2)
                        tensor.sqrt_()
                        # Add matrix operations for compute stress
                        if tensor.numel() > 10000:
                            size = int(tensor.numel() ** 0.5)
                            if size > 100:
                                try:
                                    reshaped = tensor[:size*size].view(size, size)
                                    torch.mm(reshaped, reshaped.T)
                                except:
                                    pass
                    torch.cuda.synchronize()
                
                # Phase 3: Aggressive deallocation pattern - MORE INTENSIVE
                # Delete MORE tensors in random order to create severe fragmentation
                indices_to_delete = random.sample(range(len(tensors)), 
                                                 int(len(tensors) * 0.7) if tensors else 0)
                for idx in sorted(indices_to_delete, reverse=True):
                    del tensors[idx]
                    deallocation_count += 1
                
                # Phase 4: Partial cleanup and LESS FREQUENT cache clearing - MORE INTENSIVE
                del tensors
                # Clear cache less frequently to increase fragmentation pressure
                if cycle_count % 20 == 0:
                    torch.cuda.empty_cache()
                
                # Progress logging
                if cycle_count % 100 == 0:
                    elapsed = time.time() - start_time
                    memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    log_message(
                        f"Cycle {cycle_count} | {elapsed:.0f}s/{duration}s | "
                        f"Allocs: {allocation_count} | Deallocs: {deallocation_count} | "
                        f"OOMs: {oom_count} | Mem: {memory_allocated:.2f}GB | "
                        f"Reserved: {memory_reserved:.2f}GB", 
                        log_file
                    )
                
                # MINIMAL delay for maximum intensity
                time.sleep(0.001)
                
            except RuntimeError as e:
                log_message(f"ERROR in thrashing cycle {cycle_count}: {str(e)}", log_file)
                torch.cuda.empty_cache()
                if "out of memory" not in str(e):
                    return False
        
        # Final statistics
        elapsed = time.time() - start_time
        log_message("=" * 50, log_file)
        log_message(f"Memory thrashing test completed!", log_file)
        log_message(f"Total cycles: {cycle_count}", log_file)
        log_message(f"Total allocations: {allocation_count}", log_file)
        log_message(f"Total deallocations: {deallocation_count}", log_file)
        log_message(f"OOM events (expected): {oom_count}", log_file)
        log_message(f"Duration: {elapsed:.0f} seconds", log_file)
        log_message(f"Cycles per second: {cycle_count / elapsed:.2f}", log_file)
        log_message("=" * 50, log_file)
        
        # Final cleanup
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}", log_file)
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", log_file)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 thrashing_stress.py <device_uuid> <duration_seconds> <log_file>")
        sys.exit(1)
    
    device_uuid = sys.argv[1]
    duration = int(sys.argv[2])
    log_file = sys.argv[3]
    
    success = memory_thrashing_test(device_uuid, duration, log_file)
    sys.exit(0 if success else 1)
PYTHON_EOF

chmod +x "$PYTHON_SCRIPT"
log_info "Python thrashing script created."
echo "" | tee -a "$MAIN_LOG"

# Run thrashing test on ALL MIG devices SIMULTANEOUSLY
log_info "========================================="
log_info "SIMULTANEOUS THRASHING TEST"
log_info "Starting thrashing on ALL $MIG_COUNT MIG devices in parallel"
log_info "========================================="
echo "" | tee -a "$MAIN_LOG"

# Array to store background process PIDs
declare -a WORKER_PIDS=()
device_num=0

# Launch thrashing test on each device in the background
echo "$MIG_DEVICES" | while read -r device_uuid; do
    device_num=$((device_num + 1))
    
    DEVICE_LOG="${LOG_DIR}/thrashing_device_${device_num}_${TIMESTAMP}.log"
    
    log_info "Launching thrashing worker for MIG Device $device_num/$MIG_COUNT (UUID: $device_uuid)"
    
    # Start the thrashing test in background
    python3 "$PYTHON_SCRIPT" "$device_uuid" "$TEST_DURATION" "$DEVICE_LOG" 2>&1 &
    worker_pid=$!
    echo "$worker_pid" >> "${LOG_DIR}/worker_pids_${TIMESTAMP}.tmp"
    
    log_info "  -> Worker PID: $worker_pid"
    log_info "  -> Device log: $DEVICE_LOG"
done

echo "" | tee -a "$MAIN_LOG"
log_info "All thrashing workers launched! Now monitoring..."
log_info "Test duration: $TEST_DURATION seconds"
echo "" | tee -a "$MAIN_LOG"

# Wait for all background processes to complete
log_info "Waiting for all thrashing workers to complete..."
FAILED_COUNT=0
SUCCESS_COUNT=0

if [ -f "${LOG_DIR}/worker_pids_${TIMESTAMP}.tmp" ]; then
    while read -r worker_pid; do
        if wait $worker_pid; then
            log_info "Worker PID $worker_pid completed successfully"
            SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
        else
            exit_code=$?
            log_error "Worker PID $worker_pid FAILED (exit code: $exit_code)"
            FAILED_COUNT=$((FAILED_COUNT + 1))
        fi
    done < "${LOG_DIR}/worker_pids_${TIMESTAMP}.tmp"
    
    # Clean up temp file
    rm -f "${LOG_DIR}/worker_pids_${TIMESTAMP}.tmp"
fi

echo "" | tee -a "$MAIN_LOG"

# Consolidate all device logs into main log
log_info "Consolidating device logs..."
for device_log in "${LOG_DIR}"/thrashing_device_*_${TIMESTAMP}.log; do
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
log_info "SIMULTANEOUS THRASHING TEST COMPLETED"
log_info "========================================="
log_info "Total MIG devices tested: $MIG_COUNT"
log_info "Successful workers: $SUCCESS_COUNT"
log_info "Failed workers: $FAILED_COUNT"
log_info "Test duration: $TEST_DURATION seconds"
log_info "Main log: $MAIN_LOG"
log_info "Error log: $ERROR_LOG"
log_info "Device logs: ${LOG_DIR}/thrashing_device_*_${TIMESTAMP}.log"

if [ $FAILED_COUNT -gt 0 ] || [ -s "$ERROR_LOG" ]; then
    log_warning "Some tests failed or abnormalities detected. Check logs for details."
    exit 1
else
    log_info "All thrashing tests passed successfully!"
    exit 0
fi
