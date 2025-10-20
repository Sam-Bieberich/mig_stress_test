#!/bin/bash

##############################################################################
# Intense Memory Thrashing Stress Test Script
# 
# This test maintains a persistent base load (60-70% of memory) while
# aggressively thrashing the remaining memory (20-30%).
# 
# Strategy:
# 1. Allocate 60-70% of memory as persistent base load
# 2. Rapidly allocate/deallocate chunks in the remaining space
# 3. Creates fragmentation while maintaining high memory pressure
# 4. Tests allocator under realistic high-load conditions
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
TEST_DURATION=180  # 3 minutes in seconds
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/intense_thrashing_test_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/intense_thrashing_test_errors_${TIMESTAMP}.log"
PYTHON_SCRIPT="${SCRIPT_DIR}/intense_thrashing_stress.py"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize logs
echo "==================================" | tee -a "$MAIN_LOG"
echo "Intense Thrashing Test Started: $(date)" | tee -a "$MAIN_LOG"
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

# Create the Python intense thrashing test script
log_info "Creating Python intense thrashing script..."
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

def intense_thrashing_test(device_uuid, duration, log_file):
    """
    Intense thrashing with persistent base load:
    1. Allocate 60-70% of memory as persistent base (stays for entire test)
    2. Thrash remaining 20-30% with rapid alloc/free cycles
    3. Tests allocator under sustained high memory + fragmentation
    """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        log_message(f"Starting intense thrashing test on device: {device_uuid}", log_file)
        log_message(f"Test duration: {duration} seconds", log_file)
        
        if not torch.cuda.is_available():
            log_message(f"ERROR: CUDA not available for device {device_uuid}", log_file)
            return False
        
        device = torch.device('cuda:0')
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        log_message(f"Device: {device_name}", log_file)
        log_message(f"Total memory: {total_memory / (1024**3):.2f} GB", log_file)
        
        # Phase 1: Allocate persistent base load (60-70% of memory)
        log_message("=" * 50, log_file)
        log_message("Phase 1: Allocating persistent base load (60-70%)", log_file)
        log_message("=" * 50, log_file)
        
        base_target = int(total_memory * 0.65)  # 65% persistent
        base_tensors = []
        base_allocated = 0
        
        # Use larger chunks for base load
        base_chunk_size = 100 * 1024 * 1024  # 100 MB chunks
        
        while base_allocated < base_target:
            try:
                remaining = base_target - base_allocated
                chunk_size = min(base_chunk_size, remaining)
                num_elements = chunk_size // 4
                tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
                base_tensors.append(tensor)
                base_allocated += chunk_size
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log_message(f"Reached memory limit at {base_allocated / (1024**3):.2f} GB", log_file)
                    torch.cuda.empty_cache()
                    break
                else:
                    raise
        
        torch.cuda.synchronize()
        actual_base = torch.cuda.memory_allocated(0)
        log_message(f"Base load allocated: {actual_base / (1024**3):.2f} GB ({actual_base/total_memory*100:.1f}%)", log_file)
        log_message(f"Base tensors: {len(base_tensors)}", log_file)
        
        # Calculate thrashing space
        thrashing_space = total_memory - actual_base
        log_message(f"Available for thrashing: {thrashing_space / (1024**3):.2f} GB", log_file)
        
        # Phase 2: Aggressive thrashing in remaining space
        log_message("=" * 50, log_file)
        log_message("Phase 2: Starting aggressive thrashing cycles", log_file)
        log_message("=" * 50, log_file)
        
        start_time = time.time()
        cycle_count = 0
        allocation_count = 0
        deallocation_count = 0
        oom_count = 0
        
        # Thrashing chunk sizes (smaller for rapid cycling)
        thrash_chunk_sizes = [
            5 * 1024 * 1024,    # 5 MB
            10 * 1024 * 1024,   # 10 MB
            25 * 1024 * 1024,   # 25 MB
            50 * 1024 * 1024,   # 50 MB
            100 * 1024 * 1024,  # 100 MB
        ]
        
        while time.time() - start_time < duration:
            try:
                cycle_count += 1
                thrash_tensors = []
                
                # Allocate in thrashing space
                num_allocs = random.randint(5, 15)
                for i in range(num_allocs):
                    try:
                        current_allocated = torch.cuda.memory_allocated(0)
                        available = total_memory - current_allocated
                        
                        # Stop if we're using >85% total memory
                        if current_allocated > total_memory * 0.85:
                            break
                        
                        suitable_sizes = [s for s in thrash_chunk_sizes if s <= available]
                        if not suitable_sizes:
                            break
                        
                        chunk_size = random.choice(suitable_sizes)
                        num_elements = chunk_size // 4
                        tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
                        thrash_tensors.append(tensor)
                        allocation_count += 1
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            oom_count += 1
                            torch.cuda.empty_cache()
                            break
                        else:
                            raise
                
                torch.cuda.synchronize()
                
                # Perform operations on thrashing tensors
                if thrash_tensors:
                    num_ops = min(3, len(thrash_tensors))
                    for tensor in random.sample(thrash_tensors, num_ops):
                        tensor.mul_(1.01)
                        tensor.add_(0.01)
                    torch.cuda.synchronize()
                
                # Also operate on base load occasionally to keep it active
                if cycle_count % 50 == 0 and base_tensors:
                    for tensor in random.sample(base_tensors, min(5, len(base_tensors))):
                        tensor.mul_(1.0001)
                
                # Delete most thrashing tensors (create fragmentation)
                if thrash_tensors:
                    num_to_delete = int(len(thrash_tensors) * 0.8)  # Delete 80%
                    indices_to_delete = random.sample(range(len(thrash_tensors)), num_to_delete)
                    for idx in sorted(indices_to_delete, reverse=True):
                        del thrash_tensors[idx]
                        deallocation_count += 1
                
                # Cleanup thrashing tensors
                del thrash_tensors
                
                # Occasional cache clearing
                if cycle_count % 25 == 0:
                    torch.cuda.empty_cache()
                
                # Progress logging
                if cycle_count % 200 == 0:
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
                
                # Faster thrashing
                time.sleep(0.002)
                
            except RuntimeError as e:
                error_msg = str(e)
                if "out of memory" in error_msg:
                    oom_count += 1
                    torch.cuda.empty_cache()
                    # Continue to next cycle
                else:
                    log_message(f"CRITICAL ERROR in cycle {cycle_count}: {error_msg}", log_file)
                    return False
        
        # Final statistics
        elapsed = time.time() - start_time
        log_message("=" * 50, log_file)
        log_message(f"Intense thrashing test completed!", log_file)
        log_message(f"Base load maintained: {len(base_tensors)} tensors, {actual_base / (1024**3):.2f} GB", log_file)
        log_message(f"Total thrashing cycles: {cycle_count}", log_file)
        log_message(f"Total allocations: {allocation_count}", log_file)
        log_message(f"Total deallocations: {deallocation_count}", log_file)
        log_message(f"OOM events (expected): {oom_count}", log_file)
        log_message(f"Duration: {elapsed:.0f} seconds", log_file)
        log_message(f"Cycles per second: {cycle_count / elapsed:.2f}", log_file)
        log_message("=" * 50, log_file)
        
        # Final cleanup
        del base_tensors
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}", log_file)
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", log_file)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 intense_thrashing_stress.py <device_uuid> <duration_seconds> <log_file>")
        sys.exit(1)
    
    device_uuid = sys.argv[1]
    duration = int(sys.argv[2])
    log_file = sys.argv[3]
    
    success = intense_thrashing_test(device_uuid, duration, log_file)
    sys.exit(0 if success else 1)
PYTHON_EOF

chmod +x "$PYTHON_SCRIPT"
log_info "Python intense thrashing script created."
echo "" | tee -a "$MAIN_LOG"

# Run intense thrashing test on ALL MIG devices SIMULTANEOUSLY
log_info "========================================="
log_info "SIMULTANEOUS INTENSE THRASHING TEST"
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

# Launch intense thrashing test on each device in the background
while IFS= read -r device_uuid; do
    device_num=$((device_num + 1))
    
    DEVICE_LOG="${LOG_DIR}/intense_thrashing_device_${device_num}_${TIMESTAMP}.log"
    
    log_info "Launching worker for MIG Device $device_num/$MIG_COUNT (UUID: $device_uuid)"
    
    # Start the test in background with explicit python path
    "$PYTHON_CMD" "$PYTHON_SCRIPT" "$device_uuid" "$TEST_DURATION" "$DEVICE_LOG" >> "$DEVICE_LOG" 2>&1 &
    worker_pid=$!
    WORKER_PIDS+=($worker_pid)
    
    log_info "  -> Worker PID: $worker_pid"
    log_info "  -> Device log: $DEVICE_LOG"
done < <(echo "$MIG_DEVICES")

echo "" | tee -a "$MAIN_LOG"
log_info "All workers launched! Now monitoring..."
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
for device_log in "${LOG_DIR}"/intense_thrashing_device_*_${TIMESTAMP}.log; do
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
log_info "INTENSE THRASHING TEST COMPLETED"
log_info "========================================="
log_info "Total MIG devices tested: $MIG_COUNT"
log_info "Successful workers: $SUCCESS_COUNT"
log_info "Failed workers: $FAILED_COUNT"
log_info "Test duration: $TEST_DURATION seconds"
log_info "Main log: $MAIN_LOG"
log_info "Error log: $ERROR_LOG"
log_info "Device logs: ${LOG_DIR}/intense_thrashing_device_*_${TIMESTAMP}.log"

if [ $FAILED_COUNT -gt 0 ] || [ -s "$ERROR_LOG" ]; then
    log_warning "Some tests failed or abnormalities detected. Check logs for details."
    exit 1
else
    log_info "All intense thrashing tests passed successfully!"
    exit 0
fi
