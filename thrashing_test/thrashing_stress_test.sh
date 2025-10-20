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
TEST_DURATION=180  # 3 minutes in seconds
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
        
        # Calculate adaptive chunk sizes based on available memory
        # Use smaller chunks for smaller MIG partitions
        total_gb = total_memory / (1024**3)
        
        if total_gb < 15:  # Small MIG partition
            chunk_sizes = [
                5 * 1024 * 1024,     # 5 MB
                10 * 1024 * 1024,    # 10 MB
                25 * 1024 * 1024,    # 25 MB
                50 * 1024 * 1024,    # 50 MB
                100 * 1024 * 1024,   # 100 MB
                200 * 1024 * 1024,   # 200 MB
            ]
            max_allocs = 20
            target_usage = 0.65  # Target 65% memory usage
        else:  # Larger partition
            chunk_sizes = [
                10 * 1024 * 1024,    # 10 MB
                50 * 1024 * 1024,    # 50 MB
                100 * 1024 * 1024,   # 100 MB
                200 * 1024 * 1024,   # 200 MB
                500 * 1024 * 1024,   # 500 MB
                750 * 1024 * 1024,   # 750 MB
            ]
            max_allocs = 30
            target_usage = 0.75  # Target 75% memory usage
        
        log_message(f"Adaptive configuration: max_allocs={max_allocs}, target_usage={target_usage*100}%", log_file)
        log_message(f"Chunk sizes: {[f'{s/(1024*1024):.0f}MB' for s in chunk_sizes]}", log_file)
        
        start_time = time.time()
        cycle_count = 0
        allocation_count = 0
        deallocation_count = 0
        oom_count = 0
        
        # Track GPU metrics
        gpu_temps = []
        gpu_powers = []
        
        log_message("Starting adaptive memory thrashing cycles...", log_file)
        
        while time.time() - start_time < duration:
            try:
                cycle_count += 1
                tensors = []
                
                # Get current memory status
                torch.cuda.synchronize()
                memory_allocated = torch.cuda.memory_allocated(0)
                memory_available = total_memory - memory_allocated
                
                # Phase 1: Adaptive allocation with memory awareness
                num_allocs = random.randint(5, max_allocs)
                target_memory = int(total_memory * target_usage)
                
                for i in range(num_allocs):
                    # Stop allocating if we're near target
                    current_allocated = torch.cuda.memory_allocated(0)
                    if current_allocated >= target_memory:
                        break
                    
                    try:
                        # Choose chunk size that won't exceed target
                        remaining = target_memory - current_allocated
                        suitable_sizes = [s for s in chunk_sizes if s <= remaining]
                        
                        if not suitable_sizes:
                            break
                        
                        chunk_size = random.choice(suitable_sizes)
                        num_elements = chunk_size // 4
                        tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
                        tensors.append(tensor)
                        allocation_count += 1
                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            oom_count += 1
                            log_message(f"OOM encountered in cycle {cycle_count} (expected) - recovered", log_file)
                            torch.cuda.empty_cache()
                            break
                        else:
                            raise
                
                torch.cuda.synchronize()
                
                # Phase 2: Moderate compute operations on tensors
                if tensors:
                    # Fewer operations to reduce compute overhead
                    num_ops = min(5, len(tensors))
                    for tensor in random.sample(tensors, num_ops):
                        # Basic operations
                        tensor.mul_(1.01)
                        tensor.add_(0.01)
                        # Occasional matrix operations for larger tensors
                        if cycle_count % 10 == 0 and tensor.numel() > 10000:
                            size = int(min(tensor.numel(), 1000000) ** 0.5)
                            if size > 100:
                                try:
                                    reshaped = tensor[:size*size].view(size, size)
                                    result = torch.mm(reshaped, reshaped.T)
                                    del result
                                except:
                                    pass
                    torch.cuda.synchronize()
                
                # Phase 3: Moderate deallocation pattern for fragmentation
                # Delete 60% of tensors in random order to create fragmentation
                indices_to_delete = random.sample(range(len(tensors)), 
                                                 int(len(tensors) * 0.6) if tensors else 0)
                for idx in sorted(indices_to_delete, reverse=True):
                    del tensors[idx]
                    deallocation_count += 1
                
                # Phase 4: Balanced cleanup and cache clearing
                del tensors
                # Clear cache periodically to prevent excessive fragmentation
                if cycle_count % 15 == 0:
                    torch.cuda.empty_cache()
                
                # Progress logging
                if cycle_count % 100 == 0:
                    elapsed = time.time() - start_time
                    memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    memory_reserved = torch.cuda.memory_reserved(0) / (1024**3)
                    
                    # Get GPU temperature and power
                    try:
                        temp = torch.cuda.temperature()
                        gpu_temps.append(temp)
                    except:
                        temp = 'N/A'
                    
                    try:
                        power = torch.cuda.power_draw() / 1000.0  # Convert mW to W
                        gpu_powers.append(power)
                    except:
                        power = 'N/A'
                    
                    log_message(
                        f"Cycle {cycle_count} | {elapsed:.0f}s/{duration}s | "
                        f"Allocs: {allocation_count} | Deallocs: {deallocation_count} | "
                        f"OOMs: {oom_count} | Mem: {memory_allocated:.2f}GB | "
                        f"Reserved: {memory_reserved:.2f}GB | Temp: {temp}°C | Power: {power}W", 
                        log_file
                    )
                
                # Moderate delay for sustainable intensity
                time.sleep(0.005)  # 5ms
                
            except RuntimeError as e:
                error_msg = str(e)
                if "out of memory" in error_msg:
                    oom_count += 1
                    log_message(f"OOM in cycle {cycle_count} (expected, recovered) - clearing cache", log_file)
                    torch.cuda.empty_cache()
                    # Continue to next cycle
                else:
                    log_message(f"CRITICAL ERROR in cycle {cycle_count}: {error_msg}", log_file)
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
        
        # Report GPU metrics ranges
        if gpu_temps:
            min_temp = min(gpu_temps)
            max_temp = max(gpu_temps)
            avg_temp = sum(gpu_temps) / len(gpu_temps)
            log_message(f"Temperature range: {min_temp:.1f}°C - {max_temp:.1f}°C (avg: {avg_temp:.1f}°C)", log_file)
        
        if gpu_powers:
            min_power = min(gpu_powers)
            max_power = max(gpu_powers)
            avg_power = sum(gpu_powers) / len(gpu_powers)
            log_message(f"Power range: {min_power:.1f}W - {max_power:.1f}W (avg: {avg_power:.1f}W)", log_file)
        
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

# Launch thrashing test on each device in the background
# Use process substitution to avoid subshell issues
while IFS= read -r device_uuid; do
    device_num=$((device_num + 1))
    
    DEVICE_LOG="${LOG_DIR}/thrashing_device_${device_num}_${TIMESTAMP}.log"
    
    log_info "Launching thrashing worker for MIG Device $device_num/$MIG_COUNT (UUID: $device_uuid)"
    
    # Start the thrashing test in background with explicit python path
    "$PYTHON_CMD" "$PYTHON_SCRIPT" "$device_uuid" "$TEST_DURATION" "$DEVICE_LOG" >> "$DEVICE_LOG" 2>&1 &
    worker_pid=$!
    WORKER_PIDS+=($worker_pid)
    
    log_info "  -> Worker PID: $worker_pid"
    log_info "  -> Device log: $DEVICE_LOG"
done < <(echo "$MIG_DEVICES")

echo "" | tee -a "$MAIN_LOG"
log_info "All thrashing workers launched! Now monitoring..."
log_info "Test duration: $TEST_DURATION seconds"
log_info "Worker PIDs: ${WORKER_PIDS[*]}"
echo "" | tee -a "$MAIN_LOG"

# Wait for all background processes to complete
log_info "Waiting for all thrashing workers to complete..."
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
