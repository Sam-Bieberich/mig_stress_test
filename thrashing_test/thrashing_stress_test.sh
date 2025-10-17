#!/bin/bash

##############################################################################
# Memory Thrashing Stress Test Script
# 
# This script performs aggressive memory allocation/deallocation cycles
# to stress test the GPU memory allocator and fragmentation handling.
# Tests each MIG slice with rapid alloc/free patterns for 30 minutes.
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
TEST_DURATION=1800  # 30 minutes in seconds
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
        
        # Define allocation patterns
        chunk_sizes = [
            10 * 1024 * 1024,    # 10 MB
            50 * 1024 * 1024,    # 50 MB
            100 * 1024 * 1024,   # 100 MB
            200 * 1024 * 1024,   # 200 MB
            500 * 1024 * 1024,   # 500 MB
        ]
        
        log_message("Starting aggressive memory thrashing cycles...", log_file)
        
        while time.time() - start_time < duration:
            try:
                cycle_count += 1
                tensors = []
                
                # Phase 1: Rapid allocation with varying sizes
                num_allocs = random.randint(5, 20)
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
                
                # Phase 2: Random operations on tensors
                if tensors:
                    for tensor in random.sample(tensors, min(5, len(tensors))):
                        tensor.mul_(1.01)
                        tensor.add_(0.01)
                    torch.cuda.synchronize()
                
                # Phase 3: Random deallocation pattern
                # Delete tensors in random order to create fragmentation
                indices_to_delete = random.sample(range(len(tensors)), 
                                                 len(tensors) // 2 if tensors else 0)
                for idx in sorted(indices_to_delete, reverse=True):
                    del tensors[idx]
                    deallocation_count += 1
                
                # Phase 4: Partial cleanup and cache clearing
                del tensors
                if cycle_count % 10 == 0:
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
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)
                
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

# Run thrashing test on each MIG device
device_num=0
echo "$MIG_DEVICES" | while read -r device_uuid; do
    device_num=$((device_num + 1))
    
    log_info "========================================="
    log_info "Thrashing Test - MIG Device $device_num of $MIG_COUNT"
    log_info "Device UUID: $device_uuid"
    log_info "========================================="
    
    if python3 "$PYTHON_SCRIPT" "$device_uuid" "$TEST_DURATION" "$MAIN_LOG" 2>&1 | tee -a "$MAIN_LOG"; then
        log_info "Thrashing test completed successfully for device $device_uuid"
    else
        exit_code=$?
        log_error "Thrashing test FAILED for device $device_uuid (exit code: $exit_code)"
    fi
    
    # Check for errors in dmesg
    if dmesg | tail -100 | grep -i "gpu\|nvidia\|cuda" | grep -i "error\|fail\|crash" >> "$ERROR_LOG" 2>&1; then
        log_warning "GPU-related errors found in system logs."
    fi
    
    echo "" | tee -a "$MAIN_LOG"
    
    if [ $device_num -lt $MIG_COUNT ]; then
        log_info "Pausing 10 seconds before next device..."
        sleep 10
    fi
done

# Final summary
log_info "========================================="
log_info "All memory thrashing tests completed!"
log_info "========================================="
log_info "Main log: $MAIN_LOG"
log_info "Error log: $ERROR_LOG"

if [ -s "$ERROR_LOG" ]; then
    log_warning "Abnormalities detected. Check $ERROR_LOG for details."
    exit 1
else
    log_info "No abnormalities detected. All tests passed!"
    exit 0
fi
