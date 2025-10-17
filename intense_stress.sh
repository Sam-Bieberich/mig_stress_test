#!/bin/bash

##############################################################################
# MIG Intense Stress Test Script
# 
# This script performs intensive stress testing on NVIDIA MIG slices by:
# - Testing one MIG slice at 95% memory (primary target)
# - Running all other MIG slices at 75% memory simultaneously (background load)
# - Rotating through each slice as the primary target
# - Running for 30 minutes per primary target
# Logs all activities and captures any abnormalities (memory issues, crashes).
##############################################################################

# Configuration
TEST_DURATION=1800  # 30 minutes in seconds
LOG_DIR="./stress_test_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/intense_stress_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/intense_stress_errors_${TIMESTAMP}.log"
PYTHON_SCRIPT="./mig_intense_stress.py"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize logs
echo "==================================" | tee -a "$MAIN_LOG"
echo "MIG Intense Stress Test Started: $(date)" | tee -a "$MAIN_LOG"
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

# Convert to array
MIG_ARRAY=()
while IFS= read -r device; do
    MIG_ARRAY+=("$device")
done <<< "$MIG_DEVICES"

# Count MIG devices
MIG_COUNT=${#MIG_ARRAY[@]}
log_info "Found $MIG_COUNT MIG device(s)"
for device in "${MIG_ARRAY[@]}"; do
    log_info "  - $device"
done
echo "" | tee -a "$MAIN_LOG"

# Create the Python intense stress test script
log_info "Creating Python intense stress test script..."
cat > "$PYTHON_SCRIPT" << 'PYTHON_EOF'
import torch
import sys
import time
import os
import multiprocessing as mp
from datetime import datetime
import signal

def log_message(msg, log_file):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    message = f"[PYTHON] {timestamp} - {msg}"
    print(message, flush=True)
    with open(log_file, 'a') as f:
        f.write(message + '\n')

def background_stress_worker(device_uuid, target_percent, duration, log_file, stop_event):
    """Run stress test on a background device at specified memory percentage"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        if not torch.cuda.is_available():
            log_message(f"ERROR: CUDA not available for background device {device_uuid}", log_file)
            return
        
        device = torch.device('cuda:0')
        total_memory = torch.cuda.get_device_properties(0).total_memory
        target_memory = int(total_memory * target_percent)
        
        log_message(f"Background worker starting on {device_uuid} at {target_percent*100}% ({target_memory / (1024**3):.2f} GB)", log_file)
        
        # Allocate memory
        tensors = []
        chunk_size = 100 * 1024 * 1024  # 100 MB chunks
        allocated = 0
        
        while allocated < target_memory and not stop_event.is_set():
            try:
                remaining = target_memory - allocated
                current_chunk = min(chunk_size, remaining)
                num_elements = current_chunk // 4
                
                tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
                tensors.append(tensor)
                allocated += current_chunk
            except RuntimeError as e:
                if "out of memory" in str(e):
                    break
                else:
                    log_message(f"Background worker error on {device_uuid}: {str(e)}", log_file)
                    return
        
        torch.cuda.synchronize()
        log_message(f"Background worker on {device_uuid} allocated {allocated / (1024**3):.2f} GB", log_file)
        
        # Keep memory active with operations
        iteration = 0
        start_time = time.time()
        
        while not stop_event.is_set() and (time.time() - start_time) < duration:
            try:
                for i, tensor in enumerate(tensors):
                    if stop_event.is_set():
                        break
                    tensor.mul_(1.0001)
                    tensor.add_(0.0001)
                    if i % 100 == 0:
                        torch.cuda.synchronize()
                
                iteration += 1
                time.sleep(0.1)
            except Exception as e:
                log_message(f"Background worker exception on {device_uuid}: {str(e)}", log_file)
                break
        
        # Cleanup
        del tensors
        torch.cuda.empty_cache()
        log_message(f"Background worker on {device_uuid} completed ({iteration} iterations)", log_file)
        
    except Exception as e:
        log_message(f"CRITICAL ERROR in background worker {device_uuid}: {str(e)}", log_file)

def primary_stress_test(device_uuid, duration, log_file, primary_percent=0.95):
    """Run primary stress test on target device"""
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        if not torch.cuda.is_available():
            log_message(f"ERROR: CUDA not available for primary device {device_uuid}", log_file)
            return False
        
        device = torch.device('cuda:0')
        device_name = torch.cuda.get_device_name(0)
        total_memory = torch.cuda.get_device_properties(0).total_memory
        
        log_message(f"PRIMARY: Device {device_uuid} - {device_name}", log_file)
        log_message(f"PRIMARY: Total memory: {total_memory / (1024**3):.2f} GB", log_file)
        
        target_memory = int(total_memory * primary_percent)
        log_message(f"PRIMARY: Target allocation: {target_memory / (1024**3):.2f} GB ({primary_percent*100}%)", log_file)
        
        # Allocate memory
        tensors = []
        chunk_size = 100 * 1024 * 1024
        allocated = 0
        
        log_message("PRIMARY: Allocating memory...", log_file)
        while allocated < target_memory:
            try:
                remaining = target_memory - allocated
                current_chunk = min(chunk_size, remaining)
                num_elements = current_chunk // 4
                
                tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
                tensors.append(tensor)
                allocated += current_chunk
                
                if len(tensors) % 10 == 0:
                    log_message(f"PRIMARY: Allocated {allocated / (1024**3):.2f} GB", log_file)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    log_message(f"PRIMARY: Reached memory limit at {allocated / (1024**3):.2f} GB", log_file)
                    break
                else:
                    raise
        
        torch.cuda.synchronize()
        actual_allocated = torch.cuda.memory_allocated(0)
        log_message(f"PRIMARY: Final allocated memory: {actual_allocated / (1024**3):.2f} GB", log_file)
        
        # Stress operations
        log_message("PRIMARY: Starting intensive stress operations...", log_file)
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration:
            try:
                for i, tensor in enumerate(tensors):
                    tensor.mul_(1.0001)
                    tensor.add_(0.0001)
                    if i % 100 == 0:
                        torch.cuda.synchronize()
                
                iteration += 1
                elapsed = time.time() - start_time
                
                if iteration % 10 == 0:
                    remaining = duration - elapsed
                    memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    log_message(f"PRIMARY: Progress {elapsed:.0f}s/{duration}s | Memory: {memory_used:.2f} GB | Iter: {iteration}", log_file)
                
                time.sleep(0.1)
                
            except RuntimeError as e:
                log_message(f"PRIMARY: ERROR during stress test: {str(e)}", log_file)
                return False
        
        log_message(f"PRIMARY: Test completed successfully. Total iterations: {iteration}", log_file)
        
        # Cleanup
        del tensors
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        log_message(f"PRIMARY: CRITICAL ERROR: {str(e)}", log_file)
        import traceback
        log_message(f"PRIMARY: Traceback: {traceback.format_exc()}", log_file)
        return False

def run_intense_stress_test(primary_device, all_devices, duration, log_file):
    """Run intense stress test with one primary device and multiple background devices"""
    
    # Create stop event for background workers
    stop_event = mp.Event()
    background_processes = []
    
    try:
        # Start background workers on all non-primary devices
        background_percent = 0.75
        
        log_message(f"Starting background workers at {background_percent*100}% memory...", log_file)
        for device_uuid in all_devices:
            if device_uuid != primary_device:
                p = mp.Process(
                    target=background_stress_worker,
                    args=(device_uuid, background_percent, duration + 60, log_file, stop_event)
                )
                p.start()
                background_processes.append(p)
                log_message(f"Background worker started for device {device_uuid} (PID: {p.pid})", log_file)
        
        # Give background workers time to allocate memory
        log_message("Waiting for background workers to initialize...", log_file)
        time.sleep(10)
        
        # Run primary stress test
        log_message(f"Starting PRIMARY test on device {primary_device}...", log_file)
        success = primary_stress_test(primary_device, duration, log_file)
        
        # Stop background workers
        log_message("Stopping background workers...", log_file)
        stop_event.set()
        
        # Wait for all background processes to finish
        for p in background_processes:
            p.join(timeout=30)
            if p.is_alive():
                log_message(f"WARNING: Background worker {p.pid} did not stop gracefully, terminating...", log_file)
                p.terminate()
                p.join()
        
        return success
        
    except Exception as e:
        log_message(f"ERROR in intense stress test coordinator: {str(e)}", log_file)
        stop_event.set()
        for p in background_processes:
            if p.is_alive():
                p.terminate()
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 mig_intense_stress.py <primary_device_uuid> <all_device_uuids_comma_separated> <duration_seconds>")
        sys.exit(1)
    
    primary_device = sys.argv[1]
    all_devices = sys.argv[2].split(',')
    duration = int(sys.argv[3])
    log_file = "./stress_test_logs/intense_stress_python.log"
    
    success = run_intense_stress_test(primary_device, all_devices, duration, log_file)
    sys.exit(0 if success else 1)
PYTHON_EOF

chmod +x "$PYTHON_SCRIPT"
log_info "Python intense stress test script created."
echo "" | tee -a "$MAIN_LOG"

# Create comma-separated list of all devices
ALL_DEVICES=$(IFS=,; echo "${MIG_ARRAY[*]}")

# Run intense stress test on each MIG device as primary
for device_idx in "${!MIG_ARRAY[@]}"; do
    primary_device="${MIG_ARRAY[$device_idx]}"
    device_num=$((device_idx + 1))
    
    log_info "========================================="
    log_info "INTENSE TEST - Round $device_num of $MIG_COUNT"
    log_info "Primary Device (95%): $primary_device"
    log_info "Background Devices (75%): All others"
    log_info "========================================="
    
    # Get initial status for all devices
    log_info "Initial GPU status for all devices:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv >> "$MAIN_LOG" 2>&1
    echo "" | tee -a "$MAIN_LOG"
    
    # Run the intense stress test
    log_info "Starting $TEST_DURATION second intense stress test..."
    
    if python3 "$PYTHON_SCRIPT" "$primary_device" "$ALL_DEVICES" "$TEST_DURATION" 2>&1 | tee -a "$MAIN_LOG"; then
        log_info "Intense stress test completed successfully for primary device $primary_device"
    else
        exit_code=$?
        log_error "Intense stress test FAILED for primary device $primary_device (exit code: $exit_code)"
    fi
    
    # Get final status for all devices
    log_info "Final GPU status for all devices:"
    nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free --format=csv >> "$MAIN_LOG" 2>&1
    echo "" | tee -a "$MAIN_LOG"
    
    # Check for any errors in dmesg
    log_info "Checking system logs for GPU-related errors..."
    if dmesg | tail -100 | grep -i "gpu\|nvidia\|cuda" | grep -i "error\|fail\|crash" >> "$ERROR_LOG" 2>&1; then
        log_warning "GPU-related errors found in system logs. See $ERROR_LOG for details."
    fi
    
    echo "" | tee -a "$MAIN_LOG"
    
    # Pause between rounds to let GPUs cool down and memory clear
    if [ $device_num -lt $MIG_COUNT ]; then
        log_info "Pausing 30 seconds before next round..."
        sleep 30
    fi
done

# Final summary
log_info "========================================="
log_info "All MIG intense stress tests completed!"
log_info "========================================="
log_info "Main log: $MAIN_LOG"
log_info "Error log: $ERROR_LOG"
log_info "Test completed: $(date)"

# Check if there were any errors
if [ -s "$ERROR_LOG" ]; then
    log_warning "Abnormalities detected during testing. Check $ERROR_LOG for details."
    exit 1
else
    log_info "No abnormalities detected. All intense tests passed successfully!"
    exit 0
fi
