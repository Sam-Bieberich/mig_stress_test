#!/bin/bash

##############################################################################
# Thermal Shock Test Script
# 
# Tests thermal management and power transitions by:
# - Rapidly cycling between 100% load and idle
# - Maximum compute + memory for 30 seconds
# - Complete idle for 30 seconds
# - Repeat cycles to stress thermal/power management
# - Monitor temperature, power, and clock changes
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
TEST_DURATION=1800  # 3 minutes in seconds (6 cycles of 30s hot + 30s cold)
CYCLE_DURATION=30  # 30 seconds per phase (hot or cold)
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/thermal_test_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/thermal_test_errors_${TIMESTAMP}.log"
PYTHON_SCRIPT="${SCRIPT_DIR}/thermal_stress.py"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize logs
echo "==================================" | tee -a "$MAIN_LOG"
echo "Thermal Shock Test Started: $(date)" | tee -a "$MAIN_LOG"
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

# Create the Python thermal shock test script
log_info "Creating Python thermal shock script..."
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

def thermal_shock_test(device_uuid, duration, cycle_duration, log_file):
    """
    Thermal shock test - rapid cycling between max load and idle:
    - HOT phase: Maximum compute + memory allocation for cycle_duration seconds
    - COLD phase: Complete idle (all memory freed) for cycle_duration seconds
    - Monitor temperature, power, and clock changes throughout
    """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        log_message(f"Starting thermal shock test on device: {device_uuid}", log_file)
        log_message(f"Test duration: {duration} seconds", log_file)
        log_message(f"Cycle duration: {cycle_duration}s HOT + {cycle_duration}s COLD", log_file)
        
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
        
        # Track thermal metrics
        temps = []
        powers = []
        
        while time.time() - start_time < duration:
            cycle_count += 1
            
            # ============================================
            # HOT PHASE - Maximum load
            # ============================================
            log_message("=" * 50, log_file)
            log_message(f"CYCLE {cycle_count} - HOT PHASE (Max Load)", log_file)
            log_message("=" * 50, log_file)
            
            # Allocate 90% of memory
            target_memory = int(total_memory * 0.9)
            chunk_size = 100 * 1024 * 1024  # 100 MB chunks
            tensors = []
            allocated = 0
            
            log_message("Allocating 90% of memory...", log_file)
            while allocated < target_memory:
                try:
                    num_elements = chunk_size // 4
                    t = torch.randn(num_elements, dtype=torch.float32, device=device)
                    tensors.append(t)
                    allocated += chunk_size
                except RuntimeError:
                    break
            
            torch.cuda.synchronize()
            actual_allocated = torch.cuda.memory_allocated(0) / (1024**3)
            log_message(f"Allocated: {actual_allocated:.2f} GB", log_file)
            
            # Create compute tensors
            compute_a = torch.randn(2048, 2048, dtype=torch.float32, device=device)
            compute_b = torch.randn(2048, 2048, dtype=torch.float32, device=device)
            
            # Run intense compute for cycle_duration seconds
            log_message(f"Running max compute for {cycle_duration} seconds...", log_file)
            phase_start = time.time()
            compute_iters = 0
            
            while time.time() - phase_start < cycle_duration:
                # Heavy matrix multiplication
                result = torch.mm(compute_a, compute_b)
                compute_a = result
                
                # Operations on memory tensors
                if tensors and compute_iters % 10 == 0:
                    for t in tensors[:5]:
                        t.mul_(1.001)
                
                compute_iters += 1
                
                # Sample temperature and power every 2 seconds
                if compute_iters % 100 == 0:
                    try:
                        temp = torch.cuda.temperature()
                        temps.append(temp)
                    except:
                        temp = 'N/A'
                    
                    try:
                        power = torch.cuda.power_draw() / 1000.0
                        powers.append(power)
                    except:
                        power = 'N/A'
                    
                    elapsed_phase = time.time() - phase_start
                    log_message(f"  HOT: {elapsed_phase:.0f}s | Temp: {temp}°C | Power: {power}W | Iters: {compute_iters}", log_file)
            
            hot_elapsed = time.time() - phase_start
            log_message(f"HOT phase complete: {compute_iters} iterations in {hot_elapsed:.1f}s", log_file)
            
            # Cleanup compute tensors
            del compute_a, compute_b, result
            
            # ============================================
            # COLD PHASE - Complete idle
            # ============================================
            log_message("=" * 50, log_file)
            log_message(f"CYCLE {cycle_count} - COLD PHASE (Idle)", log_file)
            log_message("=" * 50, log_file)
            
            # Free all memory
            log_message("Freeing all memory...", log_file)
            del tensors
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            remaining_mem = torch.cuda.memory_allocated(0) / (1024**3)
            log_message(f"Memory after cleanup: {remaining_mem:.2f} GB", log_file)
            
            # Idle for cycle_duration seconds
            log_message(f"Idling for {cycle_duration} seconds...", log_file)
            phase_start = time.time()
            sample_count = 0
            
            while time.time() - phase_start < cycle_duration:
                time.sleep(2)
                
                # Sample temperature and power during idle
                try:
                    temp = torch.cuda.temperature()
                    temps.append(temp)
                except:
                    temp = 'N/A'
                
                try:
                    power = torch.cuda.power_draw() / 1000.0
                    powers.append(power)
                except:
                    power = 'N/A'
                
                elapsed_phase = time.time() - phase_start
                log_message(f"  COLD: {elapsed_phase:.0f}s | Temp: {temp}°C | Power: {power}W", log_file)
                sample_count += 1
            
            cold_elapsed = time.time() - phase_start
            log_message(f"COLD phase complete: {cold_elapsed:.1f}s idle", log_file)
            log_message("", log_file)
        
        # Final statistics
        elapsed = time.time() - start_time
        log_message("=" * 50, log_file)
        log_message(f"Thermal shock test completed!", log_file)
        log_message(f"Total cycles: {cycle_count}", log_file)
        log_message(f"Duration: {elapsed:.0f} seconds", log_file)
        
        # Thermal statistics
        if temps:
            min_temp = min(temps)
            max_temp = max(temps)
            avg_temp = sum(temps) / len(temps)
            temp_delta = max_temp - min_temp
            log_message(f"Temperature: {min_temp:.1f}°C - {max_temp:.1f}°C (Δ{temp_delta:.1f}°C, avg: {avg_temp:.1f}°C)", log_file)
        
        if powers:
            min_power = min(powers)
            max_power = max(powers)
            avg_power = sum(powers) / len(powers)
            power_delta = max_power - min_power
            log_message(f"Power: {min_power:.1f}W - {max_power:.1f}W (Δ{power_delta:.1f}W, avg: {avg_power:.1f}W)", log_file)
        
        log_message("=" * 50, log_file)
        
        return True
        
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}", log_file)
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", log_file)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python3 thermal_stress.py <device_uuid> <duration_seconds> <cycle_duration> <log_file>")
        sys.exit(1)
    
    device_uuid = sys.argv[1]
    duration = int(sys.argv[2])
    cycle_duration = int(sys.argv[3])
    log_file = sys.argv[4]
    
    success = thermal_shock_test(device_uuid, duration, cycle_duration, log_file)
    sys.exit(0 if success else 1)
PYTHON_EOF

chmod +x "$PYTHON_SCRIPT"
log_info "Python thermal shock script created."
echo "" | tee -a "$MAIN_LOG"

# Run thermal shock test on ALL MIG devices SIMULTANEOUSLY
log_info "========================================="
log_info "SIMULTANEOUS THERMAL SHOCK TEST"
log_info "Starting on ALL $MIG_COUNT MIG devices in parallel"
log_info "Cycle: ${CYCLE_DURATION}s HOT + ${CYCLE_DURATION}s COLD"
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

# Launch thermal shock test on each device in the background
while IFS= read -r device_uuid; do
    device_num=$((device_num + 1))
    
    DEVICE_LOG="${LOG_DIR}/thermal_device_${device_num}_${TIMESTAMP}.log"
    
    log_info "Launching worker for MIG Device $device_num/$MIG_COUNT (UUID: $device_uuid)"
    
    # Start the test in background with explicit python path
    "$PYTHON_CMD" "$PYTHON_SCRIPT" "$device_uuid" "$TEST_DURATION" "$CYCLE_DURATION" "$DEVICE_LOG" >> "$DEVICE_LOG" 2>&1 &
    worker_pid=$!
    WORKER_PIDS+=($worker_pid)
    
    log_info "  -> Worker PID: $worker_pid"
    log_info "  -> Device log: $DEVICE_LOG"
done < <(echo "$MIG_DEVICES")

echo "" | tee -a "$MAIN_LOG"
log_info "All thermal shock workers launched! Now monitoring..."
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
for device_log in "${LOG_DIR}"/thermal_device_*_${TIMESTAMP}.log; do
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
if dmesg | tail -100 | grep -i "gpu\|nvidia\|cuda\|thermal\|throttle" | grep -i "error\|fail\|crash" >> "$ERROR_LOG" 2>&1; then
    log_warning "GPU/thermal-related errors found in system logs."
fi

# Final summary
log_info "========================================="
log_info "THERMAL SHOCK TEST COMPLETED"
log_info "========================================="
log_info "Total MIG devices tested: $MIG_COUNT"
log_info "Successful workers: $SUCCESS_COUNT"
log_info "Failed workers: $FAILED_COUNT"
log_info "Test duration: $TEST_DURATION seconds"
log_info "Main log: $MAIN_LOG"
log_info "Error log: $ERROR_LOG"
log_info "Device logs: ${LOG_DIR}/thermal_device_*_${TIMESTAMP}.log"

if [ $FAILED_COUNT -gt 0 ] || [ -s "$ERROR_LOG" ]; then
    log_warning "Some tests failed or abnormalities detected. Check logs for details."
    exit 1
else
    log_info "All thermal shock tests passed successfully!"
    exit 0
fi
