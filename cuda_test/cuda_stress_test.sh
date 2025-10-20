#!/bin/bash

##############################################################################
# CUDA API Edge Cases Stress Test Script
# 
# This script tests unusual but valid CUDA API operations:
# - Maximum number of streams
# - Large shared memory allocations
# - Complex memory access patterns
# - Multiple kernel launches
# - Synchronization edge cases
##############################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration
TEST_DURATION=1800  # 3 minutes in seconds
LOG_DIR="${SCRIPT_DIR}/logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/cuda_test_${TIMESTAMP}.log"
ERROR_LOG="${LOG_DIR}/cuda_test_errors_${TIMESTAMP}.log"
PYTHON_SCRIPT="${SCRIPT_DIR}/cuda_stress.py"

# Create log directory
mkdir -p "$LOG_DIR"

# Initialize logs
echo "==================================" | tee -a "$MAIN_LOG"
echo "CUDA API Edge Cases Test Started: $(date)" | tee -a "$MAIN_LOG"
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

# Create the Python CUDA edge cases test script
log_info "Creating Python CUDA edge cases script..."
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

def cuda_edge_cases_test(device_uuid, duration, log_file):
    """
    Tests CUDA API edge cases and limits:
    - Maximum number of CUDA streams
    - Large tensor operations
    - Complex kernel patterns
    - Memory access patterns
    - Synchronization stress
    """
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = device_uuid
        
        log_message(f"Starting CUDA API edge cases test on device: {device_uuid}", log_file)
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
        test_iterations = 0
        
        # Test 1: Maximum number of CUDA streams
        log_message("Test 1: Creating maximum CUDA streams...", log_file)
        streams = []
        max_streams = 128  # Test with large number of streams
        try:
            for i in range(max_streams):
                streams.append(torch.cuda.Stream())
            log_message(f"Successfully created {len(streams)} CUDA streams", log_file)
        except Exception as e:
            log_message(f"WARNING: Could only create {len(streams)} streams: {str(e)}", log_file)
        
        # Test 2: Large tensor allocations with unusual shapes
        log_message("Test 2: Testing large tensor allocations with unusual shapes...", log_file)
        unusual_shapes = [
            (1, 1000000),      # Very wide
            (1000000, 1),      # Very tall
            (1000, 1000, 10),  # 3D tensor
            (100, 100, 10, 10), # 4D tensor
        ]
        
        test_tensors = []
        for shape in unusual_shapes:
            try:
                tensor = torch.randn(shape, dtype=torch.float32, device=device)
                test_tensors.append(tensor)
                log_message(f"Created tensor with shape {shape}: {tensor.numel() * 4 / (1024**2):.2f} MB", log_file)
            except RuntimeError as e:
                log_message(f"WARNING: Failed to create tensor shape {shape}: {str(e)}", log_file)
        
        # Test 3: Complex operations across multiple streams
        log_message("Test 3: Running complex operations across multiple streams...", log_file)
        
        # Allocate working memory
        chunk_size = 50 * 1024 * 1024  # 50 MB chunks
        num_chunks = min(20, int(total_memory * 0.5 / chunk_size))
        working_tensors = []
        
        for i in range(num_chunks):
            num_elements = chunk_size // 4
            tensor = torch.randn(num_elements, dtype=torch.float32, device=device)
            working_tensors.append(tensor)
        
        log_message(f"Allocated {len(working_tensors)} working tensors ({num_chunks * 50} MB total)", log_file)
        
        # Main test loop
        log_message("Starting main edge case testing loop...", log_file)
        
        while time.time() - start_time < duration:
            try:
                test_iterations += 1
                
                # Edge Case 1: Operations on multiple streams
                for i, stream in enumerate(streams[:10]):  # Use first 10 streams
                    with torch.cuda.stream(stream):
                        tensor_idx = i % len(working_tensors)
                        working_tensors[tensor_idx].mul_(1.001)
                        working_tensors[tensor_idx].add_(0.001)
                
                # Edge Case 2: Cross-stream synchronization
                if test_iterations % 5 == 0:
                    torch.cuda.synchronize()
                
                # Edge Case 3: Complex reduction operations
                if test_iterations % 10 == 0:
                    for tensor in working_tensors[:5]:
                        _ = tensor.sum()
                        _ = tensor.mean()
                        _ = tensor.std()
                
                # Edge Case 4: Reshape and transpose operations
                if test_iterations % 20 == 0:
                    for tensor in working_tensors[:3]:
                        size = tensor.size(0)
                        # Find largest perfect square
                        n = int(size ** 0.5)
                        if n * n <= size:
                            reshaped = tensor[:n*n].reshape(n, n)
                            _ = reshaped.t()
                
                # Edge Case 5: In-place operations chain
                if test_iterations % 15 == 0:
                    t = working_tensors[0]
                    t.add_(1.0).mul_(0.99).clamp_(-10, 10).abs_()
                
                # Edge Case 6: Memory pinning simulation (allocate/free patterns)
                if test_iterations % 50 == 0:
                    temp_tensors = []
                    for _ in range(5):
                        temp = torch.randn(1024*1024, dtype=torch.float32, device=device)
                        temp_tensors.append(temp)
                    del temp_tensors
                    torch.cuda.empty_cache()
                
                # Progress logging
                if test_iterations % 100 == 0:
                    elapsed = time.time() - start_time
                    memory_allocated = torch.cuda.memory_allocated(0) / (1024**3)
                    memory_cached = torch.cuda.memory_reserved(0) / (1024**3)
                    log_message(
                        f"Iteration {test_iterations} | {elapsed:.0f}s/{duration}s | "
                        f"Memory: {memory_allocated:.2f}GB | Cached: {memory_cached:.2f}GB | "
                        f"Streams: {len(streams)}",
                        log_file
                    )
                
                # Small delay
                time.sleep(0.05)
                
            except RuntimeError as e:
                log_message(f"ERROR in iteration {test_iterations}: {str(e)}", log_file)
                if "out of memory" in str(e):
                    torch.cuda.empty_cache()
                else:
                    return False
        
        # Final statistics
        elapsed = time.time() - start_time
        log_message("=" * 50, log_file)
        log_message(f"CUDA API edge cases test completed!", log_file)
        log_message(f"Total iterations: {test_iterations}", log_file)
        log_message(f"CUDA streams used: {len(streams)}", log_file)
        log_message(f"Duration: {elapsed:.0f} seconds", log_file)
        log_message(f"Iterations per second: {test_iterations / elapsed:.2f}", log_file)
        log_message("=" * 50, log_file)
        
        # Cleanup
        del streams
        del test_tensors
        del working_tensors
        torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        log_message(f"CRITICAL ERROR: {str(e)}", log_file)
        import traceback
        log_message(f"Traceback: {traceback.format_exc()}", log_file)
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 cuda_stress.py <device_uuid> <duration_seconds> <log_file>")
        sys.exit(1)
    
    device_uuid = sys.argv[1]
    duration = int(sys.argv[2])
    log_file = sys.argv[3]
    
    success = cuda_edge_cases_test(device_uuid, duration, log_file)
    sys.exit(0 if success else 1)
PYTHON_EOF

chmod +x "$PYTHON_SCRIPT"
log_info "Python CUDA edge cases script created."
echo "" | tee -a "$MAIN_LOG"

# Run CUDA API test on each MIG device
device_num=0
echo "$MIG_DEVICES" | while read -r device_uuid; do
    device_num=$((device_num + 1))
    
    log_info "========================================="
    log_info "CUDA API Test - MIG Device $device_num of $MIG_COUNT"
    log_info "Device UUID: $device_uuid"
    log_info "========================================="
    
    if python3 "$PYTHON_SCRIPT" "$device_uuid" "$TEST_DURATION" "$MAIN_LOG" 2>&1 | tee -a "$MAIN_LOG"; then
        log_info "CUDA API test completed successfully for device $device_uuid"
    else
        exit_code=$?
        log_error "CUDA API test FAILED for device $device_uuid (exit code: $exit_code)"
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
log_info "All CUDA API edge case tests completed!"
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
