Stress testing files (and setup) for a device using MIG partitions (GH200)

## Overview

This repository contains scripts for setting up NVIDIA MIG (Multi-Instance GPU) partitions and performing comprehensive stress tests on each MIG slice.

## Files

- **mig_easy_setup.sh** - Quick setup script to create 7 MIG partitions
- **mig_flags.sh** - Flexible MIG management with command-line flags
- **mig_stress_test.sh** - Main stress test script that tests each MIG slice
- **run_stress_test_background.sh** - Helper script to run stress tests in the background

## MIG Setup

### Quick Setup (7 Partitions)

```bash
chmod +x mig_easy_setup.sh
./mig_easy_setup.sh
```

### Flexible Setup with Flags

```bash
chmod +x mig_flags.sh

./mig_flags.sh --enable # Enable MIG mode

./mig_flags.sh --delete # Delete old instances

./mig_flags.sh --create # Create 7 partitions

./mig_flags.sh --list # List all partitions

./mig_flags.sh --delete --create --list # Combine multiple operations
```

## Stress Testing

### Running the Stress Test

The stress test will:
- Test each MIG slice one at a time
- Allocate ~95% of available GPU memory
- Perform continuous memory operations for 30 minutes per device
- Log all activities and capture any abnormalities (crashes, memory errors, etc.)

#### Option 1: Run in Background (Recommended)

```bash
chmod +x run_stress_test_background.sh
./run_stress_test_background.sh
```

This will:
- Install PyTorch if not already installed
- Start the stress test in the background
- Provide commands to monitor progress
- Save logs to `./stress_test_logs/`

#### Option 2: Run in Foreground

```bash
chmod +x mig_stress_test.sh
./mig_stress_test.sh
```

### Monitoring the Stress Test

```bash
# Check if process is running
ps aux | grep mig_stress_test

# View live log output
tail -f stress_test_logs/stress_test_*.log

# View errors only
tail -f stress_test_logs/stress_test_errors_*.log

# Monitor GPU status
watch -n 5 nvidia-smi
```

### Understanding the Logs

**Main Log** (`stress_test_YYYYMMDD_HHMMSS.log`):
- Detailed information about each test phase
- Memory allocation progress
- Device status before/after testing
- Test iterations and progress updates

**Error Log** (`stress_test_errors_YYYYMMDD_HHMMSS.log`):
- Memory allocation failures
- Runtime errors or crashes
- System-level GPU errors from dmesg
- Any abnormalities detected during testing

### Requirements

- NVIDIA GPU with MIG support (e.g., GH200, A100, H100)
- NVIDIA drivers with MIG enabled
- Python 3.6+
- PyTorch (will be auto-installed if using `run_stress_test_background.sh`)

To manually install PyTorch:
```bash
pip3 install torch
```

### Test Duration

- **Per Device**: 30 minutes of continuous stress testing
- **Total Time**: 30 minutes × number of MIG devices + setup/cleanup time
- For 7 MIG partitions: approximately 3.5 hours total

### Stopping the Test

If running in background:
```bash
# Find the process ID
ps aux | grep mig_stress_test

# Stop the process
kill <PID>
```

Or use the PID saved in the log directory:
```bash
kill $(cat stress_test_logs/stress_test_*.pid)
```

### Interpreting Results

**Success**: 
- No entries in error log
- All devices complete 30-minute test
- Memory allocation reaches ~95% of device capacity
- No crashes or system errors

**Failure Indicators**:
- Entries in error log file
- Memory allocation errors
- Process crashes or hangs
- GPU errors in system logs (dmesg)
- Test terminates early

