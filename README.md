# MIG Stress Test Suite

Comprehensive stress testing suite for NVIDIA MIG (Multi-Instance GPU) partitions on GH200 and similar GPUs.

## Overview

This repository contains **eight different types of stress tests**, each designed to validate MIG stability under different workload patterns. Each test automatically sets up MIG partitions and can be run in the background.

## Repository Structure

```
mig_stress_test/
├── standard_test/          # Sequential single-device stress test
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── standard_stress_test.sh
├── intense_test/           # Multi-device simultaneous stress test 
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── intense_stress_test.sh
├── thrashing_test/         # Memory allocation/deallocation stress
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── thrashing_stress_test.sh
├── cuda_test/              # CUDA API edge cases and limits
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── cuda_stress_test.sh
├── intense_thrashing_test/ # Sustained high memory + thrashing
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── intense_thrashing_stress_test.sh
├── pcie_test/              # PCIe bandwidth saturation test
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── pcie_stress_test.sh
├── multiproc_test/         # Multi-process concurrent access
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── multiproc_stress_test.sh
├── thermal_test/           # Thermal shock cycling test
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── thermal_stress_test.sh
├── mig_easy_setup.sh       # Standalone MIG setup script
├── mig_flags.sh            # Flexible MIG management with flags
└── QUICK_REFERENCE.md      # Quick command reference
```

## Test Types

### 1. Standard Test (`standard_test/`)
- Tests each MIG slice **sequentially** (one at a time)
- Allocates 95% of device memory per device
- 30 minutes per device
- **Intensity:** Moderate
- **Use case:** Isolate individual MIG slice issues

### 2. Intense Test (`intense_test/`) ⚡ **RECOMMENDED**
- One MIG slice at **95% memory** (primary)
- **All other slices at 75% simultaneously** (background)
- Rotates through each slice
- **Intensity:** Very High
- **Use case:** Real-world multi-tenant production simulation

### 3. Thrashing Test (`thrashing_test/`)
- Rapid memory allocation/deallocation cycles
- Variable chunk sizes (10MB to 1GB) and random patterns
- Tests memory allocator robustness
- **Runs on ALL MIG slices simultaneously** 
- 70% fragmentation per cycle with mixed compute operations
- **Intensity:** Very High
- **Use case:** Memory fragmentation and allocator stress under concurrent load

### 4. CUDA Test (`cuda_test/`)
- Maximum CUDA streams (128+)
- Unusual tensor shapes and operations
- API edge cases and limits
- **Intensity:** High
- **Use case:** CUDA API robustness validation

### 5. Intense Thrashing Test (`intense_thrashing_test/`)
- 65% persistent base memory load per device
- 20-30% additional thrashing space for rapid alloc/free
- **Runs on ALL MIG slices simultaneously**
- Maintains sustained high memory usage (~85%)
- **Intensity:** Extreme
- **Use case:** Sustained memory pressure + fragmentation stress

### 6. PCIe Bandwidth Test (`pcie_test/`)
- Tests PCIe bandwidth fairness between MIG partitions
- 40% memory allocated for continuous H2D/D2H transfers
- Measures bidirectional bandwidth per device (GB/s)
- **Runs on ALL MIG slices simultaneously**
- **Intensity:** Very High
- **Use case:** PCIe contention and bandwidth isolation validation

### 7. Multi-Process Test (`multiproc_test/`)
- 4 worker processes per MIG device (28 total)
- Each worker type: compute, memory, transfer, or streams
- Tests multi-process handling and isolation
- **Runs on ALL MIG slices simultaneously**
- **Intensity:** Extreme
- **Use case:** Production multi-process scenarios, scheduler stress

### 8. Thermal Shock Test (`thermal_test/`)
- 30-second HOT phase (90% memory + max compute)
- 30-second COLD phase (idle)
- Monitors temperature deltas and power fluctuations
- **Runs on ALL MIG slices simultaneously**
- **Intensity:** Very High (thermal cycling)
- **Use case:** Thermal management, power stability, clock throttling validation

## Quick Start

### Step 1: Set Up MIG Partitions (One Time)

First, create MIG partitions using the setup script:

```bash
# Quick setup (creates 7 MIG partitions)
chmod +x mig_easy_setup.sh
./mig_easy_setup.sh
```

Or use the flexible flags script:

```bash
chmod +x mig_flags.sh
./mig_flags.sh --enable --delete --create --list
```

### Step 2: Run Tests

Once MIG partitions are created, run any test:

```bash
# Standard Test (sequential, one device at a time)
cd standard_test/
chmod +x run_test.sh
./run_test.sh

# Intense Test (all devices simultaneously) - RECOMMENDED FOR PRODUCTION
cd intense_test/
chmod +x run_test.sh
./run_test.sh

# Thrashing Test (memory allocator stress)
cd thrashing_test/
chmod +x run_test.sh
./run_test.sh

# CUDA API Test (edge cases and limits)
cd cuda_test/
chmod +x run_test.sh
./run_test.sh

# Intense Thrashing Test (sustained high memory + fragmentation)
cd intense_thrashing_test/
chmod +x run_test.sh
./run_test.sh

# PCIe Bandwidth Test (bandwidth fairness testing)
cd pcie_test/
chmod +x run_test.sh
./run_test.sh

# Multi-Process Test (multi-process handling)
cd multiproc_test/
chmod +x run_test.sh
./run_test.sh

# Thermal Shock Test (thermal cycling)
cd thermal_test/
chmod +x run_test.sh
./run_test.sh
```

Each `run_test.sh` script:
- ✅ Checks for existing MIG partitions
- ✅ Installs PyTorch (if needed)
- ✅ Runs test in background
- ✅ Provides monitoring commands

**Note:** MIG partitions only need to be created once. You can run multiple tests on the same MIG setup.

## Test Validation Status

| Test Type | 3-Minute Test | 30-Minute Test | Notes |
|-----------|---------------|----------------|-------|
| **Standard** | ✅ Completed | ✅ Completed | No errors - validated on GH200 |
| **Intense** | ✅ Completed | ⏳ Pending | 3-min validated on GH200 |
| **Thrashing** | ✅ Completed | ⏳ Pending | 3-min validated - no errors |
| **CUDA API** | ✅ Completed | ✅ Completed | No errors - validated on GH200 |
| **Intense Thrashing** | ✅ Completed | ⏳ Pending | 3-min validated - no errors |
| **PCIe Bandwidth** | ⏳ Pending | ⏳ Pending | New test - not yet validated |
| **Multi-Process** | ⏳ Pending | ⏳ Pending | New test - not yet validated |
| **Thermal Shock** | ⏳ Pending | ⏳ Pending | New test - not yet validated |

**Legend:**
- ✅ **Completed** - Test has been run and validated successfully
- ⏳ **Pending** - Test not yet run or validation pending
- ❌ **Failed** - Test encountered issues (if any)

**Test Configuration:**
- Hardware: GH200 Grace Hopper
- MIG Partitions: 7 instances (profile 19)
- Current test duration: 3 minutes per MIG device
- Full test duration: 30 minutes per MIG device (pending)

## Workflow Summary

```bash
# 1. Set up MIG partitions (one time):
./mig_easy_setup.sh

# 2. Run tests (can run multiple times on same MIG setup):
cd standard_test/ && ./run_test.sh
cd intense_test/ && ./run_test.sh
cd thrashing_test/ && ./run_test.sh
cd cuda_test/ && ./run_test.sh
cd intense_thrashing_test/ && ./run_test.sh
cd pcie_test/ && ./run_test.sh
cd multiproc_test/ && ./run_test.sh
cd thermal_test/ && ./run_test.sh

# 3. Monitor:
tail -f <test_folder>/logs/background_*.log
watch -n 2 nvidia-smi
```

## Monitoring Tests

All tests run in the background and create logs in their respective `logs/` directories:

```bash
# Check if process is running
ps aux | grep stress

# View live log output (from any test directory)
tail -f logs/background_*.log

# View test progress
tail -f logs/*_test_*.log

# View errors only
tail -f logs/*_errors_*.log

# Monitor ALL GPUs in real-time
watch -n 2 nvidia-smi

# Monitor memory usage
watch -n 2 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv'
```

## Test Comparison

| Feature | Standard | Intense | Thrashing | CUDA API | Intense Thrashing | PCIe | Multi-Process | Thermal |
|---------|----------|---------|-----------|----------|-------------------|------|---------------|---------|
| **Primary Focus** | Sequential stability | Multi-tenant load | Memory allocator | API robustness | Sustained pressure | Bandwidth fairness | Multi-process isolation | Thermal management |
| **Memory Pattern** | Static 95% | Mixed 95%/75% | Rapid alloc/free | Variable patterns | 65% base + 20-30% thrash | 40% transfer buffers | 4 workers/device | 90% hot / 0% cold |
| **Device Usage** | One at a time | All simultaneous | **All simultaneous** | One at a time | **All simultaneous** | **All simultaneous** | **All simultaneous** | **All simultaneous** |
| **Thermal Load** | Moderate | Very High | **Very High** | High | **Extreme** | **Very High** | **Extreme** | **Very High (cycling)** |
| **Production Realism** | Low | **Very High** | **High** | Medium | **Very High** | **Very High** | **Very High** | **High** |
| **Best For** | Baseline testing | Production validation | Memory stress | Edge case detection | Sustained load + fragmentation | PCIe contention | Scheduler stress | Power/thermal validation |

### Log Files

Each test creates logs in its `logs/` directory:

```
test_folder/logs/
├── background_TIMESTAMP.log              # Background process output
├── <test_type>_test_TIMESTAMP.log        # Main test log
└── <test_type>_test_errors_TIMESTAMP.log # Errors and warnings
```

## Stopping Tests

```bash
# Find the process
ps aux | grep <test_type>

# Stop the process
kill <PID>

# For intense test, also kill workers
pkill -f intense_stress.py

# Or use saved PID (from test's logs/ directory)
cd <test_folder>/
kill $(cat logs/*_test_*.pid)
```

## Understanding Results

### ✅ Success Indicators
- No entries in error log files
- All devices complete 30-minute test cycles
- Memory allocation reaches target percentages
- No crashes or system errors
- Stable temperatures
- Consistent iteration counts

### ❌ Failure Indicators
- Entries in error log files
- Memory allocation fails below target
- Process crashes or hangs
- GPU errors in dmesg
- Temperature throttling
- Early termination

## Troubleshooting

### Out of Memory Errors
- **Expected** when reaching allocation limits (95%)
- **Problem** if happening well below target
- Check other processes: `nvidia-smi`

### Background Workers Fail (Intense Test)
- Verify MIG configuration: `nvidia-smi -L`
- Ensure all MIG UUIDs are valid
- Check for sufficient memory on all slices

### High Temperatures
- Monitor: `nvidia-smi -q -d TEMPERATURE`
- Intense test generates most heat
- Ensure adequate cooling

### Process Hangs
- Check system logs: `dmesg | tail -100`
- May indicate driver/hardware issues
- Try reducing test duration

### MIG Setup Fails
- Ensure GPU supports MIG
- Check if MIG enabled: `nvidia-smi -i 0 --query-gpu=mig.mode.current --format=csv`
- May need system reboot after enabling MIG

## Requirements

- NVIDIA GPU with MIG support (GH200, A100, H100, etc.)
- NVIDIA drivers with MIG enabled
- Python 3.6+
- PyTorch (auto-installed by `run_test.sh` scripts)
- sudo access (for MIG configuration)

### Manual PyTorch Installation

```
module load gcc cuda
module load python3
```

## Useful Commands

```bash
# List all MIG devices
nvidia-smi -L

# Check MIG configuration
nvidia-smi mig -lgi

# Monitor temperature continuously
watch -n 1 'nvidia-smi --query-gpu=temperature.gpu --format=csv'

# Check for GPU errors in system logs
dmesg | grep -i nvidia | tail -50

# View all test logs
find . -name "*.log" -mtime -1 -exec ls -lh {} \;

# Count errors across all logs
find . -name "*errors*.log" -exec grep -i error {} \; | wc -l
```