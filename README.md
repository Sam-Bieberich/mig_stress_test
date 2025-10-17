# MIG Stress Test Suite

Comprehensive stress testing suite for NVIDIA MIG (Multi-Instance GPU) partitions on GH200 and similar GPUs.

## Overview

This repository contains **four different types of stress tests**, each designed to validate MIG stability under different workload patterns. Each test automatically sets up MIG partitions and can be run in the background.

## Repository Structure

```
mig_stress_test/
├── standard_test/          # Sequential single-device stress test
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── standard_stress_test.sh
├── intense_test/           # Multi-device simultaneous stress test  ⚡ RECOMMENDED
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── intense_stress_test.sh
├── thrashing_test/         # Memory allocation/deallocation stress
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── thrashing_stress_test.sh
├── cuda_test/              # CUDA API edge cases and limits
│   ├── run_test.sh        # Setup MIG + run test in background
│   └── cuda_stress_test.sh
├── mig_easy_setup.sh       # Standalone MIG setup script
├── mig_flags.sh            # Flexible MIG management with flags
└── QUICK_REFERENCE.md      # Quick command reference
```

## Test Types

### 1. Standard Test (`standard_test/`)
- Tests each MIG slice **sequentially** (one at a time)
- Allocates 95% of device memory per device
- 30 minutes per device
- **Intensity:** ⭐⭐⭐ Moderate
- **Use case:** Isolate individual MIG slice issues

### 2. Intense Test (`intense_test/`) ⚡ **RECOMMENDED**
- One MIG slice at **95% memory** (primary)
- **All other slices at 75% simultaneously** (background)
- Rotates through each slice
- **Intensity:** ⭐⭐⭐⭐⭐ Very High
- **Use case:** Real-world multi-tenant production simulation

### 3. Thrashing Test (`thrashing_test/`)
- Rapid memory allocation/deallocation cycles
- Variable chunk sizes and random patterns
- Tests memory allocator robustness
- **Intensity:** ⭐⭐⭐⭐ High
- **Use case:** Memory fragmentation and allocator stress

### 4. CUDA Test (`cuda_test/`)
- Maximum CUDA streams (128+)
- Unusual tensor shapes and operations
- API edge cases and limits
- **Intensity:** ⭐⭐⭐⭐ High
- **Use case:** CUDA API robustness validation

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
```

Each `run_test.sh` script:
- ✅ Checks for existing MIG partitions
- ✅ Installs PyTorch (if needed)
- ✅ Runs test in background
- ✅ Provides monitoring commands

**Note:** MIG partitions only need to be created once. You can run multiple tests on the same MIG setup.
./mig_easy_setup.sh

# Flexible setup with flags
chmod +x mig_flags.sh
./mig_flags.sh --enable --delete --create --list
```

## Workflow Summary

```
1. Set up MIG partitions (one time):
   ./mig_easy_setup.sh

2. Run tests (can run multiple times on same MIG setup):
   cd standard_test/ && ./run_test.sh
   cd intense_test/ && ./run_test.sh
   cd thrashing_test/ && ./run_test.sh
   cd cuda_test/ && ./run_test.sh

3. Monitor:
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

| Feature | Standard | Intense | Thrashing | CUDA API |
|---------|----------|---------|-----------|----------|
| **Primary Focus** | Sequential stability | Multi-tenant load | Memory allocator | API robustness |
| **Memory Pattern** | Static 95% | Mixed 95%/75% | Rapid alloc/free | Variable patterns |
| **Device Usage** | One at a time | All simultaneous | One at a time | One at a time |
| **Thermal Load** | Moderate | Very High | High | High |
| **Production Realism** | Low | **Very High** | Medium | Medium |
| **Best For** | Baseline testing | Production validation | Memory stress | Edge case detection |

## Which Test Should I Run?

### For Production Validation: 🏆
**Run the Intense Test** (`intense_test/`) - Best simulates real-world multi-tenant scenarios.

### For Comprehensive Testing: 🎯
Run all four tests in sequence:
1. Standard (baseline)
2. Intense (production simulation)  
3. Thrashing (memory stress)
4. CUDA API (edge cases)

### For Quick Validation: ⚡
Run Standard first, then Intense if Standard passes.

### Log Files

Each test creates logs in its `logs/` directory:

```
test_folder/logs/
├── background_TIMESTAMP.log              # Background process output
├── <test_type>_test_TIMESTAMP.log        # Main test log
└── <test_type>_test_errors_TIMESTAMP.log # Errors and warnings
```

**Log Contents:**
- Memory allocation progress
- Device status before/after testing
- Test iterations and progress updates
- Background worker status (intense test)
- Errors, warnings, and abnormalities
- System-level GPU errors from dmesg

### Requirements

- NVIDIA GPU with MIG support (e.g., GH200, A100, H100)
- NVIDIA drivers with MIG enabled
- Python 3.6+
- PyTorch (will be auto-installed if using `run_stress_test_background.sh`)

To manually install PyTorch:
```bash
pip3 install torch
```

## Test Duration

All tests run for **30 minutes per device**:

| Test Type | Per Device | Total (7 MIGs) | Notes |
|-----------|-----------|----------------|-------|
| Standard | 30 min | ~3.5 hours | Sequential testing |
| Intense | 30 min | ~3.5 hours | All devices active |
| Thrashing | 30 min | ~3.5 hours | Rapid alloc/free cycles |
| CUDA API | 30 min | ~3.5 hours | API edge case testing |

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

## Additional Resources

- **QUICK_REFERENCE.md** - Command quick reference guide
- See individual test folders for test-specific details

## Contributing

When adding new tests, follow the existing structure:
1. Create folder: `<test_name>_test/`
2. Add `run_test.sh` (includes MIG setup + background execution)
3. Add `<test_name>_stress_test.sh` (main test logic)
4. Update this README

## License

This is a testing suite for NVIDIA MIG validation. Use at your own risk.
