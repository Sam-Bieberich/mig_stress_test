# MIG Stress Test Suite - Summary

## Folder Structure

```
mig_stress_test/
├── standard_test/          ✅ Sequential single-device testing
├── intense_test/           ✅ Multi-device simultaneous testing (RECOMMENDED)
├── thrashing_test/         ✅ Memory allocator stress testing
├── cuda_test/              ✅ CUDA API edge case testing
├── mig_easy_setup.sh       (standalone MIG setup)
├── mig_flags.sh            (flexible MIG management)
├── README.md               (complete documentation)
└── QUICK_REFERENCE.md      (command reference)
```

## Test Descriptions

### 1. Standard Test
- **What:** Tests one MIG slice at a time at 95% memory
- **Why:** Baseline testing, isolate individual slice issues
- **Intensity:** ⭐⭐⭐ Moderate

### 2. Intense Test ⚡ **RECOMMENDED**
- **What:** One slice at 95%, all others at 75% simultaneously
- **Why:** Simulates real-world multi-tenant production workloads
- **Intensity:** ⭐⭐⭐⭐⭐ Very High

### 3. Thrashing Test
- **What:** Rapid memory allocation/deallocation cycles
- **Why:** Tests memory allocator robustness and fragmentation handling
- **Intensity:** ⭐⭐⭐⭐ High

### 4. CUDA Test
- **What:** Maximum CUDA streams, unusual tensor shapes, API edge cases
- **Why:** Validates CUDA API robustness and scheduler limits
- **Intensity:** ⭐⭐⭐⭐ High

## How to Run

Each test folder has a `run_test.sh` that does EVERYTHING:

```bash
cd <test_name>_test/
chmod +x run_test.sh
./run_test.sh
```

This automatically:
1. Enables MIG mode
2. Deletes old MIG instances
3. Creates 7 MIG partitions
4. Installs PyTorch (if needed)
5. Runs test in background
6. Provides monitoring commands

## Quick Commands

```bash
# Run recommended test for production validation
cd intense_test/ && ./run_test.sh

# Monitor progress
tail -f logs/background_*.log

# Check GPU status
watch -n 2 nvidia-smi

# Stop test
kill $(cat logs/*_test_*.pid)
```

## Test Comparison

| Feature | Standard | Intense | Thrashing | CUDA API |
|---------|----------|---------|-----------|----------|
| **Focus** | Sequential stability | Multi-tenant | Memory allocator | API robustness |
| **Devices** | One at a time | All simultaneous | One at a time | One at a time |
| **Thermal** | Moderate | Very High | High | High |
| **Production Realism** | Low | **Very High** | Medium | Medium |

## Key Features

✅ **Automatic MIG Setup** - No manual configuration needed
✅ **Background Execution** - Run and monitor remotely
✅ **Comprehensive Logging** - Detailed logs with timestamps
✅ **Error Detection** - Captures crashes, OOM, system errors
✅ **PyTorch Auto-Install** - Handles dependencies automatically
✅ **Production-Ready** - 30 min per device, 3.5 hours total for 7 MIGs

## File Organization

Each test folder contains:
- `run_test.sh` - Main entry point (setup + run in background)
- `<test_name>_stress_test.sh` - Core test logic
- `logs/` - Created automatically with timestamped logs. Not included in repository

## Success Criteria

✅ No entries in error log files
✅ All devices complete 30-minute cycles
✅ Memory reaches target percentages
✅ No crashes or system errors
✅ Stable temperatures