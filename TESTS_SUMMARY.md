# MIG Stress Test Suite - Summary

## Folder Structure

```
mig_stress_test/
├── standard_test/          ✅ Sequential single-device testing
├── intense_test/           ✅ Multi-device simultaneous testing (RECOMMENDED)
├── thrashing_test/         ✅ Memory allocator stress testing
├── cuda_test/              ✅ CUDA API edge case testing
├── intense_thrashing_test/ ✅ Sustained high memory + fragmentation stress
├── pcie_test/              ✅ PCIe bandwidth fairness testing
├── multiproc_test/         ✅ Multi-process concurrent access testing
├── thermal_test/           ✅ Thermal shock cycling testing
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

### 5. Intense Thrashing Test
- **What:** 65% persistent base load + 20-30% rapid alloc/free on all devices
- **Why:** Sustained high memory pressure + fragmentation stress
- **Intensity:** ⭐⭐⭐⭐⭐ Extreme

### 6. PCIe Bandwidth Test
- **What:** Continuous H2D/D2H transfers (40% memory) on all devices simultaneously
- **Why:** Tests PCIe bandwidth fairness and contention between MIG partitions
- **Intensity:** ⭐⭐⭐⭐⭐ Very High

### 7. Multi-Process Test
- **What:** 4 worker processes per MIG device (compute/memory/transfer/streams)
- **Why:** Tests multi-process handling, isolation, and scheduler under heavy load
- **Intensity:** ⭐⭐⭐⭐⭐ Extreme

### 8. Thermal Shock Test
- **What:** 30s hot phase (90% memory + max compute) + 30s cold phase (idle)
- **Why:** Validates thermal management, power stability, clock throttling
- **Intensity:** ⭐⭐⭐⭐⭐ Very High (thermal cycling)

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

| Feature | Standard | Intense | Thrashing | CUDA API | Intense Thrashing | PCIe | Multi-Process | Thermal |
|---------|----------|---------|-----------|----------|-------------------|------|---------------|---------|
| **Focus** | Sequential stability | Multi-tenant | Memory allocator | API robustness | Sustained pressure | Bandwidth fairness | Multi-process isolation | Thermal mgmt |
| **Devices** | One at a time | All simultaneous | All simultaneous | One at a time | **All simultaneous** | **All simultaneous** | **All simultaneous** | **All simultaneous** |
| **Thermal** | Moderate | Very High | High | High | **Extreme** | **Very High** | **Extreme** | **Very High** |
| **Production Realism** | Low | **Very High** | Medium | Medium | **Very High** | **Very High** | **Very High** | **High** |

## Key Features

✅ **Automatic MIG Setup** - No manual configuration needed
✅ **Background Execution** - Run and monitor remotely
✅ **Comprehensive Logging** - Detailed logs with timestamps
✅ **Error Detection** - Captures crashes, OOM, system errors
✅ **PyTorch Auto-Install** - Handles dependencies automatically
✅ **Production-Ready** - 30 min per device, 3.5 hours total for 7 MIGs
✅ **8 Test Types** - Covers sequential, multi-tenant, memory, API, PCIe, multi-process, and thermal scenarios
✅ **Multi-Device Testing** - Most tests run on all MIG devices simultaneously

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

## New Advanced Tests (5-8)

The newer tests target specific production failure scenarios:

- **Intense Thrashing:** Combines sustained high memory (65% base) with fragmentation stress
- **PCIe Bandwidth:** Validates bandwidth fairness - critical for multi-tenant workloads
- **Multi-Process:** Tests scheduler with 28 concurrent processes (4 per MIG device)
- **Thermal Shock:** Power/thermal cycling to validate clock stability and throttling behavior

These tests are particularly useful for:
- Finding edge cases in production environments
- Validating multi-process isolation guarantees
- Testing PCIe bandwidth contention between MIG partitions
- Verifying thermal management under rapid power fluctuations