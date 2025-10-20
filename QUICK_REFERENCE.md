# MIG Stress Test Quick Reference

## Quick Start

### Choose a Test Type

```bash
# Standard Test (Sequential, one device at a time)
cd standard_test/ && ./run_test.sh

# Intense Test (All devices simultaneously - RECOMMENDED)
cd intense_test/ && ./run_test.sh

# Thrashing Test (Memory allocator stress)
cd thrashing_test/ && ./run_test.sh

# CUDA API Test (Edge cases and limits)
cd cuda_test/ && ./run_test.sh

# Intense Thrashing Test (Sustained high memory + fragmentation)
cd intense_thrashing_test/ && ./run_test.sh

# PCIe Bandwidth Test (Bandwidth fairness between MIG partitions)
cd pcie_test/ && ./run_test.sh

# Multi-Process Test (Multi-process handling, 4 workers per device)
cd multiproc_test/ && ./run_test.sh

# Thermal Shock Test (Hot/cold thermal cycling)
cd thermal_test/ && ./run_test.sh
```

All tests automatically set up MIG partitions and run in the background. 

## Monitoring

```bash
# Check process status
ps aux | grep stress

# Live logs (adjust folder name for your test)
tail -f logs/background_*.log
tail -f logs/*_test_*.log

# Errors only
tail -f logs/*_errors_*.log

# GPU monitoring
watch -n 2 nvidia-smi

# Detailed GPU stats
watch -n 2 'nvidia-smi --query-gpu=index,name,temperature.gpu,memory.used,memory.total,utilization.gpu --format=csv'
```

## Stopping Tests

```bash
# Find and kill process
ps aux | grep stress
kill <PID>

# For multi-worker tests (intense, thrashing, multiproc, etc.)
pkill -f stress.py
pkill -f stress_test.sh

# Or use saved PID (adjust folder name)
kill $(cat logs/*_test_*.pid)
```

## Understanding Results

### ✅ Good Signs
- No error log entries
- Reaches target memory percentages
- Completes all 30-minute cycles
- Stable temperatures
- Consistent iteration counts

### ❌ Warning Signs
- Error log has entries
- Memory allocation fails below target
- Process crashes or hangs
- GPU errors in dmesg
- Temperature throttling
- Early termination

## Common Issues

### "Out of Memory" errors
- **Expected** when reaching 95% limit
- **Problem** if happening well below target
- Solution: Check `nvidia-smi` for other processes

### Background workers fail (Multi-worker tests)
- Check MIG configuration: `nvidia-smi -L`
- Verify all UUIDs are valid
- Ensure sufficient memory on all slices
- Check Python environment is accessible

### High temperatures
- Monitor: `nvidia-smi -q -d TEMPERATURE`
- Check cooling system
- Intense, multi-process, and thermal tests generate most heat
- Thermal test intentionally cycles between hot and cold

### Process hangs
- Check system logs: `dmesg | tail -100`
- May indicate driver/hardware issues
- Try reducing test duration

## Log Files

Each test creates logs in its own `logs/` directory:

```
<test_folder>/logs/
├── background_TIMESTAMP.log              # Background process output
├── <test_type>_test_TIMESTAMP.log        # Main test log
├── <test_type>_test_errors_TIMESTAMP.log # Errors and warnings
└── Device logs for multi-device tests
```

## Useful Commands

```bash
# List all MIG devices
nvidia-smi -L

# Check MIG memory usage
nvidia-smi mig -lgi

# Monitor temperature continuously
watch -n 1 nvidia-smi -q -d TEMPERATURE

# Monitor power consumption
watch -n 1 nvidia-smi -q -d POWER

# Check for GPU errors in system
dmesg | grep -i nvidia | tail -50

# View all test logs
find . -name "*.log" -mtime -1 -exec ls -lh {} \;

# Count errors across all logs
find . -name "*errors*.log" -exec grep -i error {} \; | wc -l
```

## Test Recommendations

- **Production Validation:** Run `intense_test`, `pcie_test`, and `multiproc_test`
- **Memory Stress:** Run `thrashing_test` and `intense_thrashing_test`
- **Thermal Validation:** Run `thermal_test` and monitor temperature deltas
- **Comprehensive Check:** Run all 8 tests sequentially for full validation