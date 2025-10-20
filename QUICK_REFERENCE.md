# MIG Stress Test Quick Reference

## Quick Start

### Standard Test (Sequential, One Device at a Time)
```bash
chmod +x run_stress_test_background.sh
./run_stress_test_background.sh
```
There are a suite of stress tests, and the process can be replicated in each directory. 

## Monitoring

```bash
# Check process status
ps aux | grep stress

# Live logs
tail -f stress_test_logs/intense_stress_*.log

# Errors only
tail -f stress_test_logs/*_errors_*.log

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

# For intense test, also kill workers
pkill -f mig_intense_stress.py

# Or use saved PID
kill $(cat stress_test_logs/intense_stress_*.pid)
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

### Background workers fail (Intense test)
- Check MIG configuration: `nvidia-smi -L`
- Verify all UUIDs are valid
- Ensure sufficient memory on all slices

### High temperatures
- Monitor: `nvidia-smi -q -d TEMPERATURE`
- Check cooling system
- Intense test generates more heat

### Process hangs
- Check system logs: `dmesg | tail -100`
- May indicate driver/hardware issues
- Try reducing test duration

## Log Files

```
stress_test_logs/
├── stress_test_TIMESTAMP.log          # Standard test main log
├── stress_test_errors_TIMESTAMP.log   # Standard test errors
├── intense_stress_TIMESTAMP.log       # Intense test main log
├── intense_stress_errors_TIMESTAMP.log # Intense test errors
└── intense_stress_python.log          # Worker process logs
```

## Useful Commands

```bash
# List all MIG devices
nvidia-smi -L

# Check MIG memory usage
nvidia-smi mig -lgi

# Monitor temperature continuously
watch -n 1 nvidia-smi -q -d TEMPERATURE

# Check for GPU errors in system
dmesg | grep -i nvidia | tail -50

# View all test logs
ls -lh stress_test_logs/

# Count errors across all logs
grep -i error stress_test_logs/*.log | wc -l
```