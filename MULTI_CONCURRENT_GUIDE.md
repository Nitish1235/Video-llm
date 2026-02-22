# Multi-Concurrent Generation Guide

## Overview

Yes! One worker **can** handle multiple requests simultaneously if the GPU has enough memory. The system now supports this with automatic detection and manual configuration.

## How It Works

### Auto-Detection (Default)

The system automatically detects your GPU memory and sets optimal concurrency:

- **A100 40GB+**: `MAX_CONCURRENT_GENERATIONS = 2` (can handle 2 videos simultaneously)
- **A10G/RTX 3090 24GB**: `MAX_CONCURRENT_GENERATIONS = 1` (safe default)
- **Smaller GPUs**: `MAX_CONCURRENT_GENERATIONS = 1` (conservative)

### Manual Configuration

You can override auto-detection by setting the environment variable:

```bash
# In Runpod environment variables:
MAX_CONCURRENT_GENERATIONS=2  # Allow 2 concurrent generations per worker
MAX_CONCURRENT_GENERATIONS=3  # Allow 3 concurrent generations per worker
```

## Performance Impact

### Example: 20 Requests with Different Configurations

#### Configuration 1: MAX_CONCURRENT_GENERATIONS=1 (Default)
```
5 Workers × 1 concurrent = 5 videos processing simultaneously
Time to complete 20 videos: ~4 batches × 60s = ~240 seconds
Throughput: ~5 videos/minute
```

#### Configuration 2: MAX_CONCURRENT_GENERATIONS=2 (A100 40GB)
```
5 Workers × 2 concurrent = 10 videos processing simultaneously
Time to complete 20 videos: ~2 batches × 60s = ~120 seconds
Throughput: ~10 videos/minute
```

#### Configuration 3: MAX_CONCURRENT_GENERATIONS=3 (A100 40GB, aggressive)
```
5 Workers × 3 concurrent = 15 videos processing simultaneously
Time to complete 20 videos: ~1.3 batches × 60s = ~80 seconds
Throughput: ~15 videos/minute
```

## GPU Memory Requirements

### Per Video Generation
- **Model Loading**: ~8-10 GB (one-time, cached)
- **Video Generation**: ~4-6 GB per concurrent generation
- **Total per worker**: ~8GB base + (4-6GB × concurrent_count)

### Recommended Settings

| GPU Type | GPU Memory | Recommended MAX_CONCURRENT_GENERATIONS | Max Safe |
|----------|------------|----------------------------------------|----------|
| A100 40GB | 40 GB | 2 | 3 |
| A10G 24GB | 24 GB | 1 | 2 |
| RTX 3090 | 24 GB | 1 | 2 |
| RTX 4090 | 24 GB | 1 | 2 |

## How to Configure

### Option 1: Let Auto-Detection Handle It (Recommended)

Just deploy - the system will detect your GPU and set optimal values automatically.

### Option 2: Manual Override

**In Runpod Environment Variables:**
```
MAX_CONCURRENT_GENERATIONS=2
```

**For Testing:**
```bash
# Test with 2 concurrent
MAX_CONCURRENT_GENERATIONS=2

# Test with 3 concurrent (only for A100 40GB+)
MAX_CONCURRENT_GENERATIONS=3
```

## Monitoring

### Check Current Setting

Look for this log on startup:
```
Auto-detected MAX_CONCURRENT_GENERATIONS=2 for GPU with 40.0GB
Initialized with MAX_CONCURRENT_GENERATIONS=2
```

### Monitor GPU Memory

The system logs GPU memory usage per request:
```
[Request abc123] GPU memory: allocated=12.5GB, reserved=15.2GB, available=24.8GB
```

### Watch for Warnings

If memory gets low, you'll see:
```
WARNING: Low GPU memory available (3.5GB). Consider reducing MAX_CONCURRENT_GENERATIONS.
```

### OOM Errors

If you see GPU OOM errors:
```
ERROR: GPU out of memory during video generation
ERROR: GPU memory at OOM - allocated=38.5GB, reserved=39.2GB. 
       Consider reducing MAX_CONCURRENT_GENERATIONS from 3 to 2
```

**Solution**: Reduce `MAX_CONCURRENT_GENERATIONS` by 1

## Best Practices

### 1. Start Conservative
- Begin with auto-detection (default)
- Monitor for a few hours
- Check for OOM errors

### 2. Increase Gradually
- If no OOM errors, try `MAX_CONCURRENT_GENERATIONS=2`
- Monitor memory usage
- If stable, you can try 3 (A100 only)

### 3. Monitor Closely
- Watch GPU memory logs
- Check for timeout errors
- Monitor request completion times

### 4. Balance Performance vs Stability
- Higher concurrency = faster throughput
- But higher risk of OOM errors
- Find the sweet spot for your use case

## Real-World Example

### Scenario: A100 40GB GPU, 20 Requests

**With MAX_CONCURRENT_GENERATIONS=1:**
- 5 workers × 1 = 5 concurrent
- Time: ~240 seconds
- Cost: Same (5 workers)

**With MAX_CONCURRENT_GENERATIONS=2:**
- 5 workers × 2 = 10 concurrent
- Time: ~120 seconds (2x faster!)
- Cost: Same (5 workers)
- **Better GPU utilization**

**With MAX_CONCURRENT_GENERATIONS=3:**
- 5 workers × 3 = 15 concurrent
- Time: ~80 seconds (3x faster!)
- Risk: Higher chance of OOM
- **Maximum throughput**

## Troubleshooting

### Problem: GPU OOM Errors

**Symptoms:**
- `CUDA out of memory` errors
- Requests failing randomly

**Solution:**
```bash
# Reduce concurrency
MAX_CONCURRENT_GENERATIONS=1
```

### Problem: Requests Timing Out

**Symptoms:**
- `TIMEOUT` error code
- Requests waiting too long for semaphore

**Solution:**
- Increase `Max Workers` in Runpod
- Or increase `MAX_CONCURRENT_GENERATIONS` (if GPU has memory)

### Problem: Low GPU Utilization

**Symptoms:**
- GPU memory usage < 50%
- Requests processing sequentially

**Solution:**
```bash
# Increase concurrency to better utilize GPU
MAX_CONCURRENT_GENERATIONS=2  # or 3 for A100
```

## Cost Optimization

### Current Setup (MAX_CONCURRENT_GENERATIONS=1)
- 5 workers needed for 5 concurrent videos
- Cost: 5 × $1.39/hour = $6.95/hour

### Optimized Setup (MAX_CONCURRENT_GENERATIONS=2)
- 3 workers needed for 6 concurrent videos
- Cost: 3 × $1.39/hour = $4.17/hour
- **40% cost savings!**

### Maximum Throughput (MAX_CONCURRENT_GENERATIONS=3)
- 2 workers needed for 6 concurrent videos
- Cost: 2 × $1.39/hour = $2.78/hour
- **60% cost savings!**

## Summary

✅ **Yes, one worker can handle multiple requests!**

- **Auto-detects** optimal settings based on GPU
- **Manually configurable** via `MAX_CONCURRENT_GENERATIONS`
- **Monitors** GPU memory and warns if low
- **Handles OOM** gracefully with helpful error messages

**Recommended:**
- Start with auto-detection
- Monitor for 24 hours
- Adjust based on your workload and GPU capacity
