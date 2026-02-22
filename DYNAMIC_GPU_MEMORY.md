# Dynamic GPU Memory Detection

## How It Works

The service now **dynamically checks GPU memory in real-time** to determine if it can handle more requests. It doesn't just rely on a static limit - it adapts based on actual GPU memory availability.

## Two-Layer Protection

### Layer 1: Static Semaphore Limit
- **Purpose**: Prevents too many requests from starting
- **Based on**: GPU type detection at startup
- **Example**: A100 40GB → MAX_CONCURRENT_GENERATIONS=2

### Layer 2: Dynamic Memory Check (NEW!)
- **Purpose**: Real-time GPU memory monitoring
- **Checks**: Available memory before each request
- **Action**: Waits if memory is low, proceeds when available

## How Dynamic Detection Works

### Step-by-Step Flow

```
1. Request arrives
   ↓
2. Acquire semaphore slot (Layer 1)
   ↓
3. Check GPU memory (Layer 2) ← NEW!
   ├─ If memory available (≥8GB): Proceed immediately
   └─ If memory low (<8GB): Wait and retry
   ↓
4. Process request
   ↓
5. Release semaphore + Free GPU memory
```

### Memory Check Functions

#### 1. `get_available_gpu_memory()`
```python
# Gets real-time available GPU memory
available = total_memory - reserved_memory
# Returns: 24.5 GB (example)
```

#### 2. `can_handle_more_requests()`
```python
# Checks if enough memory for another generation
required = MIN_MEMORY_REQUIRED (6GB) + SAFE_MEMORY_BUFFER (2GB) = 8GB
# Returns: True if available ≥ 8GB, False otherwise
```

#### 3. `wait_for_gpu_memory()`
```python
# Waits for memory to free up
# Checks every 1 second
# Timeout: 2 minutes
# Returns: True when memory available, False if timeout
```

## Real-World Scenarios

### Scenario 1: Normal Operation
```
Request 1 arrives → Semaphore acquired → Memory check: 20GB available ✅ → Process
Request 2 arrives → Semaphore acquired → Memory check: 12GB available ✅ → Process
Request 3 arrives → Semaphore acquired → Memory check: 4GB available ❌ → Wait...
Request 1 completes → Frees 6GB → Memory now 10GB ✅ → Request 3 proceeds
```

### Scenario 2: High Load
```
Request 1 arrives → Processing (using 6GB)
Request 2 arrives → Processing (using 6GB)
Request 3 arrives → Semaphore acquired → Memory check: 2GB available ❌
                   → Waits 5 seconds → Memory check: 2GB ❌
                   → Waits 10 seconds → Memory check: 8GB ✅ → Proceeds
```

### Scenario 3: Memory Pressure
```
Request 1 arrives → Processing
Request 2 arrives → Processing
Request 3 arrives → Semaphore acquired → Memory: 1GB ❌ → Wait...
Request 4 arrives → Semaphore acquired → Memory: 1GB ❌ → Wait...
Request 1 completes → Frees memory → Request 3 proceeds
Request 2 completes → Frees memory → Request 4 proceeds
```

## Configuration

### Memory Thresholds

```python
MIN_MEMORY_REQUIRED = 6.0  # GB needed per generation
SAFE_MEMORY_BUFFER = 2.0   # GB safety buffer
# Total required: 8GB available before allowing new request
```

### Adjustable via Code

You can modify these in `handler.py`:
```python
MIN_MEMORY_REQUIRED = 5.0  # Lower threshold (more aggressive)
SAFE_MEMORY_BUFFER = 3.0   # Higher buffer (more conservative)
```

## Logs to Monitor

### When Memory is Available
```
[Request abc123] Acquired slot. GPU memory available: 24.50GB
[Request abc123] GPU memory sufficient for additional requests
```

### When Memory is Low
```
[Request abc123] Low GPU memory (3.50GB). Waiting for memory to free up...
[Request abc123] GPU memory available. Proceeding with generation.
```

### When Memory Times Out
```
[Request abc123] Timeout waiting for GPU memory. Current available: 2.10GB
ERROR: GPU_MEMORY_UNAVAILABLE
```

## Benefits

### ✅ Prevents OOM Errors
- Checks memory **before** starting generation
- Waits if memory is insufficient
- Never starts if memory is too low

### ✅ Maximizes Throughput
- Doesn't wait unnecessarily
- Proceeds immediately when memory is available
- Adapts to actual GPU state

### ✅ Self-Regulating
- No manual tuning needed
- Automatically adapts to workload
- Handles memory spikes gracefully

## Example: 20 Requests with Dynamic Detection

### Without Dynamic Detection (Old)
```
20 requests → All acquire semaphore → 5 start processing
→ 3 hit OOM errors → Fail
→ Remaining 17 wait → Process slowly
Result: 3 failures, slow processing
```

### With Dynamic Detection (New)
```
20 requests → Acquire semaphore → Check memory
→ 5 have memory → Start immediately
→ 15 wait → Memory frees → Check again
→ Next 5 have memory → Start
→ Continue until all processed
Result: 0 failures, optimal throughput
```

## Monitoring

### Key Metrics to Watch

1. **Memory Wait Time**
   - How long requests wait for memory
   - Should be < 30 seconds typically

2. **Memory Availability**
   - Current available GPU memory
   - Should stay above 8GB for smooth operation

3. **OOM Errors**
   - Should be **zero** with dynamic detection
   - If you see OOM, reduce `MAX_CONCURRENT_GENERATIONS`

### Log Patterns

**Healthy:**
```
GPU memory available: 24.50GB
GPU memory available: 18.20GB
GPU memory available: 12.10GB
```

**Warning:**
```
Low GPU memory (3.50GB). Waiting for memory to free up...
```

**Critical:**
```
Timeout waiting for GPU memory. Current available: 1.20GB
```

## Troubleshooting

### Problem: Requests Waiting Too Long

**Symptom:**
- Requests waiting > 60 seconds for memory

**Possible Causes:**
1. `MAX_CONCURRENT_GENERATIONS` too high
2. GPU memory leaks (not freeing properly)
3. Other processes using GPU

**Solution:**
```bash
# Reduce concurrent generations
MAX_CONCURRENT_GENERATIONS=1
```

### Problem: Still Getting OOM Errors

**Symptom:**
- Occasional CUDA OOM errors

**Possible Causes:**
1. Memory threshold too low
2. Memory not freeing fast enough
3. Very large video generations

**Solution:**
- Increase `SAFE_MEMORY_BUFFER` to 3.0 or 4.0
- Reduce `MAX_CONCURRENT_GENERATIONS`

## Summary

✅ **Real-time GPU memory monitoring**
- Checks memory before each request
- Waits if memory is low
- Proceeds when memory is available

✅ **Two-layer protection**
- Semaphore (static limit)
- Memory check (dynamic limit)

✅ **Self-adapting**
- No manual configuration needed
- Automatically handles memory pressure
- Prevents OOM errors

The service now **knows** in real-time whether the GPU can handle more requests by checking actual available memory, not just a static limit!
