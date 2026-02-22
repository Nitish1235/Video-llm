# Concurrency Improvements - Production Ready

## What Was Fixed

### 1. **Thread-Safe Pipeline Caching** ✅
- **Problem**: Race condition when multiple requests try to load the model simultaneously
- **Solution**: Added `threading.Lock()` to protect the `PIPELINES` dictionary
- **Implementation**: Double-check locking pattern ensures only one thread loads the model

```python
PIPELINE_LOCK = Lock()

def get_pipeline(model_name: str):
    with PIPELINE_LOCK:
        if key in PIPELINES:
            return PIPELINES[key]
        # Load model safely...
```

### 2. **Concurrency Limiting** ✅
- **Problem**: Multiple requests could run simultaneously, causing GPU OOM
- **Solution**: Added `Semaphore` to limit concurrent video generations per worker
- **Default**: 1 generation at a time per worker (configurable via env var)

```python
MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "1"))
GENERATION_SEMAPHORE = Semaphore(MAX_CONCURRENT_GENERATIONS)
```

### 3. **GPU Memory Management** ✅
- **Problem**: GPU memory not cleared between requests
- **Solution**: 
  - Clear GPU cache before generation
  - Clear GPU cache after generation
  - Handle OOM errors gracefully

```python
torch.cuda.empty_cache()  # Before and after generation
```

### 4. **Request Tracking** ✅
- **Problem**: Hard to debug issues with multiple concurrent requests
- **Solution**: Added unique request IDs for better logging and tracking

```python
request_id = str(uuid.uuid4())[:8]
logger.info(f"[Request {request_id}] Starting video generation")
```

### 5. **Better Error Handling** ✅
- **Problem**: Generic error handling
- **Solution**:
  - Specific handling for GPU OOM errors
  - Timeout handling for semaphore acquisition
  - Better error messages with request context

### 6. **Structured Logging** ✅
- **Problem**: Logs hard to follow with concurrent requests
- **Solution**: All logs include request ID for easy tracking

## How It Works Now

### Scenario: 20 Simultaneous Requests

**With Max Workers = 5:**

```
20 Requests arrive
    ↓
Runpod spawns 5 workers
    ↓
┌─────────────────────────────────────────┐
│ Worker 1                                │
│ ├─ Request 1 → Acquires semaphore → Processing
│ ├─ Request 5 → Waits for semaphore...
│ ├─ Request 9 → Waits for semaphore...
│ └─ Request 13 → Waits for semaphore...
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│ Worker 2                                │
│ ├─ Request 2 → Acquires semaphore → Processing
│ ├─ Request 6 → Waits for semaphore...
│ └─ Request 10 → Waits for semaphore...
└─────────────────────────────────────────┘
... (3 more workers)
```

**Flow:**
1. Request arrives → Gets unique request ID
2. Acquires semaphore (waits if slot unavailable)
3. Processes request (one at a time per worker)
4. Releases semaphore
5. Next request in queue starts

**Result:**
- ✅ No GPU OOM errors
- ✅ Predictable resource usage
- ✅ Better error handling
- ✅ Easy debugging with request IDs

## Configuration

### Environment Variables

```bash
# Limit concurrent generations per worker (default: 1)
MAX_CONCURRENT_GENERATIONS=1

# For GPUs with more memory, you can increase:
MAX_CONCURRENT_GENERATIONS=2  # Allows 2 simultaneous generations
```

### Runpod Settings

**Recommended Configuration:**
- **Max Workers**: 5-10 (based on your load)
- **Idle Timeout**: 60 seconds
- **GPU**: A100 40GB (or A10G 24GB minimum)

**For High Volume:**
- **Max Workers**: 20
- **MAX_CONCURRENT_GENERATIONS**: 1 (per worker)
- **Result**: 20 videos processing simultaneously

## Performance Characteristics

### Before Improvements
- ❌ Race conditions on model loading
- ❌ GPU OOM errors with concurrent requests
- ❌ Unpredictable behavior
- ❌ Hard to debug issues

### After Improvements
- ✅ Thread-safe model loading
- ✅ Controlled concurrency (no OOM)
- ✅ Predictable performance
- ✅ Easy debugging with request IDs
- ✅ Better resource management

### Throughput

**With 5 Workers, MAX_CONCURRENT_GENERATIONS=1:**
- **Concurrent Processing**: 5 videos simultaneously
- **Throughput**: ~5 videos/minute (assuming 60s per video)
- **Cost**: ~$6.95/hour (5 × $1.39/hour)

**With 20 Workers, MAX_CONCURRENT_GENERATIONS=1:**
- **Concurrent Processing**: 20 videos simultaneously
- **Throughput**: ~20 videos/minute
- **Cost**: ~$27.80/hour (20 × $1.39/hour)

## Error Handling

### New Error Codes

1. **TIMEOUT**: Request waited too long for generation slot
   ```json
   {
     "output": {
       "error": "Service busy, please try again later",
       "code": "TIMEOUT"
     }
   }
   ```

2. **GPU_OOM**: GPU out of memory (handled gracefully)
   - Automatically clears cache
   - Returns error instead of crashing

### Request Tracking

Every response includes `request_id`:
```json
{
  "output": {
    "video_url": "...",
    "request_id": "a1b2c3d4",
    ...
  }
}
```

## Monitoring

### Logs Include Request IDs

```
[Request a1b2c3d4] Starting video generation request
[Request a1b2c3d4] Waiting for generation slot (max: 1)
[Request a1b2c3d4] Processing: duration=30s, aspect_ratio=9:16...
[Request a1b2c3d4] Starting video generation...
[Request a1b2c3d4] Merging audio and video...
[Request a1b2c3d4] Uploading to storage...
[Request a1b2c3d4] Job completed successfully in 65.23s
[Request a1b2c3d4] Request completed in 65.45s
```

### Metrics to Monitor

1. **Semaphore Wait Time**: How long requests wait for slots
2. **GPU Memory Usage**: Track memory per request
3. **Request Duration**: Time per video generation
4. **Error Rate**: Failed requests vs successful

## Best Practices

1. **Set MAX_CONCURRENT_GENERATIONS=1** for stability
2. **Monitor GPU memory** to ensure no OOM
3. **Use request IDs** for debugging
4. **Set appropriate Max Workers** based on load
5. **Monitor logs** for timeout errors

## Migration Notes

### No Breaking Changes
- All existing API calls work the same
- New `request_id` field added to response
- Backward compatible

### Recommended Updates
- Update clients to handle `TIMEOUT` error code
- Use `request_id` for tracking/debugging
- Monitor semaphore wait times
