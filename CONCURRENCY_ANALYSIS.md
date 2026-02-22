# Concurrency Analysis: What Happens with 20 Simultaneous Requests

## Current Behavior

### Scenario: 20 Video Generation Requests Arrive Simultaneously

#### 1. **Runpod Serverless Scaling**
- Runpod spawns multiple worker containers based on your **Max Workers** setting
- Each container is independent and has its own GPU
- Default behavior: Requests are distributed across workers

**Example with Max Workers = 5:**
```
20 Requests → Distributed across 5 workers
├─ Worker 1: Processes 4 requests (sequentially or concurrently)
├─ Worker 2: Processes 4 requests
├─ Worker 3: Processes 4 requests
├─ Worker 4: Processes 4 requests
└─ Worker 5: Processes 4 requests
```

#### 2. **Within Each Worker Container**

**Current Implementation:**
- ✅ **Pipeline Caching**: Model is loaded once and reused (`PIPELINES` dict)
- ❌ **No Thread Safety**: Race condition risk when multiple requests access `PIPELINES` simultaneously
- ❌ **No Concurrency Limit**: All requests process simultaneously
- ❌ **GPU Memory Contention**: Multiple requests share same GPU → OOM risk

**Request Flow in Single Worker:**
```
Request 1 arrives → get_pipeline() → Check cache → Load model → Generate → Upload
Request 2 arrives → get_pipeline() → Check cache → Use cached → Generate → Upload
Request 3 arrives → get_pipeline() → Check cache → Use cached → Generate → Upload
```

**Problem:** If Requests 1, 2, 3 arrive at the same time:
- All 3 might try to load the model simultaneously
- Race condition on `PIPELINES` dictionary
- GPU memory could overflow

#### 3. **Resource Usage Per Request**

Each request consumes:
- **GPU Memory**: ~8-12 GB (HunyuanVideo model)
- **CPU**: High during generation
- **Disk**: Temporary files (~100-500 MB per video)
- **Network**: Download audio, upload video
- **Time**: ~30-120 seconds per video

**With 20 simultaneous requests:**
- If all in 1 worker: **GPU OOM** (needs 160-240 GB!)
- If distributed across 5 workers: Each worker handles 4 requests
- **Best case**: 5 workers × 1 request = 5 videos in parallel
- **Worst case**: 1 worker × 20 requests = Queue/errors

## Current Issues

### 🔴 Critical Issues

1. **Race Condition in Pipeline Cache**
   ```python
   # UNSAFE: Multiple threads can access simultaneously
   if key in PIPELINES:  # Thread 1 checks
       return PIPELINES[key]  # Thread 2 might modify here
   ```

2. **No Concurrency Control**
   - No limit on simultaneous requests per worker
   - Can cause GPU OOM errors
   - No request queuing

3. **GPU Memory Contention**
   - Multiple requests share same GPU
   - No memory management
   - Risk of CUDA OOM errors

### 🟡 Medium Issues

4. **No Request Prioritization**
5. **No Timeout Handling for Long Queues**
6. **No Resource Monitoring**

## Recommended Solutions

### Option 1: **Runpod Configuration** (Quick Fix)

**Set Max Workers = 20** (one request per worker):
- ✅ Each request gets dedicated GPU
- ✅ No concurrency issues
- ❌ Higher cost (20 GPUs running)
- ❌ Slower cold starts

**Set Max Workers = 5, Idle Timeout = 60s**:
- ✅ Cost-effective
- ✅ Handles bursts
- ⚠️ Need to fix code for concurrency

### Option 2: **Add Thread Safety** (Code Fix)

Add locking to pipeline cache:
```python
import threading
pipeline_lock = threading.Lock()

def get_pipeline(model_name: str):
    global PIPELINES
    with pipeline_lock:
        if key in PIPELINES:
            return PIPELINES[key]
        # Load model...
```

### Option 3: **Add Concurrency Limit** (Best Practice)

Limit concurrent requests per worker:
```python
from threading import Semaphore
MAX_CONCURRENT = 1  # Only 1 video generation at a time per worker

generation_semaphore = Semaphore(MAX_CONCURRENT)

def handler(event):
    with generation_semaphore:
        # Process request
```

### Option 4: **Request Queuing** (Production Ready)

Implement a queue system:
- Queue requests when GPU is busy
- Process one at a time per worker
- Return job ID for async processing

## Expected Behavior with Fixes

### With Concurrency Limit = 1 per Worker:

**20 Requests, 5 Workers:**
```
Worker 1: Request 1 → Processing (60s) → Request 5 → Processing (60s)
Worker 2: Request 2 → Processing (60s) → Request 6 → Processing (60s)
Worker 3: Request 3 → Processing (60s) → Request 7 → Processing (60s)
Worker 4: Request 4 → Processing (60s) → Request 8 → Processing (60s)
Worker 5: Request 9 → Processing (60s) → Request 13 → Processing (60s)
...
```

**Total Time**: ~4 batches × 60s = ~240 seconds
**Throughput**: 5 videos/minute (with 5 workers)

## Recommendations

1. **Immediate**: Set `Max Workers = 5` in Runpod
2. **Short-term**: Add thread safety to pipeline cache
3. **Medium-term**: Add concurrency limit (1 per worker)
4. **Long-term**: Implement request queuing system

## Cost Implications

**Current Setup (No Limits):**
- Risk of GPU OOM → Failed requests → Wasted GPU time
- Unpredictable costs

**With Concurrency Limit:**
- Predictable: 1 request per worker = stable GPU usage
- Cost: 5 workers × $1.39/hour = $6.95/hour
- Throughput: ~5 videos/minute

**With Max Workers = 20:**
- Cost: 20 workers × $1.39/hour = $27.80/hour
- Throughput: ~20 videos/minute
- Best for high-volume production
