# Runpod Troubleshooting: Handler Not Being Called

## Problem: Request Sent But Handler Not Invoked

**Symptom:**
- Container starts successfully
- Initialization logs appear
- But no `[Request xxxxxxxx]` logs when you send a request
- Handler function never executes

## Possible Causes & Solutions

### 1. RUNPOD_HANDLER Environment Variable Not Set

**Check in Runpod Template:**
- Go to your Runpod template settings
- Verify `RUNPOD_HANDLER=handler.handler` is in environment variables
- Should be exactly: `handler.handler` (not `handler.py` or `handler`)

**Fix:**
```
Environment Variables:
RUNPOD_HANDLER=handler.handler
```

### 2. Handler File Location

**Verify:**
- `handler.py` is in `/app/` directory (WORKDIR in Dockerfile)
- File is named exactly `handler.py` (not `Handler.py` or `handler.PY`)

**Check in Dockerfile:**
```dockerfile
WORKDIR /app
COPY handler.py /app/handler.py
```

### 3. Handler Function Name

**Verify:**
- Function is named exactly `handler` (not `Handler` or `main`)
- Function signature: `def handler(event: Dict[str, Any]) -> Dict[str, Any]:`

### 4. Request Format Issue

**Verify your request format:**
```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "test",
    "audio_url": "https://example.com/audio.mp3",
    "duration": 30
  }
}
```

**Common mistakes:**
- Missing `"input"` wrapper
- Wrong endpoint URL
- Missing API key header (if API_KEY is set)

### 5. Runpod Endpoint Configuration

**Check:**
- Endpoint is deployed and active
- Worker is running (not idle/stopped)
- Endpoint URL is correct

**Verify endpoint:**
- Go to Runpod Console → Serverless → Your Endpoint
- Check status: Should be "Active"
- Check workers: Should show running workers

### 6. Check Runpod Job Logs

**Steps:**
1. Go to Runpod Console
2. Navigate to "Serverless" → Your Endpoint
3. Click on the job/request you sent
4. View logs - look for:
   - Any error messages
   - Handler invocation attempts
   - Python errors

### 7. Test Handler Directly

Add a test endpoint to verify handler works:

```python
# Add to handler.py for testing
if __name__ == "__main__":
    test_event = {
        "input": {
            "user_id": "test",
            "project_id": "test",
            "script": "test",
            "audio_url": "https://example.com/test.mp3",
            "duration": 30
        }
    }
    result = handler(test_event)
    print(result)
```

## Quick Diagnostic Checklist

- [ ] `RUNPOD_HANDLER=handler.handler` in environment variables
- [ ] `handler.py` exists in `/app/` directory
- [ ] Function named `handler` (lowercase)
- [ ] Request has `"input"` wrapper
- [ ] Endpoint is active in Runpod
- [ ] Worker is running (not idle)
- [ ] Check Runpod job logs for errors

## Expected Behavior

**When handler is called, you should see:**
```
[Request abc12345] Starting video generation request
```

**If you don't see this:**
- Handler is not being invoked
- Check Runpod configuration
- Verify environment variables
- Check Runpod job logs for errors

## Common Runpod Errors

### Error: "Handler not found"
- **Cause**: RUNPOD_HANDLER path incorrect
- **Fix**: Set `RUNPOD_HANDLER=handler.handler`

### Error: "Module not found"
- **Cause**: handler.py not in correct location
- **Fix**: Verify `COPY handler.py /app/handler.py` in Dockerfile

### Error: "Function not found"
- **Cause**: Function name mismatch
- **Fix**: Ensure function is named `handler` (lowercase)

## Next Steps

1. **Verify Runpod Template Settings:**
   - Check `RUNPOD_HANDLER=handler.handler` is set
   - Verify all environment variables

2. **Check Runpod Job Logs:**
   - Look for any error messages
   - Check if Runpod is trying to call handler

3. **Test Request Format:**
   - Ensure request has correct structure
   - Verify endpoint URL is correct

4. **Contact Runpod Support:**
   - If handler still not called after checking above
   - Provide them with your template configuration
