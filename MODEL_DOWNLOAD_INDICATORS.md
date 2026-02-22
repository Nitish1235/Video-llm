# How to Know if Model is Downloading

## Log Messages to Look For

### When Model Download Starts

You'll see these logs in sequence:

```
[Request xxxxxxxx] Starting video generation request
Loading HunyuanVideo pipeline: tencent/HunyuanVideo-1.5
⚠️  FIRST REQUEST: This may take 5-15 minutes (downloading ~8-15GB model from HuggingFace Hub)
📥 Starting model download from HuggingFace Hub: https://huggingface.co/tencent/HunyuanVideo-1.5
⏳ Downloading model files... (this can take several minutes)
```

### During Download

**What you'll see:**
- The logs above appear
- Then **silence for 5-15 minutes** (this is normal!)
- HuggingFace downloads files in the background
- No progress bars in logs (unfortunately)

**What's happening:**
- Downloading model weights (~8-15GB)
- Downloading config files
- Downloading tokenizer files
- All happening silently in background

### When Download Completes

You'll see:

```
✅ Model downloaded successfully in XXX.Xs (X.X minutes)
🔄 Now loading model to GPU...
⏳ Moving model to GPU (cuda)...
✅ Model loaded to GPU successfully
🎉 Model fully loaded in XXX.Xs (X.X minutes total)
```

## How to Check Download Progress

### Method 1: Check Cache Directory Size

The model downloads to `/cache/huggingface/hub/`. You can check:

```bash
# In Runpod logs or container shell
du -sh /cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/
```

**Expected sizes:**
- During download: Size increases gradually
- Complete: ~8-15GB total

### Method 2: Watch Logs for Completion

Look for these log messages in order:

1. ✅ `Model downloaded successfully` = Download done
2. ✅ `Model loaded to GPU successfully` = Ready to use

### Method 3: Check Network Activity

If you have access to container metrics:
- High network I/O = Downloading
- Low/zero network I/O = Download complete or not started

## Timeline Expectations

### First Request (Model Not Cached)

```
0:00 - Request arrives
0:01 - Download starts (logs appear)
0:01-15:00 - Downloading (silent, no logs)
15:00 - Download complete (logs appear)
15:01 - Loading to GPU
15:30 - Model ready, video generation starts
```

**Total: ~15-20 minutes for first request**

### Subsequent Requests (Model Cached)

```
0:00 - Request arrives
0:01 - Using cached model (instant)
0:01 - Video generation starts
```

**Total: ~1-2 minutes**

## What If You Don't See Download Logs?

### Scenario 1: No Request Sent
- **Symptom**: Only see initialization logs
- **Solution**: Send a test request to trigger download

### Scenario 2: Model Already Cached
- **Symptom**: See "Using cached pipeline" or model loads instantly
- **Solution**: This is good! Model was already downloaded

### Scenario 3: Download Failed
- **Symptom**: Error messages about download/connection
- **Solution**: Check network connectivity, HuggingFace access

## Quick Checklist

✅ **Download Started:**
- See `📥 Starting model download from HuggingFace Hub`
- See `⏳ Downloading model files...`

⏳ **Downloading (wait):**
- No new logs for 5-15 minutes (normal!)
- Container still running
- No errors

✅ **Download Complete:**
- See `✅ Model downloaded successfully`
- See `✅ Model loaded to GPU successfully`
- See `🎉 Model fully loaded`

## Troubleshooting

### Download Taking Too Long (>20 minutes)

**Possible causes:**
- Slow network connection
- HuggingFace Hub slow
- Large model files

**Check:**
```bash
# Check if files are being downloaded
ls -lh /cache/huggingface/hub/models--tencent--HunyuanVideo-1.5/
```

### No Download Logs Appear

**Possible causes:**
- Request not reaching handler
- Model already cached
- Error before download starts

**Check:**
- Look for `[Request xxxxxxxx] Starting video generation request`
- Check for any error messages
- Verify request format is correct

## Summary

**To know if model is downloading:**
1. Look for `📥 Starting model download` log
2. Wait 5-15 minutes (silent period is normal)
3. Look for `✅ Model downloaded successfully` log

**If you see the download start logs but nothing for 10+ minutes:**
- This is **normal** - model is downloading
- Be patient, it's a large download
- Check logs again in 5 minutes
