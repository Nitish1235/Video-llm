# Image-to-Video Guide: Seamless Clip Transitions

## Overview

✅ **Yes, the service supports image-to-video!** This is perfect for creating seamless transitions between video clips.

## Use Case: Continuous Video Clips

When generating multiple video clips that should flow seamlessly, you can:
1. Generate first clip (text-to-video)
2. Extract the **last frame** from the first clip
3. Use that frame as the **init image** for the next clip
4. Repeat for continuous video sequences

## How It Works

### Current Implementation

The service already supports image-to-video via the `image_url` parameter:

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "A beautiful sunset over the ocean",
    "audio_url": "https://example.com/audio1.mp3",
    "duration": 30,
    "image_url": "https://example.com/last-frame.jpg"  // ← Optional init image
  }
}
```

### Flow for Continuous Clips

```
Clip 1: Text-to-Video
  ↓
Extract last frame → Save to storage
  ↓
Clip 2: Image-to-Video (using last frame from Clip 1)
  ↓
Extract last frame → Save to storage
  ↓
Clip 3: Image-to-Video (using last frame from Clip 2)
  ↓
... and so on
```

## API Usage

### Text-to-Video (First Clip)

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "A serene mountain landscape at dawn",
    "audio_url": "https://example.com/audio1.mp3",
    "duration": 30,
    "aspect_ratio": "9:16",
    "fps": 30
  }
}
```

**Response:**
```json
{
  "output": {
    "video_url": "https://storage.googleapis.com/.../video_abc123.mp4",
    "storage_path": "autopostai-media/user-123/projects/project-abc/videos/video_abc123.mp4",
    ...
  }
}
```

### Image-to-Video (Subsequent Clips)

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "The sun rises higher, casting golden light",
    "audio_url": "https://example.com/audio2.mp3",
    "duration": 30,
    "aspect_ratio": "9:16",
    "fps": 30,
    "image_url": "https://storage.googleapis.com/.../last-frame.jpg"  // ← Last frame from Clip 1
  }
}
```

## Extracting Last Frame from Video

You'll need to extract the last frame from the generated video. Here's how:

### Option 1: Using FFmpeg (Recommended)

```bash
# Extract last frame from video
ffmpeg -i video_abc123.mp4 -vf "select='eq(n,LAST_FRAME_INDEX)'" -vframes 1 last-frame.jpg

# Or simpler - extract last frame directly
ffmpeg -sseof -1 -i video_abc123.mp4 -update 1 -q:v 1 last-frame.jpg
```

### Option 2: Using Python

```python
import cv2
import os

def extract_last_frame(video_path: str, output_path: str) -> bool:
    """Extract the last frame from a video."""
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Seek to last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        cap.release()
        return True
    
    cap.release()
    return False

# Usage
extract_last_frame("video_abc123.mp4", "last-frame.jpg")
```

### Option 3: Using PIL/ImageIO

```python
from PIL import Image
import imageio

def extract_last_frame_pil(video_path: str, output_path: str) -> bool:
    """Extract last frame using imageio."""
    try:
        reader = imageio.get_reader(video_path)
        frames = [frame for frame in reader]
        if frames:
            last_frame = Image.fromarray(frames[-1])
            last_frame.save(output_path)
            return True
    except Exception as e:
        print(f"Error: {e}")
    return False
```

## Complete Workflow Example

### Step 1: Generate First Clip

```bash
curl -X POST https://api.runpod.ai/v2/your-endpoint/runsync \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "input": {
      "user_id": "user-123",
      "project_id": "project-abc",
      "script": "A peaceful forest scene",
      "audio_url": "https://example.com/audio1.mp3",
      "duration": 30
    }
  }'
```

**Response:**
```json
{
  "output": {
    "video_url": "https://.../video_clip1.mp4",
    "storage_path": "autopostai-media/user-123/projects/project-abc/videos/video_clip1.mp4"
  }
}
```

### Step 2: Extract Last Frame

```bash
# Download video
wget "https://.../video_clip1.mp4" -O clip1.mp4

# Extract last frame
ffmpeg -sseof -1 -i clip1.mp4 -update 1 -q:v 1 last-frame.jpg

# Upload to storage (GCS/S3)
gsutil cp last-frame.jpg gs://your-bucket/frames/last-frame.jpg
```

### Step 3: Generate Second Clip (Seamless Transition)

```bash
curl -X POST https://api.runpod.ai/v2/your-endpoint/runsync \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d '{
    "input": {
      "user_id": "user-123",
      "project_id": "project-abc",
      "script": "The forest transitions to a mountain vista",
      "audio_url": "https://example.com/audio2.mp3",
      "duration": 30,
      "image_url": "https://storage.googleapis.com/your-bucket/frames/last-frame.jpg"
    }
  }'
```

## Best Practices

### 1. Frame Resolution
- Ensure the init image matches the video resolution
- Service supports: 1080x1920 (9:16), 1920x1080 (16:9), 1080x1080 (1:1)
- The init image will be resized to match if needed

### 2. Image Format
- Supported formats: JPG, PNG
- Recommended: JPG for smaller file size
- Ensure image is accessible via HTTPS URL

### 3. Seamless Transitions
- Use similar prompts for smooth transitions
- Example:
  - Clip 1: "A serene mountain landscape"
  - Clip 2: "The mountain landscape continues, now with clouds"
  - Clip 3: "Clouds drift over the mountain peaks"

### 4. Frame Extraction Timing
- Extract frame **after** video is fully generated
- Wait for video upload to complete
- Use the final frame (not a middle frame)

## Advanced: Automated Continuous Generation

### Python Script Example

```python
import requests
import subprocess
import time

def generate_continuous_clips(scripts, audio_urls, base_url, api_key):
    """Generate multiple clips with seamless transitions."""
    previous_frame_url = None
    
    for i, (script, audio_url) in enumerate(zip(scripts, audio_urls)):
        payload = {
            "input": {
                "user_id": "user-123",
                "project_id": "project-abc",
                "script": script,
                "audio_url": audio_url,
                "duration": 30,
                "aspect_ratio": "9:16"
            }
        }
        
        # Add init image if not first clip
        if previous_frame_url:
            payload["input"]["image_url"] = previous_frame_url
        
        # Generate video
        response = requests.post(
            f"{base_url}/runsync",
            headers={"x-api-key": api_key},
            json=payload
        )
        
        result = response.json()
        video_url = result["output"]["video_url"]
        
        # Extract last frame
        frame_url = extract_and_upload_frame(video_url, i)
        previous_frame_url = frame_url
        
        print(f"Clip {i+1} generated: {video_url}")
        time.sleep(1)  # Brief pause between requests

def extract_and_upload_frame(video_url, clip_num):
    """Extract last frame and upload to storage."""
    # Download video
    subprocess.run(["wget", video_url, "-O", f"clip_{clip_num}.mp4"])
    
    # Extract frame
    subprocess.run([
        "ffmpeg", "-sseof", "-1", "-i", f"clip_{clip_num}.mp4",
        "-update", "1", "-q:v", "1", f"frame_{clip_num}.jpg"
    ])
    
    # Upload to GCS (example)
    subprocess.run([
        "gsutil", "cp", f"frame_{clip_num}.jpg",
        f"gs://your-bucket/frames/frame_{clip_num}.jpg"
    ])
    
    return f"https://storage.googleapis.com/your-bucket/frames/frame_{clip_num}.jpg"
```

## Troubleshooting

### Issue: Transition Not Seamless

**Possible Causes:**
- Frame resolution mismatch
- Image quality too low
- Prompt too different from previous clip

**Solutions:**
- Ensure frame matches video resolution
- Use high-quality JPG (quality 90+)
- Use similar prompts with gradual changes

### Issue: Image Download Failed

**Error:**
```json
{
  "output": {
    "error": "Failed to load init image for image-to-video",
    "code": "IMAGE_DOWNLOAD_FAILED"
  }
}
```

**Solutions:**
- Ensure image URL is accessible via HTTPS
- Check image format (JPG/PNG)
- Verify image URL is correct
- Check network connectivity

### Issue: Video Doesn't Start from Image

**Possible Causes:**
- Model may not perfectly match the init image
- Prompt conflicts with image content

**Solutions:**
- Use prompts that describe the image content
- Example: If image shows mountains, prompt should mention mountains
- Allow some variation - perfect matching is difficult

## Response Format

### Image-to-Video Response

Same as text-to-video, but includes mode indicator:

```json
{
  "output": {
    "video_url": "https://.../video_xyz789.mp4",
    "duration": 30.5,
    "resolution": "1080x1920",
    "fps": 30.0,
    "format": "mp4",
    "file_size": 15728640,
    "storage_path": "autopostai-media/user-123/projects/project-abc/videos/video_xyz789.mp4",
    "user_id": "user-123",
    "project_id": "project-abc",
    "request_id": "a1b2c3d4"
  }
}
```

## Summary

✅ **Image-to-video is fully supported**
- Use `image_url` parameter in request
- Perfect for seamless clip transitions
- Extract last frame from previous video
- Use as init image for next clip
- Creates continuous video sequences

The service will automatically detect image-to-video mode when `image_url` is provided and generate video starting from that image!
