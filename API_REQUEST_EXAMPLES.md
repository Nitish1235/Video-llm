# API Request Examples

## Complete Request Format

### Basic Text-to-Video Request

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "A beautiful sunset over the ocean with waves crashing on the shore",
    "audio_url": "https://example.com/voiceover.mp3",
    "duration": 30
  }
}
```

### Full Request with All Options

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "A serene mountain landscape at dawn with misty valleys",
    "audio_url": "https://example.com/voiceover.mp3",
    "duration": 30,
    "aspect_ratio": "9:16",
    "style": "cinematic",
    "fps": 24,
    "seed": 42,
    "image_url": "https://example.com/init-image.jpg"
  }
}
```

## Field Descriptions

### Required Fields

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `user_id` | string | User identifier for storage path | `"user-123"` |
| `project_id` | string | Project identifier for storage path | `"project-abc"` |
| `script` | string | Video generation prompt/description | `"A beautiful sunset"` |
| `audio_url` | string | URL to audio file (MP3, M4A, etc.) | `"https://example.com/audio.mp3"` |
| `duration` | integer | Video duration in seconds (20, 30, 45, or 60) | `30` |

### Optional Fields

| Field | Type | Default | Description | Example |
|-------|------|---------|-------------|---------|
| `aspect_ratio` | string | `"9:16"` | Video aspect ratio | `"9:16"`, `"16:9"`, `"1:1"` |
| `style` | string | `null` | Video style modifier | `"cinematic"` |
| `fps` | integer | `30` (or `24` if cinematic) | Frames per second | `24`, `30` |
| `seed` | integer | `42` | Random seed for reproducibility | `12345` |
| `image_url` | string | `null` | Init image for image-to-video | `"https://example.com/image.jpg"` |
| `model` | string | `"HunyuanVideo-1.5"` | Model override | `"HunyuanVideo-1.5"` |

## Request Examples by Use Case

### 1. Vertical Video (9:16) - TikTok/Instagram Reels

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "A vibrant cityscape at night with neon lights",
    "audio_url": "https://example.com/audio.mp3",
    "duration": 30,
    "aspect_ratio": "9:16",
    "fps": 30
  }
}
```

### 2. Horizontal Video (16:9) - YouTube/Landscape

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "A peaceful forest scene with sunlight filtering through trees",
    "audio_url": "https://example.com/audio.mp3",
    "duration": 45,
    "aspect_ratio": "16:9",
    "fps": 30
  }
}
```

### 3. Square Video (1:1) - Instagram Post

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "A minimalist abstract art piece with flowing colors",
    "audio_url": "https://example.com/audio.mp3",
    "duration": 20,
    "aspect_ratio": "1:1",
    "fps": 30
  }
}
```

### 4. Cinematic Style Video

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "A dramatic mountain range at sunset",
    "audio_url": "https://example.com/audio.mp3",
    "duration": 60,
    "aspect_ratio": "16:9",
    "style": "cinematic",
    "fps": 24
  }
}
```

### 5. Image-to-Video (Seamless Transition)

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "The landscape continues with clouds drifting overhead",
    "audio_url": "https://example.com/audio2.mp3",
    "duration": 30,
    "aspect_ratio": "9:16",
    "image_url": "https://storage.googleapis.com/your-bucket/frames/last-frame.jpg"
  }
}
```

### 6. Reproducible Video (Same Seed)

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "A futuristic city with flying vehicles",
    "audio_url": "https://example.com/audio.mp3",
    "duration": 30,
    "seed": 12345
  }
}
```

## cURL Examples

### Basic Request

```bash
curl -X POST https://api.runpod.ai/v2/your-endpoint-id/runsync \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key" \
  -d '{
    "input": {
      "user_id": "user-123",
      "project_id": "project-abc",
      "script": "A beautiful sunset over the ocean",
      "audio_url": "https://example.com/voiceover.mp3",
      "duration": 30
    }
  }'
```

### Full Request with Options

```bash
curl -X POST https://api.runpod.ai/v2/your-endpoint-id/runsync \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key" \
  -d '{
    "input": {
      "user_id": "user-123",
      "project_id": "project-abc",
      "script": "A serene mountain landscape at dawn",
      "audio_url": "https://example.com/voiceover.mp3",
      "duration": 30,
      "aspect_ratio": "9:16",
      "style": "cinematic",
      "fps": 24,
      "seed": 42
    }
  }'
```

### Image-to-Video Request

```bash
curl -X POST https://api.runpod.ai/v2/your-endpoint-id/runsync \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key" \
  -d '{
    "input": {
      "user_id": "user-123",
      "project_id": "project-abc",
      "script": "The scene transitions smoothly to a new vista",
      "audio_url": "https://example.com/audio2.mp3",
      "duration": 30,
      "aspect_ratio": "9:16",
      "image_url": "https://storage.googleapis.com/your-bucket/frames/last-frame.jpg"
    }
  }'
```

## Python Examples

### Using requests library

```python
import requests

url = "https://api.runpod.ai/v2/your-endpoint-id/runsync"
headers = {
    "Content-Type": "application/json",
    "x-api-key": "your-secret-api-key"
}

payload = {
    "input": {
        "user_id": "user-123",
        "project_id": "project-abc",
        "script": "A beautiful sunset over the ocean",
        "audio_url": "https://example.com/voiceover.mp3",
        "duration": 30,
        "aspect_ratio": "9:16",
        "fps": 30
    }
}

response = requests.post(url, json=payload, headers=headers)
result = response.json()

print(result)
```

### Image-to-Video Example

```python
import requests

def generate_video_with_image(script, audio_url, image_url, user_id, project_id):
    url = "https://api.runpod.ai/v2/your-endpoint-id/runsync"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": "your-secret-api-key"
    }
    
    payload = {
        "input": {
            "user_id": user_id,
            "project_id": project_id,
            "script": script,
            "audio_url": audio_url,
            "duration": 30,
            "aspect_ratio": "9:16",
            "image_url": image_url  # Last frame from previous video
        }
    }
    
    response = requests.post(url, json=payload, headers=headers)
    return response.json()

# Usage
result = generate_video_with_image(
    script="The landscape continues with clouds",
    audio_url="https://example.com/audio2.mp3",
    image_url="https://storage.googleapis.com/bucket/frames/last-frame.jpg",
    user_id="user-123",
    project_id="project-abc"
)
```

## Response Format

### Success Response

```json
{
  "output": {
    "video_url": "https://storage.googleapis.com/bucket/autopostai-media/user-123/projects/project-abc/videos/video_abc123def456.mp4?X-Goog-Algorithm=...",
    "duration": 30.5,
    "resolution": "1080x1920",
    "fps": 30.0,
    "format": "mp4",
    "file_size": 15728640,
    "storage_path": "autopostai-media/user-123/projects/project-abc/videos/video_abc123def456.mp4",
    "user_id": "user-123",
    "project_id": "project-abc",
    "request_id": "a1b2c3d4"
  }
}
```

### Error Response

```json
{
  "output": {
    "error": "Missing required fields: user_id, project_id",
    "code": "INVALID_INPUT"
  }
}
```

## Valid Values

### Duration
- `20` seconds
- `30` seconds
- `45` seconds
- `60` seconds

### Aspect Ratio
- `"9:16"` - Vertical (1080x1920)
- `"16:9"` - Horizontal (1920x1080)
- `"1:1"` - Square (1080x1080)

### FPS
- Any positive integer
- Recommended: `24` (cinematic) or `30` (standard)

### Style
- `"cinematic"` - Automatically sets FPS to 24
- Any string (will be appended to prompt)

## Notes

1. **Audio URL**: Must be accessible via HTTPS
2. **Image URL**: Must be accessible via HTTPS (for image-to-video)
3. **Storage Path**: Videos are stored at `autopostai-media/{user_id}/projects/{project_id}/videos/`
4. **Signed URLs**: Video URLs expire after 7 days
5. **Request ID**: Included in response for tracking/debugging

## Quick Reference

```json
{
  "input": {
    "user_id": "string (required)",
    "project_id": "string (required)",
    "script": "string (required)",
    "audio_url": "string (required)",
    "duration": 20|30|45|60 (required),
    "aspect_ratio": "9:16"|"16:9"|"1:1" (optional, default: "9:16"),
    "style": "string (optional)",
    "fps": "integer (optional, default: 30)",
    "seed": "integer (optional, default: 42)",
    "image_url": "string (optional, for image-to-video)",
    "model": "string (optional, default: HunyuanVideo-1.5)"
  }
}
```
