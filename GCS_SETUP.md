# GCS Storage Setup Guide

This service uses Google Cloud Storage (GCS) for storing generated videos with a structured folder hierarchy.

## Storage Structure

Videos are stored using the following path structure:
```
autopostai-media/{user_id}/projects/{project_id}/videos/{filename}
```

Example:
```
autopostai-media/user-123/projects/project-abc/videos/video_a1b2c3d4e5f6g7h8.mp4
```

## Environment Variables

### Required for GCS

1. **GCS_BUCKET_NAME**: Name of your GCS bucket
   ```
   GCS_BUCKET_NAME=your-bucket-name
   ```

2. **GCS_CREDENTIALS_BASE64**: Base64-encoded GCS service account JSON credentials
   ```bash
   # Generate base64 from your service account JSON file:
   cat your-service-account.json | base64 -w 0
   
   # Or on macOS:
   cat your-service-account.json | base64
   
   # Set in Runpod environment:
   GCS_CREDENTIALS_BASE64=<base64-encoded-json>
   ```

### Optional

- **STORAGE_TYPE**: Set to `gcs` (default) or `s3`
- **API_KEY**: API key for authentication
- **VIDEO_MODEL_NAME**: Model name (default: `HunyuanVideo-1.5`)

## Runpod Configuration

### Step 1: Prepare Base64 Credentials

1. Download your GCS service account JSON key from Google Cloud Console
2. Encode it to base64:
   ```bash
   base64 -w 0 your-service-account.json
   ```
3. Copy the entire base64 string

### Step 2: Set Environment Variables in Runpod

In your Runpod template, add these environment variables:

```
STORAGE_TYPE=gcs
GCS_BUCKET_NAME=your-bucket-name
GCS_CREDENTIALS_BASE64=<paste-base64-string-here>
API_KEY=your-secret-api-key
VIDEO_MODEL_NAME=HunyuanVideo-1.5
```

## API Request Format

### Required Fields

- `user_id`: User identifier (string)
- `project_id`: Project identifier (string)
- `script`: Video generation prompt (string)
- `audio_url`: URL to audio file (string)
- `duration`: Video duration in seconds (20, 30, 45, or 60)

### Example Request

```json
{
  "input": {
    "user_id": "user-123",
    "project_id": "project-abc",
    "script": "A beautiful sunset over the ocean",
    "audio_url": "https://example.com/audio.mp3",
    "duration": 30,
    "aspect_ratio": "9:16",
    "style": "cinematic",
    "fps": 24,
    "seed": 42
  }
}
```

### Response Format

```json
{
  "output": {
    "video_url": "https://storage.googleapis.com/bucket/autopostai-media/user-123/projects/project-abc/videos/video_abc123.mp4?X-Goog-Algorithm=...",
    "duration": 30.5,
    "resolution": "1080x1920",
    "fps": 24.0,
    "format": "mp4",
    "file_size": 15728640,
    "storage_path": "autopostai-media/user-123/projects/project-abc/videos/video_abc123.mp4",
    "user_id": "user-123",
    "project_id": "project-abc"
  }
}
```

## GCS Service Account Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin** → **Service Accounts**
3. Create a new service account or use existing one
4. Grant **Storage Object Admin** role
5. Create a JSON key and download it
6. Encode the JSON file to base64

## Security Notes

- Never commit the base64 credentials to version control
- Use Runpod secrets/environment variables for credentials
- Signed URLs expire after 7 days
- Ensure your GCS bucket has proper IAM permissions
- Use least-privilege access for service accounts

## Troubleshooting

### Credentials Not Working

- Verify base64 encoding is correct (no line breaks)
- Check that the JSON is valid
- Ensure service account has Storage Object Admin role
- Verify bucket name is correct

### Upload Failures

- Check bucket exists and is accessible
- Verify service account has write permissions
- Check network connectivity from Runpod
- Review logs for specific error messages
