# Deployment Guide: Hunyuan Video Generation Service

This guide covers deploying the video generation service on Runpod and Modal with Google Cloud Storage (GCS) as the storage backend.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Google Cloud Storage Setup](#google-cloud-storage-setup)
3. [Runpod Deployment](#runpod-deployment)
4. [Modal Deployment](#modal-deployment)
5. [Testing](#testing)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Accounts & Tools

- **Runpod Account**: Sign up at https://www.runpod.io/
- **Modal Account**: Sign up at https://modal.com/
- **Google Cloud Platform Account**: Sign up at https://cloud.google.com/
- **Docker**: For local testing (optional)
- **Python 3.10+**: For local development (optional)

### Required Knowledge

- Basic understanding of serverless platforms
- Familiarity with environment variables
- Basic GCP console navigation

---

## Google Cloud Storage Setup

### Step 1: Create GCP Project

1. Go to [GCP Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Enter project name: `hunyuan-video-service`
4. Click "Create"

### Step 2: Enable Required APIs

1. Navigate to "APIs & Services" → "Library"
2. Enable the following APIs:
   - **Cloud Storage API**
   - **Cloud Storage JSON API**

### Step 3: Create Service Account

1. Go to "IAM & Admin" → "Service Accounts"
2. Click "Create Service Account"
3. Name: `hunyuan-video-storage`
4. Description: `Service account for video storage uploads`
5. Click "Create and Continue"
6. Grant role: **Storage Object Admin** (or **Storage Admin** for full access)
7. Click "Done"

### Step 4: Create Service Account Key

1. Click on the service account you just created
2. Go to "Keys" tab
3. Click "Add Key" → "Create new key"
4. Select **JSON** format
5. Click "Create"
6. **Save the downloaded JSON file securely** - you'll need this for Modal

### Step 5: Create GCS Bucket

1. Go to "Cloud Storage" → "Buckets"
2. Click "Create Bucket"
3. Configure:
   - **Name**: `hunyuan-video-storage` (must be globally unique)
   - **Location type**: Region
   - **Region**: Choose closest to your users (e.g., `us-central1`)
   - **Storage class**: Standard
   - **Access control**: Uniform
   - **Public access**: Prevent public access (recommended)
4. Click "Create"

### Step 6: Set Bucket CORS (Optional, for direct browser access)

1. Click on your bucket
2. Go to "Configuration" tab
3. Under "CORS", click "Edit"
4. Add CORS configuration:
```json
[
  {
    "origin": ["*"],
    "method": ["GET", "HEAD"],
    "responseHeader": ["Content-Type", "Content-Length"],
    "maxAgeSeconds": 3600
  }
]
```
5. Click "Save"

---

## Runpod Deployment

### Step 1: Prepare Docker Image

#### Option A: Build Locally (Recommended for Testing)

```bash
# Build the Docker image
docker build -t hunyuan-video-service:latest .

# Test locally (requires GPU)
docker run --gpus all \
  -e STORAGE_TYPE=gcs \
  -e GCS_BUCKET_NAME=hunyuan-video-storage \
  -e GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-creds.json \
  -e API_KEY=your-secret-api-key \
  -e VIDEO_MODEL_NAME=HunyuanVideo-1.5 \
  -v /path/to/gcp-creds.json:/secrets/gcp-creds.json:ro \
  hunyuan-video-service:latest
```

#### Option B: Push to Docker Hub

```bash
# Tag for Docker Hub
docker tag hunyuan-video-service:latest yourusername/hunyuan-video-service:latest

# Login to Docker Hub
docker login

# Push image
docker push yourusername/hunyuan-video-service:latest
```

### Step 2: Create Runpod Template

1. Log in to [Runpod Console](https://www.runpod.io/console)
2. Go to "Templates" → "New Template"
3. Configure template:

**Basic Settings:**
- **Template Name**: `hunyuan-video-service`
- **Container Image**: `yourusername/hunyuan-video-service:latest` (or use Runpod's registry)
- **Container Disk**: `50 GB` (minimum for model weights)
- **Docker Command**: Leave empty (uses CMD from Dockerfile)

**GPU Configuration:**
- **GPU Type**: `NVIDIA A100 40GB` (recommended) or `NVIDIA A10G 24GB` (minimum)
- **GPU Count**: `1`

**Environment Variables:**
Add the following environment variables:

```
STORAGE_TYPE=gcs
GCS_BUCKET_NAME=hunyuan-video-storage
GOOGLE_APPLICATION_CREDENTIALS=/secrets/gcp-creds.json
API_KEY=your-secret-api-key-here
VIDEO_MODEL_NAME=HunyuanVideo-1.5
```

**Secrets/Volumes:**
- Mount GCP credentials JSON file:
  - **Path**: `/secrets/gcp-creds.json`
  - **Content**: Paste the entire contents of your service account JSON key file

**Network:**
- **Port**: `8000` (default Runpod serverless port)
- **Handler Path**: `/handler` (Runpod will auto-detect)

**Timeout:**
- **Timeout**: `300` seconds (5 minutes)

4. Click "Create Template"

### Step 3: Deploy Serverless Endpoint

1. Go to "Serverless" → "Create Endpoint"
2. Select your template: `hunyuan-video-service`
3. Configure:
   - **Endpoint Name**: `hunyuan-video-gen`
   - **Max Workers**: `1` (start with 1, scale up as needed)
   - **Idle Timeout**: `60` seconds
   - **Flashboot**: Enabled (faster cold starts)
4. Click "Deploy"

### Step 4: Get Endpoint URL

1. After deployment, copy the **Endpoint URL**
2. Format: `https://api.runpod.ai/v2/your-endpoint-id`
3. Save this URL for API calls

---

## Modal Deployment

### Step 1: Install Modal CLI

```bash
pip install modal
```

### Step 2: Authenticate Modal

```bash
modal token new
```

This will open a browser to authenticate your Modal account.

### Step 3: Create Modal Secret for GCP Credentials

```bash
# Create secret with GCP credentials
modal secret create gcp-storage \
  GOOGLE_APPLICATION_CREDENTIALS_JSON='<paste entire JSON content here>' \
  GCS_BUCKET_NAME=hunyuan-video-storage \
  STORAGE_TYPE=gcs \
  API_KEY=your-secret-api-key-here \
  VIDEO_MODEL_NAME=HunyuanVideo-1.5
```

**Alternative: Use GCP Service Account File**

If you prefer to reference the file directly:

```bash
modal secret create gcp-storage \
  --env-file .env
```

Create `.env` file:
```bash
GOOGLE_APPLICATION_CREDENTIALS_JSON=$(cat /path/to/gcp-creds.json)
GCS_BUCKET_NAME=hunyuan-video-storage
STORAGE_TYPE=gcs
API_KEY=your-secret-api-key-here
VIDEO_MODEL_NAME=HunyuanVideo-1.5
```

### Step 4: Update modal_app.py for Secrets

The `modal_app.py` file needs to load secrets. Update the app initialization:

```python
# Add this near the top of modal_app.py after app definition
@app.function(secrets=[modal.Secret.from_name("gcp-storage")])
def setup_secrets():
    import json
    import os
    
    # If using JSON string in secret
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        creds_path = "/tmp/gcp-creds.json"
        with open(creds_path, "w") as f:
            f.write(creds_json)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
```

### Step 5: Deploy Modal App

```bash
# Deploy the app
modal deploy modal_app.py

# Or run interactively for testing
modal run modal_app.py
```

### Step 6: Create Web Endpoint (Optional)

If you want an HTTP endpoint, add to `modal_app.py`:

```python
@app.function(image=image, timeout=600)
@modal.web_endpoint(method="POST")
def generate_video_api(request: dict):
    """HTTP endpoint for video generation"""
    return modal_entrypoint(request)
```

Then deploy:
```bash
modal deploy modal_app.py
```

Modal will provide a URL like: `https://your-username--hunyuan-video-service-generate-video-api.modal.run`

---

## Testing

### Test Runpod Endpoint

```bash
curl -X POST https://api.runpod.ai/v2/your-endpoint-id/runsync \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key-here" \
  -d '{
    "input": {
      "script": "A beautiful sunset over the ocean with waves crashing on the shore",
      "audio_url": "https://example.com/voiceover.mp3",
      "duration": 30,
      "aspect_ratio": "9:16",
      "style": "cinematic",
      "fps": 24,
      "seed": 42
    }
  }'
```

### Test Modal Endpoint

```bash
curl -X POST https://your-username--hunyuan-video-service-generate-video-api.modal.run \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "script": "A beautiful sunset over the ocean with waves crashing on the shore",
      "audio_url": "https://example.com/voiceover.mp3",
      "duration": 30,
      "aspect_ratio": "9:16",
      "style": "cinematic",
      "fps": 24,
      "seed": 42,
      "api_key": "your-secret-api-key-here"
    }
  }'
```

### Expected Response

```json
{
  "output": {
    "video_url": "https://storage.googleapis.com/hunyuan-video-storage/videos/hunyuan_abc123def456.mp4?X-Goog-Algorithm=...",
    "duration": 30.5,
    "resolution": "1080x1920",
    "fps": 24.0,
    "format": "mp4",
    "file_size": 15728640
  }
}
```

### Test Image-to-Video Mode

```bash
curl -X POST https://api.runpod.ai/v2/your-endpoint-id/runsync \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-secret-api-key-here" \
  -d '{
    "input": {
      "script": "A serene mountain landscape",
      "image_url": "https://example.com/init-image.jpg",
      "audio_url": "https://example.com/voiceover.mp3",
      "duration": 30,
      "aspect_ratio": "16:9"
    }
  }'
```

---

## Environment Variables Reference

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `STORAGE_TYPE` | Storage backend | `gcs` |
| `GCS_BUCKET_NAME` | GCS bucket name | `hunyuan-video-storage` |
| `API_KEY` | API authentication key | `your-secret-key-123` |

### GCS-Specific Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to GCP service account JSON | `/secrets/gcp-creds.json` |
| `GOOGLE_APPLICATION_CREDENTIALS_JSON` | JSON content as string (Modal) | `{"type":"service_account",...}` |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIDEO_MODEL_NAME` | Model identifier | `HunyuanVideo-1.5` |
| `S3_REGION` | AWS region (if using S3) | `us-east-1` |

---

## Troubleshooting

### Common Issues

#### 1. GCS Authentication Failed

**Error**: `google.auth.exceptions.DefaultCredentialsError`

**Solutions**:
- Verify `GOOGLE_APPLICATION_CREDENTIALS` path is correct
- Ensure service account JSON has proper permissions
- Check that `GOOGLE_APPLICATION_CREDENTIALS_JSON` is valid JSON (Modal)

#### 2. Bucket Not Found

**Error**: `404 Bucket not found`

**Solutions**:
- Verify `GCS_BUCKET_NAME` matches exactly (case-sensitive)
- Ensure bucket exists in the same GCP project
- Check service account has access to the bucket

#### 3. Model Download Timeout

**Error**: `Connection timeout` or `Model loading failed`

**Solutions**:
- Increase container disk size (50GB+ recommended)
- Pre-download model in Dockerfile (already included)
- Check network connectivity in container

#### 4. GPU Out of Memory

**Error**: `CUDA out of memory`

**Solutions**:
- Use A100 40GB instead of A10G
- Reduce video resolution or duration
- Enable attention slicing (already enabled in code)

#### 5. FFmpeg Errors

**Error**: `ffmpeg failed` or `ffprobe failed`

**Solutions**:
- Verify FFmpeg is installed in container (already in Dockerfile)
- Check audio file format is supported
- Ensure audio URL is accessible

#### 6. Runpod Handler Not Found

**Error**: `Handler not found` or `404`

**Solutions**:
- Verify `RUNPOD_HANDLER=handler.handler` in environment
- Check `handler.py` is in root of container
- Ensure handler function signature matches Runpod format

#### 7. Modal Secret Not Found

**Error**: `Secret 'gcp-storage' not found`

**Solutions**:
- Create secret: `modal secret create gcp-storage ...`
- Verify secret name matches in code
- Check secret is accessible: `modal secret list`

### Debugging Tips

#### Check Logs

**Runpod**:
1. Go to "Serverless" → Your endpoint
2. Click on a job → View logs

**Modal**:
```bash
modal app logs hunyuan-video-service
```

#### Test Locally

```bash
# Test handler locally
python -c "from handler import handler; print(handler({'input': {'script': 'test', 'audio_url': 'https://example.com/test.mp3', 'duration': 20}}))"
```

#### Verify GCS Access

```python
from google.cloud import storage
client = storage.Client()
bucket = client.bucket('hunyuan-video-storage')
print(list(bucket.list_blobs(max_results=5)))
```

---

## Cost Estimation

### Runpod Costs

- **A100 40GB**: ~$1.39/hour (on-demand)
- **A10G 24GB**: ~$0.89/hour (on-demand)
- **Storage**: Included in container disk
- **Network**: Egress charges apply

### Modal Costs

- **A100 40GB**: ~$1.50/hour (on-demand)
- **Storage**: Included in image
- **Network**: Egress charges apply

### GCS Costs

- **Storage**: $0.020/GB/month (Standard)
- **Operations**: $0.05 per 10,000 Class A operations (uploads)
- **Egress**: $0.12/GB (first 10TB/month)

**Example**: 1000 videos/month @ 30MB each
- Storage: ~30GB × $0.020 = $0.60/month
- Operations: 1000 uploads × $0.05/10k = $0.005
- **Total GCS**: ~$0.61/month

---

## Security Best Practices

1. **API Keys**: Use strong, randomly generated API keys
2. **GCP Credentials**: Never commit service account JSON to git
3. **Bucket Permissions**: Use least-privilege IAM roles
4. **Signed URLs**: URLs expire after 7 days (configured in code)
5. **HTTPS Only**: Always use HTTPS for API calls
6. **Rate Limiting**: Implement rate limiting in your application layer
7. **Input Validation**: Validate all inputs before processing

---

## Scaling Considerations

### Runpod

- **Max Workers**: Increase based on concurrent request volume
- **Flashboot**: Enable for faster cold starts
- **GPU Selection**: A100 for better throughput, A10G for cost savings

### Modal

- **Concurrency Limit**: Adjust `concurrency_limit` in `@app.cls()` decorator
- **Auto-scaling**: Modal handles auto-scaling automatically
- **GPU Selection**: Change `gpu="A100"` to `gpu="A10G"` for cost savings

### GCS

- **Bucket Location**: Choose region closest to your users
- **Lifecycle Policies**: Set up automatic deletion of old videos
- **CDN**: Consider Cloud CDN for video delivery

---

## Monitoring

### Runpod Metrics

- View metrics in Runpod dashboard:
  - Request count
  - Average execution time
  - Error rate
  - GPU utilization

### Modal Metrics

```bash
# View app metrics
modal app show hunyuan-video-service

# View function metrics
modal function logs hunyuan-video-service.generate
```

### GCS Monitoring

- Enable Cloud Monitoring in GCP Console
- Set up alerts for:
  - Storage usage
  - API quota limits
  - Error rates

---

## Support

- **Runpod Docs**: https://docs.runpod.io/
- **Modal Docs**: https://modal.com/docs
- **GCS Docs**: https://cloud.google.com/storage/docs
- **HunyuanVideo**: https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5

---

## Next Steps

1. Set up monitoring and alerting
2. Implement rate limiting
3. Add request queuing for high-volume scenarios
4. Set up automated testing pipeline
5. Configure backup and disaster recovery
6. Implement cost optimization strategies
