import os
import json
import logging
import traceback
import subprocess
import tempfile
import time
import hashlib
import base64
import uuid
import threading
from typing import Any, Dict, Optional, Tuple
from threading import Lock, Semaphore

import requests
import torch
from PIL import Image

try:
    from diffusers import HunyuanVideoPipeline
except ImportError:  # fallback for older diffusers versions
    from diffusers import DiffusionPipeline as HunyuanVideoPipeline  # type: ignore

from diffusers.utils import export_to_video

STORAGE_TYPE = os.getenv("STORAGE_TYPE", "gcs").lower()
VIDEO_MODEL_NAME = os.getenv("VIDEO_MODEL_NAME", "HunyuanVideo-1.5")
API_KEY = os.getenv("API_KEY")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
GCS_CREDENTIALS_BASE64 = os.getenv("GCS_CREDENTIALS_BASE64")
GCS_CREDENTIALS_PATH = "/tmp/gcs-credentials.json"

VALID_DURATIONS = {20, 30, 45, 60}
DEFAULT_ASPECT_RATIO = "9:16"
DEFAULT_FPS = 30
DEFAULT_SEED = 42

RESOLUTION_MAP: Dict[str, Tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIPELINES: Dict[str, Any] = {}
PIPELINE_LOCK = Lock()  # Thread-safe access to PIPELINES dict

# Concurrency control: Limit simultaneous video generations per worker
# Default: Auto-detect based on GPU memory, or set manually via env var
# - A100 40GB: Can handle 2-3 concurrent generations
# - A10G 24GB: Can handle 1-2 concurrent generations
# - RTX 3090 24GB: Can handle 1-2 concurrent generations
# Set MAX_CONCURRENT_GENERATIONS env var to override auto-detection
MAX_CONCURRENT_GENERATIONS_ENV = os.getenv("MAX_CONCURRENT_GENERATIONS")
if MAX_CONCURRENT_GENERATIONS_ENV:
    MAX_CONCURRENT_GENERATIONS = int(MAX_CONCURRENT_GENERATIONS_ENV)
else:
    # Auto-detect based on GPU memory
    if torch.cuda.is_available():
        try:
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory_gb >= 40:  # A100 40GB or larger
                MAX_CONCURRENT_GENERATIONS = 2  # Safe default for large GPUs
            elif gpu_memory_gb >= 24:  # A10G, RTX 3090, etc.
                MAX_CONCURRENT_GENERATIONS = 1  # Conservative for 24GB
            else:
                MAX_CONCURRENT_GENERATIONS = 1  # Default to 1 for smaller GPUs
            logger.info(f"Auto-detected MAX_CONCURRENT_GENERATIONS={MAX_CONCURRENT_GENERATIONS} for GPU with {gpu_memory_gb:.1f}GB")
        except Exception:
            MAX_CONCURRENT_GENERATIONS = 1
            logger.warning("Failed to detect GPU memory, defaulting to MAX_CONCURRENT_GENERATIONS=1")
    else:
        MAX_CONCURRENT_GENERATIONS = 1
        logger.info("No GPU detected, MAX_CONCURRENT_GENERATIONS=1")

GENERATION_SEMAPHORE = Semaphore(MAX_CONCURRENT_GENERATIONS)

# Dynamic GPU memory thresholds (in GB)
MIN_MEMORY_REQUIRED = 6.0  # Minimum memory needed per generation
SAFE_MEMORY_BUFFER = 2.0   # Safety buffer to prevent OOM
MEMORY_CHECK_INTERVAL = 1.0  # Check memory every 1 second when waiting

logger.info(f"Initialized with MAX_CONCURRENT_GENERATIONS={MAX_CONCURRENT_GENERATIONS}")


def get_available_gpu_memory() -> Optional[float]:
    """Get available GPU memory in GB."""
    if not torch.cuda.is_available():
        return None
    try:
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        available = total - reserved
        return available
    except Exception:
        return None


def can_handle_more_requests() -> bool:
    """Check if GPU has enough memory to handle another request."""
    available = get_available_gpu_memory()
    if available is None:
        return True  # No GPU, assume CPU can handle it
    
    # Check if we have enough memory for another generation
    return available >= (MIN_MEMORY_REQUIRED + SAFE_MEMORY_BUFFER)


def wait_for_gpu_memory(timeout: float = 300.0) -> bool:
    """Wait for GPU memory to become available."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        if can_handle_more_requests():
            return True
        time.sleep(MEMORY_CHECK_INTERVAL)
    return False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("hunyuan_video_service")


def setup_gcs_credentials() -> bool:
    """Setup GCS credentials from base64 encoded environment variable."""
    global GCS_CREDENTIALS_PATH
    
    if not GCS_CREDENTIALS_BASE64:
        logger.error("GCS_CREDENTIALS_BASE64 environment variable is not set")
        return False
    
    try:
        # Decode base64 credentials
        credentials_json = base64.b64decode(GCS_CREDENTIALS_BASE64).decode("utf-8")
        
        # Validate JSON
        json.loads(credentials_json)
        
        # Write to file
        os.makedirs(os.path.dirname(GCS_CREDENTIALS_PATH), exist_ok=True)
        with open(GCS_CREDENTIALS_PATH, "w") as f:
            f.write(credentials_json)
        
        # Set environment variable for google-cloud-storage
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_CREDENTIALS_PATH
        
        logger.info(f"GCS credentials successfully loaded from base64")
        return True
    except Exception as exc:
        logger.error(f"Failed to setup GCS credentials: {exc}")
        logger.debug(traceback.format_exc())
        return False


# Initialize GCS credentials on module load if using GCS
if STORAGE_TYPE == "gcs":
    setup_gcs_credentials()


def get_headers_from_event(event: Dict[str, Any]) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    evt_headers = event.get("headers") or {}
    if isinstance(evt_headers, dict):
        headers.update(evt_headers)

    input_obj = event.get("input") or {}
    input_headers = input_obj.get("_headers") or {}
    if isinstance(input_headers, dict):
        headers.update(input_headers)

    normalized = {str(k).lower(): str(v) for k, v in headers.items()}
    return normalized


def validate_api_key(event: Dict[str, Any]) -> bool:
    if not API_KEY:
        logger.warning("API_KEY not set; authentication is DISABLED")
        return True
    headers = get_headers_from_event(event)
    provided = headers.get("x-api-key")
    if provided != API_KEY:
        logger.warning("API key validation failed")
        return False
    return True


def get_pipeline(model_name: str) -> HunyuanVideoPipeline:
    """Thread-safe pipeline loader with caching."""
    global PIPELINES

    key = model_name.strip()
    if not key:
        key = VIDEO_MODEL_NAME

    # Thread-safe check and load
    with PIPELINE_LOCK:
        # Double-check pattern: check again after acquiring lock
        if key in PIPELINES:
            logger.debug(f"Using cached pipeline: {key}")
            return PIPELINES[key]

        hf_id = key
        if "/" not in hf_id:
            hf_id = f"tencent/{hf_id}"

        logger.info(f"Loading HunyuanVideo pipeline: {hf_id} on device={DEVICE}")
        start = time.time()

        try:
            pipe = HunyuanVideoPipeline.from_pretrained(
                hf_id,
                torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            )
            if hasattr(pipe, "enable_attention_slicing"):
                pipe.enable_attention_slicing()  # type: ignore[attr-defined]
            pipe.to(DEVICE)

            load_time = time.time() - start
            logger.info(f"Loaded model {hf_id} in {load_time:.2f}s")

            if torch.cuda.is_available():
                total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({total:.2f} GB total)")

            PIPELINES[key] = pipe
            return pipe
        except Exception as exc:
            logger.error(f"Failed to load pipeline {hf_id}: {exc}")
            raise


def download_file(url: str, suffix: str) -> Optional[str]:
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir="/tmp") as tmp:
            tmp_path = tmp.name
        logger.info(f"Downloading from URL: {url}")
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        with open(tmp_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Downloaded file to: {tmp_path}")
        return tmp_path
    except Exception as exc:
        logger.error(f"Download failed for {url}: {exc}")
        logger.debug(traceback.format_exc())
        return None


def load_init_image(image_url: str) -> Optional[Image.Image]:
    """Load init image for image-to-video generation.
    
    Supports: JPG, JPEG, PNG formats
    Automatically converts to RGB format.
    """
    # Try common image formats
    for suffix in [".jpg", ".jpeg", ".png", ".webp"]:
        img_path = download_file(image_url, suffix=suffix)
        if img_path:
            break
    else:
        # If no suffix matches, try downloading without specific suffix
        img_path = download_file(image_url, suffix=".jpg")
    
    if not img_path:
        logger.error(f"Failed to download image from {image_url}")
        return None
    
    try:
        img = Image.open(img_path).convert("RGB")
        logger.info(f"Loaded init image: {img.size[0]}x{img.size[1]} pixels")
        return img
    except Exception as exc:
        logger.error(f"Failed to load image from {img_path}: {exc}")
        logger.debug(traceback.format_exc())
        return None
    finally:
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except OSError:
            pass


def get_audio_duration(audio_path: str) -> Optional[float]:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            audio_path,
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if res.returncode != 0:
            logger.error(f"ffprobe failed: {res.stderr}")
            return None
        duration_str = res.stdout.strip()
        if not duration_str:
            return None
        duration = float(duration_str)
        logger.info(f"Audio duration: {duration:.3f}s")
        return duration
    except Exception as exc:
        logger.error(f"Failed to get audio duration: {exc}")
        logger.debug(traceback.format_exc())
        return None


def generate_video(
    pipe: HunyuanVideoPipeline,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int,
    init_image: Optional[Image.Image] = None,
    request_id: Optional[str] = None,
) -> Optional[str]:
    """Generate video with GPU memory management."""
    req_id = f"[Request {request_id}] " if request_id else ""
    try:
        logger.info(
            f"{req_id}Generating video: {width}x{height}, frames={num_frames}, fps={fps}, "
            f"mode={'image-to-video' if init_image is not None else 'text-to-video'}"
        )
        
        # Clear GPU cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            alloc_before = torch.cuda.memory_allocated() / (1024**3)
            reserved_before = torch.cuda.memory_reserved() / (1024**3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available = total_memory - reserved_before
            logger.debug(
                f"{req_id}GPU memory: allocated={alloc_before:.2f}GB, "
                f"reserved={reserved_before:.2f}GB, available={available:.2f}GB"
            )
            
            # Warn if memory is getting low
            if available < 4.0:  # Less than 4GB available
                logger.warning(
                    f"{req_id}Low GPU memory available ({available:.2f}GB). "
                    f"Consider reducing MAX_CONCURRENT_GENERATIONS."
                )
        
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        start = time.time()

        call_kwargs: Dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_frames": num_frames,
            "num_inference_steps": 30,
            "generator": gen,
        }
        if init_image is not None:
            call_kwargs["image"] = init_image

        with torch.no_grad():
            out = pipe(**call_kwargs)

        frames = out.frames[0]  # type: ignore[index]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp") as tmp_vid:
            video_path = tmp_vid.name

        export_to_video(frames, video_path, fps=fps)

        elapsed = time.time() - start
        logger.info(f"{req_id}Video generated at {video_path} in {elapsed:.2f}s")

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            available = total - reserved
            logger.info(
                f"{req_id}GPU memory after generation - allocated: {alloc:.2f}GB, "
                f"reserved: {reserved:.2f}GB, available: {available:.2f}GB"
            )
            # Clear cache after generation to free memory
            torch.cuda.empty_cache()
            
            # Log if memory is now available for more requests
            if can_handle_more_requests():
                logger.debug(f"{req_id}GPU memory sufficient for additional requests")
            else:
                logger.warning(
                    f"{req_id}GPU memory still low ({available:.2f}GB). "
                    f"May need to wait before processing next request."
                )

        return video_path
    except torch.cuda.OutOfMemoryError as oom_exc:
        logger.error(f"{req_id}GPU out of memory during video generation: {oom_exc}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Log current memory state
            try:
                alloc = torch.cuda.memory_allocated() / (1024**3)
                reserved = torch.cuda.memory_reserved() / (1024**3)
                logger.error(
                    f"{req_id}GPU memory at OOM - allocated: {alloc:.2f}GB, reserved: {reserved:.2f}GB. "
                    f"Consider reducing MAX_CONCURRENT_GENERATIONS from {MAX_CONCURRENT_GENERATIONS} to 1"
                )
            except Exception:
                pass
        return None
    except Exception as exc:
        logger.error(f"{req_id}Video generation failed: {exc}")
        logger.debug(traceback.format_exc())
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None


def merge_audio_video(
    video_path: str,
    audio_path: str,
    target_duration: float,
) -> Optional[str]:
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False, dir="/tmp") as tmp_out:
            out_path = tmp_out.name

        logger.info(
            f"Merging audio {audio_path} into video {video_path}, target duration={target_duration:.3f}s"
        )

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-i",
            audio_path,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-t",
            str(target_duration),
            "-shortest",
            out_path,
        ]

        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=240,
        )
        if res.returncode != 0:
            logger.error(f"ffmpeg failed: {res.stderr}")
            try:
                os.remove(out_path)
            except OSError:
                pass
            return None

        logger.info(f"Audio-video merged to {out_path}")
        return out_path
    except subprocess.TimeoutExpired:
        logger.error("ffmpeg merge timed out")
        return None
    except Exception as exc:
        logger.error(f"Audio-video merge failed: {exc}")
        logger.debug(traceback.format_exc())
        return None


def get_video_metadata(video_path: str) -> Optional[Dict[str, Any]]:
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,r_frame_rate,duration",
            "-show_entries",
            "format=size",
            "-of",
            "json",
            video_path,
        ]
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=15,
        )
        if res.returncode != 0:
            logger.error(f"ffprobe metadata failed: {res.stderr}")
            return None

        data = json.loads(res.stdout)
        if not data.get("streams"):
            return None
        stream = data["streams"][0]
        fmt = data.get("format", {})

        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        duration = float(stream.get("duration", 0.0))
        size = int(fmt.get("size", 0))

        fps_str = stream.get("r_frame_rate", "30/1")
        num, den = fps_str.split("/")
        fps = float(num) / float(den) if float(den) != 0 else 30.0

        return {
            "width": width,
            "height": height,
            "duration": duration,
            "fps": fps,
            "size": size,
        }
    except Exception as exc:
        logger.error(f"Failed to read video metadata: {exc}")
        logger.debug(traceback.format_exc())
        return None


def upload_to_s3(file_path: str, object_name: str) -> Optional[str]:
    try:
        if not (AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY and S3_BUCKET_NAME):
            logger.error("S3 configuration is incomplete")
            return None
        import boto3

        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=S3_REGION,
        )
        logger.info(f"Uploading {file_path} to s3://{S3_BUCKET_NAME}/{object_name}")
        s3.upload_file(file_path, S3_BUCKET_NAME, object_name, ExtraArgs={"ContentType": "video/mp4"})

        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": S3_BUCKET_NAME, "Key": object_name},
            ExpiresIn=7 * 24 * 3600,
        )
        logger.info(f"S3 upload complete, presigned URL generated")
        return url
    except Exception as exc:
        logger.error(f"S3 upload failed: {exc}")
        logger.debug(traceback.format_exc())
        return None


def upload_to_gcs(file_path: str, object_name: str) -> Optional[str]:
    try:
        if not GCS_BUCKET_NAME:
            logger.error("GCS bucket name is not configured")
            return None
        
        # Ensure credentials are set up
        if not os.path.exists(GCS_CREDENTIALS_PATH):
            if not setup_gcs_credentials():
                logger.error("Failed to setup GCS credentials")
                return None
        
        from google.cloud import storage as gcs_storage  # type: ignore[import]

        client = gcs_storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(object_name)
        
        logger.info(f"Uploading {file_path} to gs://{GCS_BUCKET_NAME}/{object_name}")
        
        # Upload with metadata
        blob.upload_from_filename(
            file_path,
            content_type="video/mp4",
            timeout=300,  # 5 minute timeout
        )
        
        # Generate signed URL (valid for 7 days)
        url = blob.generate_signed_url(
            version="v4",
            expiration=7 * 24 * 3600,
            method="GET",
        )
        
        logger.info(f"GCS upload complete: {object_name} ({blob.size} bytes)")
        return url
    except Exception as exc:
        logger.error(f"GCS upload failed: {exc}")
        logger.debug(traceback.format_exc())
        return None


def upload_to_storage(file_path: str, object_name: str) -> Optional[str]:
    if STORAGE_TYPE == "s3":
        return upload_to_s3(file_path, object_name)
    if STORAGE_TYPE == "gcs":
        return upload_to_gcs(file_path, object_name)
    logger.error(f"Unsupported STORAGE_TYPE: {STORAGE_TYPE}")
    return None


def cleanup_files(paths: Any) -> None:
    if not paths:
        return
    for p in paths:
        if not p:
            continue
        try:
            if os.path.exists(p):
                os.remove(p)
                logger.info(f"Deleted temp file: {p}")
        except Exception as exc:
            logger.warning(f"Failed to delete temp file {p}: {exc}")


def build_error(code: str, message: str) -> Dict[str, Any]:
    return {"output": {"error": message, "code": code}}


def handler(event: Dict[str, Any]) -> Dict[str, Any]:
    """Main handler with concurrency control and request tracking."""
    start_time = time.time()
    tmp_files: list[str] = []
    request_id = str(uuid.uuid4())[:8]
    
    logger.info(f"[Request {request_id}] Starting video generation request")

    try:
        if not validate_api_key(event):
            logger.warning(f"[Request {request_id}] Authentication failed")
            return build_error("AUTH_FAILED", "Unauthorized")
        
        # Dynamic concurrency control: Check both semaphore AND GPU memory
        logger.debug(f"[Request {request_id}] Waiting for generation slot (max: {MAX_CONCURRENT_GENERATIONS})")
        
        # First, acquire semaphore (respects max concurrent limit)
        if not GENERATION_SEMAPHORE.acquire(timeout=300):  # 5 minute timeout
            logger.error(f"[Request {request_id}] Timeout waiting for generation slot")
            return build_error("TIMEOUT", "Service busy, please try again later")
        
        try:
            # Then, check if GPU has enough memory (dynamic check)
            available_memory = get_available_gpu_memory()
            if available_memory is not None:
                logger.debug(
                    f"[Request {request_id}] Acquired slot. GPU memory available: {available_memory:.2f}GB"
                )
                
                if not can_handle_more_requests():
                    logger.warning(
                        f"[Request {request_id}] Low GPU memory ({available_memory:.2f}GB). "
                        f"Waiting for memory to free up..."
                    )
                    # Wait for memory to become available
                    if not wait_for_gpu_memory(timeout=120):  # Wait up to 2 minutes for memory
                        logger.error(
                            f"[Request {request_id}] Timeout waiting for GPU memory. "
                            f"Current available: {available_memory:.2f}GB"
                        )
                        return build_error(
                            "GPU_MEMORY_UNAVAILABLE",
                            "GPU memory is currently unavailable. Please try again later."
                        )
                    logger.info(
                        f"[Request {request_id}] GPU memory available. Proceeding with generation."
                    )
        except Exception as exc:
            logger.warning(f"[Request {request_id}] Error checking GPU memory: {exc}. Proceeding anyway.")
        
        try:
            return _process_request(event, request_id, tmp_files, start_time)
        finally:
            # Always release semaphore
            GENERATION_SEMAPHORE.release()
            logger.debug(f"[Request {request_id}] Released generation slot")
            
    except Exception as exc:
        logger.error(f"[Request {request_id}] Unhandled exception: {exc}")
        logger.debug(traceback.format_exc())
        return build_error("INTERNAL_ERROR", "Internal server error")
    finally:
        cleanup_files(tmp_files)
        elapsed = time.time() - start_time
        logger.info(f"[Request {request_id}] Request completed in {elapsed:.2f}s")


def _process_request(
    event: Dict[str, Any],
    request_id: str,
    tmp_files: list[str],
    start_time: float,
) -> Dict[str, Any]:
    """Process video generation request (called within semaphore)."""
    input_payload = event.get("input") or {}
    script = input_payload.get("script")
    audio_url = input_payload.get("audio_url")
    duration = input_payload.get("duration")
    aspect_ratio = input_payload.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
    style = input_payload.get("style")
    fps = input_payload.get("fps")
    seed = input_payload.get("seed", DEFAULT_SEED)
    model_override = input_payload.get("model")
    image_url = input_payload.get("image_url")  # optional for image-to-video
    
    # Extract user and project IDs for storage path
    user_id = input_payload.get("user_id")
    project_id = input_payload.get("project_id")
    
    if not user_id or not project_id:
        return build_error(
            "INVALID_INPUT",
            "Missing required fields: user_id, project_id",
        )
    
    # Validate user_id and project_id format
    if not isinstance(user_id, str) or not user_id.strip():
        return build_error("INVALID_INPUT", "user_id must be a non-empty string")
    if not isinstance(project_id, str) or not project_id.strip():
        return build_error("INVALID_INPUT", "project_id must be a non-empty string")

    if script is None or audio_url is None or duration is None:
        return build_error(
            "INVALID_INPUT",
            "Missing required fields: script, audio_url, duration",
        )

    if not isinstance(script, str) or not script.strip():
        return build_error("INVALID_INPUT", "script must be a non-empty string")

    try:
        duration = int(duration)
    except Exception:
        return build_error("INVALID_DURATION", "duration must be an integer")

    if duration not in VALID_DURATIONS:
        return build_error(
            "INVALID_DURATION",
            f"duration must be one of {sorted(VALID_DURATIONS)}",
        )

    if aspect_ratio not in RESOLUTION_MAP:
        return build_error(
            "INVALID_ASPECT_RATIO",
            f"aspect_ratio must be one of {list(RESOLUTION_MAP.keys())}",
        )

    if fps is None:
        if isinstance(style, str) and style.lower() == "cinematic":
            fps_val = 24
        else:
            fps_val = DEFAULT_FPS
    else:
        try:
            fps_val = int(fps)
        except Exception:
            return build_error("INVALID_FPS", "fps must be an integer")
        if fps_val <= 0:
            return build_error("INVALID_FPS", "fps must be positive")

    try:
        seed_val = int(seed)
    except Exception:
        seed_val = DEFAULT_SEED

    width, height = RESOLUTION_MAP[aspect_ratio]

    logger.info(
        f"[Request {request_id}] Processing: duration={duration}s, aspect_ratio={aspect_ratio}, "
        f"fps={fps_val}, seed={seed_val}, model={model_override or VIDEO_MODEL_NAME}, "
        f"mode={'image+text' if image_url else 'text-only'}"
    )

    audio_path = download_file(audio_url, suffix=".m4a")
    if not audio_path:
        return build_error("AUDIO_DOWNLOAD_FAILED", "Failed to download audio")
    tmp_files.append(audio_path)

    audio_duration = get_audio_duration(audio_path)
    if audio_duration is None or audio_duration <= 0:
        logger.warning(f"[Request {request_id}] Could not determine audio duration; falling back to requested duration")
        target_duration = float(duration)
    else:
        target_duration = float(audio_duration)

    if abs(target_duration - duration) > 1.0:
        logger.info(
            f"[Request {request_id}] Audio duration ({target_duration:.2f}s) differs from requested duration "
            f"({duration}s) by >1s; using audio duration for final video length."
        )

    num_frames = max(1, int(round(target_duration * fps_val)))

    model_name = model_override or VIDEO_MODEL_NAME
    pipe = get_pipeline(model_name)

    init_image = None
    if image_url:
        init_image = load_init_image(image_url)
        if init_image is None:
            return build_error("IMAGE_DOWNLOAD_FAILED", "Failed to load init image for image-to-video")

    full_prompt = script.strip()
    if style:
        full_prompt = f"{full_prompt} Style: {style}"

    logger.info(f"[Request {request_id}] Starting video generation...")
    video_path = generate_video(
        pipe=pipe,
        prompt=full_prompt,
        width=width,
        height=height,
        num_frames=num_frames,
        fps=fps_val,
        seed=seed_val,
        init_image=init_image,
        request_id=request_id,
    )
    if not video_path:
        return build_error("GENERATION_FAILED", "Video generation failed")
    tmp_files.append(video_path)

    logger.info(f"[Request {request_id}] Merging audio and video...")
    merged_path = merge_audio_video(
        video_path=video_path,
        audio_path=audio_path,
        target_duration=target_duration,
    )
    if not merged_path:
        return build_error("MERGE_FAILED", "Failed to merge audio and video")
    tmp_files.append(merged_path)

    meta = get_video_metadata(merged_path)
    if not meta:
        return build_error("METADATA_FAILED", "Failed to read video metadata")

    # Generate unique filename with better hash
    file_hash = hashlib.sha256(
        f"{uuid.uuid4()}_{time.time()}_{seed_val}_{duration}".encode("utf-8")
    ).hexdigest()[:16]
    
    # Use structured path: autopostai-media/{user_id}/projects/{project_id}/videos/{filename}
    filename = f"video_{file_hash}.mp4"
    object_name = f"autopostai-media/{user_id.strip()}/projects/{project_id.strip()}/videos/{filename}"

    logger.info(f"[Request {request_id}] Uploading to storage: {object_name}")
    url = upload_to_storage(merged_path, object_name)
    if not url:
        return build_error("UPLOAD_FAILED", "Failed to upload video to storage")

    elapsed = time.time() - start_time
    logger.info(f"[Request {request_id}] Job completed successfully in {elapsed:.2f}s")

    response = {
        "video_url": url,
        "duration": float(meta["duration"]),
        "resolution": f"{meta['width']}x{meta['height']}",
        "fps": float(meta["fps"]),
        "format": "mp4",
        "file_size": int(meta["size"]),
        "storage_path": object_name,
        "user_id": user_id,
        "project_id": project_id,
        "request_id": request_id,
    }
    return {"output": response}


if __name__ == "__main__":
    # Simple local smoke test harness (no real model call)
    print("Handler module loaded. This entrypoint is for local testing only.")
