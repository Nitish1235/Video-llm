import os
import json
import logging
import traceback
import subprocess
import tempfile
import time
import hashlib
from typing import Any, Dict, Optional, Tuple

import requests
import torch
from PIL import Image

import modal

try:
    from diffusers import HunyuanVideoPipeline
except ImportError:  # fallback for older diffusers versions
    from diffusers import DiffusionPipeline as HunyuanVideoPipeline  # type: ignore

from diffusers.utils import export_to_video

STORAGE_TYPE = os.getenv("STORAGE_TYPE", "s3").lower()
VIDEO_MODEL_NAME = os.getenv("VIDEO_MODEL_NAME", "HunyuanVideo-1.5")
API_KEY = os.getenv("API_KEY")

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION", "us-east-1")

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

VALID_DURATIONS = {20, 30, 45, 60}
DEFAULT_ASPECT_RATIO = "9:16"
DEFAULT_FPS = 30
DEFAULT_SEED = 42

RESOLUTION_MAP: Dict[str, Tuple[int, int]] = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
    "1:1": (1080, 1080),
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("hunyuan_video_modal")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "git", "wget", "curl")
    .pip_install(
        "torch==2.1.2",
        "torchvision==0.16.2",
        "torchaudio==2.1.2",
        "diffusers==0.30.1",
        "transformers==4.40.2",
        "safetensors==0.4.4",
        "accelerate==0.32.0",
        "Pillow==10.2.0",
        "requests==2.31.0",
        "boto3==1.34.144",
        "google-cloud-storage==2.14.0",
        "numpy==1.24.3",
        "ffmpeg-python==0.2.0",
    )
)

app = modal.App("hunyuan-video-service", image=image)


def _download_file(url: str, suffix: str) -> Optional[str]:
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


def _load_init_image(image_url: str) -> Optional[Image.Image]:
    img_path = _download_file(image_url, suffix=".png")
    if not img_path:
        return None
    try:
        img = Image.open(img_path).convert("RGB")
        return img
    except Exception as exc:
        logger.error(f"Failed to load image from {img_path}: {exc}")
        return None
    finally:
        try:
            if os.path.exists(img_path):
                os.remove(img_path)
        except OSError:
            pass


def _get_audio_duration(audio_path: str) -> Optional[float]:
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


def _generate_video(
    pipe: HunyuanVideoPipeline,
    prompt: str,
    width: int,
    height: int,
    num_frames: int,
    fps: int,
    seed: int,
    init_image: Optional[Image.Image] = None,
) -> Optional[str]:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(
            f"Generating video: {width}x{height}, frames={num_frames}, fps={fps}, "
            f"mode={'image-to-video' if init_image is not None else 'text-to-video'} on {device}"
        )
        gen = torch.Generator(device=device).manual_seed(seed)
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
        logger.info(f"Video generated at {video_path} in {elapsed:.2f}s")

        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated() / (1024**3)
            reserved = torch.cuda.memory_reserved() / (1024**3)
            logger.info(
                f"GPU memory used - allocated: {alloc:.2f} GB, reserved: {reserved:.2f} GB"
            )

        return video_path
    except Exception as exc:
        logger.error(f"Video generation failed: {exc}")
        logger.debug(traceback.format_exc())
        return None


def _merge_audio_video(
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


def _get_video_metadata(video_path: str) -> Optional[Dict[str, Any]]:
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


def _upload_to_s3(file_path: str, object_name: str) -> Optional[str]:
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


def _upload_to_gcs(file_path: str, object_name: str) -> Optional[str]:
    try:
        if not GCS_BUCKET_NAME:
            logger.error("GCS bucket name is not configured")
            return None
        from google.cloud import storage as gcs_storage  # type: ignore[import]

        client = gcs_storage.Client()
        bucket = client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(object_name)
        logger.info(f"Uploading {file_path} to gs://{GCS_BUCKET_NAME}/{object_name}")
        blob.upload_from_filename(file_path, content_type="video/mp4")
        url = blob.generate_signed_url(
            version="v4",
            expiration=7 * 24 * 3600,
            method="GET",
        )
        logger.info("GCS upload complete, signed URL generated")
        return url
    except Exception as exc:
        logger.error(f"GCS upload failed: {exc}")
        logger.debug(traceback.format_exc())
        return None


def _upload_to_storage(file_path: str, object_name: str) -> Optional[str]:
    if STORAGE_TYPE == "s3":
        return _upload_to_s3(file_path, object_name)
    if STORAGE_TYPE == "gcs":
        return _upload_to_gcs(file_path, object_name)
    logger.error(f"Unsupported STORAGE_TYPE: {STORAGE_TYPE}")
    return None


def _cleanup_files(paths: Any) -> None:
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


def _build_error(code: str, message: str) -> Dict[str, Any]:
    return {"error": message, "code": code}


def _validate_api_key_from_input(input_payload: Dict[str, Any]) -> bool:
    if not API_KEY:
        logger.warning("API_KEY not set; authentication is DISABLED")
        return True
    provided = input_payload.get("api_key") or input_payload.get("x-api-key")
    if provided != API_KEY:
        logger.warning("API key validation failed (Modal)")
        return False
    return True


@app.cls(gpu="A100", timeout=600, concurrency_limit=1)
class HunyuanVideoWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipelines: Dict[str, Any] = {}

    def _get_pipeline(self, model_name: str) -> HunyuanVideoPipeline:
        key = model_name.strip() or VIDEO_MODEL_NAME
        if key in self.pipelines:
            return self.pipelines[key]

        hf_id = key
        if "/" not in hf_id:
            hf_id = f"tencent/{hf_id}"

        logger.info(f"[Modal] Loading HunyuanVideo pipeline: {hf_id} on device={self.device}")
        start = time.time()

        pipe = HunyuanVideoPipeline.from_pretrained(
            hf_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()  # type: ignore[attr-defined]
        pipe.to(self.device)

        load_time = time.time() - start
        logger.info(f"[Modal] Loaded model {hf_id} in {load_time:.2f}s")
        if torch.cuda.is_available():
            total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"[Modal] GPU: {torch.cuda.get_device_name(0)} ({total:.2f} GB total)")

        self.pipelines[key] = pipe
        return pipe

    @modal.method()
    def generate(self, input_payload: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        tmp_files: list[str] = []

        try:
            if not _validate_api_key_from_input(input_payload):
                return _build_error("AUTH_FAILED", "Unauthorized")

            script = input_payload.get("script")
            audio_url = input_payload.get("audio_url")
            duration = input_payload.get("duration")
            aspect_ratio = input_payload.get("aspect_ratio", DEFAULT_ASPECT_RATIO)
            style = input_payload.get("style")
            fps = input_payload.get("fps")
            seed = input_payload.get("seed", DEFAULT_SEED)
            model_override = input_payload.get("model")
            image_url = input_payload.get("image_url")  # optional

            if script is None or audio_url is None or duration is None:
                return _build_error(
                    "INVALID_INPUT",
                    "Missing required fields: script, audio_url, duration",
                )

            if not isinstance(script, str) or not script.strip():
                return _build_error("INVALID_INPUT", "script must be a non-empty string")

            try:
                duration = int(duration)
            except Exception:
                return _build_error("INVALID_DURATION", "duration must be an integer")

            if duration not in VALID_DURATIONS:
                return _build_error(
                    "INVALID_DURATION",
                    f"duration must be one of {sorted(VALID_DURATIONS)}",
                )

            if aspect_ratio not in RESOLUTION_MAP:
                return _build_error(
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
                    return _build_error("INVALID_FPS", "fps must be an integer")
                if fps_val <= 0:
                    return _build_error("INVALID_FPS", "fps must be positive")

            try:
                seed_val = int(seed)
            except Exception:
                seed_val = DEFAULT_SEED

            width, height = RESOLUTION_MAP[aspect_ratio]

            logger.info(
                f"[Modal] Request: duration={duration}s, aspect_ratio={aspect_ratio}, "
                f"fps={fps_val}, seed={seed_val}, model={model_override or VIDEO_MODEL_NAME}, "
                f"mode={'image+text' if image_url else 'text-only'}"
            )

            audio_path = _download_file(audio_url, suffix=".m4a")
            if not audio_path:
                return _build_error("AUDIO_DOWNLOAD_FAILED", "Failed to download audio")
            tmp_files.append(audio_path)

            audio_duration = _get_audio_duration(audio_path)
            if audio_duration is None or audio_duration <= 0:
                logger.warning(
                    "[Modal] Could not determine audio duration; falling back to requested duration"
                )
                target_duration = float(duration)
            else:
                target_duration = float(audio_duration)

            if abs(target_duration - duration) > 1.0:
                logger.info(
                    f"[Modal] Audio duration ({target_duration:.2f}s) differs from requested duration "
                    f"({duration}s) by >1s; using audio duration for final video length."
                )

            num_frames = max(1, int(round(target_duration * fps_val)))

            model_name = model_override or VIDEO_MODEL_NAME
            pipe = self._get_pipeline(model_name)

            init_image = None
            if image_url:
                init_image = _load_init_image(image_url)
                if init_image is None:
                    return _build_error(
                        "IMAGE_DOWNLOAD_FAILED",
                        "Failed to load init image for image-to-video",
                    )

            full_prompt = script.strip()
            if style:
                full_prompt = f"{full_prompt} Style: {style}"

            video_path = _generate_video(
                pipe=pipe,
                prompt=full_prompt,
                width=width,
                height=height,
                num_frames=num_frames,
                fps=fps_val,
                seed=seed_val,
                init_image=init_image,
            )
            if not video_path:
                return _build_error("GENERATION_FAILED", "Video generation failed")
            tmp_files.append(video_path)

            merged_path = _merge_audio_video(
                video_path=video_path,
                audio_path=audio_path,
                target_duration=target_duration,
            )
            if not merged_path:
                return _build_error("MERGE_FAILED", "Failed to merge audio and video")
            tmp_files.append(merged_path)

            meta = _get_video_metadata(merged_path)
            if not meta:
                return _build_error("METADATA_FAILED", "Failed to read video metadata")

            file_hash = hashlib.md5(
                f"{time.time()}_{seed_val}_{duration}".encode("utf-8")
            ).hexdigest()[:12]
            object_name = f"videos/hunyuan_{file_hash}.mp4"

            url = _upload_to_storage(merged_path, object_name)
            if not url:
                return _build_error("UPLOAD_FAILED", "Failed to upload video to storage")

            elapsed = time.time() - start_time
            logger.info(f"[Modal] Job completed in {elapsed:.2f}s")

            return {
                "video_url": url,
                "duration": float(meta["duration"]),
                "resolution": f"{meta['width']}x{meta['height']}",
                "fps": float(meta["fps"]),
                "format": "mp4",
                "file_size": int(meta["size"]),
            }

        except Exception as exc:
            logger.error(f"[Modal] Unhandled exception: {exc}")
            logger.debug(traceback.format_exc())
            return _build_error("INTERNAL_ERROR", "Internal server error")
        finally:
            _cleanup_files(tmp_files)


@app.function(image=image, timeout=600)
def modal_entrypoint(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Modal entrypoint that mirrors the Runpod input schema.
    Expects {"input": {...}} and returns {"output": {...}}.
    """
    input_payload = event.get("input") or {}
    worker = HunyuanVideoWorker()
    result = worker.generate.remote(input_payload)

    if "error" in result:
        return {"output": result}
    return {"output": result}


if __name__ == "__main__":
    print("Modal app module loaded. Use `modal run modal_app.py::modal_entrypoint` to invoke.")
