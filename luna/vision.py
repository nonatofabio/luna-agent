"""Vision system: camera capture and image analysis via LLM."""

from __future__ import annotations

import asyncio
import base64
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import openai

from luna.config import VisionConfig, LLMConfig
from luna.observe import get_logger, log_event, log_duration

logger = get_logger("vision")


class VisionService:
    """Captures images from a USB webcam and analyzes them via a vision LLM.

    Supports two backends:
    1. Local llama-server (if the loaded model supports vision)
    2. OpenAI GPT-4o fallback (requires OPENAI_API_KEY)

    Camera and LLM client are lazy-initialized on first use.
    """

    def __init__(self, config: VisionConfig, llm_config: LLMConfig) -> None:
        self.config = config
        self._llm_config = llm_config
        self._capture_dir = Path(config.capture_dir)
        self._capture_dir.mkdir(parents=True, exist_ok=True)
        self._cap = None  # cv2.VideoCapture, lazy
        self._client: openai.AsyncOpenAI | None = None
        self._model: str = ""
        self._backend: str = ""  # "local" or "openai"
        log_event(logger, "vision_initialized",
                  device=config.device, capture_dir=config.capture_dir)

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------

    def _ensure_camera(self) -> Any:
        """Open the camera on first use. Returns cv2.VideoCapture."""
        if self._cap is not None:
            return self._cap
        import cv2
        self._cap = cv2.VideoCapture(self.config.device)
        if not self._cap.isOpened():
            self._cap = None
            raise RuntimeError(
                f"Cannot open camera /dev/video{self.config.device}. "
                "Check that the webcam is connected and not in use."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        log_event(logger, "camera_opened", device=self.config.device)
        return self._cap

    def _ensure_llm(self) -> tuple[openai.AsyncOpenAI, str]:
        """Set up the vision LLM client on first use."""
        if self._client is not None:
            return self._client, self._model

        # Try local llama-server first
        endpoint = self.config.llm_endpoint or self._llm_config.endpoint
        model = self.config.llm_model or self._llm_config.model

        if self.config.fallback_api_key:
            # Use OpenAI GPT-4o as the vision backend
            self._client = openai.AsyncOpenAI(api_key=self.config.fallback_api_key)
            self._model = "gpt-4o"
            self._backend = "openai"
            log_event(logger, "vision_llm_ready", backend="openai", model=self._model)
        else:
            # Use local llama-server (same as main LLM)
            self._client = openai.AsyncOpenAI(
                base_url=endpoint,
                api_key="not-needed",
            )
            self._model = model
            self._backend = "local"
            log_event(logger, "vision_llm_ready", backend="local",
                      endpoint=endpoint, model=model)

        return self._client, self._model

    # ------------------------------------------------------------------
    # Camera operations
    # ------------------------------------------------------------------

    async def capture_image(self, filename: str | None = None) -> str:
        """Capture a single frame from the webcam. Returns the saved file path."""
        import cv2

        def _capture() -> str:
            cap = self._ensure_camera()
            # Warm up: discard a few frames so auto-exposure settles
            for _ in range(5):
                cap.read()
            ret, frame = cap.read()
            if not ret or frame is None:
                raise RuntimeError("Failed to capture frame from camera.")

            if filename:
                name = filename if filename.endswith((".jpg", ".png")) else f"{filename}.jpg"
            else:
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                name = f"capture_{ts}.jpg"

            path = self._capture_dir / name
            cv2.imwrite(str(path), frame)
            log_event(logger, "image_captured", path=str(path),
                      width=frame.shape[1], height=frame.shape[0])
            return str(path)

        return await asyncio.to_thread(_capture)

    # ------------------------------------------------------------------
    # Image analysis via LLM
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_image(image_path: str) -> str:
        """Read an image file and return its base64 encoding."""
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("utf-8")

    async def _ask_vision(self, image_path: str, prompt: str) -> str:
        """Send an image + prompt to the vision LLM and return the response text."""
        client, model = self._ensure_llm()
        b64 = self._encode_image(image_path)

        # Determine MIME type
        suffix = Path(image_path).suffix.lower()
        mime = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                "png": "image/png", "webp": "image/webp"}.get(
            suffix.lstrip("."), "image/jpeg")

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                        },
                    },
                ],
            }
        ]

        with log_duration(logger, "vision_llm_call", model=model, backend=self._backend):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=1024,
                    temperature=0.3,
                )
                result = response.choices[0].message.content or ""
                log_event(logger, "vision_llm_response",
                          result_len=len(result),
                          prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                          completion_tokens=response.usage.completion_tokens if response.usage else 0)
                return result
            except openai.BadRequestError as e:
                # Model doesn't support vision — give a clear error
                error_msg = str(e)
                if "image" in error_msg.lower() or "vision" in error_msg.lower() or "multimodal" in error_msg.lower():
                    return (
                        f"Error: The current LLM ({model}) does not support vision/image input. "
                        "Set OPENAI_API_KEY env var to enable GPT-4o fallback, or load a "
                        "vision-capable model (e.g., Llama 3.2 Vision, Qwen2-VL) in llama-server."
                    )
                raise

    async def analyze_image(self, image_path: str, prompt: str | None = None) -> str:
        """Analyze an image: describe what's visible.

        Args:
            image_path: Path to the image file.
            prompt: Custom prompt. Defaults to general scene description.
        """
        if not Path(image_path).exists():
            return f"Error: Image not found: {image_path}"

        if prompt is None:
            prompt = (
                "Describe this image in detail. Include: the main subject, "
                "notable objects, colors, lighting, any text visible, and the "
                "overall scene or setting."
            )
        return await self._ask_vision(image_path, prompt)

    async def detect_objects(self, image_path: str) -> str:
        """Detect and list objects visible in the image."""
        if not Path(image_path).exists():
            return f"Error: Image not found: {image_path}"

        prompt = (
            "List every distinct object you can identify in this image. "
            "For each object, provide:\n"
            "- Name of the object\n"
            "- Approximate location (e.g., center, top-left, background)\n"
            "- Any notable attributes (color, size, state)\n\n"
            "Format as a numbered list. Be thorough but only list objects "
            "you can clearly identify."
        )
        return await self._ask_vision(image_path, prompt)

    async def extract_text(self, image_path: str) -> str:
        """Extract any visible text (OCR-style) from the image."""
        if not Path(image_path).exists():
            return f"Error: Image not found: {image_path}"

        prompt = (
            "Extract ALL text visible in this image. Reproduce the text exactly "
            "as it appears, preserving line breaks and formatting where possible. "
            "If there are multiple text regions, separate them with blank lines "
            "and note their approximate location. If no text is visible, say "
            "'No text detected.'"
        )
        return await self._ask_vision(image_path, prompt)

    async def capture_and_analyze(self, prompt: str | None = None) -> str:
        """Capture a frame and immediately analyze it. Convenience method."""
        path = await self.capture_image()
        analysis = await self.analyze_image(path, prompt)
        return f"[Captured: {path}]\n\n{analysis}"

    # ------------------------------------------------------------------
    # Image serving (for sharing with users)
    # ------------------------------------------------------------------

    def get_image_url(self, image_path: str) -> str:
        """Return a URL where the image can be accessed.

        For now, returns the local file path. In a full deployment this
        would return an HTTP URL from an image-serving endpoint.
        """
        return f"file://{image_path}"

    def list_captures(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent captured images."""
        images = sorted(
            self._capture_dir.glob("*.jpg"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:limit]
        results = []
        for img in images:
            stat = img.stat()
            results.append({
                "path": str(img),
                "filename": img.name,
                "size_kb": round(stat.st_size / 1024, 1),
                "captured_at": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            })
        return results

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release camera resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            log_event(logger, "camera_released")
