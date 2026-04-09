"""Tests for the vision system."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from luna.config import VisionConfig, LLMConfig


# ── VisionService unit tests (camera mocked) ──────────────────────


@pytest.fixture
def vision_config(tmp_path):
    return VisionConfig(
        enabled=True,
        device=0,
        capture_dir=str(tmp_path / "captures"),
        width=640,
        height=480,
    )


@pytest.fixture
def llm_config():
    return LLMConfig(endpoint="http://localhost:8001/v1", model="test-model")


@pytest.fixture
def vision_service(vision_config, llm_config):
    from luna.vision import VisionService
    return VisionService(vision_config, llm_config)


def test_capture_dir_created(vision_config, llm_config):
    """VisionService creates the capture directory on init."""
    from luna.vision import VisionService
    svc = VisionService(vision_config, llm_config)
    assert Path(vision_config.capture_dir).exists()


async def test_capture_image(vision_service, tmp_path):
    """capture_image saves a JPEG to the capture dir."""
    import numpy as np

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    fake_frame[100:200, 100:200] = [0, 255, 0]  # green square

    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.read.return_value = (True, fake_frame)

    with patch("cv2.VideoCapture", return_value=mock_cap), \
         patch("cv2.imwrite") as mock_write:
        mock_write.return_value = True
        # Make imwrite actually write a file so we can check
        def real_write(path, frame):
            Path(path).write_bytes(b"FAKEJPEG")
            return True
        mock_write.side_effect = real_write

        path = await vision_service.capture_image("test_shot")
        assert "test_shot.jpg" in path
        assert Path(path).exists()


async def test_capture_image_no_camera(vision_service):
    """capture_image raises RuntimeError when camera can't open."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        with pytest.raises(RuntimeError, match="Cannot open camera"):
            await vision_service.capture_image()


async def test_analyze_missing_image(vision_service):
    """analyze_image returns error for non-existent file."""
    result = await vision_service.analyze_image("/nonexistent/photo.jpg")
    assert "Error: Image not found" in result


async def test_detect_objects_missing_image(vision_service):
    """detect_objects returns error for non-existent file."""
    result = await vision_service.detect_objects("/nonexistent/photo.jpg")
    assert "Error: Image not found" in result


async def test_extract_text_missing_image(vision_service):
    """extract_text returns error for non-existent file."""
    result = await vision_service.extract_text("/nonexistent/photo.jpg")
    assert "Error: Image not found" in result


async def test_list_captures_empty(vision_service):
    """list_captures returns empty list when no images exist."""
    captures = vision_service.list_captures()
    assert captures == []


async def test_list_captures_with_files(vision_service):
    """list_captures finds existing .jpg files."""
    cap_dir = Path(vision_service.config.capture_dir)
    (cap_dir / "img1.jpg").write_bytes(b"JPEG1")
    (cap_dir / "img2.jpg").write_bytes(b"JPEG2")
    (cap_dir / "not_image.txt").write_text("nope")

    captures = vision_service.list_captures()
    assert len(captures) == 2
    assert all(c["filename"].endswith(".jpg") for c in captures)


def test_close_releases_camera(vision_service):
    """close() releases the camera if it was opened."""
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    vision_service._cap = mock_cap
    vision_service.close()
    mock_cap.release.assert_called_once()
    assert vision_service._cap is None


def test_close_noop_when_no_camera(vision_service):
    """close() is safe to call when camera was never opened."""
    vision_service.close()  # should not raise


# ── Tool handler tests ─────────────────────────────────────────────


async def test_vision_tools_disabled():
    """Vision tools return error when vision is not enabled."""
    from luna.tools import (
        _tool_vision_capture,
        _tool_vision_analyze,
        _tool_vision_detect_objects,
        _tool_vision_extract_text,
        _tool_vision_list_captures,
    )
    import luna.tools as tools_mod
    old = tools_mod._vision_service
    tools_mod._vision_service = None
    try:
        for handler in [
            _tool_vision_capture,
            _tool_vision_analyze,
            _tool_vision_detect_objects,
            _tool_vision_extract_text,
            _tool_vision_list_captures,
        ]:
            result = await handler({})
            assert "not enabled" in result.lower()
    finally:
        tools_mod._vision_service = old
