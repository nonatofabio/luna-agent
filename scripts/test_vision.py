#!/usr/bin/env python3
"""Standalone vision system test script.

Usage:
    # Activate venv first
    source .venv/bin/activate

    # Test camera capture only (no LLM needed)
    python scripts/test_vision.py capture

    # Test full analysis (requires vision-capable LLM or OPENAI_API_KEY)
    python scripts/test_vision.py analyze

    # Test object detection
    python scripts/test_vision.py detect

    # Test text extraction (OCR)
    python scripts/test_vision.py ocr

    # Analyze an existing image file
    python scripts/test_vision.py analyze /path/to/image.jpg

    # List captured images
    python scripts/test_vision.py list
"""

import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from luna.config import load_config
from luna.vision import VisionService


async def main():
    config = load_config()

    if not config.vision.enabled:
        print("Vision is disabled in config.toml. Set [vision] enabled = true")
        sys.exit(1)

    vision = VisionService(config.vision, config.llm)

    command = sys.argv[1] if len(sys.argv) > 1 else "capture"
    image_path = sys.argv[2] if len(sys.argv) > 2 else None

    try:
        if command == "capture":
            print("Capturing image from webcam...")
            t0 = time.monotonic()
            path = await vision.capture_image()
            elapsed = time.monotonic() - t0
            print(f"Saved: {path} ({elapsed:.1f}s)")

        elif command == "analyze":
            if image_path:
                print(f"Analyzing: {image_path}")
            else:
                print("Capturing and analyzing...")
            t0 = time.monotonic()
            if image_path:
                result = await vision.analyze_image(image_path)
            else:
                result = await vision.capture_and_analyze()
            elapsed = time.monotonic() - t0
            print(f"\n{result}\n\n({elapsed:.1f}s)")

        elif command == "detect":
            if not image_path:
                print("Capturing image first...")
                image_path = await vision.capture_image()
                print(f"Saved: {image_path}")
            print(f"Detecting objects in: {image_path}")
            t0 = time.monotonic()
            result = await vision.detect_objects(image_path)
            elapsed = time.monotonic() - t0
            print(f"\n{result}\n\n({elapsed:.1f}s)")

        elif command == "ocr":
            if not image_path:
                print("Capturing image first...")
                image_path = await vision.capture_image()
                print(f"Saved: {image_path}")
            print(f"Extracting text from: {image_path}")
            t0 = time.monotonic()
            result = await vision.extract_text(image_path)
            elapsed = time.monotonic() - t0
            print(f"\n{result}\n\n({elapsed:.1f}s)")

        elif command == "list":
            captures = vision.list_captures()
            if not captures:
                print("No captured images found.")
            else:
                print(f"Found {len(captures)} captures:")
                for c in captures:
                    print(f"  {c['filename']}  {c['size_kb']:>7} KB  {c['captured_at']}")
                    print(f"    {c['path']}")

        else:
            print(f"Unknown command: {command}")
            print("Commands: capture, analyze, detect, ocr, list")
            sys.exit(1)

    finally:
        vision.close()


if __name__ == "__main__":
    asyncio.run(main())
