#!/usr/bin/env python3
"""
Generate American English TTS audio for phoneme test cases.

Requires OPENAI_API_KEY and ffmpeg on PATH.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
import sys
from openai import OpenAI

sys.path.append(str(Path(__file__).resolve().parents[1]))
from scripts.phoneme_suite_cases import TEST_CASES  # type: ignore


def synthesize_case(client: OpenAI, slug: str, text: str, out_dir: Path) -> None:
    wav_path = out_dir / f"{slug}.wav"
    if wav_path.exists():
        print(f"[skip] {slug} already exists")
        return

    mp3_path = out_dir / f"{slug}.mp3"
    print(f"[tts] Generating {slug}: {text}")
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )
    mp3_path.write_bytes(response.read())

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(mp3_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(wav_path),
        ],
        check=True,
    )
    print(f"[ok] wrote {wav_path}")


def main() -> None:
    client = OpenAI()
    out_dir = Path("tests/audio/phoneme_suite")
    out_dir.mkdir(parents=True, exist_ok=True)

    for case in TEST_CASES:
        synthesize_case(client, case["slug"], case["expected_text"], out_dir)


if __name__ == "__main__":
    main()
