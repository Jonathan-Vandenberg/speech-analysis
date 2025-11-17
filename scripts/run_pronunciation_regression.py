#!/usr/bin/env python3
"""Generate TTS samples and run pronunciation regression tests."""
import json
import os
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import sys

from dotenv import load_dotenv
from fastapi.testclient import TestClient
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from main.app import app

load_dotenv()

API_KEY = os.getenv("AUDIO_ANALYSIS_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("AUDIO_ANALYSIS_API_KEY missing")
if not OPENAI_KEY:
    raise SystemExit("OPENAI_API_KEY missing")

client = TestClient(app)
tts_client = OpenAI(api_key=OPENAI_KEY)
audio_dir = Path("tests/audio/regression")
audio_dir.mkdir(parents=True, exist_ok=True)

TEST_CASES = [
    {"name": "basic_match", "category": "Basic", "type": "match",
     "expected": "Hello world.", "actual": "Hello world."},
    {"name": "basic_mismatch", "category": "Basic", "type": "mismatch",
     "expected": "Hello world.", "actual": "Goodbye universe."},
    {"name": "consonant_match", "category": "Consonant", "type": "match",
     "expected": "Peter Piper picked a peck of pickled peppers.",
     "actual": "Peter Piper picked a peck of pickled peppers."},
    {"name": "consonant_mismatch", "category": "Consonant", "type": "mismatch",
     "expected": "Peter Piper picked a peck of pickled peppers.",
     "actual": "Betty Botter bought some bitter butter."},
    {"name": "vowel_match", "category": "Vowel", "type": "match",
     "expected": "A unique audio of open ocean echoes.",
     "actual": "A unique audio of open ocean echoes."},
    {"name": "vowel_mismatch", "category": "Vowel", "type": "mismatch",
     "expected": "A unique audio of open ocean echoes.",
     "actual": "Under autumn umbrellas we gather quietly."},
    {"name": "long_match", "category": "Long", "type": "match",
     "expected": "In the grand library of Alexandria, scholars gathered to debate complex problems while gentle breezes carried the scent of papyrus.",
     "actual": "In the grand library of Alexandria, scholars gathered to debate complex problems while gentle breezes carried the scent of papyrus."},
    {"name": "long_mismatch", "category": "Long", "type": "mismatch",
     "expected": "In the grand library of Alexandria, scholars gathered to debate complex problems while gentle breezes carried the scent of papyrus.",
     "actual": "Modern satellites orbit Earth while engineers in mission control whisper about telemetry and solar winds."},
    {"name": "short_single_match", "category": "Short", "type": "match",
     "expected": "Fox.", "actual": "Fox."},
    {"name": "short_single_mismatch", "category": "Short", "type": "mismatch",
     "expected": "Fox.", "actual": "Dog."},
    {"name": "word_twister_match", "category": "Twister", "type": "match",
     "expected": "She sells seashells by the seashore.",
     "actual": "She sells seashells by the seashore."},
    {"name": "word_twister_mismatch", "category": "Twister", "type": "mismatch",
     "expected": "She sells seashells by the seashore.",
     "actual": "How much wood would a woodchuck chuck if a woodchuck could chuck wood."},
    {"name": "th_cluster_match", "category": "Cluster", "type": "match",
     "expected": "This they them these Thursday.",
     "actual": "This they them these Thursday."},
    {"name": "th_cluster_mismatch", "category": "Cluster", "type": "mismatch",
     "expected": "This they them these Thursday.",
     "actual": "Random words without the th cluster present anywhere."},
    {"name": "st_final_match", "category": "Cluster", "type": "match",
     "expected": "Past fast last mast.",
     "actual": "Past fast last mast."},
    {"name": "st_final_mismatch", "category": "Cluster", "type": "mismatch",
     "expected": "Past fast last mast.",
     "actual": "These phrases avoid the sharp st ending altogether."},
    {"name": "complicated_match", "category": "Complicated", "type": "match",
     "expected": "Quantum engineers synchronize entangled particles to teleport encrypted keys across vast networks.",
     "actual": "Quantum engineers synchronize entangled particles to teleport encrypted keys across vast networks."},
    {"name": "complicated_mismatch", "category": "Complicated", "type": "mismatch",
     "expected": "Quantum engineers synchronize entangled particles to teleport encrypted keys across vast networks.",
     "actual": "Gardeners quietly trim bonsai trees while rain taps rhythmically against the greenhouse glass."},
]


def synthesize(text: str, slug: str) -> Path:
    wav_path = audio_dir / f"{slug}.wav"
    if wav_path.exists():
        return wav_path
    mp3_path = audio_dir / f"{slug}.mp3"
    response = tts_client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="alloy",
        input=text,
    )
    mp3_path.write_bytes(response.read())
    subprocess.run([
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
    ], check=True)
    return wav_path


def run_case(case: dict) -> dict:
    wav_path = synthesize(case["actual"], case["name"])
    with wav_path.open("rb") as audio_file:
        files = {"file": (wav_path.name, audio_file.read(), "audio/wav")}
    data = {"expected_text": case["expected"]}
    headers = {"Authorization": f"Bearer {API_KEY}"}
    response = client.post("/analyze/pronunciation", data=data, files=files, headers=headers)
    if response.status_code != 200:
        raise RuntimeError(f"Test {case['name']} failed: {response.status_code} {response.text}")
    payload = response.json()
    pron = payload.get("pronunciation", {})
    words = pron.get("words", [])
    non_zero = [w for w in words if (w.get("word_score") or 0) > 0.01]
    zero_words = [w for w in words if (w.get("word_score") or 0) <= 0.01]
    return {
        "name": case["name"],
        "category": case["category"],
        "type": case["type"],
        "expected": case["expected"],
        "actual": case["actual"],
        "overall": pron.get("overall_score", 0.0),
        "non_zero": [w["word_text"] for w in non_zero],
        "zero_words": [w["word_text"] for w in zero_words],
        "payload": payload,
    }


def build_report(results: list[dict]) -> str:
    lines = []
    lines.append("# Pronunciation Test Report")
    lines.append("")
    lines.append(f"_Generated on {datetime.utcnow().isoformat()}Z_")
    lines.append("")
    lines.append("## Approach Summary")
    lines.append("- Montreal Forced Aligner validates timing and dictionary coverage.")
    lines.append("- Allosaurus phoneme recognition + feature alignment provide per-phoneme scoring.")
    lines.append("- Whisper transcription gates scoring so words that were never spoken stay at 0%.")
    lines.append("- Random penalties were removed; missing phonemes now score 0 by design.")
    lines.append("")
    lines.append("## Test Matrix")
    lines.append("| Test | Category | Type | Overall % | Non-zero Words | Zeroed Words |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for res in results:
        non_zero = ", ".join(res["non_zero"]) or "—"
        zero_words = ", ".join(res["zero_words"]) or "—"
        lines.append(
            f"| {res['name']} | {res['category']} | {res['type']} | "
            f"{res['overall']:.2f} | {non_zero} | {zero_words} |"
        )
    lines.append("")
    lines.append("## JSON Snapshots")
    lines.append("```")
    lines.append(json.dumps(results, indent=2))
    lines.append("```")
    return "\n".join(lines)


def main() -> None:
    results: list[dict] = []
    for case in TEST_CASES:
        print(f"Running {case['name']} ({case['type']}) ...")
        res = run_case(case)
        results.append(res)
        print(f"  -> overall {res['overall']:.2f}")
    report = build_report(results)
    report_path = Path("PRONUNCIATION_TEST_REPORT.md")
    report_path.write_text(report, encoding="utf-8")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
