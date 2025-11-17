#!/usr/bin/env python3
"""
Run phoneme suite audio clips through /analyze/pronunciation twice and record results.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any

import sys
from fastapi.testclient import TestClient

sys.path.append(str(Path(__file__).resolve().parents[1]))
from main.app import app  # type: ignore
from scripts.phoneme_suite_cases import TEST_CASES


def run_case(client: TestClient, wav_path: Path, expected_text: str, repeat: int = 2) -> List[Dict[str, Any]]:
    outcomes: List[Dict[str, Any]] = []
    for attempt in range(repeat):
        with wav_path.open("rb") as audio_file:
            files = {"file": (wav_path.name, audio_file.read(), "audio/wav")}
        data = {"expected_text": expected_text}
        response = client.post("/analyze/pronunciation", data=data, files=files, headers={"Authorization": "Bearer test-key"})
        response.raise_for_status()
        payload = response.json()
        outcomes.append(payload)
        print(f"  attempt {attempt+1}: overall {payload['pronunciation']['overall_score']:.2f}")
    return outcomes


def summarize(outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    scores = [o["pronunciation"]["overall_score"] for o in outcomes]
    avg = sum(scores) / len(scores)
    words = {}
    for outcome in outcomes:
        for word in outcome["pronunciation"]["words"]:
            words.setdefault(word["word_text"], []).append(word["word_score"])
    word_avg = {w: sum(vals) / len(vals) for w, vals in words.items()}
    return {"attempts": len(outcomes), "scores": scores, "average": avg, "word_averages": word_avg}


def main() -> None:
    audio_dir = Path("tests/audio/phoneme_suite")
    report_path = Path("PHONEME_SUITE_REPORT.md")
    client = TestClient(app)

    results = []
    lines = ["# Phoneme Suite Report", ""]

    for case in TEST_CASES:
        slug = case["slug"]
        wav_path = audio_dir / f"{slug}.wav"
        if not wav_path.exists():
            raise FileNotFoundError(f"Missing audio for {slug}: run build_phoneme_suite.py first.")
        print(f"Running case '{slug}' – {case['description']}")
        outcomes = run_case(client, wav_path, case["expected_text"])
        summary = summarize(outcomes)
        results.append({"case": case, "outcomes": outcomes, "summary": summary})
        lines.append(f"## {case['description']} ({slug})")
        lines.append(f"- Expected: `{case['expected_text']}`")
        lines.append(f"- Average score: {summary['average']:.2f}")
        lines.append(f"- Attempts: {', '.join(f'{s:.2f}' for s in summary['scores'])}")
        word_bits = ", ".join(f"{w}={score:.1f}" for w, score in summary["word_averages"].items())
        lines.append(f"- Word averages: {word_bits}")
        lines.append("")

    lines.append("## JSON Results")
    lines.append("```json")
    lines.append(json.dumps(results, indent=2))
    lines.append("```")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
