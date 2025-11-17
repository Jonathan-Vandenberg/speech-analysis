import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
from textgrid import TextGrid, IntervalTier

from .utils_text import (
    tokenize_words,
    phonemize_words_en,
    normalize_ipa_preserve_diphthongs,
    ipa_to_arpabet_symbol,
)

logger = logging.getLogger("speech_analyzer")


@dataclass
class MFAPhonemeInterval:
    label: str
    start: float
    end: float


@dataclass
class MFAWordAlignment:
    label: str
    start: float
    end: float
    phonemes: List[MFAPhonemeInterval]


class MFAAlignmentError(Exception):
    """Raised when Montreal Forced Aligner alignment fails."""


def _normalize_label(label: str) -> str:
    return re.sub(r"[^a-z0-9']+", "", (label or "").lower())


def _generate_dictionary(transcript: str, directory: Path) -> Optional[Path]:
    words = [w.lower() for w in tokenize_words(transcript)]
    if not words:
        return None
    dict_path = directory / "generated.dict"
    ipa_sequences = phonemize_words_en(" ".join(words))
    seen = set()
    vowel_phones = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
    with dict_path.open("w", encoding="utf-8") as dict_f:
        for word, ipa_seq in zip(words, ipa_sequences):
            if not word or word in seen:
                continue
            seen.add(word)
            normalized = normalize_ipa_preserve_diphthongs(ipa_seq)
            arpabet: List[str] = []
            for sym in normalized:
                mapped = ipa_to_arpabet_symbol(sym)
                if mapped:
                    if mapped in vowel_phones and not mapped[-1:].isdigit():
                        mapped = f"{mapped}0"
                    arpabet.append(mapped)
                elif len(sym) > 1:
                    for ch in sym:
                        mapped_ch = ipa_to_arpabet_symbol(ch)
                        if mapped_ch:
                            if mapped_ch in vowel_phones and not mapped_ch[-1:].isdigit():
                                mapped_ch = f"{mapped_ch}0"
                            arpabet.append(mapped_ch)
            if arpabet:
                dict_f.write(f"{word}\t{' '.join(arpabet)}\n")
    if dict_path.exists() and dict_path.stat().st_size > 0:
        logger.info("Generated temporary MFA dictionary with %d unique words", len(seen))
        return dict_path
    return None


def run_mfa_alignment(audio: np.ndarray, transcript: str, sample_rate: int = 16000) -> List[MFAWordAlignment]:
    """Align audio to transcript text using the Montreal Forced Aligner."""
    transcript = (transcript or "").strip()
    if not transcript:
        raise MFAAlignmentError("Transcript text is required for Montreal Forced Aligner.")
    if audio.size == 0:
        raise MFAAlignmentError("Audio data is empty, cannot align transcript.")

    dictionary_source = (
        os.getenv("MFA_DICTIONARY_PATH")
        or os.getenv("MFA_DICTIONARY")
        or None
    )
    acoustic_model_source = (
        os.getenv("MFA_ACOUSTIC_MODEL_PATH")
        or os.getenv("MFA_ACOUSTIC_MODEL")
        or "english_mfa"
    )
    mfa_binary = os.getenv("MFA_BINARY", "mfa")
    mfa_bin_dir = str(Path(mfa_binary).parent) if os.path.sep in mfa_binary else None
    num_jobs = os.getenv("MFA_NUM_JOBS", "1")
    beam = os.getenv("MFA_BEAM")
    retry_beam = os.getenv("MFA_RETRY_BEAM")
    child_env = os.environ.copy()
    if mfa_bin_dir:
        child_env["PATH"] = f"{mfa_bin_dir}:{child_env.get('PATH', '')}"

    with tempfile.TemporaryDirectory(prefix="mfa_pron_") as temp_dir:
        temp_path = Path(temp_dir)
        corpus_dir = temp_path / "corpus"
        output_dir = temp_path / "aligned"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_path = corpus_dir / "utterance.wav"
        text_path = corpus_dir / "utterance.lab"

        logger.info("Writing temporary audio/text corpus for MFA alignment at %s", corpus_dir)
        sf.write(audio_path, audio, sample_rate)
        text_path.write_text(transcript + "\n", encoding="utf-8")
        if dictionary_source is None:
            generated_dict = _generate_dictionary(transcript, temp_path)
            if generated_dict is None:
                raise MFAAlignmentError("Could not build dictionary for transcript.")
            dictionary_source = str(generated_dict)

        cmd = [
            mfa_binary,
            "align",
            str(corpus_dir),
            dictionary_source,
            acoustic_model_source,
            str(output_dir),
            "--clean",
            "--overwrite",
            "--disable_mp",
            "--num_jobs",
            str(num_jobs),
        ]
        if beam:
            cmd.extend(["--beam", str(beam)])
        if retry_beam:
            cmd.extend(["--retry_beam", str(retry_beam)])
        if os.getenv("MFA_PHONE_SET"):
            cmd.extend(["--phone_set", os.getenv("MFA_PHONE_SET")])

        logger.info("Running Montreal Forced Aligner: %s", " ".join(cmd))
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                timeout=int(os.getenv("MFA_TIMEOUT_SECONDS", "180")),
                env=child_env,
            )
        except FileNotFoundError as exc:
            raise MFAAlignmentError(
                "Montreal Forced Aligner CLI not found. Install it with `pip install montreal-forced-aligner` "
                "and ensure the `mfa` command is on PATH."
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise MFAAlignmentError("Montreal Forced Aligner timed out during alignment.") from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or exc.stdout or "Unknown MFA error"
            raise MFAAlignmentError(f"MFA alignment failed: {stderr.strip()[:500]}") from exc

        textgrid_files = sorted(output_dir.rglob("*.TextGrid"))
        if not textgrid_files:
            raise MFAAlignmentError("MFA alignment succeeded but no TextGrid files were produced.")

        textgrid_path = textgrid_files[0]
        logger.info("Parsing MFA TextGrid output at %s", textgrid_path)
        tg = TextGrid.fromFile(str(textgrid_path))
        word_tier = _find_tier(tg, {"word", "words"})
        phone_tier = _find_tier(tg, {"phone", "phones", "phoneme", "phonemes"})

        if word_tier is None or phone_tier is None:
            raise MFAAlignmentError("MFA output missing expected word/phone tiers.")

        alignments: List[MFAWordAlignment] = []
        epsilon = 1e-4
        phone_intervals = [
            MFAPhonemeInterval(label=p.mark.strip(), start=float(p.minTime), end=float(p.maxTime))
            for p in phone_tier
        ]

        for interval in word_tier:
            label = interval.mark.strip()
            if not label:
                continue
            normalized = _normalize_label(label)
            if not normalized or normalized in {"sil", "sp"}:
                continue
            start = float(interval.minTime)
            end = float(interval.maxTime)
            phones_for_word: List[MFAPhonemeInterval] = []
            for phone in phone_intervals:
                if phone.start + epsilon < start or phone.end - epsilon > end:
                    continue
                phone_label = phone.label
                phone_norm = _normalize_label(phone_label)
                if not phone_norm or phone_norm in {"sil", "sp"}:
                    continue
                phones_for_word.append(phone)
            alignments.append(
                MFAWordAlignment(
                    label=label,
                    start=start,
                    end=end,
                    phonemes=phones_for_word,
                )
            )

        if not alignments:
            raise MFAAlignmentError("No spoken words detected by Montreal Forced Aligner.")

        alignments.sort(key=lambda item: item.start)
        return alignments


def _find_tier(grid: TextGrid, names: set[str]) -> IntervalTier | None:
    """Locate a TextGrid tier by name."""
    for tier in grid.tiers:
        name = (tier.name or "").strip().lower()
        if name in names:
            return tier  # type: ignore[return-value]
    return None
