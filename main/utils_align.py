from typing import List, Optional, Tuple

import logging
import numpy as np

logger = logging.getLogger("speech_analyzer")


def random_low_score() -> float:
    """Return a flat 0 score for missing/poor matches."""
    return 0.0

OPTIONAL_SILENT_PHONEMES = {"h"}

def deletion_penalty(phoneme: str) -> float:
    if phoneme in OPTIONAL_SILENT_PHONEMES:
        return 60.0
    return random_low_score()

PHONEME_FEATURES: dict[str, dict[str, object]] = {
    # Vowels (height, backness, rounded)
    "i": {"type": "vowel", "height": "high", "backness": "front", "rounded": False},
    "ɪ": {"type": "vowel", "height": "near-high", "backness": "front", "rounded": False},
    "e": {"type": "vowel", "height": "mid", "backness": "front", "rounded": False},
    "ɛ": {"type": "vowel", "height": "open-mid", "backness": "front", "rounded": False},
    "æ": {"type": "vowel", "height": "near-open", "backness": "front", "rounded": False},
    "a": {"type": "vowel", "height": "open", "backness": "front", "rounded": False},
    "ɑ": {"type": "vowel", "height": "open", "backness": "back", "rounded": False},
    "ɒ": {"type": "vowel", "height": "open", "backness": "back", "rounded": True},
    "ɔ": {"type": "vowel", "height": "open-mid", "backness": "back", "rounded": True},
    "o": {"type": "vowel", "height": "mid", "backness": "back", "rounded": True},
    "u": {"type": "vowel", "height": "high", "backness": "back", "rounded": True},
    "ʊ": {"type": "vowel", "height": "near-high", "backness": "back", "rounded": True},
    "ə": {"type": "vowel", "height": "mid", "backness": "central", "rounded": False},
    "ɜ": {"type": "vowel", "height": "mid", "backness": "central", "rounded": False},
    "ʌ": {"type": "vowel", "height": "open-mid", "backness": "central", "rounded": False},
    "ɚ": {"type": "vowel", "height": "mid", "backness": "central", "rounded": False},
    "ɝ": {"type": "vowel", "height": "mid", "backness": "central", "rounded": False},
    "eɪ": {"type": "vowel", "height": "mid", "backness": "front", "rounded": False},
    "aɪ": {"type": "vowel", "height": "open", "backness": "front", "rounded": False},
    "oʊ": {"type": "vowel", "height": "mid", "backness": "back", "rounded": True},
    "aʊ": {"type": "vowel", "height": "open", "backness": "back", "rounded": True},
    "ɔɪ": {"type": "vowel", "height": "open-mid", "backness": "back", "rounded": True},
    "ᵻ": {"type": "vowel", "height": "mid", "backness": "central", "rounded": False},

    # Consonants (place, manner, voice)
    "p": {"type": "consonant", "place": "bilabial", "manner": "plosive", "voice": False},
    "b": {"type": "consonant", "place": "bilabial", "manner": "plosive", "voice": True},
    "m": {"type": "consonant", "place": "bilabial", "manner": "nasal", "voice": True},
    "f": {"type": "consonant", "place": "labiodental", "manner": "fricative", "voice": False},
    "v": {"type": "consonant", "place": "labiodental", "manner": "fricative", "voice": True},
    "w": {"type": "consonant", "place": "labial-velar", "manner": "approximant", "voice": True},
    "t": {"type": "consonant", "place": "alveolar", "manner": "plosive", "voice": False},
    "d": {"type": "consonant", "place": "alveolar", "manner": "plosive", "voice": True},
    "n": {"type": "consonant", "place": "alveolar", "manner": "nasal", "voice": True},
    "s": {"type": "consonant", "place": "alveolar", "manner": "fricative", "voice": False},
    "z": {"type": "consonant", "place": "alveolar", "manner": "fricative", "voice": True},
    "l": {"type": "consonant", "place": "alveolar", "manner": "lateral", "voice": True},
    "r": {"type": "consonant", "place": "alveolar", "manner": "approximant", "voice": True},
    "ɹ": {"type": "consonant", "place": "alveolar", "manner": "approximant", "voice": True},
    "θ": {"type": "consonant", "place": "dental", "manner": "fricative", "voice": False},
    "ð": {"type": "consonant", "place": "dental", "manner": "fricative", "voice": True},
    "ʃ": {"type": "consonant", "place": "postalveolar", "manner": "fricative", "voice": False},
    "ʒ": {"type": "consonant", "place": "postalveolar", "manner": "fricative", "voice": True},
    "tʃ": {"type": "consonant", "place": "postalveolar", "manner": "affricate", "voice": False},
    "dʒ": {"type": "consonant", "place": "postalveolar", "manner": "affricate", "voice": True},
    "j": {"type": "consonant", "place": "palatal", "manner": "approximant", "voice": True},
    "k": {"type": "consonant", "place": "velar", "manner": "plosive", "voice": False},
    "g": {"type": "consonant", "place": "velar", "manner": "plosive", "voice": True},
    "ŋ": {"type": "consonant", "place": "velar", "manner": "nasal", "voice": True},
    "h": {"type": "consonant", "place": "glottal", "manner": "fricative", "voice": False},
}

PHONEME_ALIAS_SETS = [
    {"h", "∅"},
    {"ə", "ʌ", "ɐ", "ɑ", "a"},
    {"ɜ", "ɚ", "ɝ", "ər", "ɜː", "ʊ", "u", "uː"},
    {"o", "oʊ", "əʊ", "ɔ", "ɔː", "oː", "ə"},
    {"e", "eɪ", "ɛ", "eː"},
    {"i", "iː", "ɪ"},
    {"aɪ", "ɑɪ", "ai"},
    {"aʊ", "ɑʊ", "au"},
    {"p", "pʰ", "b"},
    {"t", "tʰ", "d", "ɾ"},
    {"k", "kʰ", "g"},
    {"v", "b"},
    {"f", "p"},
]

PHONEME_ALIAS_LOOKUP: dict[str, set[int]] = {}
for idx, group in enumerate(PHONEME_ALIAS_SETS):
    for symbol in group:
        PHONEME_ALIAS_LOOKUP.setdefault(symbol, set()).add(idx)


def _normalize_symbol(symbol: str) -> str:
    replacements = {
        "ɝ": "ɚ",
        "ɛr": "ɛ",
        "ʊr": "ʊ",
        "ur": "u",
        "ər": "ə",
    }
    return replacements.get(symbol, symbol)


def _phoneme_similarity(p1: str, p2: str) -> float:
    if p1 == p2:
        return 0.99
    s1 = _normalize_symbol(p1)
    s2 = _normalize_symbol(p2)
    info1 = PHONEME_FEATURES.get(s1)
    info2 = PHONEME_FEATURES.get(s2)
    if not info1 or not info2:
        return 0.15
    if info1["type"] != info2["type"]:
        return 0.15
    if info1["type"] == "vowel":
        score = 0.35
        if info1["height"] == info2["height"]:
            score += 0.25
        if info1["backness"] == info2["backness"]:
            score += 0.25
        if info1["rounded"] == info2["rounded"]:
            score += 0.10
        return min(score, 0.98)
    score = 0.30
    if info1.get("place") == info2.get("place"):
        score += 0.30
    if info1.get("manner") == info2.get("manner"):
        score += 0.30
    if info1.get("voice") == info2.get("voice"):
        score += 0.10
    return min(score, 0.95)


_distance_cache: dict[tuple[str, str], float] = {}


def ipa_feature_distance(p1: str, p2: str) -> float:
    if p1 == p2:
        return 0.0
    groups1 = PHONEME_ALIAS_LOOKUP.get(p1)
    groups2 = PHONEME_ALIAS_LOOKUP.get(p2)
    if groups1 and groups2 and groups1 & groups2:
        return 0.05
    key = (p1, p2)
    if key in _distance_cache:
        return _distance_cache[key]
    similarity = _phoneme_similarity(p1, p2)
    distance = 1.0 - similarity
    _distance_cache[key] = distance
    return distance



def align_and_score(
    expected_ipa: List[str],
    recognized_ipa: List[str],
    detailed_logging: bool = True,
) -> tuple[List[float], List[tuple[Optional[str], Optional[str], Optional[float]]]]:
    """Align and score phonemes with optional detailed logging."""
    original_level = logger.level
    if detailed_logging:
        logger.setLevel(logging.DEBUG)
        logger.debug("🔧 ALIGNMENT INPUT:")
        logger.debug(f"  Expected: {expected_ipa} ({len(expected_ipa)} phonemes)")
        logger.debug(f"  Recognized: {recognized_ipa} ({len(recognized_ipa)} phonemes)")

    try:
        gap_penalty = 0.9
        n = len(expected_ipa)
        m = len(recognized_ipa)
        if n == 0:
            if detailed_logging:
                logger.debug("  ⚠️ No expected phonemes, returning empty alignment")
            return [], []

        score = np.zeros((n + 1, m + 1), dtype=float)
        ptr = np.zeros((n + 1, m + 1), dtype=int)

        for i in range(1, n + 1):
            score[i, 0] = score[i - 1, 0] - gap_penalty
            ptr[i, 0] = 1
        for j in range(1, m + 1):
            score[0, j] = score[0, j - 1] - gap_penalty
            ptr[0, j] = 2

        if detailed_logging:
            logger.debug("🔬 ALIGNMENT MATRIX CALCULATION:")
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                expected_ph = expected_ipa[i - 1]
                recognized_ph = recognized_ipa[j - 1]
                distance = ipa_feature_distance(expected_ph, recognized_ph)
                sub_cost = 1.0 - (1.0 - distance)
                match_score = score[i - 1, j - 1] - sub_cost
                delete_score = score[i - 1, j] - gap_penalty
                insert_score = score[i, j - 1] - gap_penalty
                best = max(match_score, delete_score, insert_score)
                score[i, j] = best
                ptr[i, j] = 0 if best == match_score else (1 if best == delete_score else 2)
                if detailed_logging and n <= 6 and m <= 6:
                    similarity_pct = (1.0 - distance) * 100
                    operation = (
                        "MATCH" if best == match_score else (
                            "DELETE" if best == delete_score else "INSERT"
                        )
                    )
                    logger.debug(
                        f"    [{i},{j}] '{expected_ph}' vs '{recognized_ph}': "
                        f"distance={distance:.3f}, similarity={similarity_pct:.1f}%, operation={operation}"
                    )

        i, j = n, m
        aligned_scores: List[float] = []
        align_pairs: List[tuple[Optional[str], Optional[str], Optional[float]]] = []

        if detailed_logging:
            logger.debug("🔄 TRACEBACK PROCESS:")
        step = 0
        while i > 0 and j > 0:
            step += 1
            dir_ = ptr[i, j]
            if dir_ == 0:
                expected_ph = expected_ipa[i - 1]
                recognized_ph = recognized_ipa[j - 1]
                dist = ipa_feature_distance(expected_ph, recognized_ph)
                sc01 = 1.0 - dist
                aligned_scores.append(sc01)
                align_pairs.append((expected_ph, recognized_ph, sc01 * 100.0))
                if detailed_logging:
                    logger.debug(
                        f"  Step {step}: MATCH '{expected_ph}' ↔ '{recognized_ph}' (similarity: {sc01*100:.1f}%)"
                    )
                i -= 1
                j -= 1
            elif dir_ == 1:
                expected_ph = expected_ipa[i - 1]
                low_score = deletion_penalty(expected_ph)
                aligned_scores.append(low_score / 100.0)
                align_pairs.append((expected_ph, "∅", low_score))
                if detailed_logging:
                    logger.debug(
                        f"  Step {step}: DELETE '{expected_ph}' (not said, score: {low_score:.1f}%)"
                    )
                i -= 1
            else:
                recognized_ph = recognized_ipa[j - 1]
                align_pairs.append(("∅", recognized_ph, None))
                if detailed_logging:
                    logger.debug(f"  Step {step}: INSERT '{recognized_ph}' (extra phoneme said)")
                j -= 1

        while i > 0:
            step += 1
            expected_ph = expected_ipa[i - 1]
            low_score = deletion_penalty(expected_ph)
            aligned_scores.append(low_score / 100.0)
            align_pairs.append((expected_ph, "∅", low_score))
            if detailed_logging:
                logger.debug(
                    f"  Step {step}: DELETE '{expected_ph}' (remaining expected, score: {low_score:.1f}%)"
                )
            i -= 1

        aligned_scores.reverse()
        align_pairs.reverse()
        if detailed_logging:
            logger.debug("📋 RAW ALIGNMENT RESULTS:")
            for idx, (exp, rec, score) in enumerate(align_pairs, start=1):
                logger.debug(f"  {idx}. '{exp}' vs '{rec}' = {score}%")

        final_scores = [s * 100.0 for s in aligned_scores]
        return final_scores, align_pairs
    finally:
        if detailed_logging:
            logger.setLevel(original_level)
