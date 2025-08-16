from typing import List, Optional, Tuple

import numpy as np
import panphon
import logging

logger = logging.getLogger("speech_analyzer")


_feature_table: Optional[panphon.FeatureTable] = None


def _lazy_feature_table() -> panphon.FeatureTable:
    global _feature_table
    if _feature_table is None:
        _feature_table = panphon.FeatureTable()
    return _feature_table


# Heuristic confusion similarities to soften near-matches (0..1 similarity)
CONFUSION_SIMILARITY: dict[tuple[str, str], float] = {
    ("o", "ɔ"): 0.85, ("ʌ", "ə"): 0.8, ("ɒ", "ɑ"): 0.85,
    ("a", "ɑ"): 0.9,
    ("i", "ɪ"): 0.85, ("e", "ɛ"): 0.85, ("ʊ", "u"): 0.8,
    ("j", "i"): 0.75, ("r", "ɹ"): 0.9, ("n", "ɴ"): 0.85,
    ("ʃ", "ɕ"): 0.85, ("tʃ", "tɕ"): 0.85, ("dʒ", "dʑ"): 0.85,
    ("h", "x"): 0.7, ("θ", "ð"): 0.7,
}


_distance_cache: dict[tuple[str, str], float] = {}


def ipa_feature_distance(p1: str, p2: str) -> float:
    key = (p1, p2)
    if key in _distance_cache:
        return _distance_cache[key]
    ft = _lazy_feature_table()
    if p1 == p2:
        _distance_cache[key] = 0.0
        return 0.0
    if (p1, p2) in CONFUSION_SIMILARITY:
        val = 1.0 - CONFUSION_SIMILARITY[(p1, p2)]
        _distance_cache[key] = val
        return val
    if (p2, p1) in CONFUSION_SIMILARITY:
        val = 1.0 - CONFUSION_SIMILARITY[(p2, p1)]
        _distance_cache[key] = val
        return val
    try:
        v1_list = ft.word_to_vector_list(p1)
        v2_list = ft.word_to_vector_list(p2)
        if not v1_list or not v2_list:
            _distance_cache[key] = 1.0
            return 1.0
        v1 = np.mean(np.array(v1_list, dtype=float), axis=0)
        v2 = np.mean(np.array(v2_list, dtype=float), axis=0)
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            _distance_cache[key] = 1.0
            return 1.0
        cos_sim = float(np.dot(v1, v2) / denom)
        val = float(1.0 - (cos_sim + 1.0) / 2.0)
        _distance_cache[key] = val
        return val
    except Exception:
        logger.debug("Panphon distance failed for (%s,%s)", p1, p2)
        _distance_cache[key] = 1.0
        return 1.0


def align_and_score(expected_ipa: List[str], recognized_ipa: List[str]) -> tuple[List[float], List[tuple[Optional[str], Optional[str], Optional[float]]]]:
    # Needleman–Wunsch with feature-based substitution cost mapped to [0,1]
    gap_penalty = 0.9
    n = len(expected_ipa)
    m = len(recognized_ipa)
    if n == 0:
        return [], []

    score = np.zeros((n + 1, m + 1), dtype=float)
    ptr = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] - gap_penalty
        ptr[i, 0] = 1
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] - gap_penalty
        ptr[0, j] = 2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = 1.0 - (1.0 - ipa_feature_distance(expected_ipa[i - 1], recognized_ipa[j - 1]))
            match_score = score[i - 1, j - 1] - sub_cost
            delete_score = score[i - 1, j] - gap_penalty
            insert_score = score[i, j - 1] - gap_penalty
            best = max(match_score, delete_score, insert_score)
            score[i, j] = best
            ptr[i, j] = 0 if best == match_score else (1 if best == delete_score else 2)

    i, j = n, m
    aligned_scores: List[float] = []
    align_pairs: List[tuple[Optional[str], Optional[str], Optional[float]]] = []
    while i > 0 and j > 0:
        dir_ = ptr[i, j]
        if dir_ == 0:
            dist = ipa_feature_distance(expected_ipa[i - 1], recognized_ipa[j - 1])
            sc01 = 1.0 - dist
            aligned_scores.append(sc01)
            align_pairs.append((expected_ipa[i - 1], recognized_ipa[j - 1], sc01 * 100.0))
            i -= 1; j -= 1
        elif dir_ == 1:
            aligned_scores.append(0.0)
            align_pairs.append((expected_ipa[i - 1], "∅", 0.0))
            i -= 1
        else:
            align_pairs.append(("∅", recognized_ipa[j - 1], None))
            j -= 1
    while i > 0:
        aligned_scores.append(0.0)
        align_pairs.append((expected_ipa[i - 1], "∅", 0.0))
        i -= 1

    aligned_scores.reverse()
    align_pairs.reverse()
    scores_100 = [float(np.clip(s * 100.0, 0.0, 100.0)) for s in aligned_scores]
    return scores_100, align_pairs


