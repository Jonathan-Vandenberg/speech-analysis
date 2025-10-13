from typing import List, Optional, Tuple

import numpy as np
import panphon
import logging
import random

logger = logging.getLogger("speech_analyzer")


def random_low_score() -> float:
    """Generate a random score between 10-20% for missing/poor matches instead of harsh 0%."""
    return random.uniform(10.0, 20.0)


_feature_table: Optional[panphon.FeatureTable] = None


def _lazy_feature_table() -> panphon.FeatureTable:
    global _feature_table
    if _feature_table is None:
        _feature_table = panphon.FeatureTable()
    return _feature_table


# Heuristic confusion similarities - STRICTER for pronunciation assessment
# Higher values = more similar = lower distance = higher score
# Reduced values for more accurate pronunciation scoring
CONFUSION_SIMILARITY: dict[tuple[str, str], float] = {
    # Very close vowel pairs (high similarity)
    ("ɒ", "ɑ"): 0.85, ("a", "ɑ"): 0.80, ("i", "ɪ"): 0.75, ("e", "ɛ"): 0.75, ("ʊ", "u"): 0.75,
    
    # DIPHTHONG PARTIAL MATCHES - reduced for stricter scoring
    # aɪ diphthong (my, I, high, time) - reduced partial credit
    ("aɪ", "a"): 0.45, ("a", "aɪ"): 0.45,   # "a" is 45% of "aɪ" (was 78%)
    ("aɪ", "ɪ"): 0.45, ("ɪ", "aɪ"): 0.45,   # "ɪ" is 45% of "aɪ" (was 78%)
    ("aɪ", "æ"): 0.55, ("æ", "aɪ"): 0.55,   # close "a" sound (was 85%)
    
    # eɪ diphthong (name, day, great, say) - reduced partial credit
    ("eɪ", "e"): 0.45, ("e", "eɪ"): 0.45,   # "e" is 45% of "eɪ" (was 78%)
    ("eɪ", "ɪ"): 0.45, ("ɪ", "eɪ"): 0.45,   # "ɪ" is 45% of "eɪ" (was 78%)
    ("eɪ", "ɛ"): 0.55, ("ɛ", "eɪ"): 0.55,   # close "e" sound (was 85%)
    
    # aʊ diphthong (now, house, out) - reduced partial credit
    ("aʊ", "a"): 0.45, ("a", "aʊ"): 0.45,   # "a" is 45% of "aʊ" (was 78%)
    ("aʊ", "ʊ"): 0.45, ("ʊ", "aʊ"): 0.45,   # "ʊ" is 45% of "aʊ" (was 78%)
    ("aʊ", "æ"): 0.55, ("æ", "aʊ"): 0.55,   # close "a" sound (was 85%)
    
    # oʊ diphthong (go, show, no) - reduced partial credit  
    ("oʊ", "o"): 0.45, ("o", "oʊ"): 0.45,   # "o" is 45% of "oʊ" (was 78%)
    ("oʊ", "ʊ"): 0.45, ("ʊ", "oʊ"): 0.45,   # "ʊ" is 45% of "oʊ" (was 78%)
    ("oʊ", "ɔ"): 0.55, ("ɔ", "oʊ"): 0.55,   # close "o" sound (was 85%)
    
    # ɔɪ diphthong (boy, voice, choice) - reduced partial credit
    ("ɔɪ", "ɔ"): 0.45, ("ɔ", "ɔɪ"): 0.45,   # "ɔ" is 45% of "ɔɪ" (was 78%)
    ("ɔɪ", "ɪ"): 0.45, ("ɪ", "ɔɪ"): 0.45,   # "ɪ" is 45% of "ɔɪ" (was 78%)
    
    # Medium similarity pairs - reduced
    ("æ", "ɛ"): 0.65, ("ɑ", "ʌ"): 0.60, ("o", "u"): 0.70, ("i", "e"): 0.60,
    ("ɪ", "ɛ"): 0.55, ("ʊ", "ɔ"): 0.60, ("ə", "ɪ"): 0.50,
    
    # Consonant similarities - more conservative
    ("j", "i"): 0.50, ("r", "ɹ"): 0.85, ("n", "ɴ"): 0.80,
    ("ʃ", "ɕ"): 0.75, ("tʃ", "tɕ"): 0.75, ("dʒ", "dʑ"): 0.75,
    ("h", "x"): 0.50, ("θ", "ð"): 0.40,  # Very different sounds
    ("f", "θ"): 0.35, ("v", "ð"): 0.35,  # Different articulation
    ("p", "b"): 0.60, ("t", "d"): 0.65, ("k", "ɡ"): 0.60,
    ("s", "z"): 0.70, ("ʃ", "ʒ"): 0.75, ("m", "n"): 0.55,
    
    # Very different consonants - much lower scores
    ("w", "h"): 0.15,  # /w/ vs /h/ are very different (was not mapped)
    ("l", "r"): 0.40, ("w", "v"): 0.35, ("j", "dʒ"): 0.25,
}


_distance_cache: dict[tuple[str, str], float] = {}


def ipa_feature_distance(p1: str, p2: str) -> float:
    # Clear cache for testing - remove this line in production
    global _distance_cache
    _distance_cache = {}  # Clear cache to test new distance calculation
    
    key = (p1, p2)
    if key in _distance_cache:
        return _distance_cache[key]
    ft = _lazy_feature_table()
    if p1 == p2:
        # Perfect matches get excellent scores (98-99%)
        # Use hash for deterministic "randomness" based on phoneme
        variance = (hash(p1) % 100) / 2000.0  # 0.0-0.05 range
        distance = min(0.01, variance)  # Cap at 1% distance (98-99% score)
        _distance_cache[key] = distance
        return distance
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
        
        # Handle empty vector lists more intelligently
        if not v1_list and not v2_list:
            # Both phonemes unknown to Panphon - use character similarity
            if p1 == p2:
                val = 0.0
            else:
                val = 0.8  # High distance for unknown phonemes
            logger.debug("Both phonemes (%s,%s) unknown to Panphon, using character similarity: %f", p1, p2, val)
            _distance_cache[key] = val
            return val
            
        elif not v1_list or not v2_list:
            # One phoneme unknown - try to normalize it
            unknown_phoneme = p1 if not v1_list else p2
            known_phoneme = p2 if not v1_list else p1
            
            # Try common normalizations for unknown phonemes
            normalized = normalize_unknown_phoneme(unknown_phoneme)
            if normalized != unknown_phoneme:
                logger.debug("Normalized unknown phoneme %s to %s", unknown_phoneme, normalized)
                # Retry with normalized phoneme
                return ipa_feature_distance(normalized if not v1_list else known_phoneme, 
                                          known_phoneme if not v1_list else normalized)
            
            # If normalization didn't help, use moderate distance
            val = 0.6  # Moderate distance for unknown vs known phoneme
            logger.debug("Phoneme %s unknown to Panphon (vs %s), using moderate distance: %f", unknown_phoneme, known_phoneme, val)
            _distance_cache[key] = val
            return val
        
        # Both phonemes have vectors - use standard Panphon analysis
        # Convert feature vectors to numeric, handling '+', '-', '0' 
        def convert_features(feature_list):
            numeric_vectors = []
            for vector in feature_list:
                numeric_vector = []
                for feature in vector:
                    if feature == '+':
                        numeric_vector.append(1.0)
                    elif feature == '-':
                        numeric_vector.append(-1.0)
                    elif feature == '0':
                        numeric_vector.append(0.0)
                    else:
                        # Handle unexpected feature values
                        try:
                            numeric_vector.append(float(feature))
                        except ValueError:
                            numeric_vector.append(0.0)  # Default to neutral
                numeric_vectors.append(numeric_vector)
            return numeric_vectors
        
        v1_numeric = convert_features(v1_list)
        v2_numeric = convert_features(v2_list)
        
        v1 = np.mean(np.array(v1_numeric, dtype=float), axis=0)
        v2 = np.mean(np.array(v2_numeric, dtype=float), axis=0)
        
        denom = (np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0:
            _distance_cache[key] = 1.0
            return 1.0
            
        cos_sim = float(np.dot(v1, v2) / denom)
        # Original calculation: val = float(1.0 - (cos_sim + 1.0) / 2.0)
        # This was too generous - apply much stricter distance calculation
        
        # Transform cosine similarity to distance with stricter thresholds
        # cosine similarity ranges from -1 to 1
        normalized_sim = (cos_sim + 1.0) / 2.0  # Now 0 to 1
        original_distance = 1.0 - normalized_sim  # What the old calculation would give
        
        # Apply exponential penalty to make scoring much stricter
        # For pronunciation, completely different phonemes should get near-zero scores
        if normalized_sim < 0.4:  # Completely different phonemes - annihilate them  
            # Apply devastating penalty - these should be 5-10%
            strict_distance = 1.0 - (normalized_sim ** 8.0)  # Devastating penalty
            penalty_type = "DEVASTATING"
        elif normalized_sim < 0.6:  # Very different phonemes - crush them
            # Apply brutal penalty - these should be near zero
            strict_distance = 1.0 - (normalized_sim ** 6.0)  # Extremely aggressive penalty
            penalty_type = "BRUTAL"
        elif normalized_sim < 0.75:  # Different phonemes - severe penalty
            # Apply severe penalty
            strict_distance = 1.0 - (normalized_sim ** 4.0)  # Very aggressive penalty
            penalty_type = "SEVERE"
        elif normalized_sim < 0.9:  # Moderately different - moderate penalty
            # Apply moderate penalty
            strict_distance = 1.0 - (normalized_sim ** 2.0)  # Moderate reduction
            penalty_type = "MODERATE"  
        else:  # Similar phonemes - keep normal
            strict_distance = 1.0 - normalized_sim
            penalty_type = "NONE"
        
        val = min(1.0, strict_distance)  # Ensure we don't exceed 1.0
        
        # Log the distance calculation for analysis
        if logger.isEnabledFor(logging.DEBUG):
            old_similarity = (1.0 - original_distance) * 100
            new_similarity = (1.0 - val) * 100
            logger.debug(f"  📏 Distance '{p1}' vs '{p2}': {old_similarity:.1f}% → {new_similarity:.1f}% (penalty: {penalty_type})")
        
        _distance_cache[key] = val
        return val
        
    except Exception as exc:
        # True Panphon errors (not just empty vectors)
        logger.warning("Panphon error for phonemes (%s,%s): %s", p1, p2, str(exc))
        
        # Fallback to simple character comparison for panphon failures
        if p1 == p2:
            fallback_val = 0.0
        elif len(p1) == 1 and len(p2) == 1:
            # Simple phonetic similarity heuristics
            fallback_val = 0.5 if abs(ord(p1) - ord(p2)) < 10 else 1.0
        else:
            fallback_val = 1.0
        
        logger.debug("Using fallback distance for (%s,%s): %f", p1, p2, fallback_val)
        _distance_cache[key] = fallback_val
        return fallback_val


def normalize_unknown_phoneme(phoneme: str) -> str:
    """Normalize unknown phonemes to standard IPA that Panphon recognizes."""
    # Common Wav2Vec2 -> Standard IPA mappings
    normalizations = {
        'ɚ': 'ər',    # R-colored schwa -> schwa + r
        'ɝ': 'ər',    # R-colored schwa (stressed) -> schwa + r  
        'ɑr': 'ɑ',    # Remove r-coloring, use base vowel
        'ɛr': 'ɛ',    # Remove r-coloring, use base vowel
        'ɪr': 'ɪ',    # Remove r-coloring, use base vowel  
        'ɔr': 'ɔ',    # Remove r-coloring, use base vowel
        'ʊr': 'ʊ',    # Remove r-coloring, use base vowel
        'ɑː': 'ɑ',    # Remove length marker
        'iː': 'i',    # Remove length marker
        'uː': 'u',    # Remove length marker
        'ɔː': 'ɔ',    # Remove length marker
        'eː': 'e',    # Remove length marker
    }
    
    return normalizations.get(phoneme, phoneme)


def align_and_score(expected_ipa: List[str], recognized_ipa: List[str]) -> tuple[List[float], List[tuple[Optional[str], Optional[str], Optional[float]]]]:
    """Align and score phonemes with detailed logging."""
    # Temporarily set logging to DEBUG for detailed analysis
    original_level = logger.level
    logger.setLevel(logging.DEBUG)
    
    logger.debug(f"🔧 ALIGNMENT INPUT:")
    logger.debug(f"  Expected: {expected_ipa} ({len(expected_ipa)} phonemes)")
    logger.debug(f"  Recognized: {recognized_ipa} ({len(recognized_ipa)} phonemes)")
    
    # Needleman–Wunsch with feature-based substitution cost mapped to [0,1]
    gap_penalty = 0.9
    n = len(expected_ipa)
    m = len(recognized_ipa)
    if n == 0:
        logger.debug("  ⚠️ No expected phonemes, returning empty alignment")
        return [], []

    score = np.zeros((n + 1, m + 1), dtype=float)
    ptr = np.zeros((n + 1, m + 1), dtype=int)
    
    # Initialize scoring matrix
    for i in range(1, n + 1):
        score[i, 0] = score[i - 1, 0] - gap_penalty
        ptr[i, 0] = 1
    for j in range(1, m + 1):
        score[0, j] = score[0, j - 1] - gap_penalty
        ptr[0, j] = 2
    
    # Fill scoring matrix with detailed logging for small alignments
    logger.debug(f"🔬 ALIGNMENT MATRIX CALCULATION:")
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            expected_ph = expected_ipa[i - 1]
            recognized_ph = recognized_ipa[j - 1]
            
            # Get phoneme distance
            distance = ipa_feature_distance(expected_ph, recognized_ph)
            sub_cost = 1.0 - (1.0 - distance)
            
            match_score = score[i - 1, j - 1] - sub_cost
            delete_score = score[i - 1, j] - gap_penalty
            insert_score = score[i, j - 1] - gap_penalty
            best = max(match_score, delete_score, insert_score)
            
            score[i, j] = best
            ptr[i, j] = 0 if best == match_score else (1 if best == delete_score else 2)
            
            # Log details for small matrices (< 6 phonemes each)
            if n <= 6 and m <= 6:
                similarity_pct = (1.0 - distance) * 100
                operation = "MATCH" if best == match_score else ("DELETE" if best == delete_score else "INSERT")
                logger.debug(f"    [{i},{j}] '{expected_ph}' vs '{recognized_ph}': distance={distance:.3f}, similarity={similarity_pct:.1f}%, operation={operation}")

    # Traceback to get alignment
    i, j = n, m
    aligned_scores: List[float] = []
    align_pairs: List[tuple[Optional[str], Optional[str], Optional[float]]] = []
    
    logger.debug(f"🔄 TRACEBACK PROCESS:")
    step = 0
    while i > 0 and j > 0:
        step += 1
        dir_ = ptr[i, j]
        if dir_ == 0:  # Match/substitution
            expected_ph = expected_ipa[i - 1]
            recognized_ph = recognized_ipa[j - 1]
            dist = ipa_feature_distance(expected_ph, recognized_ph)
            sc01 = 1.0 - dist
            aligned_scores.append(sc01)
            align_pairs.append((expected_ph, recognized_ph, sc01 * 100.0))
            logger.debug(f"  Step {step}: MATCH '{expected_ph}' ↔ '{recognized_ph}' (similarity: {sc01*100:.1f}%)")
            i -= 1; j -= 1
        elif dir_ == 1:  # Deletion
            expected_ph = expected_ipa[i - 1]
            low_score = random_low_score()
            aligned_scores.append(low_score / 100.0)
            align_pairs.append((expected_ph, "∅", low_score))
            logger.debug(f"  Step {step}: DELETE '{expected_ph}' (not said, score: {low_score:.1f}%)")
            i -= 1
        else:  # Insertion
            recognized_ph = recognized_ipa[j - 1]
            align_pairs.append(("∅", recognized_ph, None))
            logger.debug(f"  Step {step}: INSERT '{recognized_ph}' (extra phoneme said)")
            j -= 1
    
    # Handle remaining expected phonemes (deletions)
    while i > 0:
        step += 1
        expected_ph = expected_ipa[i - 1]
        low_score = random_low_score()
        aligned_scores.append(low_score / 100.0)
        align_pairs.append((expected_ph, "∅", low_score))
        logger.debug(f"  Step {step}: DELETE '{expected_ph}' (remaining expected, score: {low_score:.1f}%)")
        i -= 1

    aligned_scores.reverse()
    align_pairs.reverse()
    
    logger.debug(f"📋 RAW ALIGNMENT RESULTS:")
    for i, (exp, rec, score) in enumerate(align_pairs):
        logger.debug(f"  {i+1}. '{exp}' vs '{rec}' = {score}%")
    
    # Apply strict scoring for pronunciation accuracy assessment
    # Completely different phonemes should get very low scores
    boosted_scores = []
    boosted_pairs = []
    
    logger.debug(f"⚡ SCORE BOOSTING PROCESS:")
    
    for i, s in enumerate(aligned_scores):
        # Convert to percentage
        base_score = s * 100.0
        
        logger.debug(f"  Phoneme {i+1}: base_score = {base_score:.1f}%")
        
        # Absolutely ruthless scoring for pronunciation assessment
        # Wrong phonemes should get single-digit scores, period.
        if base_score < 10.0:  # Completely wrong - NO boost whatsoever
            final_score = max(5.0, base_score)  # Keep it brutal, minimum 5%
            logger.debug(f"    → DEVASTATING (< 10%): {base_score:.1f}% → {final_score:.1f}% (NO BOOST)")
        elif base_score < 20.0:  # Very wrong - tiny boost only
            final_score = base_score + 0.5  # Almost no boost
            logger.debug(f"    → BRUTAL (< 20%): {base_score:.1f}% + 0.5% = {final_score:.1f}%")
        elif base_score < 35.0:  # Wrong - very small boost
            final_score = base_score + 1.0  # Tiny boost
            logger.debug(f"    → VERY WRONG (< 35%): {base_score:.1f}% + 1% = {final_score:.1f}%")
        elif base_score < 50.0:  # Somewhat different - small boost
            final_score = base_score + 2.0
            logger.debug(f"    → WRONG (< 50%): {base_score:.1f}% + 2% = {final_score:.1f}%")
        elif base_score < 70.0:  # Medium - normal boost
            final_score = base_score + 5.0
            logger.debug(f"    → MEDIUM (< 70%): {base_score:.1f}% + 5% = {final_score:.1f}%")
        elif base_score < 85.0:  # Good - larger boost
            final_score = base_score + 8.0
            logger.debug(f"    → GOOD (< 85%): {base_score:.1f}% + 8% = {final_score:.1f}%")
        else:  # Excellent - maximum boost
            final_score = min(99.0, base_score + 10.0)
            logger.debug(f"    → EXCELLENT (≥ 85%): {base_score:.1f}% + 10% = {final_score:.1f}%")
        
        # Ensure minimum is very low for bad matches
        final_score = max(5.0, min(99.0, final_score))
        boosted_scores.append(final_score)
        
        # Update align_pairs with boosted scores
        if i < len(align_pairs):
            expected, recognized, original_score = align_pairs[i]
            if expected not in (None, "∅") and recognized not in (None, "∅"):
                boosted_pairs.append((expected, recognized, final_score))
                logger.debug(f"    → FINAL PAIR: '{expected}' vs '{recognized}' = {final_score:.1f}%")
            elif recognized == "∅":
                # Deletion: use the already-random low score (10-20%) with slight boost
                deletion_score = min(25.0, original_score + 3.0) if original_score else random_low_score()
                boosted_pairs.append((expected, recognized, deletion_score))
                logger.debug(f"    → DELETION: '{expected}' (not said) = {deletion_score:.1f}%")
            else:
                # Insertion: keep as None (extra recognized phoneme)
                boosted_pairs.append((expected, recognized, None))
                logger.debug(f"    → INSERTION: '{recognized}' (extra phoneme)")
        else:
            boosted_pairs.append(align_pairs[i])

    # Handle any remaining pairs
    while len(boosted_pairs) < len(align_pairs):
        boosted_pairs.append(align_pairs[len(boosted_pairs)])

    logger.debug(f"🏁 FINAL BOOSTED RESULTS:")
    logger.debug(f"  Scores: {[f'{s:.1f}%' for s in boosted_scores]}")
    logger.debug(f"  Average: {sum(boosted_scores) / len(boosted_scores):.1f}%" if boosted_scores else "  No scores")
    for i, (exp, rec, score) in enumerate(boosted_pairs):
        logger.debug(f"  {i+1}. '{exp}' vs '{rec}' = {score}%")

    # Restore original logging level
    logger.setLevel(original_level)
    
    return boosted_scores, boosted_pairs


