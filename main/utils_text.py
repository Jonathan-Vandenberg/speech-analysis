import re
import unicodedata
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger("speech_analyzer")


def tokenize_words(text: str) -> List[str]:
    return [w for w in re.findall(r"[A-Za-z]+'?[A-Za-z]+|[A-Za-z]+", text or "")] 


def split_diphthongs(seq: List[str]) -> List[str]:
    out: List[str] = []
    for p in seq:
        if p == "eɪ":
            out += ["e", "ɪ"]
        elif p == "oʊ":
            out += ["o", "ʊ"]
        elif p == "aɪ":
            out += ["a", "ɪ"]
        elif p == "aʊ":
            out += ["a", "ʊ"]
        elif p == "ɔɪ":
            out += ["ɔ", "ɪ"]
        elif p == "ɪə":
            out += ["ɪ", "ə"]
        elif p == "ʊə":
            out += ["ʊ", "ə"]
        else:
            out.append(p)
    return out


def normalize_ipa(seq: List[str]) -> List[str]:
    out: List[str] = []
    for p in split_diphthongs(seq):
        nf = unicodedata.normalize("NFD", p)
        base = "".join(ch for ch in nf if not unicodedata.combining(ch) and ch not in {"ː", "ˑ"})
        base = (base
            .replace("tɕ", "tʃ").replace("tɕʰ", "tʃ").replace("tʂ", "tʃ")
            .replace("dʑ", "dʒ")
            .replace("ɕ", "ʃ").replace("ʂ", "ʃ").replace("ʐ", "ʒ")
            .replace("ɹ", "r").replace("ɾ", "r")
            .replace("x", "h").replace("y", "j")
            .replace("ɴ", "n").replace("ɫ", "l")
            .replace("ɒ", "ɑ").replace("ɤ", "ʌ")
        )
        if base:
            out.append(base)
    return out


# Lightweight English fallback phonemizer using g2p_en (ARPABET → IPA)
_ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "r", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ",
}


def arpabet_tokens_to_ipa(tokens: List[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        t = t.strip().upper()
        if not t:
            continue
        # remove stress digits from vowels
        if t[-1:].isdigit():
            t = t[:-1]
        if t in _ARPABET_TO_IPA:
            out.append(_ARPABET_TO_IPA[t])
    return out


_phonemize_cache: Dict[str, List[str]] = {}


def phonemize_words_en(text: str) -> List[List[str]]:
    words = tokenize_words(text)
    ipa_per_word: List[List[str]] = []
    # Try phonemizer/espeak first (if available)
    try:
        from phonemizer import phonemize
        from phonemizer.separator import Separator
        # Attempt to help locate espeak-ng on macOS Homebrew installs
        import os
        import glob
        if not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
            for p in (
                "/opt/homebrew/Cellar/espeak-ng/*/lib/libespeak-ng.1.dylib",
                "/opt/homebrew/lib/libespeak-ng.1.dylib",
                "/opt/homebrew/lib/libespeak-ng.dylib",
                "/usr/local/lib/libespeak-ng.dylib",
            ):
                matches = glob.glob(p)
                if matches:
                    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = matches[0]
                    logger.info(f"Found espeak-ng library at: {matches[0]}")
                    break
        # Batch phonemize as a list of words to preserve boundaries reliably
        ipa_list = phonemize(
            words,
            language="en-us",
            backend="espeak",
            strip=True,
            with_stress=False,
            separator=Separator(phone=" ", word="|"),
        )
        if isinstance(ipa_list, str):
            ipa_list = [ipa_list]
        # Ensure 1:1 with input words; if library returns fewer/more, fall back per word
        if len(ipa_list) != len(words):
            ipa_list = [
                phonemize(
                    w,
                    language="en-us",
                    backend="espeak",
                    strip=True,
                    with_stress=False,
                    separator=Separator(phone=" ", word="|"),
                )
                for w in words
            ]
        for seg in ipa_list:
            key = seg
            if key in _phonemize_cache:
                phones = _phonemize_cache[key]
            else:
                phones = [p for p in str(seg).strip().split() if p]
                _phonemize_cache[key] = phones
            ipa_per_word.append(phones)
        return ipa_per_word
    except Exception as exc:
        logger.warning("Phonemizer/espeak unavailable, falling back to g2p_en: %s", exc)
    # Fallback: g2p_en
    try:
        from g2p_en import G2p
        # Ensure required NLTK data exists (for POS tagger used by g2p_en)
        try:
            import nltk
            for pkg in ("averaged_perceptron_tagger_eng", "averaged_perceptron_tagger", "punkt"):
                try:
                    nltk.data.find(f"taggers/{pkg}") if "tagger" in pkg else nltk.data.find(f"tokenizers/{pkg}")
                except LookupError:
                    nltk.download(pkg, quiet=True)
        except Exception as exc_nltk:
            logger.warning("NLTK bootstrap issue: %s", exc_nltk)
        g2p = G2p()
        # Cache results per word to avoid recomputation across calls
        for w in words:
            if w in _phonemize_cache:
                ipa_per_word.append(_phonemize_cache[w])
                continue
            arpa = [t for t in g2p(w) if re.fullmatch(r"[A-Za-z]+\d?", t)]
            ipa = arpabet_tokens_to_ipa(arpa)
            _phonemize_cache[w] = ipa
            ipa_per_word.append(ipa)
        return ipa_per_word
    except Exception as exc:
        logger.warning("g2p_en fallback failed: %s", exc)
        return [[] for _ in words]


# Phoneme recognition from audio using wav2vec2
_phoneme_processor: Optional[object] = None
_phoneme_model: Optional[object] = None


def _lazy_wav2vec2_phoneme():
    """Load wav2vec2 phoneme recognition model lazily."""
    global _phoneme_processor, _phoneme_model
    if _phoneme_processor is None or _phoneme_model is None:
        try:
            # Set up espeak-ng path if needed
            import os
            import glob
            if not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
                for p in (
                    "/opt/homebrew/Cellar/espeak-ng/*/lib/libespeak-ng.1.dylib",
                    "/opt/homebrew/lib/libespeak-ng.1.dylib",
                    "/opt/homebrew/lib/libespeak-ng.dylib",
                    "/usr/local/lib/libespeak-ng.dylib",
                ):
                    matches = glob.glob(p)
                    if matches:
                        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = matches[0]
                        logger.info(f"Found espeak-ng library at: {matches[0]}")
                        break
            
            from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
            _phoneme_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
            _phoneme_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-lv-60-espeak-cv-ft")
            _phoneme_model.eval()
            logger.info("Loaded wav2vec2 phoneme recognition model")
        except Exception as exc:
            logger.warning("Failed to load wav2vec2 phoneme model: %s", exc)
            raise
    return _phoneme_processor, _phoneme_model


def dedupe_phonemes(phonemes: List[str]) -> List[str]:
    """Remove consecutive duplicate phonemes to clean up wav2vec2 output."""
    if not phonemes:
        return []
    
    deduped = [phonemes[0]]
    for i in range(1, len(phonemes)):
        if phonemes[i] != phonemes[i-1]:
            deduped.append(phonemes[i])
    return deduped


def clean_phoneme_sequence(phonemes: List[str]) -> List[str]:
    """Advanced cleaning of phoneme sequences to fix wav2vec2 artifacts."""
    if not phonemes:
        return []
    
    cleaned = []
    i = 0
    while i < len(phonemes):
        current = phonemes[i]
        
        # Skip very short or invalid phonemes
        if len(current.strip()) == 0:
            i += 1
            continue
            
        # Look ahead to detect common artifacts
        if i < len(phonemes) - 1:
            next_phoneme = phonemes[i + 1]
            
            # Pattern: vowel + same vowel (e.g., "e ɪ ɪ" → "e ɪ")
            if current in ["e", "o", "a", "i", "u"] and next_phoneme == "ɪ" and current == "e":
                # Special case: "e ɪ" is actually the diphthong /eɪ/
                cleaned.append("eɪ")
                i += 2
                continue
            elif current in ["o"] and next_phoneme == "ʊ":
                # Special case: "o ʊ" is actually the diphthong /oʊ/
                cleaned.append("oʊ") 
                i += 2
                continue
            elif current in ["a"] and next_phoneme == "ɪ":
                # Special case: "a ɪ" is actually the diphthong /aɪ/
                cleaned.append("aɪ")
                i += 2
                continue
            elif current in ["a"] and next_phoneme == "ʊ":
                # Special case: "a ʊ" is actually the diphthong /aʊ/
                cleaned.append("aʊ")
                i += 2
                continue
        
        cleaned.append(current)
        i += 1
    
    # Final deduplication
    return dedupe_phonemes(cleaned)


def improved_espeak_to_ipa(phone: str) -> str:
    """Convert espeak phonemes to IPA with better mapping."""
    # Enhanced espeak to IPA conversion with more vowel variants
    mapping = {
        # Consonants
        "tS": "tʃ", "dZ": "dʒ", "N": "ŋ", "T": "θ", "D": "ð", 
        "S": "ʃ", "Z": "ʒ", "h": "h", "x": "x", "?": "ʔ", "J": "j", "w": "w",
        
        # Vowels - more comprehensive mapping
        "@": "ə", "3": "ɜ", "A": "ɑ", "I": "ɪ", "O": "ɔ", "U": "ʊ",
        "i": "i", "u": "u", "e": "e", "o": "o", "a": "a",
        
        # Common wav2vec2 variants that might appear
        "uu": "u", "ii": "i", "oo": "o", "aa": "a", "ee": "e",
        "V": "ʌ", "Q": "ɒ", "E": "ɛ", "Y": "y",
        
        # Diphthongs
        "aI": "aɪ", "eI": "eɪ", "oU": "oʊ", "aU": "aʊ", "OI": "ɔɪ", 
        "I@": "ɪə", "e@": "eə", "u@": "ʊə",
        
        # Syllabic consonants
        "r=": "ɝ", "l=": "l̩", "n=": "n̩", "m=": "m̩",
    }
    
    # Apply mappings
    for espeak, ipa in mapping.items():
        phone = phone.replace(espeak, ipa)
    
    # Remove stress markers and other diacritics
    phone = phone.replace("'", "").replace(",", "").replace(":", "")
    
    return phone


def phonemes_from_audio(audio_16k: np.ndarray) -> List[str]:
    """Extract phonemes directly from 16kHz mono audio using wav2vec2."""
    try:
        import torch
        processor, model = _lazy_wav2vec2_phoneme()
        
        with torch.no_grad():
            inputs = processor(audio_16k, sampling_rate=16000, return_tensors="pt", padding=True)
            logits = model(inputs.input_values).logits
            
            # Apply confidence thresholding - only keep high-confidence predictions
            probs = torch.softmax(logits, dim=-1)
            max_probs, ids = torch.max(probs, dim=-1)
            
            # Use a lower confidence threshold to preserve vowels (they're often quieter)
            confidence_threshold = 0.1
            filtered_ids = []
            for i, (prob, id_val) in enumerate(zip(max_probs[0], ids[0])):
                if prob >= confidence_threshold:
                    filtered_ids.append(id_val.item())
            
            # Always use all predictions for vowel preservation, then filter later
            text = processor.batch_decode(ids)[0]
        
        # Extract individual phonemes, filter out word boundaries and empty tokens
        phones = [p for p in text.strip().split() if p and p != "|" and len(p.strip()) > 0]
        
        # Convert espeak phonemes to normalized IPA with improved mapping
        ipa_phones = []
        for phone in phones:
            normalized = improved_espeak_to_ipa(phone)
            if normalized and len(normalized.strip()) > 0:
                ipa_phones.append(normalized)
        
        # Apply advanced cleaning to fix artifacts and create proper diphthongs
        ipa_phones = clean_phoneme_sequence(ipa_phones)
        
        # Final IPA normalization
        final_ipa = normalize_ipa(ipa_phones)
        return final_ipa
        
    except Exception as exc:
        logger.warning("Phoneme recognition from audio failed: %s", exc)
        return []



