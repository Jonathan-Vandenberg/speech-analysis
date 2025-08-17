import re
import unicodedata
from typing import List, Dict, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger("speech_analyzer")


def number_to_words(num_str: str) -> str:
    """Convert a number string to its spoken form."""
    try:
        num = int(num_str)
        # Handle common numbers that appear in speech
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        if num == 0:
            return "zero"
        elif 1 <= num <= 9:
            return ones[num]
        elif 10 <= num <= 19:
            return teens[num - 10]
        elif 20 <= num <= 99:
            if num % 10 == 0:
                return tens[num // 10]
            else:
                return tens[num // 10] + " " + ones[num % 10]
        elif 100 <= num <= 999:
            if num % 100 == 0:
                return ones[num // 100] + " hundred"
            else:
                return ones[num // 100] + " hundred " + number_to_words(str(num % 100))
        elif 1000 <= num <= 999999:
            if num % 1000 == 0:
                return number_to_words(str(num // 1000)) + " thousand"
            else:
                return number_to_words(str(num // 1000)) + " thousand " + number_to_words(str(num % 1000))
        else:
            # For very large numbers, just return the original
            return num_str
    except ValueError:
        # If it's not a valid integer, return as-is
        return num_str


def normalize_text_for_phonemization(text: str) -> str:
    """Normalize text by converting numbers to their spoken form."""
    
    def replace_ordinal(match):
        """Convert ordinal numbers to words."""
        num = match.group(1)
        suffix = match.group(2).lower()
        
        # Convert base number to words
        base_words = number_to_words(num)
        
        # Handle ordinal endings
        if base_words.endswith('one'):
            return base_words[:-3] + 'first'
        elif base_words.endswith('two'):
            return base_words[:-3] + 'second'
        elif base_words.endswith('three'):
            return base_words[:-5] + 'third'
        elif base_words.endswith('five'):
            return base_words[:-4] + 'fifth'
        elif base_words.endswith('eight'):
            return base_words[:-5] + 'eighth'
        elif base_words.endswith('nine'):
            return base_words[:-4] + 'ninth'
        elif base_words.endswith('twelve'):
            return base_words[:-6] + 'twelfth'
        elif base_words.endswith('twenty'):
            return base_words[:-6] + 'twentieth'
        elif base_words.endswith('thirty'):
            return base_words[:-6] + 'thirtieth'
        elif base_words.endswith('forty'):
            return base_words[:-5] + 'fortieth'
        elif base_words.endswith('fifty'):
            return base_words[:-5] + 'fiftieth'
        elif base_words.endswith('sixty'):
            return base_words[:-5] + 'sixtieth'
        elif base_words.endswith('seventy'):
            return base_words[:-7] + 'seventieth'
        elif base_words.endswith('eighty'):
            return base_words[:-6] + 'eightieth'
        elif base_words.endswith('ninety'):
            return base_words[:-6] + 'ninetieth'
        else:
            # Default: add 'th' to the end
            return base_words + 'th'
    
    def replace_number(match):
        return number_to_words(match.group())
    
    # First replace ordinal numbers (1st, 2nd, 3rd, 4th, etc.)
    normalized = re.sub(r'\b(\d+)(st|nd|rd|th)\b', replace_ordinal, text, flags=re.IGNORECASE)
    
    # Then replace standalone numbers with their word forms
    normalized = re.sub(r'\b\d+\b', replace_number, normalized)
    
    return normalized


def tokenize_words(text: str) -> List[str]:
    # Updated to handle alphanumeric tokens (numbers + letters) like "17th", "1st", "2nd", etc.
    return [w for w in re.findall(r"[A-Za-z0-9]+'?[A-Za-z0-9]*|[A-Za-z0-9]+", text or "")] 


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


def normalize_ipa_preserve_diphthongs(seq: List[str]) -> List[str]:
    """Normalize IPA but preserve diphthongs (for Allosaurus which we already fixed)."""
    out: List[str] = []
    for p in seq:  # Skip split_diphthongs() to preserve our reconstructed diphthongs
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
    # Normalize numbers to their spoken form before phonemization
    normalized_text = normalize_text_for_phonemization(text)
    logger.debug(f"Normalized text for phonemization: '{text}' -> '{normalized_text}'")
    
    words = tokenize_words(normalized_text)
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

# Allosaurus phoneme recognition model
_allosaurus_model: Optional[object] = None


def _lazy_allosaurus_model():
    """Lazy load Allosaurus phoneme recognition model."""
    global _allosaurus_model
    if _allosaurus_model is None:
        try:
            from allosaurus.app import read_recognizer
            logger.info("Loading Allosaurus phoneme recognition model")
            _allosaurus_model = read_recognizer()
            logger.info("Allosaurus model loaded successfully")
        except ImportError:
            raise ImportError("Allosaurus not installed. Install with: pip install allosaurus")
        except Exception as e:
            raise RuntimeError(f"Failed to load Allosaurus model: {e}")
    return _allosaurus_model


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


def fix_allosaurus_oversegmentation(phonemes: List[str]) -> List[str]:
    """Fix Allosaurus over-segmentation and duplicate vowels to create proper English phonemes."""
    if not phonemes:
        return phonemes
    
    logger.debug(f"🔧 Starting fix with {len(phonemes)} phonemes: {phonemes[:10]}...")
    
    # First pass: remove consecutive duplicates 
    deduplicated = []
    prev = None
    for phone in phonemes:
        if phone != prev:
            deduplicated.append(phone)
        prev = phone
    
    logger.debug(f"🔧 After deduplication: {len(deduplicated)} phonemes: {deduplicated[:10]}...")
    
    # Second pass: reconstruct diphthongs
    fixed = []
    i = 0
    
    while i < len(deduplicated):
        current = deduplicated[i]
        next_phone = deduplicated[i + 1] if i + 1 < len(deduplicated) else None
        
        # Reconstruct common English diphthongs from Allosaurus segments
        if current == "a" and next_phone == "ɪ":
            fixed.append("aɪ")  # price diphthong (my, I, time)
            logger.debug(f"🎭 Combined a+ɪ → aɪ at position {i}")
            i += 2
        elif current == "e" and next_phone == "ɪ":
            fixed.append("eɪ")  # face diphthong (name, day, play)
            logger.debug(f"🎭 Combined e+ɪ → eɪ at position {i}")
            i += 2
        elif current == "a" and next_phone == "ʊ":
            fixed.append("aʊ")  # mouth diphthong (now, house)
            logger.debug(f"🎭 Combined a+ʊ → aʊ at position {i}")
            i += 2
        elif current == "o" and next_phone == "ʊ":
            fixed.append("oʊ")  # goat diphthong (go, show)
            logger.debug(f"🎭 Combined o+ʊ → oʊ at position {i}")
            i += 2
        elif current == "ɔ" and next_phone == "ɪ":
            fixed.append("ɔɪ")  # choice diphthong (boy, voice)
            logger.debug(f"🎭 Combined ɔ+ɪ → ɔɪ at position {i}")
            i += 2
        elif current == "ɪ" and next_phone == "ə":
            fixed.append("ɪə")  # near diphthong (here, near)
            logger.debug(f"🎭 Combined ɪ+ə → ɪə at position {i}")
            i += 2
        elif current == "ɛ" and next_phone == "ə":
            fixed.append("ɛə")  # square diphthong (there, care)
            logger.debug(f"🎭 Combined ɛ+ə → ɛə at position {i}")
            i += 2
        elif current == "ʊ" and next_phone == "ə":
            fixed.append("ʊə")  # cure diphthong (sure, tour)
            logger.debug(f"🎭 Combined ʊ+ə → ʊə at position {i}")
            i += 2
        else:
            # Keep single phoneme
            fixed.append(current)
            i += 1
    
    logger.debug(f"🔧 Final fixed phonemes: {len(fixed)} phonemes: {fixed[:10]}...")
    return fixed


def reconstruct_diphthongs(phonemes: List[str]) -> List[str]:
    """Reconstruct common English diphthongs that wav2vec2 might have separated."""
    if not phonemes:
        return phonemes
    
    reconstructed = []
    i = 0
    
    while i < len(phonemes):
        current = phonemes[i]
        next_phoneme = phonemes[i + 1] if i + 1 < len(phonemes) else None
        
        # Common English diphthongs to reconstruct
        if current == "a" and next_phoneme == "ɪ":
            reconstructed.append("aɪ")  # like in "my", "I", "high"
            i += 2
        elif current == "eɪ" or (current == "e" and next_phoneme == "ɪ"):
            reconstructed.append("eɪ")  # like in "name", "day", "great"
            i += 2 if next_phoneme == "ɪ" else 1
        elif current == "aʊ" or (current == "a" and next_phoneme == "ʊ"):
            reconstructed.append("aʊ")  # like in "now", "house"
            i += 2 if next_phoneme == "ʊ" else 1
        elif current == "oʊ" or (current == "o" and next_phoneme == "ʊ"):
            reconstructed.append("oʊ")  # like in "go", "show"
            i += 2 if next_phoneme == "ʊ" else 1
        elif current == "ɔɪ" or (current == "ɔ" and next_phoneme == "ɪ"):
            reconstructed.append("ɔɪ")  # like in "boy", "voice"
            i += 2 if next_phoneme == "ɪ" else 1
        elif current == "ɪə" or (current == "ɪ" and next_phoneme == "ə"):
            reconstructed.append("ɪə")  # like in "here", "near"
            i += 2 if next_phoneme == "ə" else 1
        elif current == "ʊə" or (current == "ʊ" and next_phoneme == "ə"):
            reconstructed.append("ʊə")  # like in "sure", "tour"
            i += 2 if next_phoneme == "ə" else 1
        elif current == "ɛə" or (current == "ɛ" and next_phoneme == "ə"):
            reconstructed.append("ɛə")  # like in "there", "care"
            i += 2 if next_phoneme == "ə" else 1
        else:
            # Handle single vowels that might be missing
            if current in ["a", "e", "i", "o", "u"] and len(current) == 1:
                # Map single vowels to more likely IPA equivalents
                vowel_map = {
                    "a": "æ",  # trap vowel
                    "e": "ɛ",  # dress vowel  
                    "i": "ɪ",  # kit vowel
                    "o": "ɒ",  # lot vowel
                    "u": "ʊ"   # foot vowel
                }
                reconstructed.append(vowel_map.get(current, current))
            else:
                reconstructed.append(current)
            i += 1
    
    return reconstructed


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


def allosaurus_to_oxford_ipa(phone: str) -> str:
    """Convert Allosaurus IPA phonemes to Oxford English IPA system."""
    # Allosaurus uses standard IPA, but we need to map to Oxford English system
    mapping = {
        # Consonants - usually direct mapping
        "b": "b", "d": "d", "f": "f", "g": "g", "h": "h", "k": "k", "l": "l", 
        "m": "m", "n": "n", "p": "p", "r": "r", "s": "s", "t": "t", "v": "v", 
        "w": "w", "z": "z", "j": "j", "ŋ": "ŋ", "θ": "θ", "ð": "ð", "ʃ": "ʃ", 
        "ʒ": "ʒ", "ʧ": "ʧ", "ʤ": "ʤ", "tʃ": "ʧ", "dʒ": "ʤ",
        
        # Vowels - map to Oxford English IPA
        "ə": "ə",      # schwa
        "ɛ": "ɛ",      # dress  
        "ɪ": "ɪ",      # kit
        "i": "iː",     # fleece (Allosaurus often uses 'i' for long vowel)
        "ɑ": "ɑ",      # palm/start
        "ɔ": "ɔ",      # thought
        "ʊ": "ʊ",      # foot
        "u": "uː",     # goose (Allosaurus often uses 'u' for long vowel)
        "æ": "æ",      # trap
        "ʌ": "ʌ",      # strut
        "ɒ": "ɒ",      # lot (UK)
        "ɜ": "ɜː",     # nurse
        "e": "eɪ",     # Allosaurus 'e' often represents the face diphthong
        "o": "oʊ",     # Allosaurus 'o' often represents the goat diphthong
        "a": "aɪ",     # Allosaurus 'a' sometimes represents price diphthong
        
        # Common Allosaurus diphthongs  
        "aɪ": "aɪ",    # price
        "eɪ": "eɪ",    # face 
        "ɔɪ": "ɔɪ",    # choice
        "aʊ": "aʊ",    # mouth
        "oʊ": "oʊ",    # goat (US)
        "əʊ": "əʊ",    # goat (UK)
        "ɪə": "ɪə",    # near
        "ɛə": "ɛə",    # square
        "ʊə": "ʊə",    # cure
        
        # Long vowels from Allosaurus
        "iː": "iː",    # fleece
        "uː": "uː",    # goose
        "ɑː": "ɑː",    # palm
        "ɔː": "ɔː",    # thought
        "ɜː": "ɜː",    # nurse
        
        # R-colored vowels (common in Allosaurus US English)
        "ɚ": "ər",     # unstressed syllabic r
        "ɝ": "ɜː",     # stressed r-colored vowel
        
        # Syllabic consonants
        "n̩": "n",      # syllabic n
        "m̩": "m",      # syllabic m
        "l̩": "l",      # syllabic l
        
        # Handle stress markers and diacritics
        "ˈ": "",       # primary stress (remove)
        "ˌ": "",       # secondary stress (remove)
        "ː": "",       # length marker (handle separately)
    }
    
    # Clean the phoneme first
    clean_phone = phone.strip()
    
    # Handle length markers
    if clean_phone.endswith("ː"):
        base = clean_phone[:-1]
        if base in ["i", "u", "ɑ", "ɔ", "ɜ"]:
            return base + "ː"
        else:
            clean_phone = base
    
    # Apply mapping
    return mapping.get(clean_phone, clean_phone)


def improved_espeak_to_ipa(phone: str) -> str:
    """Convert espeak phonemes to Oxford English IPA with better mapping."""
    # Oxford English IPA mapping based on Language Confidence system
    mapping = {
        # Consonants (Oxford IPA standard)
        "tS": "ʧ", "dZ": "ʤ", "N": "ŋ", "T": "θ", "D": "ð", 
        "S": "ʃ", "Z": "ʒ", "h": "h", "x": "h", "?": "ʔ", "J": "j", "w": "w",
        "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g",
        "f": "f", "v": "v", "s": "s", "z": "z", "m": "m", "n": "n", "l": "l", "r": "r",
        
        # Vowels - Oxford English IPA system
        "@": "ə",      # schwa (cup)
        "3": "ɜ",      # nurse 
        "A": "ɑ",      # father
        "I": "ɪ",      # sit
        "O": "ɔ",      # hawk
        "U": "ʊ",      # put
        "i": "i",      # cosy
        "u": "u",      # goose
        "e": "ɛ",      # dress
        "o": "ɒ",      # hot (UK)
        "a": "æ",      # trap (US) or "a" (UK)
        "V": "ʌ",      # cup
        "Q": "ɒ",      # hot
        "E": "ɛ",      # dress
        
        # Long vowels
        "A:": "ɑː",    # father (long)
        "3:": "əː",    # nurse (long) 
        "i:": "iː",    # fleece
        "O:": "ɔː",    # hawk (long)
        "u:": "uː",    # goose (long)
        
        # Diphthongs (Oxford system) - multiple variations for wav2vec2
        "aI": "aɪ",    # price (US) or "ʌɪ" (UK)
        "AI": "aɪ",    # alternative
        "aj": "aɪ",    # alternative variant
        "eI": "eɪ",    # face
        "EI": "eɪ",    # alternative
        "ej": "eɪ",    # alternative variant
        "OI": "ɔɪ",    # choice
        "oj": "ɔɪ",    # alternative
        "aU": "aʊ",    # mouth
        "AU": "aʊ",    # alternative
        "aw": "aʊ",    # alternative variant
        "oU": "oʊ",    # goat (US) or "əʊ" (UK)
        "OU": "oʊ",    # alternative
        "ow": "oʊ",    # alternative variant
        "@U": "əʊ",    # UK variant
        "I@": "ɪə",    # here
        "e@": "ɛː",    # square
        "u@": "ʊə",    # cure
        
        # Additional vowel mappings wav2vec2 might produce
        "ɐ": "æ",      # near-open central vowel → trap
        "ɚ": "ər",     # r-colored schwa
        
        # Clean up common wav2vec2 artifacts
        "uu": "u", "ii": "i", "oo": "o", "aa": "a", "ee": "e",
        
        # Syllabic consonants
        "r=": "ər", "l=": "l", "n=": "n", "m=": "m",
    }
    
    # Apply mappings
    for espeak, ipa in mapping.items():
        phone = phone.replace(espeak, ipa)
    
    # Remove stress markers and other diacritics
    phone = phone.replace("'", "").replace(",", "").replace(":", "")
    
    return phone


def phonemes_from_audio(audio_16k: np.ndarray) -> List[str]:
    """Extract phonemes directly from 16kHz mono audio with intelligent reconstruction."""
    try:
        # Try Allosaurus first if available - MUCH better for phoneme recognition
        return phonemes_from_audio_allosaurus(audio_16k)
    except Exception as e:
        logger.warning(f"Allosaurus not available ({e}), using enhanced wav2vec2 with intelligent reconstruction")
        return phonemes_from_audio_wav2vec2(audio_16k)


def phonemes_from_audio_allosaurus(audio_16k: np.ndarray) -> List[str]:
    """Extract phonemes using Allosaurus - superior model for phoneme recognition."""
    try:
        import tempfile
        import soundfile as sf
        
        # Get cached Allosaurus model
        model = _lazy_allosaurus_model()
        
        # Enhanced audio preprocessing
        if len(audio_16k) == 0:
            return []
            
        # Normalize audio for Allosaurus (expects [-1, 1] range)
        audio_normalized = audio_16k / (np.max(np.abs(audio_16k)) + 1e-8)
        
        # Apply light noise reduction
        noise_threshold = 0.02
        audio_cleaned = np.where(np.abs(audio_normalized) < noise_threshold, 0, audio_normalized)
        
        # Scale to optimal range
        peak_amplitude = np.max(np.abs(audio_cleaned))
        if peak_amplitude > 0:
            audio_final = 0.9 * audio_cleaned / peak_amplitude
        else:
            audio_final = audio_cleaned
        
        # Allosaurus expects a wav file path, so create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            sf.write(temp_wav.name, audio_final, 16000)
            temp_wav_path = temp_wav.name
        
        try:
            # Run Allosaurus phoneme recognition with English language
            phones_raw = model.recognize(temp_wav_path, lang_id="eng")
            logger.debug(f"Raw Allosaurus output: {phones_raw}")
            
            # Parse Allosaurus output - it's a simple space-separated phoneme sequence
            phones = []
            for phone in phones_raw.strip().split():
                phone = phone.strip()
                if phone and phone not in ["<unk>", "<sil>", "<spn>", ""]:
                    phones.append(phone)
            
            logger.debug(f"Extracted Allosaurus phones: {phones[:20]}...")
            
            # Convert Allosaurus IPA to Oxford English IPA
            ipa_phones = []
            for phone in phones:
                normalized = allosaurus_to_oxford_ipa(phone)
                if normalized and len(normalized.strip()) > 0:
                    ipa_phones.append(normalized)
            
            logger.debug(f"Oxford IPA phones: {ipa_phones[:20]}...")
            
            # Post-process to fix Allosaurus over-segmentation and create proper diphthongs
            logger.debug(f"🔧 Before fix: {ipa_phones[:10]}...")
            processed_phones = fix_allosaurus_oversegmentation(ipa_phones)
            logger.debug(f"🔧 After fix: {processed_phones[:10]}...")
            
            # Clean and normalize (preserve diphthongs that we just reconstructed)
            final_ipa = normalize_ipa_preserve_diphthongs(processed_phones)
            logger.info(f"🦎 Allosaurus extraction: {len(phones)} raw → {len(ipa_phones)} mapped → {len(processed_phones)} processed → {len(final_ipa)} final")
            
            # Debug: Check if fix actually worked
            has_consecutive_dups = any(processed_phones[i] == processed_phones[i+1] for i in range(len(processed_phones)-1))
            diphthongs_found = [p for p in processed_phones if len(p) > 1 and any(v in p for v in 'aeiou')]
            logger.info(f"🔧 Fix results: consecutive_duplicates={has_consecutive_dups}, diphthongs={diphthongs_found[:5]}")
            
            return final_ipa
            
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_wav_path)
            except:
                pass
        
    except Exception as exc:
        logger.warning("Allosaurus phoneme recognition failed: %s", exc)
        raise exc


def phonemes_from_audio_wav2vec2(audio_16k: np.ndarray) -> List[str]:
    """Extract phonemes using wav2vec2 (fallback method)."""
    try:
        import torch
        processor, model = _lazy_wav2vec2_phoneme()
        
        # Enhanced audio preprocessing for better phoneme extraction
        if len(audio_16k) == 0:
            return []
            
        # 1. Normalize audio amplitude
        audio_normalized = audio_16k / (np.max(np.abs(audio_16k)) + 1e-8)
        
        # 2. Apply light noise reduction
        # Remove very quiet segments that might be noise
        noise_threshold = 0.02
        audio_cleaned = np.where(np.abs(audio_normalized) < noise_threshold, 0, audio_normalized)
        
        # 3. Ensure proper amplitude range for wav2vec2
        # The model expects values roughly in [-1, 1] range
        peak_amplitude = np.max(np.abs(audio_cleaned))
        if peak_amplitude > 0:
            # Scale to use 80% of the range to avoid clipping
            audio_normalized = 0.8 * audio_cleaned / peak_amplitude
        else:
            audio_normalized = audio_cleaned
        
        with torch.no_grad():
            inputs = processor(audio_normalized, sampling_rate=16000, return_tensors="pt", padding=True)
            logits = model(inputs.input_values).logits
            
            # Try multiple decoding strategies for better phoneme detection
            try:
                # Method 1: CTC beam search
                from torch.nn import functional as F
                log_probs = F.log_softmax(logits, dim=-1)
                
                # Use a confidence threshold to filter weak predictions
                confidence_threshold = 0.1
                probs = torch.exp(log_probs)
                max_probs, ids = torch.max(probs, dim=-1)
                
                # Mask out low-confidence predictions
                ids = ids.squeeze()
                confidences = max_probs.squeeze()
                
                # Only keep high-confidence phonemes
                confident_ids = ids[confidences > confidence_threshold]
                
                # Decode with better handling
                if len(confident_ids) > 0:
                    text = processor.batch_decode(confident_ids.unsqueeze(0))[0]
                else:
                    # Fallback to regular argmax if threshold too strict
                    ids = torch.argmax(log_probs, dim=-1)
                    text = processor.batch_decode(ids)[0]
                    
            except Exception as e:
                logger.debug(f"CTC beam search failed, using fallback: {e}")
                # Fallback to simple argmax
                ids = torch.argmax(logits, dim=-1)
                text = processor.batch_decode(ids)[0]
        
        logger.debug(f"Raw wav2vec2 output: {text}")
        
        # Extract individual phonemes, filter out word boundaries and empty tokens
        phones = [p for p in text.strip().split() if p and p != "|" and len(p.strip()) > 0]
        logger.debug(f"Extracted phones: {phones[:20]}...")
        
        # Convert espeak phonemes to Oxford IPA with improved mapping
        ipa_phones = []
        for phone in phones:
            normalized = improved_espeak_to_ipa(phone)
            if normalized and len(normalized.strip()) > 0 and normalized not in ["'", ",", ":"]:
                ipa_phones.append(normalized)
        
        logger.debug(f"IPA phones after mapping: {ipa_phones[:20]}...")
        
        # Post-process to reconstruct diphthongs that wav2vec2 might have separated
        ipa_phones = reconstruct_diphthongs(ipa_phones)
        
        logger.debug(f"IPA phones before cleaning: {ipa_phones[:20]}...")
        
        # Apply advanced cleaning to fix artifacts and create proper diphthongs
        ipa_phones = clean_phoneme_sequence(ipa_phones)
        
        # Additional cleaning for Oxford IPA consistency
        cleaned_phones = []
        for phone in ipa_phones:
            # Map to Oxford IPA standard phonemes
            if phone in ["b", "ʧ", "d", "ð", "f", "g", "h", "ʤ", "k", "l", "m", "n", "ŋ", "p", "r", "s", "ʃ", "t", "θ", "v", "w", "j", "z", "ʒ"]:
                cleaned_phones.append(phone)  # Valid consonants
            elif phone in ["ɛ", "ə", "ɪ", "i", "ʊ", "ɑ", "ɔ", "æ", "u", "a", "ɒ", "ʌ", "ɑː", "ɛː", "əː", "iː", "ɔː", "uː"]:
                cleaned_phones.append(phone)  # Valid vowels
            elif phone in ["aʊ", "eɪ", "ɔɪ", "aɪ", "oʊ", "ʌɪ", "əʊ", "ɪə", "ʊə"]:
                cleaned_phones.append(phone)  # Valid diphthongs
            else:
                # Try to map unknown phonemes to closest Oxford IPA equivalent
                if len(phone) == 1:
                    if phone in "aeiou":
                        # Map basic vowels to schwa as fallback
                        cleaned_phones.append("ə")
                    elif phone.isalpha():
                        # Keep consonant-like sounds
                        cleaned_phones.append(phone)
        
        logger.debug(f"Final cleaned phones: {cleaned_phones[:20]}...")
        
        # Final IPA normalization using our existing function
        final_ipa = normalize_ipa(cleaned_phones)
        logger.debug(f"Final normalized IPA: {final_ipa[:20]}...")
        logger.info(f"🔍 wav2vec2 extraction: {len(phones)} raw → {len(ipa_phones)} mapped → {len(final_ipa)} final")
        
        return final_ipa
        
    except Exception as exc:
        logger.warning("wav2vec2 phoneme recognition failed: %s", exc)
        return []



