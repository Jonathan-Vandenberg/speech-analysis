import re
import unicodedata
from typing import List, Dict, Tuple
import logging

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
        if not os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
            for p in (
                "/opt/homebrew/opt/espeak-ng/lib/libespeak-ng.dylib",
                "/opt/homebrew/lib/libespeak-ng.dylib",
                "/usr/local/lib/libespeak-ng.dylib",
            ):
                if os.path.exists(p):
                    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = p
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



