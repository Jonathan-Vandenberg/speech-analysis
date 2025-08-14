import re
import unicodedata
from typing import List


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


