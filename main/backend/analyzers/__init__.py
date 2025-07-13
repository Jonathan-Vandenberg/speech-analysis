"""
Speech Analysis Modules

This package contains modular analyzers for different aspects of speech analysis:
- Pronunciation analysis (Professional forced alignment + Legacy Vosk-based)
- Relevance analysis  
- Grammar correction
- IELTS/CEFR scoring
- Freestyle speech analysis
"""

# Professional forced alignment analyzer (recommended)
from .pronunciation_mfa import ProfessionalPronunciationAnalyzer

# Legacy Vosk-based analyzer (for backward compatibility)
from .pronunciation import PronunciationAnalyzer as LegacyPronunciationAnalyzer

# Use the professional analyzer as the default
PronunciationAnalyzer = ProfessionalPronunciationAnalyzer

# Other analyzers
from .relevance import RelevanceAnalyzer
from .grammar import GrammarCorrector
from .scoring import SpeechScorer
from .freestyle import FreestyleSpeechAnalyzer

__all__ = [
    'PronunciationAnalyzer',           # Professional forced alignment (default)
    'ProfessionalPronunciationAnalyzer',
    'LegacyPronunciationAnalyzer',     # Legacy Vosk-based analyzer
    'RelevanceAnalyzer', 
    'GrammarCorrector',
    'SpeechScorer',
    'FreestyleSpeechAnalyzer'
] 