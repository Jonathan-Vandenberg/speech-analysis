import os
import re
import json
import urllib.request
from typing import List, Dict, Optional, Set
from pathlib import Path
import pickle

class ProductionPhonemeEngine:
    """
    Production-quality phoneme engine using CMU Pronouncing Dictionary and comprehensive fallbacks
    """
    
    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'phoneme_data')
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Main pronunciation dictionaries
        self.cmu_dict = {}
        self.british_dict = {}
        self.word_variants = {}
        
        # Phoneme mappings
        self.arpabet_to_ipa = {}
        self.setup_phoneme_mappings()
        
        # Load pronunciation data
        self.load_pronunciation_data()
        
        print(f"✅ Production Phoneme Engine loaded with {len(self.cmu_dict)} words")
    
    def setup_phoneme_mappings(self):
        """Setup comprehensive ARPABET to IPA mappings"""
        self.arpabet_to_ipa = {
            # Vowels - monophthongs
            'AA': 'ɑː',    # father, lot
            'AE': 'æ',     # cat, trap
            'AH': 'ʌ',     # cut, strut
            'AO': 'ɔː',    # caught, thought
            'AW': 'aʊ',    # mouth, how
            'AX': 'ə',     # about (unstressed)
            'AXR': 'ɚ',    # letter (unstressed r-colored)
            'AY': 'aɪ',    # price, time
            'EH': 'ɛ',     # dress, bet
            'ER': 'ɜːr',   # nurse, word
            'EY': 'eɪ',    # face, day
            'IH': 'ɪ',     # kit, bit
            'IX': 'ɨ',     # roses (unstressed)
            'IY': 'iː',    # fleece, see
            'OW': 'oʊ',    # goat, note
            'OY': 'ɔɪ',    # choice, boy
            'UH': 'ʊ',     # foot, good
            'UW': 'uː',    # goose, blue
            'UX': 'ʉ',     # tuna (unstressed)
            
            # Consonants
            'B': 'b',      # bad
            'CH': 'tʃ',    # chair
            'D': 'd',      # did
            'DH': 'ð',     # this, that
            'DX': 'ɾ',     # butter (flap)
            'EL': 'l̩',     # bottle (syllabic l)
            'EM': 'm̩',     # rhythm (syllabic m)
            'EN': 'n̩',     # button (syllabic n)
            'F': 'f',      # five
            'G': 'ɡ',      # give
            'HH': 'h',     # house
            'JH': 'dʒ',    # just
            'K': 'k',      # cat
            'L': 'l',      # left
            'M': 'm',      # man
            'N': 'n',      # no
            'NG': 'ŋ',     # sing
            'NX': 'ɾ̃',     # winner (nasalized flap)
            'P': 'p',      # put
            'Q': 'ʔ',      # glottal stop
            'R': 'r',      # right
            'S': 's',      # say
            'SH': 'ʃ',     # she
            'T': 't',      # take
            'TH': 'θ',     # think
            'V': 'v',      # very
            'W': 'w',      # way
            'WH': 'ʍ',     # which (in dialects that distinguish)
            'Y': 'j',      # yes
            'Z': 'z',      # zoo
            'ZH': 'ʒ',     # measure
        }
    
    def load_pronunciation_data(self):
        """Load CMU dictionary and other pronunciation resources"""
        try:
            # Load CMU Pronouncing Dictionary
            cmu_file = self.data_dir / "cmudict.dict"
            if not cmu_file.exists():
                self.download_cmu_dict()
            
            self.load_cmu_dict(cmu_file)
            
            # Load British pronunciation variants
            british_file = self.data_dir / "british_variants.json"
            if not british_file.exists():
                self.create_british_variants()
            
            self.load_british_variants(british_file)
            
            # Load common word variants and contractions
            self.load_word_variants()
            
        except Exception as e:
            print(f"⚠️ Error loading pronunciation data: {e}")
            print("📦 Using minimal built-in dictionary")
            self.load_minimal_dict()
    
    def download_cmu_dict(self):
        """Download CMU Pronouncing Dictionary"""
        try:
            print("📥 Downloading CMU Pronouncing Dictionary...")
            url = "https://raw.githubusercontent.com/cmusphinx/cmudict/master/cmudict.dict"
            cmu_file = self.data_dir / "cmudict.dict"
            
            urllib.request.urlretrieve(url, cmu_file)
            print("✅ CMU Dictionary downloaded successfully")
            
        except Exception as e:
            print(f"⚠️ Failed to download CMU dict: {e}")
            print("📦 Creating minimal dictionary")
            self.create_minimal_cmu_dict()
    
    def create_minimal_cmu_dict(self):
        """Create a minimal CMU dictionary with essential words"""
        minimal_dict = {
            'THE': ['DH', 'AH'],
            'AND': ['AH', 'N', 'D'],
            'OF': ['AH', 'V'],
            'TO': ['T', 'UW'],
            'A': ['AH'],
            'IN': ['IH', 'N'],
            'IS': ['IH', 'Z'],
            'YOU': ['Y', 'UW'],
            'THAT': ['DH', 'AE', 'T'],
            'IT': ['IH', 'T'],
            'HE': ['HH', 'IY'],
            'WAS': ['W', 'AH', 'Z'],
            'FOR': ['F', 'AO', 'R'],
            'ON': ['AO', 'N'],
            'ARE': ['AA', 'R'],
            'AS': ['AE', 'Z'],
            'WITH': ['W', 'IH', 'DH'],
            'HIS': ['HH', 'IH', 'Z'],
            'THEY': ['DH', 'EY'],
            'I': ['AY'],
            'AT': ['AE', 'T'],
            'BE': ['B', 'IY'],
            'THIS': ['DH', 'IH', 'S'],
            'HAVE': ['HH', 'AE', 'V'],
            'FROM': ['F', 'R', 'AH', 'M'],
            'OR': ['AO', 'R'],
            'ONE': ['W', 'AH', 'N'],
            'HAD': ['HH', 'AE', 'D'],
            'BY': ['B', 'AY'],
            'WORD': ['W', 'ER', 'D'],
            'BUT': ['B', 'AH', 'T'],
            'NOT': ['N', 'AO', 'T'],
            'WHAT': ['W', 'AH', 'T'],
            'ALL': ['AO', 'L'],
            'WERE': ['W', 'ER'],
            'WE': ['W', 'IY'],
            'WHEN': ['W', 'EH', 'N'],
            'YOUR': ['Y', 'AO', 'R'],
            'CAN': ['K', 'AE', 'N'],
            'SAID': ['S', 'EH', 'D'],
            'THERE': ['DH', 'EH', 'R'],
            'EACH': ['IY', 'CH'],
            'WHICH': ['W', 'IH', 'CH'],
            'SHE': ['SH', 'IY'],
            'DO': ['D', 'UW'],
            'HOW': ['HH', 'AW'],
            'THEIR': ['DH', 'EH', 'R'],
            'IF': ['IH', 'F'],
            'WILL': ['W', 'IH', 'L'],
            'UP': ['AH', 'P'],
            'OTHER': ['AH', 'DH', 'ER'],
            'ABOUT': ['AH', 'B', 'AW', 'T'],
            'OUT': ['AW', 'T'],
            'MANY': ['M', 'EH', 'N', 'IY'],
            'THEN': ['DH', 'EH', 'N'],
            'THEM': ['DH', 'EH', 'M'],
            'THESE': ['DH', 'IY', 'Z'],
            'SO': ['S', 'OW'],
            'SOME': ['S', 'AH', 'M'],
            'HER': ['HH', 'ER'],
            'WOULD': ['W', 'UH', 'D'],
            'MAKE': ['M', 'EY', 'K'],
            'LIKE': ['L', 'AY', 'K'],
            'INTO': ['IH', 'N', 'T', 'UW'],
            'HIM': ['HH', 'IH', 'M'],
            'TIME': ['T', 'AY', 'M'],
            'TWO': ['T', 'UW'],
            'MORE': ['M', 'AO', 'R'],
            'GO': ['G', 'OW'],
            'NO': ['N', 'OW'],
            'WAY': ['W', 'EY'],
            'MAY': ['M', 'EY'],
            'DAY': ['D', 'EY'],
            'USE': ['Y', 'UW', 'Z'],
            'MAN': ['M', 'AE', 'N'],
            'NEW': ['N', 'UW'],
            'NOW': ['N', 'AW'],
            'OLD': ['OW', 'L', 'D'],
            'SEE': ['S', 'IY'],
            'HIM': ['HH', 'IH', 'M'],
            'TWO': ['T', 'UW'],
            'HOW': ['HH', 'AW'],
            'ITS': ['IH', 'T', 'S'],
            'WHO': ['HH', 'UW'],
            'OIL': ['OY', 'L'],
            'SIT': ['S', 'IH', 'T'],
            'SET': ['S', 'EH', 'T'],
            'RUN': ['R', 'AH', 'N'],
            'EAT': ['IY', 'T'],
            'FAR': ['F', 'AA', 'R'],
            'SEA': ['S', 'IY'],
            'EYE': ['AY'],
            'HELLO': ['HH', 'AH', 'L', 'OW'],
            'WORLD': ['W', 'ER', 'L', 'D'],
            'GOOD': ['G', 'UH', 'D'],
            'BAD': ['B', 'AE', 'D'],
            'VERY': ['V', 'EH', 'R', 'IY'],
            'WELL': ['W', 'EH', 'L'],
            'THINK': ['TH', 'IH', 'NG', 'K'],
            'KNOW': ['N', 'OW'],
            'CALL': ['K', 'AO', 'L'],
            'FIRST': ['F', 'ER', 'S', 'T'],
            'WORK': ['W', 'ER', 'K'],
            'LONG': ['L', 'AO', 'NG'],
            'LITTLE': ['L', 'IH', 'T', 'AH', 'L'],
            'VERY': ['V', 'EH', 'R', 'IY'],
            'AFTER': ['AE', 'F', 'T', 'ER'],
            'RIGHT': ['R', 'AY', 'T'],
            'MOVE': ['M', 'UW', 'V'],
            'MUCH': ['M', 'AH', 'CH'],
            'WHERE': ['W', 'EH', 'R'],
            'THROUGH': ['TH', 'R', 'UW'],
            'BACK': ['B', 'AE', 'K'],
            'YEARS': ['Y', 'IH', 'R', 'Z'],
            'MOST': ['M', 'OW', 'S', 'T'],
            'CAME': ['K', 'EY', 'M'],
            'SHOW': ['SH', 'OW'],
            'EVERY': ['EH', 'V', 'R', 'IY'],
            'GOOD': ['G', 'UH', 'D'],
            'THOSE': ['DH', 'OW', 'Z'],
            'PEOPLE': ['P', 'IY', 'P', 'AH', 'L'],
            'MR': ['M', 'IH', 'S', 'T', 'ER'],
            'THEY': ['DH', 'EY'],
            'REALLY': ['R', 'IY', 'L', 'IY']
        }
        
        cmu_file = self.data_dir / "cmudict.dict"
        with open(cmu_file, 'w') as f:
            for word, phones in minimal_dict.items():
                f.write(f"{word}  {' '.join(phones)}\n")
        
        print("📦 Minimal CMU dictionary created")
    
    def load_cmu_dict(self, cmu_file: Path):
        """Load CMU dictionary from file"""
        try:
            with open(cmu_file, 'r', encoding='latin-1') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith(';;;'):
                        # Handle variant pronunciations (words ending with (1), (2), etc.)
                        if '  ' in line:
                            word_part, phones_part = line.split('  ', 1)
                            
                            # Extract base word (remove variant markers)
                            word = re.sub(r'\(\d+\)$', '', word_part).strip()
                            
                            # Parse phones, removing stress markers for base storage
                            phones = phones_part.split()
                            base_phones = [re.sub(r'\d+$', '', phone) for phone in phones]
                            
                            # Store main pronunciation (first variant or no variant)
                            if word not in self.cmu_dict or '(1)' in word_part:
                                self.cmu_dict[word] = base_phones
                            
                            # Store variants
                            if word not in self.word_variants:
                                self.word_variants[word] = []
                            self.word_variants[word].append(base_phones)
            
            print(f"✅ Loaded {len(self.cmu_dict)} words from CMU dictionary")
            
        except Exception as e:
            print(f"⚠️ Error loading CMU dict: {e}")
            self.load_minimal_dict()
    
    def create_british_variants(self):
        """Create British pronunciation variants for common differences"""
        british_variants = {
            # Common US vs UK differences
            'DANCE': ['D', 'AA', 'N', 'S'],      # US: æ, UK: ɑː
            'BATH': ['B', 'AA', 'TH'],           # US: æ, UK: ɑː
            'LAUGH': ['L', 'AA', 'F'],           # US: æ, UK: ɑː
            'ASK': ['AA', 'S', 'K'],             # US: æ, UK: ɑː
            'ANSWER': ['AA', 'N', 'S', 'ER'],    # US: æ, UK: ɑː
            'CASTLE': ['K', 'AA', 'S', 'AH', 'L'], # US: æ, UK: ɑː
            'FAST': ['F', 'AA', 'S', 'T'],       # US: æ, UK: ɑː
            'PLANT': ['P', 'L', 'AA', 'N', 'T'], # US: æ, UK: ɑː
            'BRANCH': ['B', 'R', 'AA', 'N', 'CH'], # US: æ, UK: ɑː
            'EXAMPLE': ['IH', 'G', 'Z', 'AA', 'M', 'P', 'AH', 'L'], # US: æ, UK: ɑː
            
            # R-dropping in British English
            'CAR': ['K', 'AA'],                  # UK: no final R
            'PARK': ['P', 'AA', 'K'],            # UK: no R before consonant
            'BIRD': ['B', 'ER', 'D'],            # Different vowel
            'NURSE': ['N', 'ER', 'S'],           # Different vowel
            
            # LOT-THOUGHT merger differences
            'LOT': ['L', 'AO', 'T'],             # UK: ɒ
            'HOT': ['HH', 'AO', 'T'],            # UK: ɒ
            'GOT': ['G', 'AO', 'T'],             # UK: ɒ
            
            # Other vowel differences
            'SCHEDULE': ['SH', 'EH', 'D', 'Y', 'UW', 'L'], # UK: ʃ vs US: sk
            'PRIVACY': ['P', 'R', 'IH', 'V', 'AH', 'S', 'IY'], # UK: ɪ vs US: aɪ
        }
        
        british_file = self.data_dir / "british_variants.json"
        with open(british_file, 'w') as f:
            json.dump(british_variants, f, indent=2)
        
        print("🇬🇧 Created British pronunciation variants")
    
    def load_british_variants(self, british_file: Path):
        """Load British pronunciation variants"""
        try:
            with open(british_file, 'r') as f:
                self.british_dict = json.load(f)
            print(f"🇬🇧 Loaded {len(self.british_dict)} British variants")
        except Exception as e:
            print(f"⚠️ Error loading British variants: {e}")
    
    def load_word_variants(self):
        """Load common word variants, contractions, and morphological forms"""
        # Common contractions
        contractions = {
            "CAN'T": ["K", "AE", "N", "T"],
            "WON'T": ["W", "OW", "N", "T"],
            "DON'T": ["D", "OW", "N", "T"],
            "I'M": ["AY", "M"],
            "YOU'RE": ["Y", "UH", "R"],
            "IT'S": ["IH", "T", "S"],
            "THAT'S": ["DH", "AE", "T", "S"],
            "THERE'S": ["DH", "EH", "R", "S"],
            "WE'RE": ["W", "IH", "R"],
            "THEY'RE": ["DH", "EH", "R"],
            "I'VE": ["AY", "V"],
            "YOU'VE": ["Y", "UW", "V"],
            "WE'VE": ["W", "IY", "V"],
            "THEY'VE": ["DH", "EY", "V"],
            "I'LL": ["AY", "L"],
            "YOU'LL": ["Y", "UW", "L"],
            "HE'LL": ["HH", "IY", "L"],
            "SHE'LL": ["SH", "IY", "L"],
            "WE'LL": ["W", "IY", "L"],
            "THEY'LL": ["DH", "EY", "L"],
            "WOULD'VE": ["W", "UH", "D", "AH", "V"],
            "COULD'VE": ["K", "UH", "D", "AH", "V"],
            "SHOULD'VE": ["SH", "UH", "D", "AH", "V"],
            "ISN'T": ["IH", "Z", "AH", "N", "T"],
            "AREN'T": ["AA", "R", "AH", "N", "T"],
            "WASN'T": ["W", "AA", "Z", "AH", "N", "T"],
            "WEREN'T": ["W", "ER", "AH", "N", "T"],
            "HAVEN'T": ["HH", "AE", "V", "AH", "N", "T"],
            "HASN'T": ["HH", "AE", "Z", "AH", "N", "T"],
            "HADN'T": ["HH", "AE", "D", "AH", "N", "T"],
            "WOULDN'T": ["W", "UH", "D", "AH", "N", "T"],
            "COULDN'T": ["K", "UH", "D", "AH", "N", "T"],
            "SHOULDN'T": ["SH", "UH", "D", "AH", "N", "T"],
        }
        
        # Add contractions to main dictionary
        for word, phones in contractions.items():
            self.cmu_dict[word] = phones
        
        print(f"📝 Loaded {len(contractions)} contractions")
    
    def load_minimal_dict(self):
        """Fallback minimal dictionary"""
        self.cmu_dict = {
            'THE': ['DH', 'AH'],
            'HELLO': ['HH', 'AH', 'L', 'OW'],
            'WORLD': ['W', 'ER', 'L', 'D'],
            'GOOD': ['G', 'UH', 'D'],
            'BAD': ['B', 'AE', 'D'],
            'YES': ['Y', 'EH', 'S'],
            'NO': ['N', 'OW'],
            'THANK': ['TH', 'AE', 'NG', 'K'],
            'YOU': ['Y', 'UW'],
            'PLEASE': ['P', 'L', 'IY', 'Z'],
            'SORRY': ['S', 'AO', 'R', 'IY'],
            'WATER': ['W', 'AO', 'T', 'ER'],
            'FOOD': ['F', 'UW', 'D'],
            'HOUSE': ['HH', 'AW', 'S'],
            'CAR': ['K', 'AA', 'R'],
            'WORK': ['W', 'ER', 'K'],
            'TIME': ['T', 'AY', 'M'],
            'PEOPLE': ['P', 'IY', 'P', 'AH', 'L'],
            'THINK': ['TH', 'IH', 'NG', 'K'],
            'KNOW': ['N', 'OW']
        }
        print("📦 Using minimal fallback dictionary")
    
    def get_phonemes(self, word: str, accent: str = 'us') -> List[str]:
        """
        Get phonemes for a word with comprehensive fallbacks
        
        Args:
            word: The word to get phonemes for
            accent: 'us' or 'uk' for accent-specific pronunciations
            
        Returns:
            List of IPA phonemes
        """
        word_upper = word.upper().strip()
        
        # Remove common punctuation
        word_clean = re.sub(r'[^\w\'-]', '', word_upper)
        
        # Try exact match first
        arpabet_phones = None
        
        # 1. Try accent-specific pronunciation
        if accent == 'uk' and word_clean in self.british_dict:
            arpabet_phones = self.british_dict[word_clean]
        
        # 2. Try main CMU dictionary
        elif word_clean in self.cmu_dict:
            arpabet_phones = self.cmu_dict[word_clean]
        
        # 3. Try without apostrophes/hyphens
        elif word_clean.replace("'", "").replace("-", "") in self.cmu_dict:
            arpabet_phones = self.cmu_dict[word_clean.replace("'", "").replace("-", "")]
        
        # 4. Try morphological analysis
        elif arpabet_phones is None:
            arpabet_phones = self.try_morphological_analysis(word_clean)
        
        # 5. Fallback to G2P system
        if arpabet_phones is None:
            return self.fallback_g2p_system(word_clean)
        
        # Convert ARPABET to IPA
        ipa_phonemes = []
        for phone in arpabet_phones:
            # Remove stress markers and convert
            base_phone = re.sub(r'\d+$', '', phone)
            if base_phone in self.arpabet_to_ipa:
                ipa_phonemes.append(self.arpabet_to_ipa[base_phone])
            else:
                print(f"⚠️ Unknown ARPABET phone: {base_phone}")
                # Try to map unknown phones
                ipa_phonemes.append(self.map_unknown_phone(base_phone))
        
        return ipa_phonemes
    
    def try_morphological_analysis(self, word: str) -> Optional[List[str]]:
        """Try to find pronunciation through morphological analysis"""
        
        # Handle common suffixes
        suffixes = {
            'S': ['S'],           # plural, 3rd person
            'ES': ['IH', 'Z'],    # wishes, boxes
            'ED': ['D'],          # past tense (regular)
            'ING': ['IH', 'NG'],  # present participle
            'LY': ['L', 'IY'],    # adverbs
            'ER': ['ER'],         # comparative
            'EST': ['AH', 'S', 'T'], # superlative
            'TION': ['SH', 'AH', 'N'], # -tion endings
            'SION': ['ZH', 'AH', 'N'], # -sion endings
            'NESS': ['N', 'AH', 'S'],  # -ness endings
            'MENT': ['M', 'AH', 'N', 'T'], # -ment endings
            'FUL': ['F', 'AH', 'L'],   # -ful endings
            'LESS': ['L', 'AH', 'S'],  # -less endings
        }
        
        for suffix, suffix_phones in suffixes.items():
            if word.endswith(suffix) and len(word) > len(suffix):
                root = word[:-len(suffix)]
                
                # Try to find root word
                if root in self.cmu_dict:
                    return self.cmu_dict[root] + suffix_phones
                
                # Try common transformations
                # Doubling final consonant (running -> run + ning)
                if len(root) > 1 and root[-1] == root[-2]:
                    single_root = root[:-1]
                    if single_root in self.cmu_dict:
                        return self.cmu_dict[single_root] + suffix_phones
                
                # Y to I transformation (happier -> happy + er)
                if suffix in ['ER', 'EST', 'NESS'] and root.endswith('I'):
                    y_root = root[:-1] + 'Y'
                    if y_root in self.cmu_dict:
                        return self.cmu_dict[y_root] + suffix_phones
        
        # Handle common prefixes
        prefixes = {
            'UN': ['AH', 'N'],
            'RE': ['R', 'IY'],
            'PRE': ['P', 'R', 'IY'],
            'DIS': ['D', 'IH', 'S'],
            'MIS': ['M', 'IH', 'S'],
            'OVER': ['OW', 'V', 'ER'],
            'UNDER': ['AH', 'N', 'D', 'ER'],
            'OUT': ['AW', 'T'],
            'UP': ['AH', 'P'],
        }
        
        for prefix, prefix_phones in prefixes.items():
            if word.startswith(prefix) and len(word) > len(prefix):
                root = word[len(prefix):]
                if root in self.cmu_dict:
                    return prefix_phones + self.cmu_dict[root]
        
        return None
    
    def fallback_g2p_system(self, word: str) -> List[str]:
        """Enhanced G2P system with better coverage"""
        
        # This is the same enhanced G2P from the original code but returns IPA directly
        phonemes = []
        i = 0
        word_lower = word.lower()
        
        # Enhanced G2P rules with IPA output
        g2p_rules = [
            # Silent letters and special cases
            ('mb$', 'm'),           ('bt$', 't'),           ('mn$', 'm'),
            ('gn$', 'n'),           ('wr', 'r'),            ('kn', 'n'),
            ('ps', 's'),            ('pt', 't'),
            
            # Complex consonant clusters  
            ('tch', 'tʃ'),          ('dge', 'dʒ'),          ('sch', 'sk'),
            ('qu', 'kw'),           ('x', 'ks'),
            
            # TH sounds (context dependent)
            ('th', 'θ'),            # Default voiceless
            
            # NG combinations
            ('ng$', 'ŋ'),           ('nge$', 'ndʒ'),        ('ng([bcdfghjklmnpqrstvwxyz])', 'ŋg'),
            
            # SH and CH sounds
            ('sh', 'ʃ'),            ('ch', 'tʃ'),           ('ck', 'k'),
            
            # PH and GH
            ('ph', 'f'),            ('gh$', ''),            ('ght$', 't'),
            ('gh', 'g'),
            
            # Double letters
            ('bb', 'b'), ('cc', 'k'), ('dd', 'd'), ('ff', 'f'),
            ('gg', 'g'), ('ll', 'l'), ('mm', 'm'), ('nn', 'n'),
            ('pp', 'p'), ('rr', 'r'), ('ss', 's'), ('tt', 't'), ('zz', 'z'),
            
            # Vowel combinations
            ('ai', 'eɪ'),           ('ay', 'eɪ'),           ('ea', 'iː'),
            ('ee', 'iː'),           ('ei', 'eɪ'),           ('ey', 'eɪ'),
            ('ie', 'iː'),           ('oa', 'oʊ'),           ('oe', 'oʊ'),
            ('oo', 'uː'),           ('ou', 'aʊ'),           ('ow', 'aʊ'),
            ('ue', 'uː'),           ('ui', 'uː'),
            
            # R-colored vowels
            ('ar', 'ɑːr'),          ('er', 'ɜːr'),          ('ir', 'ɜːr'),
            ('or', 'ɔːr'),          ('ur', 'ɜːr'),
            
            # Common endings
            ('tion', 'ʃən'),        ('sion', 'ʃən'),        ('ture', 'tʃər'),
            ('sure', 'ʃər'),        ('ed$', 'd'),           ('ing$', 'ɪŋ'),
            ('ly$', 'li'),          ('ness$', 'nəs'),       ('ment$', 'mənt'),
            
            # Y as vowel
            ('y$', 'i'),            ('y([bcdfghjklmnpqrstvwxz])', 'ɪ'),
            
            # Single consonants
            ('b', 'b'), ('c', 'k'), ('d', 'd'), ('f', 'f'), ('g', 'g'),
            ('h', 'h'), ('j', 'dʒ'), ('k', 'k'), ('l', 'l'), ('m', 'm'),
            ('n', 'n'), ('p', 'p'), ('r', 'r'), ('s', 's'), ('t', 't'),
            ('v', 'v'), ('w', 'w'), ('z', 'z'),
            
            # Single vowels
            ('a', 'æ'),             ('e', 'ɛ'),             ('i', 'ɪ'),
            ('o', 'ɔ'),             ('u', 'ʌ'),
        ]
        
        # Handle special TH cases (voiced)
        voiced_th_words = ['the', 'this', 'that', 'they', 'them', 'their', 'there', 'then', 'than', 'though', 'through']
        if word_lower in voiced_th_words:
            word_lower = word_lower.replace('th', 'ð')
        
        # Apply G2P rules
        while i < len(word_lower):
            matched = False
            
            for pattern, replacement in g2p_rules:
                import re
                
                if pattern.endswith('$'):
                    pattern_regex = pattern[:-1] + '$'
                    remaining_word = word_lower[i:]
                    match = re.match(pattern_regex, remaining_word)
                    if match:
                        match_len = len(match.group(0))
                        if replacement:
                            phonemes.append(replacement)
                        i += match_len
                        matched = True
                        break
                elif '(' in pattern:
                    remaining_word = word_lower[i:]
                    match = re.match(pattern, remaining_word)
                    if match:
                        match_len = len(match.group(0))
                        if replacement:
                            phonemes.append(replacement)
                        i += match_len
                        matched = True
                        break
                else:
                    if word_lower[i:].startswith(pattern):
                        if replacement:
                            phonemes.append(replacement)
                        i += len(pattern)
                        matched = True
                        break
            
            if not matched:
                i += 1
        
        return phonemes if phonemes else ['ʌ', 'n', 'k', 'n', 'oʊ', 'n']  # "unknown"
    
    def map_unknown_phone(self, phone: str) -> str:
        """Map unknown ARPABET phones to reasonable IPA equivalents"""
        # Common mappings for phones not in main dictionary
        unknown_mappings = {
            'Q': 'ʔ',      # Glottal stop
            'DX': 'ɾ',     # Flap T
            'NX': 'ɾ̃',     # Nasal flap
            'WH': 'ʍ',     # Voiceless W
            'EL': 'l̩',     # Syllabic L
            'EM': 'm̩',     # Syllabic M
            'EN': 'n̩',     # Syllabic N
        }
        
        return unknown_mappings.get(phone, phone.lower())
    
    def get_word_variants(self, word: str) -> List[List[str]]:
        """Get all pronunciation variants for a word"""
        word_upper = word.upper().strip()
        
        variants = []
        
        # Add main pronunciation
        main_pronunciation = self.get_phonemes(word)
        if main_pronunciation:
            variants.append(main_pronunciation)
        
        # Add stored variants
        if word_upper in self.word_variants:
            for variant_phones in self.word_variants[word_upper]:
                ipa_variant = []
                for phone in variant_phones:
                    base_phone = re.sub(r'\d+$', '', phone)
                    if base_phone in self.arpabet_to_ipa:
                        ipa_variant.append(self.arpabet_to_ipa[base_phone])
                
                if ipa_variant and ipa_variant not in variants:
                    variants.append(ipa_variant)
        
        return variants if variants else [self.get_phonemes(word)]
    
    def validate_installation(self) -> Dict[str, bool]:
        """Validate that the phoneme engine is properly installed"""
        status = {
            'cmu_dict_loaded': len(self.cmu_dict) > 100,
            'arpabet_mappings': len(self.arpabet_to_ipa) > 30,
            'british_variants': len(self.british_dict) > 0,
            'word_variants': len(self.word_variants) > 0,
            'g2p_fallback': True,  # Always available
        }
        
        # Test pronunciation generation
        try:
            test_words = ['hello', 'world', 'pronunciation', 'testing']
            test_results = []
            for word in test_words:
                phonemes = self.get_phonemes(word)
                test_results.append(len(phonemes) > 0)
            
            status['pronunciation_generation'] = all(test_results)
        except Exception:
            status['pronunciation_generation'] = False
        
        overall_status = all(status.values())
        
        print(f"🔍 Phoneme Engine Validation:")
        for component, ok in status.items():
            emoji = "✅" if ok else "❌"
            print(f"  {emoji} {component}: {ok}")
        
        print(f"🎯 Overall Status: {'✅ READY' if overall_status else '⚠️ ISSUES DETECTED'}")
        
        return status

# Global instance for use throughout the application
_phoneme_engine = None

def get_phoneme_engine() -> ProductionPhonemeEngine:
    """Get global phoneme engine instance (singleton pattern)"""
    global _phoneme_engine
    if _phoneme_engine is None:
        _phoneme_engine = ProductionPhonemeEngine()
    return _phoneme_engine 