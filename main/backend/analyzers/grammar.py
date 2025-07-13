import httpx
import json
import re
from typing import List, Dict, Any
from models.responses import GrammarCorrection, GrammarDifference

class GrammarCorrector:
    """
    Advanced grammar correction with IELTS-style analysis including differences tracking,
    lexical analysis, and model answers for different band levels
    """
    
    def __init__(self, openai_api_key: str, openai_url: str):
        self.api_key = openai_api_key
        self.api_url = openai_url
        self.model = "gpt-3.5-turbo"  # Much faster than gpt-4-turbo-preview
    
    async def correct_grammar(self, text: str, question: str = None) -> GrammarCorrection:
        """
        Correct grammar and provide comprehensive IELTS-style analysis
        """
        if not self.api_key:
            return self._fallback_grammar_correction(text)
        
        try:
            prompt = self._create_advanced_grammar_prompt(text, question)
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.api_url,
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model": self.model,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert IELTS examiner with deep knowledge of assessment criteria. Your task is to provide comprehensive analysis of spoken responses with a focus on both grammar and lexical resource (vocabulary, word choice, collocation, idiomatic expression, etc). Your feedback should be constructive and specific. For model answers, use realistic vocabulary and grammar appropriate for each band level, showing clear progression in complexity, accuracy, and richness of language. Include markup in the model answers to highlight specific language features that contribute to that band score, with brief explanations."
                            },
                            {
                                "role": "user", 
                                "content": prompt
                            }
                        ],
                        "max_tokens": 1500,
                        "temperature": 0.3
                    },
                    timeout=25.0
                )
                
                if response.status_code == 200:
                    ai_response = response.json()
                    content = ai_response["choices"][0]["message"]["content"]
                    print(f"🔍 Grammar AI response received: {len(content)} characters")
                    print(f"🔍 First 200 chars: {content[:200]}...")
                    result = self._parse_advanced_grammar_response(content, text)
                    print(f"🔍 Grammar result - corrected text: '{result.corrected_text[:100]}...'")
                    print(f"🔍 Grammar result - differences count: {len(result.differences)}")
                    return result
                else:
                    print(f"⚠️ OpenAI API error: {response.status_code}")
                    print(f"⚠️ Response content: {response.text[:200]}...")
                    return self._fallback_grammar_correction(text)
                    
        except httpx.TimeoutException:
            print(f"⚠️ OpenAI API timeout after 25 seconds")
            return self._fallback_grammar_correction(text)
        except httpx.RequestError as e:
            print(f"⚠️ Network error in grammar correction: {e}")
            return self._fallback_grammar_correction(text)
        except json.JSONDecodeError as e:
            print(f"⚠️ Invalid JSON response from OpenAI: {e}")
            return self._fallback_grammar_correction(text)
        except Exception as e:
            print(f"⚠️ Error in advanced grammar correction: {e}")
            return self._fallback_grammar_correction(text)
    
    def _create_advanced_grammar_prompt(self, text: str, question: str = None) -> str:
        """Create comprehensive prompt for grammar and lexical analysis focused on spoken language"""
        question_context = f'\nQuestion: "{question}"' if question else ""
        
        return f"""
CRITICAL: This is transcribed speech with filler words. Your job is to:
1. REMOVE all filler words (um, uh, uhh, err, ah, eh, hmm, etc.)
2. Fix only clear grammatical errors
3. PRESERVE the speaker's vocabulary level, style, and word choices completely

REMOVE THESE FILLER WORDS:
- um, umm, ummm
- uh, uhh, uhhh  
- err, errr
- ah, ahh (when used as hesitation)
- eh, ehh
- hmm, hm

PRESERVE COMPLETELY:
- All vocabulary choices (keep "cherished avocation" not "beloved hobby")
- Sentence structure and complexity
- Speaking style and tone
- Advanced or sophisticated language
- Natural speech patterns

FIX ONLY these specific grammar errors:
- Subject-verb disagreement ("I likes" → "I like")
- Wrong verb tenses ("I goed" → "I went") 
- Missing articles ("I play guitar" → "I play the guitar" - only if clearly needed)
- Wrong prepositions ("different than" → "different from" - only if clearly wrong)
- Clear word form errors ("beautifuly" → "beautifully")

DO NOT change:
- Advanced vocabulary to simpler words
- Complex sentences to simple ones
- Formal language to informal
- Technical terms or sophisticated expressions
- Natural speech variations

{question_context}
Original Speech: {text}

EXAMPLE:
Input: "I, um, really enjoy, uh, playing the guitar and, err, painting."
Output correctedText: "I really enjoy playing the guitar and painting."

{{
  "correctedText": "clean text with filler words removed and grammar errors fixed",
  "lexicalAnalysis": "brief assessment of vocabulary sophistication",
  "strengths": ["specific good language features"],
  "improvements": ["only actual grammar issues if any exist"],
  "lexicalBandScore": 7.5,
  "grammarScore": 85,
  "modelAnswers": {{
    "band4": "how a band 4 speaker might express similar ideas",
    "band5": "how a band 5 speaker might express similar ideas", 
    "band6": "how a band 6 speaker might express similar ideas",
    "band7": "how a band 7 speaker might express similar ideas",
    "band8": "how a band 8 speaker might express similar ideas",
    "band9": "how a band 9 speaker might express similar ideas"
  }}
}}
"""
    
    def _parse_advanced_grammar_response(self, content: str, original_text: str) -> GrammarCorrection:
        """Parse comprehensive AI grammar response"""
        try:
            print(f"🔍 Parsing grammar response...")
            content = content.strip()
            if not content.startswith('{'):
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    content = content[start:end]
                    print(f"🔍 Extracted JSON from response: {content[:100]}...")
            
            data = json.loads(content)
            print(f"🔍 JSON parsed successfully. Keys: {list(data.keys())}")
            corrected_text = data.get("correctedText", original_text)
            print(f"🔍 Corrected text from AI: '{corrected_text[:100]}...'")
            
            # Generate differences between original and corrected text
            differences = self._find_differences(original_text, corrected_text)
            print(f"🔍 Found {len(differences)} differences initially")
            
            # If the texts are very similar (minimal changes), be more conservative about differences
            if len(differences) > len(original_text.split()) * 0.3:  # If more than 30% of words flagged
                print(f"⚠️ Too many differences detected ({len(differences)} vs {len(original_text.split())} words) - may be a rewrite instead of correction")
                # Fallback to simpler difference detection for minimal changes
                differences = self._find_minimal_differences(original_text, corrected_text)
                print(f"🔍 After minimal detection: {len(differences)} differences")
            
            # Create tagged text with grammar mistake markup
            tagged_text = self._create_tagged_text(original_text, differences)
            
            # Parse strengths and improvements - handle both string and list formats
            strengths_raw = data.get("strengths", [])
            if isinstance(strengths_raw, str):
                # Split by bullet points, line breaks, or numbered lists
                strengths = [s.strip() for s in re.split(r'[•\-\*\n]|(?:\d+\.)', strengths_raw) if s.strip()]
            else:
                strengths = strengths_raw if isinstance(strengths_raw, list) else []
            
            improvements_raw = data.get("improvements", [])
            if isinstance(improvements_raw, str):
                # Split by bullet points, line breaks, or numbered lists
                improvements = [s.strip() for s in re.split(r'[•\-\*\n]|(?:\d+\.)', improvements_raw) if s.strip()]
            else:
                improvements = improvements_raw if isinstance(improvements_raw, list) else []
            
            # Parse model answers - handle both string and object formats
            model_answers_raw = data.get("modelAnswers", {})
            model_answers = {}
            
            for band_level, answer_data in model_answers_raw.items():
                if isinstance(answer_data, dict) and 'text' in answer_data:
                    # Extract text from dictionary format
                    model_answers[band_level] = answer_data['text']
                elif isinstance(answer_data, str):
                    # Already in string format
                    model_answers[band_level] = answer_data
                else:
                    # Convert to string as fallback
                    model_answers[band_level] = str(answer_data)
            
            result = GrammarCorrection(
                original_text=original_text,
                corrected_text=corrected_text,
                differences=differences,
                tagged_text=tagged_text,
                lexical_analysis=data.get("lexicalAnalysis", ""),
                strengths=strengths,
                improvements=improvements,
                lexical_band_score=float(data.get("lexicalBandScore", 7.5)),
                model_answers=model_answers,
                grammar_score=float(data.get("grammarScore", 85))
            )
            print(f"✅ Grammar correction created successfully!")
            print(f"✅ Final corrected text: '{result.corrected_text[:100]}...'")
            print(f"✅ Final differences: {len(result.differences)}")
            print(f"✅ Final grammar score: {result.grammar_score}")
            return result
            
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error in grammar response: {e}")
            return self._fallback_grammar_correction(original_text)
        except Exception as e:
            print(f"⚠️ Error parsing grammar response: {e}")
            return self._fallback_grammar_correction(original_text)
    
    def _find_differences(self, original: str, corrected: str) -> List[GrammarDifference]:
        """Find differences between original and corrected text"""
        original_words = original.split()
        corrected_words = corrected.split()
        
        differences = []
        i = 0
        j = 0
        
        while i < len(original_words) or j < len(corrected_words):
            if i >= len(original_words):
                # Addition at the end
                differences.append(GrammarDifference(
                    type='addition',
                    original=None,
                    corrected=corrected_words[j],
                    position=i
                ))
                j += 1
            elif j >= len(corrected_words):
                # Deletion at the end
                differences.append(GrammarDifference(
                    type='deletion',
                    original=original_words[i],
                    corrected=None,
                    position=i
                ))
                i += 1
            elif original_words[i].lower() == corrected_words[j].lower():
                # Words match (ignoring case)
                i += 1
                j += 1
            else:
                # Check for substitution or lookahead
                found_match = False
                look_ahead_limit = 3
                
                # Look ahead in original for matches
                for look in range(1, min(look_ahead_limit + 1, len(original_words) - i)):
                    if original_words[i + look].lower() == corrected_words[j].lower():
                        # Found match ahead in original, words were deleted
                        for k in range(look):
                            differences.append(GrammarDifference(
                                type='deletion',
                                original=original_words[i + k],
                                corrected=None,
                                position=i + k
                            ))
                        i += look
                        found_match = True
                        break
                
                if not found_match:
                    # Look ahead in corrected for matches
                    for look in range(1, min(look_ahead_limit + 1, len(corrected_words) - j)):
                        if original_words[i].lower() == corrected_words[j + look].lower():
                            # Found match ahead in corrected, words were added
                            for k in range(look):
                                differences.append(GrammarDifference(
                                    type='addition',
                                    original=None,
                                    corrected=corrected_words[j + k],
                                    position=i
                                ))
                            j += look
                            found_match = True
                            break
                
                if not found_match:
                    # It's a substitution
                    differences.append(GrammarDifference(
                        type='substitution',
                        original=original_words[i],
                        corrected=corrected_words[j],
                        position=i
                    ))
                    i += 1
                    j += 1
        
        return differences
    
    def _find_minimal_differences(self, original: str, corrected: str) -> List[GrammarDifference]:
        """Find only clear, minimal differences - used when texts are very similar"""
        original_words = original.split()
        corrected_words = corrected.split()
        
        differences = []
        
        # Only mark differences if the words are clearly different
        min_length = min(len(original_words), len(corrected_words))
        
        for i in range(min_length):
            orig_word = original_words[i].strip('.,!?:;')
            corr_word = corrected_words[i].strip('.,!?:;')
            
            # Only mark as different if they're clearly different words (not just punctuation)
            if orig_word.lower() != corr_word.lower():
                differences.append(GrammarDifference(
                    type='substitution',
                    original=original_words[i],
                    corrected=corrected_words[i],
                    position=i
                ))
        
        # Handle length differences (additions/deletions) only for clear cases
        if len(corrected_words) > len(original_words):
            for i in range(len(original_words), len(corrected_words)):
                differences.append(GrammarDifference(
                    type='addition',
                    original=None,
                    corrected=corrected_words[i],
                    position=i
                ))
        elif len(original_words) > len(corrected_words):
            for i in range(len(corrected_words), len(original_words)):
                differences.append(GrammarDifference(
                    type='deletion',
                    original=original_words[i],
                    corrected=None,
                    position=i
                ))
        
        return differences
    
    def _create_tagged_text(self, original: str, differences: List[GrammarDifference]) -> str:
        """Create tagged text with grammar mistake markup"""
        if not differences:
            return original
        
        # Sort differences by position in descending order to avoid position shifts
        sorted_differences = sorted(differences, key=lambda x: x.position, reverse=True)
        
        # Split text into words
        words = original.split()
        
        # Process each difference
        for diff in sorted_differences:
            if diff.position < len(words):
                if diff.type == 'substitution':
                    words[diff.position] = f'<grammar-mistake correction="{diff.corrected}">{words[diff.position]}</grammar-mistake>'
                elif diff.type == 'deletion':
                    words[diff.position] = f'<grammar-mistake correction="">{words[diff.position]}</grammar-mistake>'
                # Additions don't exist in the original text, so we skip them
        
        return ' '.join(words)
    
    def _fallback_grammar_correction(self, text: str) -> GrammarCorrection:
        """Basic grammar analysis when AI is unavailable"""
        corrected_text = text
        differences = []
        
        # Basic corrections
        basic_fixes = [
            (r'\bi\b', 'I'),  # Capitalize I
            (r'\s+', ' '),    # Multiple spaces
        ]
        
        position = 0
        for pattern, replacement in basic_fixes:
            if re.search(pattern, corrected_text):
                old_text = corrected_text
                corrected_text = re.sub(pattern, replacement, corrected_text)
                if old_text != corrected_text:
                    differences.append(GrammarDifference(
                        type='substitution',
                        original=pattern,
                        corrected=replacement,
                        position=position
                    ))
                    position += 1
        
        # Calculate basic grammar score
        score = 85 - (len(differences) * 5)
        score = max(50, min(100, score))
        
        tagged_text = self._create_tagged_text(text, differences)
        
        return GrammarCorrection(
            original_text=text,
            corrected_text=corrected_text.strip(),
            differences=differences,
            tagged_text=tagged_text,
            lexical_analysis="AI analysis unavailable. Basic grammar check performed.",
            strengths=["Response provided"],
            improvements=["Enable AI analysis for detailed feedback"],
            lexical_band_score=5.0,
            model_answers={},
            grammar_score=float(score)
        ) 