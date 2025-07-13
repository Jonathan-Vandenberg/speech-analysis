import httpx
import json
from typing import Tuple
from models.responses import RelevanceAnalysis

class RelevanceAnalyzer:
    """
    Analyzes how relevant a spoken answer is to a given question using AI
    """
    
    def __init__(self, openai_api_key: str, openai_url: str):
        self.api_key = openai_api_key
        self.api_url = openai_url
        self.model = "gpt-3.5-turbo"  # Much faster than gpt-4-turbo-preview
    
    async def analyze_relevance(self, question: str, answer: str) -> RelevanceAnalysis:
        """
        Analyze how relevant the answer is to the question
        """
        if not self.api_key:
            return self._fallback_relevance_analysis(question, answer)
        
        try:
            prompt = self._create_relevance_prompt(question, answer)
            
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
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        "max_tokens": 300,
                        "temperature": 0.3
                    },
                    timeout=15.0
                )
                
                if response.status_code == 200:
                    ai_response = response.json()
                    content = ai_response["choices"][0]["message"]["content"]
                    return self._parse_ai_response(content)
                else:
                    print(f"⚠️ OpenAI API error: {response.status_code}")
                    print(f"⚠️ Response content: {response.text[:200]}...")
                    return self._fallback_relevance_analysis(question, answer)
                    
        except httpx.TimeoutException:
            print(f"⚠️ OpenAI API timeout after 15 seconds")
            return self._fallback_relevance_analysis(question, answer)
        except httpx.RequestError as e:
            print(f"⚠️ Network error in relevance analysis: {e}")
            return self._fallback_relevance_analysis(question, answer)
        except json.JSONDecodeError as e:
            print(f"⚠️ Invalid JSON response from OpenAI: {e}")
            return self._fallback_relevance_analysis(question, answer)
        except Exception as e:
            print(f"⚠️ Error in relevance analysis: {e}")
            return self._fallback_relevance_analysis(question, answer)
    
    def _create_relevance_prompt(self, question: str, answer: str) -> str:
        """Create a structured prompt for relevance analysis"""
        return f"""
Rate answer relevance to question (0-100).

Question: "{question}"
Answer: "{answer}"

Scoring: 90-100=comprehensive, 80-89=good, 70-79=adequate, 60-69=partial, 50-59=limited, 0-49=off-topic

{{
    "relevance_score": [0-100],
    "explanation": "brief reason",
    "key_points_covered": ["point1", "point2"],
    "missing_points": ["missing1", "missing2"]
}}
"""
    
    def _parse_ai_response(self, content: str) -> RelevanceAnalysis:
        """Parse the AI response into a RelevanceAnalysis object"""
        try:
            # Try to extract JSON from the response
            content = content.strip()
            if not content.startswith('{'):
                # Find JSON block if it's wrapped in other text
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    content = content[start:end]
            
            data = json.loads(content)
            
            return RelevanceAnalysis(
                relevance_score=float(data.get("relevance_score", 50)),
                explanation=data.get("explanation", "AI analysis completed"),
                key_points_covered=data.get("key_points_covered", []),
                missing_points=data.get("missing_points", [])
            )
            
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse AI response JSON: {e}")
            return self._fallback_relevance_analysis("", "")
    
    def _fallback_relevance_analysis(self, question: str, answer: str) -> RelevanceAnalysis:
        """Fallback relevance analysis when AI is not available"""
        # Basic keyword matching fallback
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        question_words = question_words - stop_words
        answer_words = answer_words - stop_words
        
        if not question_words:
            overlap_score = 50.0
        else:
            # Calculate word overlap
            overlap = len(question_words.intersection(answer_words))
            overlap_score = min(100.0, (overlap / len(question_words)) * 100)
        
        # Adjust based on answer length (very short answers are penalized)
        length_factor = min(1.0, len(answer.split()) / 10.0)
        final_score = overlap_score * length_factor
        
        return RelevanceAnalysis(
            relevance_score=round(final_score, 1),
            explanation="Basic keyword matching analysis (AI unavailable)",
            key_points_covered=list(question_words.intersection(answer_words)),
            missing_points=list(question_words - answer_words)
        ) 