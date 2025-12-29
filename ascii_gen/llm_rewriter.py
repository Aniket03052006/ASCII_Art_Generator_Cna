"""
LLM-Powered Prompt Rewriter
Uses Gemini/Groq to dynamically rewrite user prompts for optimal ASCII art generation.

This is an advanced layer on top of the rule-based prompt_engineering.py.
"""

import os
from typing import Optional, Tuple

# Try to import Google's new GenAI
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Try to import Groq as fallback
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


# System prompt for the LLM
SYSTEM_PROMPT = """You are an expert at rewriting prompts for ASCII art generation.

Your task: Take the user's prompt and rewrite it to produce CLEAR, SIMPLE ASCII art.

RULES:
1. Convert abstract concepts to concrete visuals (e.g., "freedom" → "eagle with wings spread wide")
2. Simplify complex scenes - focus on 1-3 main elements max
3. Add explicit visual features (e.g., "cat" → "cat with pointed triangle ears, round body, curved tail")
4. Specify spatial relationships clearly (e.g., "on top of", "to the left of")
5. Always request: "simple line art, black outline only, white background, bold thick lines"
6. Avoid: textures, gradients, shading, 3D, photorealistic, colors

OUTPUT FORMAT:
Return ONLY the rewritten prompt, nothing else. Keep it under 100 words."""


class LLMPromptRewriter:
    """Rewrites prompts using LLM for better ASCII art generation."""
    
    def __init__(self, gemini_key: Optional[str] = None, groq_key: Optional[str] = None):
        self.gemini_key = gemini_key or os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        self.groq_key = groq_key or os.environ.get('GROQ_API_KEY')
        
        self.gemini_client = None
        self.groq_client = None
        
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize available LLM clients."""
        # Try Gemini first (preferred)
        if GEMINI_AVAILABLE and self.gemini_key:
            try:
                self.gemini_client = genai.Client(api_key=self.gemini_key)
                print("✅ Gemini LLM initialized")
            except Exception as e:
                print(f"⚠️ Gemini setup failed: {e}")
        
        # Always try Groq as fallback option
        if GROQ_AVAILABLE and self.groq_key:
            try:
                self.groq_client = Groq(api_key=self.groq_key)
                print("✅ Groq LLM initialized (fallback)")
            except Exception as e:
                print(f"⚠️ Groq setup failed: {e}")
    
    @property
    def is_available(self) -> bool:
        """Check if any LLM is available."""
        return self.gemini_client is not None or self.groq_client is not None
    
    def rewrite(self, prompt: str) -> Tuple[str, bool]:
        """Rewrite the prompt using LLM. Falls back to Groq if Gemini fails."""
        if not self.is_available:
            return prompt, False
        
        # Try Gemini first
        if self.gemini_client:
            try:
                return self._rewrite_gemini(prompt), True
            except Exception as e:
                print(f"⚠️ Gemini failed ({str(e)[:50]}...), trying Groq...")
        
        # Fallback to Groq
        if self.groq_client:
            try:
                return self._rewrite_groq(prompt), True
            except Exception as e:
                print(f"⚠️ Groq also failed: {e}")
        
        return prompt, False
    
    def _rewrite_gemini(self, prompt: str) -> str:
        """Rewrite using Gemini."""
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser prompt: \"{prompt}\"\n\nRewritten prompt:"
        
        response = self.gemini_client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=150,
                temperature=0.3,
            )
        )
        
        rewritten = response.text.strip().strip('"\'')
        return rewritten
    
    def _rewrite_groq(self, prompt: str) -> str:
        """Rewrite using Groq."""
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Rewrite this prompt: "{prompt}"'}
            ],
            max_tokens=150,
            temperature=0.3,
        )
        
        rewritten = response.choices[0].message.content.strip().strip('"\'')
        return rewritten


# Singleton instance
_rewriter = None

def get_rewriter() -> LLMPromptRewriter:
    """Get or create the LLM rewriter instance."""
    global _rewriter
    if _rewriter is None:
        _rewriter = LLMPromptRewriter()
    return _rewriter


def llm_rewrite_prompt(prompt: str) -> Tuple[str, bool]:
    """
    Public API: Rewrite a prompt using LLM.
    
    Returns:
        Tuple of (rewritten_prompt, was_rewritten_by_llm)
    """
    rewriter = get_rewriter()
    return rewriter.rewrite(prompt)


if __name__ == "__main__":
    # Test the rewriter
    print("Testing LLM Prompt Rewriter")
    print("=" * 50)
    
    test_prompts = [
        "freedom",
        "happiness",
        "cat on a chair",
        "thing on stuff",
        "photorealistic 3D rainbow holographic dancing cat",
    ]
    
    rewriter = LLMPromptRewriter()
    
    if not rewriter.is_available:
        print("❌ No LLM available. Set GEMINI_API_KEY or GROQ_API_KEY")
    else:
        for prompt in test_prompts:
            rewritten, success = rewriter.rewrite(prompt)
            print(f"\n📝 Original: {prompt}")
            print(f"✨ Rewritten: {rewritten}")
