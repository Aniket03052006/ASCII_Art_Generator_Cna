"""
LLM-Powered Prompt Rewriter v2
Uses Gemini/Groq to dynamically rewrite user prompts for optimal ASCII art generation.

Key Features:
- Few-shot examples for consistent output
- Chain-of-thought reasoning
- Complexity detection and simplification
- Negative prompt generation
- Fallback chain: Gemini → Groq → Rule-based
"""

import os
import re
import json
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass

# Try to import Google's GenAI SDK
try:
    import google.generativeai as genai
    from google.generativeai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Try to import Groq as fallback
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False


@dataclass
class RewriteResult:
    """Result of prompt rewriting."""
    original: str
    rewritten: str
    negative_prompt: str
    was_llm_rewritten: bool
    complexity_score: float
    simplification_applied: bool
    classification: str = "organic" # organic | structure | text
    semantic_palette: list = None
    logs: list = None


# =============================================================================
# Enhanced System Prompt with Few-Shot Examples and Chain-of-Thought
# =============================================================================
SYSTEM_PROMPT_V2 = """You are an expert ASCII Art prompt engineer. 
Your goal is to optimize prompts for a FLUX.1 diffusion model to generate images that convert perfectly to ASCII.

## YOUR MISSION
Transform ANY user prompt into one that produces CLEAR, HIGH-CONTRAST images perfect for ASCII conversion.

## CRITICAL CONSTRAINTS FOR ASCII ART
ASCII art requires images with:
- BOLD, CONTINUOUS BLACK OUTLINES (no broken lines)
- CLEAR BOUNDARIES: Separate subject from background completely
- SIMPLE GEOMETRIC SHAPES (circles, lines, triangles) - *EXCEPT FOR PORTRAITS*
- PORTRAITS/FACES: Keep facial features (eyes, nose, mouth) distinct; DO NOT SIMPLIFY FACES into icons.
- HIGH CONTRAST (absolutely no subtle gradients or textures)
- NEATNESS: Prioritize clean, solid lines over detail
- RECOGNIZABILITY: Prominent identifying features must be exagerated
- 1-3 MAIN SUBJECTS maximum (if more, select the most important, but try to keep the core interaction)
- CLEAR SPATIAL SEPARATION between elements (do not overlap subjects)

## REASONING PROCESS (think step-by-step)
1. IDENTIFY: What are the key subjects in the USER's prompt? List them exactly.
2. SIMPLIFY: Remove textures, colors, 3D effects, background clutter.
3. CONCRETIZE: Convert abstract concepts to drawable icons
4. SPECIFY: Add explicit visual features for EACH subject from the prompt
5. ACTION -> VISUAL: Convert verbs to static cues (e.g., "running" -> "legs extended, leaning forward")
6. STRUCTURE: Define spatial layout if multiple subjects

## FEW-SHOT EXAMPLES

### Example 0: Action (Chasing)
INPUT: "cat chasing a rat"
REASONING: "Chasing" implies motion and direction. Need two distinct subjects.
OUTPUT: "LEFT: A large cat silhouette with legs EXTENDED in running stride, leaning forward. RIGHT: A small rat silhouette running away. Cat is behind Rat. Bold horizontal speed lines indicating fast motion. Clear separation, side-view profile."

### Example 1: Abstract Concept
INPUT: "freedom"
REASONING: Freedom is abstract → iconic representation is a bird with spread wings
OUTPUT: "A majestic eagle with wings spread WIDE horizontally, soaring bird silhouette viewed from front, simple bold black outline on pure white background, no texture, flat 2D icon style"

### Example 2: Complex Scene → Simplified
INPUT: "a beautiful sunset over the ocean with sailboats and flying seagulls"
REASONING: Too many elements → pick 1-2 focal points, remove gradients
OUTPUT: "Simple silhouette of a single sailboat on calm water, large setting sun circle behind it, bold black outlines only, high contrast, minimalist line art"

### Example 3: Vague/Minimal Input
INPUT: "cat"
REASONING: Need specific anatomical features for recognizable ASCII
OUTPUT: "A cute cat sitting upright, round head with two pointed triangle ears on top, oval body, curved tail wrapping around, simple black line art on white, thick bold outlines, cartoon cat icon"

### Example 4: Action/Motion → Static Pose
INPUT: "dog running in the park"
REASONING: Motion is hard in ASCII → freeze to dynamic pose
OUTPUT: "A dog in mid-stride pose with legs extended, simple side-view silhouette, bold black outline, no background details, athletic frozen motion, clean vector style"

### Example 5: Multiple Subjects
INPUT: "cat and dog together"
REASONING: Need clear separation, distinct features for each
OUTPUT: "LEFT: a cat silhouette with pointed triangle ears and curved tail. RIGHT: a dog silhouette with floppy ears and wagging tail. Both sitting, generous white space between them, simple black outlines only"

### Example 6: Already Good Prompt (minimal changes)
INPUT: "simple line drawing of a house with triangular roof"
REASONING: Already well-structured, just reinforce ASCII-friendly style
OUTPUT: "Simple house icon with triangular roof pointing up, rectangular body, one door in center, two square windows, thick bold black lines on white background, flat 2D architectural icon"

### Example 7: Technology/Objects
INPUT: "computer"
REASONING: Computer → desktop monitor with rectangular screen and keyboard
OUTPUT: "A desktop computer monitor icon: large rectangle for screen, smaller rectangle stand at bottom, separate rectangular keyboard below, simple black outlines, flat 2D tech icon style"

### Example 8: Vehicle
INPUT: "car"
REASONING: Car → side view silhouette with wheels and body
OUTPUT: "A car silhouette viewed from side: rectangular body with curved roof, two circular wheels at bottom, rectangular windows, simple bold outline, minimal details"

## OUTPUT FORMAT
Return a JSON object with the following keys:
- `complexity_score`: A float between 0.0 and 1.0 indicating the prompt's complexity.
- `classification`: A string, one of "organic", "structure", "face", or "text".
- `semantic_palette`: A list of 10-15 Unicode characters that visually match the subject's texture.
- `rewritten_prompt`: The optimized prompt for ASCII art generation (under 80 words).
- `negative_prompt`: An optional negative prompt string.
- `missing_subjects`: A list of subjects from the original prompt that were not explicitly included in the rewritten prompt.

Example JSON Output:
```json
{
    "complexity_score": 0.7,
    "classification": "face",
    "semantic_palette": [".", ":", "-", "=", "+", "*", "#", "%", "@", "8"],
    "rewritten_prompt": "High contrast portrait of a wise old man, deep wrinkles, strong directional lighting from side, distinct eyes and beard, stipple shading style, clean black lines",
    "negative_prompt": "photorealistic, 3D render, blurry, soft focus, gradient, shading, texture, noise, grain, colors, colorful, rainbow, multiple colors, complex background, busy, cluttered, low contrast, gray, dim, dark overall, shadows, realistic lighting, ray tracing, subsurface scattering, photograph, photo, camera, lens flare, bokeh, watermark, text, signature, logo",
    "missing_subjects": []
}
```
"""


# Negative prompt to include with image generation
NEGATIVE_PROMPT_TEMPLATE = """photorealistic, 3D render, blurry, soft focus, gradient, 
shading, texture, noise, grain, colors, colorful, rainbow, 
multiple colors, complex background, busy, cluttered, 
low contrast, gray, dim, dark overall, shadows, 
realistic lighting, ray tracing, subsurface scattering,
photograph, photo, camera, lens flare, bokeh,
watermark, text, signature, logo"""


# =============================================================================
# Complexity Detection
# =============================================================================
COMPLEXITY_INDICATORS = {
    # High complexity keywords (add to score)
    "high": [
        r"\b(photorealistic|hyper[-\s]?realistic|ultra[-\s]?detailed)\b",
        r"\b(3[dD]|three[-\s]?dimensional)\b",
        r"\b(gradient|shading|lighting|shadows|reflections)\b",
        r"\b(texture|textured|fur|feathers|scales)\b",
        r"\b(multiple|many|several|group of|crowd)\b",
        r"\b(background|environment|scene|landscape)\b",
        r"\b(dancing|running|flying|moving|animated)\b",
        r"\b(rainbow|colorful|vibrant|neon|holographic)\b",
    ],
    # Low complexity (reduce score)
    "low": [
        r"\b(simple|minimal|minimalist|basic)\b",
        r"\b(line art|lineart|outline|silhouette)\b",
        r"\b(black and white|b&w|monochrome)\b",
        r"\b(icon|logo|symbol|geometric)\b",
        r"\b(flat|2[dD]|two[-\s]?dimensional)\b",
    ]
}


def calculate_complexity(prompt: str) -> float:
    """
    Calculate complexity score for a prompt.
    
    Returns:
        Score from 0.0 (simple) to 1.0 (extremely complex)
    """
    score = 0.5  # Start neutral
    prompt_lower = prompt.lower()
    
    # Add for complexity indicators
    for pattern in COMPLEXITY_INDICATORS["high"]:
        if re.search(pattern, prompt_lower, re.IGNORECASE):
            score += 0.1
    
    # Subtract for simplicity indicators
    for pattern in COMPLEXITY_INDICATORS["low"]:
        if re.search(pattern, prompt_lower, re.IGNORECASE):
            score -= 0.1
    
    # Word count penalty (longer = more complex)
    word_count = len(prompt.split())
    if word_count > 20:
        score += 0.1
    if word_count > 40:
        score += 0.1
    
    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))


def needs_simplification(complexity_score: float) -> bool:
    """Check if prompt needs aggressive simplification."""
    return complexity_score > 0.6


# =============================================================================
# Attend-and-Excite: Subject Extraction and Verification
# =============================================================================
# Inspired by "Attend-and-Excite" (Chefer et al., SIGGRAPH 2023)
# Ensures ALL subjects mentioned in the prompt appear in the rewritten output
# Addresses "catastrophic neglect" where subjects are omitted

# Common nouns that should be preserved as subjects
SUBJECT_PATTERNS = [
    # Animals
    r'\b(cat|dog|bird|fish|horse|cow|pig|sheep|lion|tiger|bear|elephant|monkey|mouse|rat|rabbit|fox|wolf|deer|eagle|owl|snake|frog|butterfly|bee|ant)\b',
    # People
    r'\b(person|man|woman|boy|girl|child|baby|human|people|warrior|soldier|king|queen|prince|princess|knight)\b',
    # Objects
    r'\b(car|truck|bus|bike|bicycle|motorcycle|plane|airplane|boat|ship|train|computer|phone|laptop|keyboard|monitor|table|chair|desk|bed|house|building|tree|flower|sun|moon|star|mountain|river|ocean|sea|lake)\b',
    # Food
    r'\b(apple|banana|orange|pizza|burger|cake|bread|coffee|tea|water|wine|beer)\b',
    # Symbols
    r'\b(heart|star|circle|triangle|square|diamond|cross|arrow|lightning)\b',
]


def extract_subjects(prompt: str) -> list:
    """
    Extract key subjects from a prompt using pattern matching.
    
    Based on Attend-and-Excite principle: identify ALL subjects
    that must appear in the final output.
    
    Returns:
        List of subject words found in the prompt
    """
    subjects = []
    prompt_lower = prompt.lower()
    
    for pattern in SUBJECT_PATTERNS:
        matches = re.findall(pattern, prompt_lower, re.IGNORECASE)
        subjects.extend(matches)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_subjects = []
    for s in subjects:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique_subjects.append(s)
    
    return unique_subjects


def verify_subjects_present(original: str, rewritten: str) -> tuple:
    """
    Verify that all subjects from the original prompt appear in the rewritten version.
    
    Implements the "Attend-and-Excite" verification step.
    
    Returns:
        (all_present: bool, missing_subjects: list)
    """
    original_subjects = extract_subjects(original)
    rewritten_lower = rewritten.lower()
    
    missing = []
    for subject in original_subjects:
        # Check if subject or close variant appears
        if subject.lower() not in rewritten_lower:
            # Check for plurals/variants
            variants = [subject, subject + 's', subject + 'es', subject[:-1] if subject.endswith('s') else subject]
            if not any(v.lower() in rewritten_lower for v in variants):
                missing.append(subject)
    
    return (len(missing) == 0, missing)


def inject_missing_subjects(rewritten: str, missing_subjects: list) -> str:
    """
    Inject missing subjects into the rewritten prompt.
    
    Ensures semantic faithfulness by explicitly adding neglected subjects.
    """
    if not missing_subjects:
        return rewritten
    
    # Create injection phrase
    subjects_str = ", ".join(missing_subjects)
    injection = f" Also include: {subjects_str} with clear distinct outlines."
    
    return rewritten.strip() + injection


# =============================================================================
# LLM Prompt Rewriter Class
# =============================================================================
class LLMPromptRewriter:
    """Rewrites prompts using LLM for better ASCII art generation."""
    
    def __init__(
        self, 
        gemini_key: Optional[str] = None, 
        groq_key: Optional[str] = None,
        enable_negative_prompt: bool = True,
    ):
        # Hardcoded API keys (fallbacks if not provided via params or env)
        HARDCODED_GEMINI_KEY = "AIzaSyCFldCZ-Inftl3-efdMuop4ne2-ggCYVzY"
        HARDCODED_GROQ_KEY = "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"
        
        self.gemini_key = (
            gemini_key or 
            os.environ.get('GEMINI_API_KEY') or 
            os.environ.get('GOOGLE_API_KEY') or 
            HARDCODED_GEMINI_KEY
        )
        self.groq_key = (
            groq_key or 
            os.environ.get('GROQ_API_KEY') or 
            HARDCODED_GROQ_KEY
        )
        self.enable_negative_prompt = enable_negative_prompt
        
        self.gemini_client = None
        self.groq_client = None
        
        self._setup_clients()
    
    def _setup_clients(self):
        """Initialize available LLM clients."""
        # Try Gemini first (preferred)
        if GEMINI_AVAILABLE and self.gemini_key:
            try:
                # Using genai.configure for API key setup
                genai.configure(api_key=self.gemini_key)
                # Initialize the model directly, no need for genai.Client
                self.gemini_client = genai.GenerativeModel(model_name="gemini-1.5-flash")
                print("✅ Gemini LLM initialized (v2 enhanced)")
            except Exception as e:
                print(f"⚠️ Gemini setup failed: {e}")
                self.gemini_client = None # Ensure client is None if setup fails
        
        # Always try Groq as fallback option
        if GROQ_AVAILABLE and self.groq_key:
            try:
                self.groq_client = Groq(api_key=self.groq_key)
                print("✅ Groq LLM initialized (fallback)")
            except Exception as e:
                print(f"⚠️ Groq setup failed: {e}")
                self.groq_client = None # Ensure client is None if setup fails
    
    def update_keys(self, gemini_key: Optional[str] = None, groq_key: Optional[str] = None):
        """Update API keys at runtime (useful for web UI)."""
        if gemini_key:
            self.gemini_key = gemini_key
        if groq_key:
            self.groq_key = groq_key
        self._setup_clients()
    
    @property
    def is_available(self) -> bool:
        """Check if any LLM is available."""
        return self.gemini_client is not None or self.groq_client is not None
    
    def get_status(self) -> Dict[str, bool]:
        """Get status of LLM backends."""
        return {
            "gemini_available": self.gemini_client is not None,
            "groq_available": self.groq_client is not None,
            "any_available": self.is_available,
        }
    
    def rewrite(self, prompt: str) -> RewriteResult:
        """
        Rewrite the prompt using LLM with complexity analysis.
        
        Returns:
            RewriteResult with all metadata
        """
        logs = []
        logs.append(f"Analyzing prompt: '{prompt}'")

        # Initial complexity calculation (used if LLM fails or for comparison)
        initial_complexity = calculate_complexity(prompt)
        initial_needs_simplify = needs_simplification(initial_complexity)
        logs.append(f"Initial Complexity Score: {initial_complexity:.2f} ({'High' if initial_needs_simplify else 'Normal'})")
        
        llm_result: Optional[RewriteResult] = None
        
        if self.is_available:
            # Try Gemini first
            if self.gemini_client:
                try:
                    llm_result = self._rewrite_gemini(prompt, logs)
                    logs.append(f"🤖 LLM Rewrite Success via Gemini")
                except Exception as e:
                    logs.append(f"⚠️ Gemini failed ({str(e)[:100]}...), trying Groq...")
            
            # Fallback to Groq if Gemini failed or wasn't available
            if llm_result is None and self.groq_client:
                try:
                    llm_result = self._rewrite_groq(prompt, logs)
                    logs.append(f"🤖 LLM Rewrite Success via Groq")
                except Exception as e:
                    logs.append(f"⚠️ Groq also failed: {e}")
        
        # If LLM rewriting failed, create a default result
        if llm_result is None:
            logs.append("ℹ️ Using rule-based rewrite (LLM unavailable/failed)")
            rewritten_text = prompt # Fallback to original prompt
            negative_text = NEGATIVE_PROMPT_TEMPLATE if self.enable_negative_prompt else ""
            llm_result = RewriteResult(
                original=prompt,
                rewritten=rewritten_text,
                negative_prompt=negative_text,
                was_llm_rewritten=False,
                complexity_score=initial_complexity,
                simplification_applied=initial_needs_simplify,
                classification="organic", # Default classification
                semantic_palette=["#", "@", "%", ".", ":", "+", "*", "-", "="], # Default palette
                logs=logs
            )
        
        # === Attend-and-Excite: Subject Verification ===
        # Ensure ALL subjects from original prompt appear in rewritten version
        all_present, missing = verify_subjects_present(prompt, llm_result.rewritten)
        if not all_present and missing:
            logs.append(f"⚠️ Attend-and-Excite: Missing subjects detected: {missing}")
            llm_result.rewritten = inject_missing_subjects(llm_result.rewritten, missing)
            logs.append(f"   ✅ Injected missing subjects into prompt")
            # Update missing_subjects in the result if the LLM provided it
            if "missing_subjects" in llm_result.logs[-1]: # Check if LLM output included this
                llm_result.logs[-1]["missing_subjects"] = missing # This is a bit hacky, better to pass missing_subjects to LLM
        
        # Ensure negative prompt is included if enabled
        if self.enable_negative_prompt and not llm_result.negative_prompt:
            llm_result.negative_prompt = NEGATIVE_PROMPT_TEMPLATE
            logs.append("✅ Injected default negative prompt as LLM did not provide one.")

        # Update the logs in the final result
        llm_result.logs = logs
        
        return llm_result
    
    def _rewrite_gemini(self, prompt: str, logs: list) -> RewriteResult:
        """Rewrite using Gemini with enhanced prompt and JSON output."""
        full_prompt = f"{SYSTEM_PROMPT_V2}\n\n---\n\nUSER INPUT: \"{prompt}\""
        
        response = self.gemini_client.generate_content(
            contents=[{"role": "user", "parts": [{"text": full_prompt}]}],
            generation_config=types.GenerationConfig(
                max_output_tokens=500, # Increased for JSON output
                temperature=0.4,
                response_mime_type="application/json" # Request JSON output
            )
        )
        
        response_text = response.text.strip()
        
        # Parse JSON
        try:
            data = json.loads(response_text)
            
            rewritten = data.get("rewritten_prompt", prompt)
            negative = data.get("negative_prompt", "")
            complexity = data.get("complexity_score", 0.5)
            classification = data.get("classification", "organic")
            palette = data.get("semantic_palette", ["#", "@", "%", ".", ":"])
            
            logs.append(f"Classified as: {classification.upper()}")
            logs.append(f"🎨 Palette: {''.join(palette[:8])}...")
            
            return RewriteResult(
                original=prompt,
                rewritten=rewritten,
                negative_prompt=negative,
                was_llm_rewritten=True,
                complexity_score=complexity,
                simplification_applied=needs_simplification(complexity),
                classification=classification,
                semantic_palette=palette,
                logs=logs
            )
        except json.JSONDecodeError:
            logs.append(f"⚠️ Gemini JSON Parse Failed, falling back to raw text: {response_text[:100]}...")
            # Fallback to a basic RewriteResult if JSON parsing fails
            return RewriteResult(
                original=prompt,
                rewritten=response_text, # Use raw response as rewritten
                negative_prompt="",
                was_llm_rewritten=True,
                complexity_score=calculate_complexity(prompt),
                simplification_applied=needs_simplification(calculate_complexity(prompt)),
                classification="organic",
                logs=logs
            )
    
    def _rewrite_groq(self, prompt: str, logs: list) -> RewriteResult:
        """Rewrite using Groq with enhanced prompt and JSON output."""
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_V2},
                {"role": "user", "content": f'Rewrite this prompt for ASCII art: "{prompt}"'}
            ],
            max_tokens=500, # Increased for JSON output
            temperature=0.4,
            response_format={"type": "json_object"} # Request JSON output
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON
        try:
            # Clean markdown code blocks if present
            clean_response = response_text.replace('```json', '').replace('```', '').strip()
            data = json.loads(clean_response)
            
            rewritten = data.get("rewritten_prompt", prompt)
            negative = data.get("negative_prompt", "")
            complexity = data.get("complexity_score", 0.5)
            classification = data.get("classification", "organic")
            palette = data.get("semantic_palette", ["#", "@", "%", ".", ":"])
            
            logs.append(f"Classified as: {classification.upper()}")
            logs.append(f"🎨 Palette: {''.join(palette[:8])}...")
            
            return RewriteResult(
                original=prompt,
                rewritten=rewritten,
                negative_prompt=negative,
                was_llm_rewritten=True,
                complexity_score=complexity,
                simplification_applied=needs_simplification(complexity),
                classification=classification,
                semantic_palette=palette,
                logs=logs
            )
        except json.JSONDecodeError:
            logs.append(f"⚠️ Groq JSON Parse Failed, falling back to raw text: {response_text[:100]}...")
            # Fallback to a basic RewriteResult if JSON parsing fails
            return RewriteResult(
                original=prompt,
                rewritten=response_text, # Use raw response as rewritten
                negative_prompt="",
                was_llm_rewritten=True,
                complexity_score=calculate_complexity(prompt),
                simplification_applied=needs_simplification(calculate_complexity(prompt)),
                classification="organic",
                logs=logs
            )


# =============================================================================
# Singleton and Public API
# =============================================================================
_rewriter: Optional[LLMPromptRewriter] = None


def get_rewriter() -> LLMPromptRewriter:
    """Get or create the LLM rewriter instance."""
    global _rewriter
    if _rewriter is None:
        _rewriter = LLMPromptRewriter()
    return _rewriter


def set_api_keys(gemini_key: Optional[str] = None, groq_key: Optional[str] = None):
    """Update API keys for the rewriter (useful for web UI)."""
    rewriter = get_rewriter()
    rewriter.update_keys(gemini_key, groq_key)


def llm_rewrite_prompt(prompt: str) -> Tuple[str, bool]:
    """
    Public API: Rewrite a prompt using LLM.
    
    Returns:
        Tuple of (rewritten_prompt, was_rewritten_by_llm)
    """
    rewriter = get_rewriter()
    result = rewriter.rewrite(prompt)
    return result.rewritten, result.was_llm_rewritten


def llm_rewrite_prompt_full(prompt: str) -> RewriteResult:
    """
    Public API: Full rewrite with all metadata.
    
    Returns:
        RewriteResult with original, rewritten, negative prompt, etc.
    """
    rewriter = get_rewriter()
    return rewriter.rewrite(prompt)


def get_negative_prompt() -> str:
    """Get the standard negative prompt for ASCII art generation."""
    return NEGATIVE_PROMPT_TEMPLATE


# =============================================================================
# CLI Test
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("LLM Prompt Rewriter v2 - Test Suite")
    print("=" * 60)
    
    test_prompts = [
        # Abstract concepts
        "freedom",
        "happiness",
        "love",
        
        # Simple subjects
        "cat",
        "house",
        
        # Complex prompts (should trigger simplification)
        "a photorealistic 3D render of a rainbow holographic dancing cat with sparkles",
        "beautiful sunset over the ocean with multiple sailboats and flying seagulls",
        
        # Action prompts
        "dog running in the park",
        "bird flying through clouds",
        
        # Multi-subject
        "cat and dog together",
        "moon orbiting earth",
        
        # Vague
        "thing on stuff",
        "something nice",
    ]
    
    rewriter = LLMPromptRewriter()
    
    print(f"\nStatus: {rewriter.get_status()}")
    print("-" * 60)
    
    for prompt in test_prompts:
        result = rewriter.rewrite(prompt)
        
        print(f"\n📝 Original: {prompt}")
        print(f"   Complexity: {result.complexity_score:.2f} {'⚠️ SIMPLIFY' if result.simplification_applied else ''}")
        print(f"   Classification: {result.classification.upper()}")
        print(f"   LLM Used: {'✅' if result.was_llm_rewritten else '❌'}")
        print(f"✨ Rewritten: {result.rewritten[:80]}...")
        if result.negative_prompt:
            print(f"   Negative: {result.negative_prompt[:80]}...")
        print(f"   Logs: {result.logs}")


