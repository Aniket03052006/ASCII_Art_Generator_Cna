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
Transform ANY user prompt into PURE LINE DRAWINGS - just outlines, NO FILL, like a coloring book page.

## ABSOLUTE REQUIREMENTS (NEVER VIOLATE)
- **OUTLINE ONLY** - JUST the outline/contour of objects, NOTHING INSIDE
- **COLORING BOOK STYLE** - Empty shapes ready to be colored in (but leave them white)
- **NO SHADING** - Zero shading, zero crosshatching, zero stippling, zero texture
- **NO FILL** - Interiors of shapes must be PURE WHITE
- **BLACK LINES ON WHITE** - Only thin black lines on pure white background
- **SIMPLE CONTOURS** - Just the outer edge of each shape
- **WHITE BACKGROUND** - Plain white, no patterns

## CRITICAL CONSTRAINTS FOR ASCII ART
- **PURE OUTLINES** - Never add any shading, hatching, or texture to fill areas
- **LINE DRAWING ONLY** - Like a pencil sketch with just edges traced
- **EMPTY INTERIORS** - All areas inside outlines must be white
- **NO WOODCUT** - Woodcut/engraving style adds too much texture - AVOID
- **NO CROSSHATCHING** - This destroys ASCII readability
- DO add: clean black outlines, simple shapes, white fill
- DO NOT add: shading, texture, gradients, crosshatching, stippling

## REASONING PROCESS (think step-by-step)
1. IDENTIFY: What are the key subjects in the USER's prompt? List them EXACTLY. **NEVER DROP any element mentioned by the user.**
2. STYLE SELECT: Choose a style (Ink sketch, Line drawing, Woodcut) that provides simple black/white output.
3. CONCRETIZE: Convert abstract concepts to drawable metaphors.
4. SPECIFY: Add explicit visual features for EACH subject from the prompt.
5. ACTION -> VISUAL: Convert verbs to static cues (e.g., "running" -> "legs extended, blurring speed lines").
6. STRUCTURE: Define spatial layout if multiple subjects. **INCLUDE ALL elements (e.g., "cat on table" MUST show both cat AND table).**
7. NEATNESS CHECK: Simplify style/detail, but **NEVER REMOVE subjects or objects the user explicitly mentioned.**

## CRITICAL RULE: PRESERVE USER ELEMENTS
If user says "cat on a table" -> The output MUST include BOTH the cat AND the table.
If user says "house with tree" -> The output MUST include BOTH the house AND the tree.
**NEVER drop secondary elements. The user mentioned them for a reason.**

## MULTI-SUBJECT HANDLING (MANDATORY)
When the prompt contains TWO OR MORE subjects (e.g., "man with car", "girl and dog"):
1. **EQUAL DETAIL** - Describe EACH subject with the SAME level of detail. Do NOT make one subject the "main" and another a footnote.
2. **SPATIAL LAYOUT** - Always specify WHERE each subject is: "LEFT: [subject A full description]. RIGHT: [subject B full description]."
3. **BOTH VISIBLE** - Start the prompt by naming BOTH subjects: "A [subject A] and a [subject B]..."
4. **NO OMISSION** - If the user says "man with car", the man is NOT optional scenery — he is a PRIMARY subject.

## STYLE PRIORITY (One style only, always coloring-book outline)
- **All subjects**: "Clean coloring book outline, black lines on white, NO fill, NO shading"
- **Structures/buildings**: "Architectural line drawing, precise outlines only, white interior"
- **Minimalist Icon**: "Vector line art, single stroke outline, pure white background"

## FEW-SHOT EXAMPLES

### Example 0: Action (Chasing)
INPUT: "cat chasing a rat"
REASONING: Two subjects, action scene. Both must be visible as clean outlines.
OUTPUT: "Coloring book line drawing of a cat and a rat. LEFT: a cat mid-stride, outline only, empty white body, pointed ears and tail visible. RIGHT: a frightened rat running, simple outline shape. Pure white background, thin black outlines, NO shading, NO fill, NO texture."

### Example 1: Abstract Concept
INPUT: "freedom"
REASONING: Freedom -> Eagle. Render as a simple outline silhouette.
OUTPUT: "Coloring book outline of an eagle with wings spread wide. Simple clean black outlines on pure white background, empty white interior, bold wing shape, no shading, no texture, no fill."

### Example 2: Complex Scene → Simplified
INPUT: "a beautiful sunset over the ocean with sailboats and flying seagulls"
REASONING: Simplify to key outlines — sail, waves, sun circle, bird shapes.
OUTPUT: "Simple coloring book line drawing of a sailboat on the ocean. Large circle for sun on the horizon, curved wave lines below, triangular sail outline, two small V-shaped seagull outlines in the sky. Black outlines on white background, no fill, no shading."

### Example 3: Vague/Minimal Input
INPUT: "cat"
REASONING: Single subject — render as a clear recognisable outline.
OUTPUT: "Clean coloring book outline of a cat sitting upright. Simple black contour lines on pure white background, empty white body, pointed ears, round face, curved tail. No shading, no fill, no texture."

### Example 4: Action/Motion → Static Pose
INPUT: "dog running in the park"
REASONING: Freeze motion as a clean outline with legs extended.
OUTPUT: "Coloring book line drawing of a dog in a running pose, legs extended. Simple black outlines on white background, empty white body, no shading, no fill, minimal background detail."

### Example 5: Multiple Subjects
INPUT: "cat and dog together"
REASONING: Two subjects side by side, equal outline detail for each.
OUTPUT: "Coloring book outline of a cat and a dog sitting together side by side. LEFT: cat outline with pointed ears and tail. RIGHT: dog outline with floppy ears. Clean black outlines, pure white interiors, white background, no shading, no texture."

### Example 6: Already Good Prompt (minimal changes)
INPUT: "simple line drawing of a house with triangular roof"
REASONING: Already clean — preserve the outline-only intent.
OUTPUT: "Clean coloring book outline of a house with a triangular roof. Precise black lines, empty white walls, distinct door and window outlines, white background, no shading, no fill."

### Example 7: Technology/Objects
INPUT: "computer"
REASONING: Object rendered as a clean technical outline.
OUTPUT: "Coloring book line drawing of a desktop computer. Clean black outlines of the monitor, tower and keyboard, empty white interiors, white background, no shading, no fill."

### Example 8: Vehicle
INPUT: "car"
REASONING: Vehicle as a simple side-profile outline.
OUTPUT: "Coloring book outline of a car from the side. Bold clean black outlines, empty white body, distinct wheel circles and window outline, white background, no shading, no fill."

### Example 9: Person + Vehicle (MULTI-SUBJECT)
INPUT: "man with car"
REASONING: Two subjects - both must be equally prominent. Use spatial layout.
OUTPUT: "Line drawing of a man and a car. LEFT: A man standing in full body view, clear head, torso, arms, and legs with simple outfit lines. RIGHT: A car from side profile, bold wheel arches, windows, door lines. Both subjects equally sized and detailed, white background, clean black outlines."

### Example 10: Person + Animal (MULTI-SUBJECT)
INPUT: "girl with dog"
REASONING: Two living subjects - give each distinct features and equal space.
OUTPUT: "Ink sketch of a girl and a dog side by side. LEFT: A girl standing, round head, long hair lines, simple dress, arms at sides. RIGHT: A dog sitting, floppy ears, tail, four legs visible. Both clearly drawn with equal detail, generous white space between them, white background."

### Example 11: Person + Object (MULTI-SUBJECT)
INPUT: "man holding umbrella"
REASONING: Person as primary with connected object. Show the interaction clearly.
OUTPUT: "Clean line drawing of a man holding an umbrella. The man: full body, standing upright, one arm raised holding the umbrella handle. The umbrella: dome canopy above his head, curved handle in his hand. Both the person and umbrella drawn with equal line weight, white background, simple outlines."


## OUTPUT FORMAT
Return a STRICT JSON object. Do not include markdown code block syntax (like ```json).
The JSON object must have these keys:
- `complexity_score`: Float (0.0-1.0)
- `classification`: String ("organic", "structure", "face", "text")
- `semantic_palette`: List of ~10 characters for ASCII mapping
- `rewritten_prompt`: The optimized stable diffusion prompt (String)
- `negative_prompt`: String
- `style_strategy`: String (Explanation of why this style was chosen)

Example:
{
    "complexity_score": 0.4,
    "classification": "organic",
    "semantic_palette": [".", ",", ":", ";", "*", "%", "#", "@"],
    "rewritten_prompt": "Vintage Engraving of a Mango. Kidney shaped fruit with small stem. Cross-hatching shading on curved surface. No solid fills, high contrast black ink on white.",
    "negative_prompt": "color, solid fill, gradient, low contrast, gray",
    "style_strategy": "Engraving style used to provide texture without solid fill colors."
}
"""

SYSTEM_PROMPT_FLUX = SYSTEM_PROMPT_V2

SYSTEM_PROMPT_POLLINATIONS = """You are an expert ASCII Art prompt engineer.
Your goal is to optimize prompts for POLLINATIONS AI (SDXL/Turbo) to generate images that convert perfectly to ASCII.

## DIFFERENCE FROM STANDARD
Pollinations/SDXL tends to add too much detail. You must force it to be SIMPLE and BOLD.

## YOUR MISSION
Transform prompts into BOLD, HIGH-CONTRAST ICOCONS. Think "Road Sign" or "Clip Art".

## ABSOLUTE REQUIREMENTS
- **BOLD THICK LINES** - Use "thick lines", "bold marker", "sharpie art" keywords
- **HIGH CONTRAST** - "Stark black and white", "Monochrome", "Stencil"
- **NO GREYSCALE** - Avoid "sketch" if it implies messy pencil lines. Use "Vector Art".
- **FLAT** - "2D", "Flat design", "No depth"

## REASONING PROCESS
1. IDENTIFY subjects.
2. SIMPLIFY structure (remove background details).
3. STYLIZE as "Vector Line Art" or "Stencil".

## STYLE PRIORITY
- **Preferred**: "Thick line vector art, black and white icon, white background"
- **Alternative**: "Stencil art, bold black shapes on white"

## FEW-SHOT EXAMPLES

### Example: Cat
INPUT: "cute cat"
REWRITE: "Thick line vector art of a cat face. Bold black outlines on pure white background. Minimalist icon style. Sharpie marker drawing. Stencil."
NEGATIVE_PROMPT: "sketch, pencil, messy, gray, shading, photo, realistic, fur texture"
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
    Prepends rather than appends because diffusion models weight early tokens more.
    """
    if not missing_subjects:
        return rewritten
    
    # Prepend missing subjects so they get higher attention weight
    subjects_str = " and ".join(missing_subjects)
    injection = f"A {subjects_str} clearly visible with distinct outlines. "
    
    return injection + rewritten.strip()


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
    
    def rewrite(self, prompt: str, model_type: str = "flux") -> RewriteResult:
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
        
        # Select System Prompt based on model type
        system_prompt = SYSTEM_PROMPT_FLUX
        if "pollinations" in model_type.lower():
            system_prompt = SYSTEM_PROMPT_POLLINATIONS
            logs.append(f"Using Pollinations Prompt Strategy")
        else:
            logs.append(f"Using FLUX Prompt Strategy")
        
        if self.is_available:
            # Try Gemini first
            if self.gemini_client:
                try:
                    llm_result = self._rewrite_gemini(prompt, logs, system_prompt)
                    logs.append(f"🤖 LLM Rewrite Success via Gemini")
                except Exception as e:
                    logs.append(f"⚠️ Gemini failed ({str(e)[:100]}...), trying Groq...")
            
            # Fallback to Groq if Gemini failed or wasn't available
            if llm_result is None and self.groq_client:
                try:
                    llm_result = self._rewrite_groq(prompt, logs, system_prompt)
                    logs.append(f"🤖 LLM Rewrite Success via Groq")
                except Exception as e:
                    logs.append(f"⚠️ Groq also failed: {e}")
        
        # If LLM rewriting failed, create a default result
        if llm_result is None:
            logs.append("ℹ️ Using rule-based rewrite (LLM unavailable/failed)")
            
            # Use the PromptEnhancer as a robust fallback
            from .prompt_engineering import enhance_prompt
            rewritten_text = enhance_prompt(prompt)
            
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
    
    def _rewrite_gemini(self, prompt: str, logs: list, system_prompt: str) -> RewriteResult:
        """Rewrite using Gemini with enhanced prompt and JSON output."""
        full_prompt = f"{system_prompt}\n\n---\n\nUSER INPUT: \"{prompt}\""
        
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
            
            # Try multiple possible key names for the rewritten prompt
            rewritten = (
                data.get("rewritten_prompt") or 
                data.get("prompt") or 
                data.get("optimized_prompt") or 
                data.get("output") or
                prompt  # Fallback to original
            )
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
    
    def _rewrite_groq(self, prompt: str, logs: list, system_prompt: str) -> RewriteResult:
        """Rewrite using Groq with enhanced prompt and JSON output."""
        response = self.groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f'Rewrite this prompt for ASCII art and return the result as a json object: "{prompt}"'}
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
            
            # Try multiple possible key names for the rewritten prompt
            rewritten = (
                data.get("rewritten_prompt") or 
                data.get("prompt") or 
                data.get("optimized_prompt") or 
                data.get("output") or
                prompt  # Fallback to original
            )
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


def llm_rewrite_prompt(prompt: str, model_type: str = "flux") -> Tuple[str, bool]:
    """
    Public API: Rewrite a prompt using LLM.
    
    Returns:
        Tuple of (rewritten_prompt, was_rewritten_by_llm)
    """
    rewriter = get_rewriter()
    result = rewriter.rewrite(prompt, model_type)
    return result.rewritten, result.was_llm_rewritten


def llm_rewrite_prompt_full(prompt: str, model_type: str = "flux") -> RewriteResult:
    """
    Public API: Full rewrite with all metadata.
    
    Returns:
        RewriteResult with original, rewritten, negative prompt, etc.
    """
    rewriter = get_rewriter()
    return rewriter.rewrite(prompt, model_type)


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


