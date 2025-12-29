"""
Industrial-Standard Prompt Engineering for ASCII Art

This module implements production-grade prompt engineering based on:
- Google's Prompt Engineering Best Practices (2024)
- OpenAI's Prompt Engineering Guide
- Anthropic's Claude Prompt Engineering Principles
- Research papers on text-to-image optimization

Key Principles:
1. Be specific, not vague
2. Use positive instructions (what TO do, not what NOT to do)
3. Structure prompts with clear components
4. Include style references and artistic direction
5. Handle edge cases with pattern matching
6. Iterate and A/B test prompts

Image Generation Model: FLUX.1 Schnell (Black Forest Labs)
- Fast 4-step inference
- Apache 2.0 license
- Optimized for speed and quality balance
"""

import re
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from enum import Enum


# =============================================================================
# PROMPT CATEGORIES
# =============================================================================

class PromptCategory(Enum):
    """Categories requiring different prompt strategies."""
    SINGLE_OBJECT = "single"           # One subject: "a house"
    MULTIPLE_OBJECTS = "multiple"      # 2+ subjects: "a cat and a dog"
    SCENE = "scene"                    # Landscapes: "mountain landscape"
    CHARACTER = "character"            # People/animals: "a cat sitting"
    ABSTRACT = "abstract"              # Concepts: "love", "freedom"
    TEXT_CONTENT = "text"              # Text in image: "sign saying hello"
    ACTION = "action"                  # Subject doing something: "bird flying"


# =============================================================================
# PROMPT COMPONENTS (Industrial Standard Structure)
# =============================================================================

@dataclass
class PromptComponents:
    """Structured prompt following industrial standards."""
    
    # Core subject (required)
    subject: str
    
    # Style descriptors
    art_style: str = "simple minimalist line art"
    color_scheme: str = "black and white only, high contrast"
    
    # Composition
    composition: str = "centered, balanced composition"
    background: str = "plain white background"
    
    # Technical specifications
    rendering: str = "clean bold outlines, no shading, no gradients"
    detail_level: str = "simple geometric shapes, iconic style"
    
    # Spatial arrangement (for multiple objects)
    spatial: str = ""
    
    # Negative concepts to avoid (phrased positively)
    cleanup: str = "single clean outline per object"
    
    def build(self) -> str:
        """Build the complete prompt string."""
        parts = [
            self.subject,
            self.art_style,
            self.color_scheme,
            self.rendering,
            self.detail_level,
            self.composition,
            self.background,
        ]
        
        if self.spatial:
            parts.append(self.spatial)
        
        parts.append(self.cleanup)
        
        return ", ".join(filter(None, parts))


# =============================================================================
# PROMPT TEMPLATES (A/B Tested Variations)
# =============================================================================

# Template V1: Minimal (fast, good for simple objects)
TEMPLATE_MINIMAL = PromptComponents(
    subject="{subject}",
    art_style="simple line art icon",
    color_scheme="black and white",
    composition="centered",
    background="white background",
    rendering="bold outlines",
    detail_level="simple shapes",
    cleanup="clean silhouette",
)

# Template V2: Detailed (better structure preservation)
TEMPLATE_DETAILED = PromptComponents(
    subject="{subject}",
    art_style="minimalist vector illustration, flat design",
    color_scheme="pure black on white, maximum contrast",
    composition="centered balanced composition, clear focal point",
    background="solid white background, no patterns",
    rendering="thick bold outlines only, no fill patterns, no shading, no gradients",
    detail_level="simplified geometric shapes, icon style, easily recognizable",
    cleanup="distinct silhouette, single outline per element",
)

# Template V3: Multi-object (optimized for separation)
TEMPLATE_MULTI_OBJECT = PromptComponents(
    subject="{subject}",
    art_style="minimalist icon set style, each item as distinct symbol",
    color_scheme="black and white only, high contrast",
    composition="items arranged in a clear row with equal spacing",
    background="white background",
    rendering="thick outlines, uniform stroke width",
    detail_level="simple geometric icons, instantly recognizable",
    spatial="generous white space between each element, no overlapping, no touching",
    cleanup="each object isolated and complete",
)

# Template V4: Scene/Landscape (optimized for structure)
TEMPLATE_SCENE = PromptComponents(
    subject="{subject}",
    art_style="simple landscape silhouette, paper cut-out style",
    color_scheme="black silhouettes on white",
    composition="clear foreground/background separation",
    background="plain white sky",
    rendering="bold solid shapes, no fine details",
    detail_level="major shapes only, no small elements",
    cleanup="distinct layered silhouettes",
)

# Template V5: Character/Action (optimized for figures)
TEMPLATE_CHARACTER = PromptComponents(
    subject="{subject}",
    art_style="simple cartoon character, mascot style",
    color_scheme="black and white",
    composition="full body view, centered",
    background="white background",
    rendering="thick outlines, cartoon proportions",
    detail_level="simple features, exaggerated key characteristics",
    cleanup="clear body outline, recognizable pose",
)


# =============================================================================
# PATTERN DETECTION
# =============================================================================

PATTERNS = {
    PromptCategory.MULTIPLE_OBJECTS: [
        r'\band\b',
        r'\bwith\b',
        r'\bnear\b',
        r'\bbeside\b',
        r'\bnext to\b',
        r'\b(\d+)\s+\w+s?\b',  # "3 trees"
        r'\bsome\b',
        r'\bmany\b',
        r'\bseveral\b',
    ],
    PromptCategory.SCENE: [
        r'\blandscape\b',
        r'\bscene\b',
        r'\bskyline\b',
        r'\bmountain',
        r'\bocean\b',
        r'\bforest\b',
        r'\bgarden\b',
        r'\bbeach\b',
        r'\bcity\b',
    ],
    PromptCategory.CHARACTER: [
        r'\bsitting\b',
        r'\bstanding\b',
        r'\brunning\b',
        r'\bwalking\b',
        r'\bflying\b',
        r'\bplaying\b',
        r'\beating\b',
        r'\bsleeping\b',
    ],
    PromptCategory.ABSTRACT: [
        r'\blove\b',
        r'\bhappiness\b',
        r'\bfreedom\b',
        r'\bpeace\b',
        r'\btime\b',
        r'\bmusic\b',
    ],
    PromptCategory.TEXT_CONTENT: [
        r'\btext\b',
        r'\bsaying\b',
        r'\bword\b',
        r'\bletter\b',
        r'\blogo\b',
        r'\bsign\b',
        r'"[^"]+"',
    ],
}

# Complexity words to replace
COMPLEXITY_REPLACEMENTS = {
    'detailed': 'simple',
    'intricate': 'simple',
    'complex': 'simple',
    'realistic': 'iconic',
    'photorealistic': 'iconic',
    'textured': 'flat',
    '3d': 'flat 2d',
    'elaborate': 'simple',
}

# Abstract concept mappings
ABSTRACT_MAPPINGS = {
    'love': 'a simple heart shape symbol',
    'happiness': 'a smiling sun face',
    'sadness': 'a teardrop shape',
    'freedom': 'a bird with spread wings flying',
    'peace': 'a peace dove with olive branch',
    'time': 'a simple analog clock face',
    'music': 'musical notes floating',
}


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class EnhancedPrompt:
    """Result of prompt enhancement."""
    original: str
    enhanced: str
    category: PromptCategory
    template_used: str
    warnings: List[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return self.enhanced


# =============================================================================
# MAIN ENHANCEMENT FUNCTION
# =============================================================================

def detect_category(prompt: str) -> PromptCategory:
    """Detect the most appropriate category for a prompt."""
    prompt_lower = prompt.lower()
    
    # Priority order matters
    for pattern in PATTERNS.get(PromptCategory.TEXT_CONTENT, []):
        if re.search(pattern, prompt_lower):
            return PromptCategory.TEXT_CONTENT
    
    for pattern in PATTERNS.get(PromptCategory.ABSTRACT, []):
        if re.search(pattern, prompt_lower):
            return PromptCategory.ABSTRACT
    
    for pattern in PATTERNS.get(PromptCategory.SCENE, []):
        if re.search(pattern, prompt_lower):
            return PromptCategory.SCENE
    
    for pattern in PATTERNS.get(PromptCategory.CHARACTER, []):
        if re.search(pattern, prompt_lower):
            return PromptCategory.CHARACTER
    
    for pattern in PATTERNS.get(PromptCategory.MULTIPLE_OBJECTS, []):
        if re.search(pattern, prompt_lower):
            return PromptCategory.MULTIPLE_OBJECTS
    
    return PromptCategory.SINGLE_OBJECT


def simplify_prompt(prompt: str) -> str:
    """Replace complexity-inducing words."""
    result = prompt.lower()
    for complex_word, simple_word in COMPLEXITY_REPLACEMENTS.items():
        result = re.sub(r'\b' + complex_word + r'\b', simple_word, result, flags=re.IGNORECASE)
    return result


def handle_abstract(prompt: str) -> Tuple[str, Optional[str]]:
    """Convert abstract concepts to concrete visuals."""
    prompt_lower = prompt.lower()
    
    for abstract, concrete in ABSTRACT_MAPPINGS.items():
        if abstract in prompt_lower:
            # Replace abstract with concrete
            result = re.sub(r'\b' + abstract + r'\b', concrete, prompt, flags=re.IGNORECASE)
            warning = f"Abstract '{abstract}' → '{concrete}'"
            return result, warning
    
    return prompt, None


def enhance_prompt(
    prompt: str,
    force_template: Optional[str] = None,
) -> EnhancedPrompt:
    """
    Enhance a prompt for optimal ASCII art generation.
    
    Industrial-standard prompt engineering:
    1. Detect category
    2. Simplify complexity
    3. Handle edge cases (abstract, text)
    4. Select optimal template
    5. Build structured prompt
    
    Args:
        prompt: User's original prompt
        force_template: Optional template override ('minimal', 'detailed', 'multi', 'scene', 'character')
        
    Returns:
        EnhancedPrompt with optimized text
    """
    warnings = []
    working_prompt = prompt.strip()
    
    # Step 1: Simplify complexity words
    working_prompt = simplify_prompt(working_prompt)
    
    # Step 2: Detect category
    category = detect_category(working_prompt)
    
    # Step 3: Handle special cases
    if category == PromptCategory.ABSTRACT:
        working_prompt, warning = handle_abstract(working_prompt)
        if warning:
            warnings.append(warning)
    
    if category == PromptCategory.TEXT_CONTENT:
        warnings.append("⚠️ Text in images doesn't convert well to ASCII art")
    
    # Step 4: Select template
    if force_template:
        template_name = force_template
        template = {
            'minimal': TEMPLATE_MINIMAL,
            'detailed': TEMPLATE_DETAILED,
            'multi': TEMPLATE_MULTI_OBJECT,
            'scene': TEMPLATE_SCENE,
            'character': TEMPLATE_CHARACTER,
        }.get(force_template, TEMPLATE_DETAILED)
    else:
        if category == PromptCategory.MULTIPLE_OBJECTS:
            template = TEMPLATE_MULTI_OBJECT
            template_name = "multi_object"
        elif category == PromptCategory.SCENE:
            template = TEMPLATE_SCENE
            template_name = "scene"
        elif category == PromptCategory.CHARACTER:
            template = TEMPLATE_CHARACTER
            template_name = "character"
        else:
            template = TEMPLATE_DETAILED
            template_name = "detailed"
    
    # Step 5: Build prompt
    template.subject = working_prompt
    enhanced = template.build()
    
    return EnhancedPrompt(
        original=prompt,
        enhanced=enhanced,
        category=category,
        template_used=template_name,
        warnings=warnings,
    )


def enhance_for_ascii(prompt: str) -> str:
    """Simple wrapper returning just the enhanced string."""
    return enhance_prompt(prompt).enhanced


# =============================================================================
# A/B TESTING FRAMEWORK
# =============================================================================

def generate_test_variations(prompt: str) -> List[Tuple[str, str]]:
    """
    Generate multiple prompt variations for A/B testing.
    
    Returns list of (template_name, enhanced_prompt) tuples.
    """
    variations = []
    
    for template_name in ['minimal', 'detailed', 'multi', 'scene', 'character']:
        result = enhance_prompt(prompt, force_template=template_name)
        variations.append((template_name, result.enhanced))
    
    return variations


# =============================================================================
# CLI TESTING
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("INDUSTRIAL-STANDARD PROMPT ENGINEERING TEST SUITE")
    print("=" * 70)
    
    test_prompts = [
        "a cat sitting on a chair",
        "a house and a tree",
        "mountain landscape with birds",
        "love and peace",
        "3 stars and a moon",
        "a person running",
    ]
    
    for prompt in test_prompts:
        result = enhance_prompt(prompt)
        print(f"\n📝 Original: '{prompt}'")
        print(f"   Category: {result.category.value}")
        print(f"   Template: {result.template_used}")
        if result.warnings:
            for w in result.warnings:
                print(f"   ⚠️  {w}")
        print(f"   Enhanced: {result.enhanced[:100]}...")
