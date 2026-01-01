
import re
from typing import Dict, List, Optional, Tuple, NamedTuple

class CompositionMatch(NamedTuple):
    subject_a: str
    preposition: str
    subject_b: str
    template_key: str

class CompositionHandler:
    """
    Handles multi-object compositions by detecting patterns like "X on Y", "X next to Y".
    Provides structured prompt expansion to ensure clear boundaries and separation.
    """

    def __init__(self):
        # Preposition to Template Mapping
        self.templates = {
            "on": "vertical_stack",
            "on top of": "vertical_stack",
            "atop": "vertical_stack",
            
            "under": "vertical_stack_reverse",
            "beneath": "vertical_stack_reverse",
            "below": "vertical_stack_reverse",
            "underneath": "vertical_stack_reverse",
            
            "next to": "horizontal_side_by_side",
            "beside": "horizontal_side_by_side",
            "by": "horizontal_side_by_side",
            "with": "horizontal_side_by_side",
            
            "inside": "containment",
            "in": "containment",
            "within": "containment",
        }
        
        # Regex to capture "Subject A [prep] Subject B"
        # We try to match longer prepositions first
        preps = sorted(self.templates.keys(), key=len, reverse=True)
        prep_pattern = "|".join([re.escape(p) for p in preps])
        self.pattern = re.compile(rf"^(?P<subj_a>.+?)\s+\b(?P<prep>{prep_pattern})\b\s+(?P<subj_b>.+)$", re.IGNORECASE)

    def detect_composition(self, prompt: str) -> Optional[CompositionMatch]:
        """
        Detects if the prompt fits a simple binary composition pattern.
        """
        match = self.pattern.match(prompt.strip())
        if match:
            return CompositionMatch(
                subject_a=match.group("subj_a").strip(),
                preposition=match.group("prep").lower(),
                subject_b=match.group("subj_b").strip(),
                template_key=self.templates.get(match.group("prep").lower(), "default")
            )
        return None

    def format_composition(self, match: CompositionMatch, enhancer_func) -> str:
        """
        Formats the detected composition into a strong visual prompt.
        Accepts an `enhancer_func` to recursively enhance the subjects.
        """
        subj_a_desc = enhancer_func(match.subject_a)
        subj_b_desc = enhancer_func(match.subject_b)
        
        # Clean up subjects (remove "SUBJECT: " prefix if it exists from recursive call)
        subj_a_clean = self._clean_subject(subj_a_desc)
        subj_b_clean = self._clean_subject(subj_b_desc)

        if match.template_key == "vertical_stack":
            return (
                f"vertical composition: TOP OBJECT: {subj_a_clean}, BOTTOM OBJECT: {subj_b_clean}. "
                f"Clear separation between top and bottom. "
                f"The {match.subject_a} is resting clearly ON TOP OF the {match.subject_b}. "
                f"No merging of lines."
            )
        
        elif match.template_key == "vertical_stack_reverse":
            return (
                f"vertical composition: TOP OBJECT: {subj_b_clean}, BOTTOM OBJECT: {subj_a_clean}. "
                f"The {match.subject_a} is visible BENEATH the {match.subject_b}."
            )
            
        elif match.template_key == "horizontal_side_by_side":
            return (
                f"side-by-side composition: LEFT: {subj_a_clean}, RIGHT: {subj_b_clean}. "
                f"Distinct gap between them. No overlapping."
            )
            
        elif match.template_key == "containment":
            return (
                f"container view: OUTER OBJECT: {subj_b_clean}, INNER OBJECT: {subj_a_clean}. "
                f"The {match.subject_a} is clearly visible INSIDE the bounds of {match.subject_b}."
            )
            
        return f"{subj_a_clean} {match.preposition} {subj_b_clean}"

    def _clean_subject(self, text: str) -> str:
        """Helper to extract core description if the enhancer returns a full prompt."""
        # This assumes the enhancer returns something like "style, comp, SUBJECT: core, ..."
        # We try to extract the "SUBJECT: ..." part or just use the text
        if "SUBJECT:" in text:
            try:
                # varied formats, but usually SUBJECT: <desc>,
                parts = text.split("SUBJECT:")[1]
                # take until next comma if possible, or usually the SUBJECT part is the core
                # But our v3 enhancer puts it in middle.
                # Let's just use the whole thing but strip style prefixes if possible?
                # Actually, recursively enhancing might be too much if it adds styles again.
                # For now, let's just return the text but simplified
                return text
            except:
                return text
        return text

# Singleton
composition_handler = CompositionHandler()
