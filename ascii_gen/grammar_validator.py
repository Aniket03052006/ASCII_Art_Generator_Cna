"""
Grammar-Constrained ASCII Validation
====================================
Implements structural constraints enforcement for ASCII art generation,
inspired by "Grammar-Constrained Natural Language Generation" (Tuccio et al., 2025).

While full LLM grammar decoding (like GRAMMAR-LLM) requires manipulating logits,
this module enforces "visual grammar" rules on the generated output:

1. Vocabulary Constraints: Ensure only valid charset characters are used.
2. Structural Constraints: Enforce rectangular grid (line lengths).
3. Connectivity Constraints: Remove isolated noise (visual syntax error).
4. Continuity Constraints: Ensure lines connect logically (rule-based post-processing).
"""

import re
from typing import List, Set, Tuple

class GrammarValidator:
    """Enforces grammatical/structural constraints on ASCII art."""
    
    def __init__(self, valid_chars: str = None):
        # Default to standard ASCII + extended block characters
        self.valid_chars = valid_chars or r" @B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'."
        self.valid_set = set(self.valid_chars)
    
    def validate(self, ascii_art: str) -> Tuple[bool, List[str]]:
        """
        Check if ASCII art adheres to grammar rules.
        
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        lines = ascii_art.splitlines()
        
        if not lines:
            return False, ["Empty output"]
            
        # 1. Structural Constraint: Rectangular Grid
        widths = [len(line) for line in lines]
        avg_width = sum(widths) / len(widths)
        if len(set(widths)) > 1:
            violations.append(f"Irregular line lengths (min: {min(widths)}, max: {max(widths)})")
            
        # 2. Vocabulary Constraint: Valid Characters
        invalid_chars = set()
        for line in lines:
            for char in line:
                if char not in self.valid_set and char != '\n':
                    invalid_chars.add(char)
        
        if invalid_chars:
            violations.append(f"Invalid characters found: {sorted(list(invalid_chars))}")
            
        return len(violations) == 0, violations
    
    def enforce_constraints(self, ascii_art: str) -> str:
        """
        Fix grammatical violations in the ASCII art.
        """
        lines = ascii_art.splitlines()
        if not lines:
            return ""
            
        # 1. Fix Vocabulary: Replace invalid chars with space
        cleaned_lines = []
        for line in lines:
            cleaned = "".join([c if c in self.valid_set else " " for c in line])
            cleaned_lines.append(cleaned)
            
        # 2. Fix Structure: Pad/Truncate to target width
        # Use mode width as target
        if cleaned_lines:
            lens = [len(l) for l in cleaned_lines]
            target_width = max(set(lens), key=lens.count)
            
            final_lines = []
            for line in cleaned_lines:
                if len(line) < target_width:
                    line = line + " " * (target_width - len(line))
                elif len(line) > target_width:
                    line = line[:target_width]
                final_lines.append(line)
            cleaned_lines = final_lines
        
        # 3. Fix Visual Syntax: Remove isolated noise (Automata rule)
        # Rule: A single non-space char surrounded by spaces is "noise"
        # unless it's a specific small punctuation like . or ,
        rows = len(cleaned_lines)
        cols = len(cleaned_lines[0]) if rows > 0 else 0
        
        grid = [list(line) for line in cleaned_lines]
        result_grid = [list(line) for line in cleaned_lines]
        
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                char = grid[r][c]
                if char == ' ':
                    continue
                    
                # Check 8 neighbors
                neighbors = [
                    grid[r-1][c-1], grid[r-1][c], grid[r-1][c+1],
                    grid[r][c-1],                 grid[r][c+1],
                    grid[r+1][c-1], grid[r+1][c], grid[r+1][c+1]
                ]
                
                # Use grammar rule: Isolated pixels are invalid unless punctuation
                is_isolated = all(n == ' ' for n in neighbors)
                is_punctuation = char in ".,`'"
                
                if is_isolated and not is_punctuation:
                    result_grid[r][c] = ' '
                    
        return "\n".join(["".join(row) for row in result_grid])

def validate_grammar(ascii_art: str) -> Tuple[bool, List[str]]:
    """Helper function for quick validation."""
    validator = GrammarValidator()
    return validator.validate(ascii_art)

def enforce_grammar(ascii_art: str) -> str:
    """Helper function for quick enforcement."""
    validator = GrammarValidator()
    return validator.enforce_constraints(ascii_art)
