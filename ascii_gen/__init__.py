"""
Prompt-to-ASCII Art Generator

A complete pipeline for converting text prompts to ASCII art using:
- Stable Diffusion (SDXL-Turbo) for image generation
- ControlNet for structural guidance
- AISS (Log-Polar) + Random Forest for structural character mapping

Optimized for Apple Silicon (M4) with MPS acceleration.
"""

__version__ = "0.1.0"
__author__ = "ASCII_Gen"

from .charsets import CharacterSet, get_charset
from .pipeline import PromptToASCII
from .result import ASCIIResult

__all__ = [
    "CharacterSet",
    "get_charset", 
    "PromptToASCII",
    "ASCIIResult",
]
