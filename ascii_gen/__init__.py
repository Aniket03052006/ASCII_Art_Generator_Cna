"""
Prompt-to-ASCII Art Generator

A complete pipeline for converting text prompts to ASCII art using:
- FLUX.1 Schnell / SDXL-Turbo for image generation
- AISS (Log-Polar) + Random Forest + CNN for structural mapping
- Production-trained models with edge-aligned data

Optimized for Apple Silicon (M4) with MPS acceleration.
"""

__version__ = "0.3.0"
__author__ = "ASCII_Gen"

from .charsets import CharacterSet, get_charset
from .pipeline import PromptToASCII, image_to_ascii, prompt_to_ascii
from .result import ASCIIResult
from .cnn_mapper import CNNMapper, create_cnn_mapper
from .production_training import ProductionCNNMapper, ProductionRFMapper

__all__ = [
    "CharacterSet",
    "get_charset", 
    "PromptToASCII",
    "ASCIIResult",
    "CNNMapper",
    "create_cnn_mapper",
    "image_to_ascii",
    "prompt_to_ascii",
    "ProductionCNNMapper",
    "ProductionRFMapper",
]
