"""
Quality Metrics for ASCII Art

Provides metrics to evaluate ASCII art quality:
- SSIM (Structural Similarity Index)
- MSE (Mean Squared Error)
- Character diversity (entropy)
"""

from typing import Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from .charsets import CharacterSet


def render_ascii_to_image(
    ascii_art: str,
    font_size: int = 12,
    bg_color: int = 255,
    fg_color: int = 0,
) -> Image.Image:
    """
    Render ASCII art text to a PIL Image for comparison.
    
    Args:
        ascii_art: Multi-line ASCII art string
        font_size: Font size for rendering
        bg_color: Background color (0-255)
        fg_color: Foreground color (0-255)
        
    Returns:
        Rendered PIL Image
    """
    lines = ascii_art.split('\n')
    if not lines:
        return Image.new('L', (1, 1), color=bg_color)
    
    # Calculate image dimensions
    max_width = max(len(line) for line in lines)
    height = len(lines)
    
    # Get font
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
    except:
        font = ImageFont.load_default()
    
    # Estimate character dimensions
    char_w = font_size * 0.6
    char_h = font_size * 1.2
    
    img_w = int(max_width * char_w) + 10
    img_h = int(height * char_h) + 10
    
    # Create image
    img = Image.new('L', (img_w, img_h), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Draw text
    y = 5
    for line in lines:
        draw.text((5, y), line, fill=fg_color, font=font)
        y += char_h
    
    return img


def compute_ssim(
    original: Image.Image,
    ascii_art: str,
    tile_size: Tuple[int, int] = (10, 16),
) -> float:
    """
    Compute Structural Similarity Index between original and ASCII rendering.
    
    Args:
        original: Original PIL Image
        ascii_art: Generated ASCII art string
        tile_size: Tile size used for conversion
        
    Returns:
        SSIM score (0-1, higher is better)
    """
    # Render ASCII to image
    ascii_img = render_ascii_to_image(ascii_art)
    
    # Convert original to grayscale
    if original.mode != 'L':
        original = original.convert('L')
    
    # Resize to match dimensions
    orig_array = np.array(original.resize(ascii_img.size))
    ascii_array = np.array(ascii_img)
    
    # Compute SSIM
    score = ssim(orig_array, ascii_array, data_range=255)
    
    return score


def compute_mse(
    original: Image.Image,
    ascii_art: str,
) -> float:
    """
    Compute Mean Squared Error between original and ASCII rendering.
    
    Args:
        original: Original PIL Image
        ascii_art: Generated ASCII art string
        
    Returns:
        MSE score (lower is better)
    """
    ascii_img = render_ascii_to_image(ascii_art)
    
    if original.mode != 'L':
        original = original.convert('L')
    
    orig_array = np.array(original.resize(ascii_img.size))
    ascii_array = np.array(ascii_img)
    
    return mse(orig_array, ascii_array)


def character_diversity(ascii_art: str) -> float:
    """
    Compute Shannon entropy of character distribution.
    
    Higher entropy = more diverse character usage.
    
    Args:
        ascii_art: ASCII art string
        
    Returns:
        Entropy value (higher = more diverse)
    """
    # Remove newlines and count characters
    chars = ascii_art.replace('\n', '')
    if not chars:
        return 0.0
    
    # Count frequencies
    unique, counts = np.unique(list(chars), return_counts=True)
    probabilities = counts / len(chars)
    
    # Compute entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return entropy


def edge_preservation_score(
    original: Image.Image,
    ascii_art: str,
    canny_low: int = 50,
    canny_high: int = 150,
) -> float:
    """
    Measure how well edges are preserved in the ASCII output.
    
    Args:
        original: Original PIL Image
        ascii_art: Generated ASCII art
        canny_low: Canny low threshold
        canny_high: Canny high threshold
        
    Returns:
        Edge preservation score (0-1)
    """
    import cv2
    
    # Get edges from original
    if original.mode != 'L':
        original = original.convert('L')
    orig_array = np.array(original)
    orig_edges = cv2.Canny(orig_array, canny_low, canny_high)
    
    # Render and get edges from ASCII
    ascii_img = render_ascii_to_image(ascii_art)
    ascii_array = np.array(ascii_img.resize(original.size))
    ascii_edges = cv2.Canny(ascii_array, canny_low, canny_high)
    
    # Compute overlap
    orig_edge_pixels = np.sum(orig_edges > 0)
    if orig_edge_pixels == 0:
        return 1.0
    
    overlap = np.sum((orig_edges > 0) & (ascii_edges > 0))
    
    return overlap / orig_edge_pixels


def compare_mappers(
    original: Image.Image,
    aiss_ascii: str,
    rf_ascii: str,
) -> dict:
    """
    Compare AISS and Random Forest outputs on all metrics.
    
    Args:
        original: Original image
        aiss_ascii: ASCII art from AISS mapper
        rf_ascii: ASCII art from Random Forest mapper
        
    Returns:
        Dictionary with comparison metrics
    """
    return {
        'aiss': {
            'ssim': compute_ssim(original, aiss_ascii),
            'mse': compute_mse(original, aiss_ascii),
            'diversity': character_diversity(aiss_ascii),
            'edge_preservation': edge_preservation_score(original, aiss_ascii),
        },
        'random_forest': {
            'ssim': compute_ssim(original, rf_ascii),
            'mse': compute_mse(original, rf_ascii),
            'diversity': character_diversity(rf_ascii),
            'edge_preservation': edge_preservation_score(original, rf_ascii),
        }
    }
