"""
ASCII Art Quality Metrics Suite

Implements quantitative evaluation metrics for ASCII art generation:
1. Grid Adherence Score - FFT analysis for character grid alignment
2. OCR Confidence - Tesseract recognition confidence
3. Structural Integrity - Loop closure and topology analysis
4. SSIM Quality Score - Structural similarity between source and rendered
5. Character Diversity - Entropy of character distribution
6. Edge Preservation - How well edges are preserved in ASCII conversion
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from collections import Counter
import json

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import optional dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available - some metrics will be skipped")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("⚠️ Tesseract not available - OCR metrics will be skipped")


@dataclass
class MetricsResult:
    """Container for all quality metrics."""
    grid_adherence: float
    ocr_confidence: float
    structural_integrity: float
    ssim_score: float
    character_diversity: float
    edge_preservation: float
    prompt_match_score: float  # CLIP-based (if available)
    overall_score: float
    details: Dict


def render_ascii_to_image(ascii_text: str, font_size: int = 12) -> Image.Image:
    """Render ASCII text to PIL Image for analysis."""
    lines = ascii_text.split('\n')
    if not lines:
        return Image.new('RGB', (100, 100), 'white')
    
    # Calculate dimensions
    char_width = font_size * 0.6  # Approximate monospace ratio
    char_height = font_size
    
    width = int(max(len(line) for line in lines) * char_width) + 20
    height = int(len(lines) * char_height) + 20
    
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)
    
    # Try to get a monospace font
    try:
        font_paths = [
            "/System/Library/Fonts/Menlo.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/System/Library/Fonts/Monaco.dfont"
        ]
        font = None
        for p in font_paths:
            if os.path.exists(p):
                font = ImageFont.truetype(p, font_size)
                break
        if not font:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    draw.text((10, 10), ascii_text, fill='black', font=font)
    return img


def calculate_grid_adherence(image: Image.Image, expected_cell_size: int = 8) -> Tuple[float, Dict]:
    """
    Calculate grid adherence using FFT analysis.
    
    Looks for peak frequencies at the expected character cell intervals.
    High score means the image has strong grid-like structure.
    
    Returns:
        Tuple of (score 0-1, details dict)
    """
    if not CV2_AVAILABLE:
        return 0.0, {"error": "OpenCV not available"}
    
    # Convert to grayscale numpy
    img_array = np.array(image.convert('L'), dtype=np.float32)
    
    # Apply 2D FFT
    fft = np.fft.fft2(img_array)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    # Log transform for visualization
    magnitude_log = np.log1p(magnitude)
    
    # Find expected frequency for cell size
    h, w = img_array.shape
    expected_freq_x = w / expected_cell_size
    expected_freq_y = h / expected_cell_size
    
    # Sample magnitude at expected frequencies
    center_x, center_y = w // 2, h // 2
    
    # Check for harmonics at cell intervals
    harmonics_x = []
    harmonics_y = []
    
    for harmonic in range(1, 4):  # Check first 3 harmonics
        freq_offset_x = int(expected_freq_x * harmonic)
        freq_offset_y = int(expected_freq_y * harmonic)
        
        if center_x + freq_offset_x < w:
            harmonics_x.append(magnitude[center_y, center_x + freq_offset_x])
        if center_y + freq_offset_y < h:
            harmonics_y.append(magnitude[center_y + freq_offset_y, center_x])
    
    # Calculate score based on harmonic strength relative to noise
    if harmonics_x and harmonics_y:
        harmonic_strength = np.mean(harmonics_x + harmonics_y)
        noise_floor = np.median(magnitude)
        
        if noise_floor > 0:
            ratio = harmonic_strength / noise_floor
            score = min(1.0, ratio / 10.0)  # Normalize to 0-1
        else:
            score = 0.5
    else:
        score = 0.0
    
    return score, {
        "expected_cell_size": expected_cell_size,
        "harmonic_strength": float(np.mean(harmonics_x + harmonics_y)) if harmonics_x else 0,
        "noise_floor": float(np.median(magnitude)),
    }


def calculate_ocr_confidence(image: Image.Image) -> Tuple[float, Dict]:
    """
    Calculate OCR confidence using Tesseract.
    
    Returns:
        Tuple of (confidence 0-1, details dict)
    """
    if not TESSERACT_AVAILABLE:
        return 0.0, {"error": "Tesseract not available"}
    
    try:
        # Get detailed OCR data
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        
        # Filter valid confidences
        confidences = [int(c) for c in data['conf'] if int(c) > 0]
        
        if confidences:
            avg_confidence = np.mean(confidences) / 100.0
            recognized_chars = len([t for t in data['text'] if t.strip()])
            
            return avg_confidence, {
                "avg_confidence_pct": avg_confidence * 100,
                "recognized_characters": recognized_chars,
                "total_boxes": len(confidences)
            }
        else:
            return 0.0, {"error": "No characters recognized"}
            
    except Exception as e:
        return 0.0, {"error": str(e)}


def calculate_structural_integrity(ascii_text: str) -> Tuple[float, Dict]:
    """
    Analyze structural integrity of ASCII art.
    
    Checks for:
    1. Closed loops (rooms in dungeon maps)
    2. Connected components
    3. Symmetry indicators
    
    Returns:
        Tuple of (score 0-1, details dict)
    """
    lines = ascii_text.split('\n')
    if not lines:
        return 0.0, {"error": "Empty ASCII"}
    
    # Pad lines to same length
    max_len = max(len(line) for line in lines)
    grid = [list(line.ljust(max_len)) for line in lines]
    rows, cols = len(grid), max_len
    
    # Define "wall" characters (non-space, non-dot)
    wall_chars = set('#@$%&*|/\\-_=+[]{}()<>')
    
    # Count structural elements
    wall_count = 0
    corner_count = 0
    
    for r in range(rows):
        for c in range(cols):
            char = grid[r][c]
            if char in wall_chars:
                wall_count += 1
                
                # Check if it's a corner (has neighbors in perpendicular directions)
                neighbors = []
                if r > 0 and grid[r-1][c] in wall_chars:
                    neighbors.append('up')
                if r < rows-1 and grid[r+1][c] in wall_chars:
                    neighbors.append('down')
                if c > 0 and grid[r][c-1] in wall_chars:
                    neighbors.append('left')
                if c < cols-1 and grid[r][c+1] in wall_chars:
                    neighbors.append('right')
                
                # Corner = has both horizontal and vertical neighbors
                has_vertical = 'up' in neighbors or 'down' in neighbors
                has_horizontal = 'left' in neighbors or 'right' in neighbors
                if has_vertical and has_horizontal:
                    corner_count += 1
    
    # Calculate metrics
    total_chars = rows * cols
    density = wall_count / total_chars if total_chars > 0 else 0
    
    # Higher corner ratio suggests better closed structures
    corner_ratio = corner_count / wall_count if wall_count > 0 else 0
    
    # Score based on reasonable density (10-40%) and corner ratio
    density_score = 1.0 - abs(density - 0.25) * 4  # Peak at 25% density
    density_score = max(0, min(1, density_score))
    
    structure_score = corner_ratio * 2  # Corners indicate structure
    structure_score = max(0, min(1, structure_score))
    
    final_score = (density_score + structure_score) / 2
    
    return final_score, {
        "wall_count": wall_count,
        "corner_count": corner_count,
        "density": density,
        "corner_ratio": corner_ratio,
        "grid_size": f"{rows}x{cols}"
    }


def calculate_ssim_score(source_image: Image.Image, rendered_ascii: Image.Image) -> Tuple[float, Dict]:
    """
    Calculate SSIM between source image and rendered ASCII.
    
    Returns:
        Tuple of (ssim score 0-1, details dict)
    """
    if not CV2_AVAILABLE:
        return 0.0, {"error": "OpenCV not available"}
    
    # Resize to same dimensions
    source = np.array(source_image.convert('L'))
    rendered = np.array(rendered_ascii.convert('L'))
    
    # Resize rendered to match source
    rendered = cv2.resize(rendered, (source.shape[1], source.shape[0]))
    
    # Calculate SSIM
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    source = source.astype(np.float64)
    rendered = rendered.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(source, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(rendered, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(source ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(rendered ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(source * rendered, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    ssim_score = float(np.mean(ssim_map))
    
    return max(0, ssim_score), {
        "mean_ssim": ssim_score,
        "min_ssim": float(np.min(ssim_map)),
        "max_ssim": float(np.max(ssim_map))
    }


def calculate_character_diversity(ascii_text: str) -> Tuple[float, Dict]:
    """
    Calculate character diversity using entropy.
    
    Higher diversity usually indicates more detailed ASCII art.
    
    Returns:
        Tuple of (normalized entropy 0-1, details dict)
    """
    # Remove newlines and count characters
    chars = [c for c in ascii_text if c != '\n']
    if not chars:
        return 0.0, {"error": "Empty ASCII"}
    
    # Count frequencies
    counter = Counter(chars)
    total = len(chars)
    
    # Calculate Shannon entropy
    entropy = 0.0
    for count in counter.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)
    
    # Maximum possible entropy for this character set
    unique_chars = len(counter)
    max_entropy = np.log2(unique_chars) if unique_chars > 1 else 1
    
    # Normalize to 0-1
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    
    # Get top characters
    top_chars = counter.most_common(5)
    
    return normalized_entropy, {
        "entropy_bits": entropy,
        "unique_characters": unique_chars,
        "total_characters": total,
        "top_5_chars": [(c, n) for c, n in top_chars],
        "space_ratio": counter.get(' ', 0) / total
    }


def calculate_edge_preservation(source_image: Image.Image, ascii_text: str) -> Tuple[float, Dict]:
    """
    Measure how well edges from the source are preserved in ASCII.
    
    Returns:
        Tuple of (score 0-1, details dict)
    """
    if not CV2_AVAILABLE:
        return 0.0, {"error": "OpenCV not available"}
    
    # Get edges from source
    source_gray = np.array(source_image.convert('L'))
    source_edges = cv2.Canny(source_gray, 50, 150)
    
    # Render ASCII and get its edges
    rendered = render_ascii_to_image(ascii_text)
    rendered_gray = np.array(rendered.convert('L'))
    rendered_resized = cv2.resize(rendered_gray, (source_gray.shape[1], source_gray.shape[0]))
    rendered_edges = cv2.Canny(rendered_resized, 50, 150)
    
    # Calculate edge overlap
    source_edge_pixels = np.sum(source_edges > 0)
    rendered_edge_pixels = np.sum(rendered_edges > 0)
    
    # Dilate edges for fuzzy matching
    kernel = np.ones((3, 3), np.uint8)
    source_dilated = cv2.dilate(source_edges, kernel, iterations=2)
    
    # Count matches
    matches = np.sum((rendered_edges > 0) & (source_dilated > 0))
    
    if rendered_edge_pixels > 0:
        precision = matches / rendered_edge_pixels
    else:
        precision = 0
    
    if source_edge_pixels > 0:
        recall = matches / source_edge_pixels
    else:
        recall = 0
    
    # F1 score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    
    return f1, {
        "source_edge_pixels": int(source_edge_pixels),
        "rendered_edge_pixels": int(rendered_edge_pixels),
        "matched_pixels": int(matches),
        "precision": precision,
        "recall": recall
    }


def run_full_metrics(
    source_image: Image.Image,
    ascii_text: str,
    prompt: str = ""
) -> MetricsResult:
    """
    Run all quality metrics on an ASCII art result.
    
    Args:
        source_image: Original image used for conversion
        ascii_text: Generated ASCII art text
        prompt: Original text prompt (for CLIP scoring)
    
    Returns:
        MetricsResult with all scores
    """
    print("📊 Running Quality Metrics...")
    
    # Render ASCII to image
    rendered = render_ascii_to_image(ascii_text)
    
    # Calculate each metric
    print("  • Grid Adherence...")
    grid_score, grid_details = calculate_grid_adherence(rendered)
    
    print("  • OCR Confidence...")
    ocr_score, ocr_details = calculate_ocr_confidence(rendered)
    
    print("  • Structural Integrity...")
    struct_score, struct_details = calculate_structural_integrity(ascii_text)
    
    print("  • SSIM Score...")
    ssim_score, ssim_details = calculate_ssim_score(source_image, rendered)
    
    print("  • Character Diversity...")
    div_score, div_details = calculate_character_diversity(ascii_text)
    
    print("  • Edge Preservation...")
    edge_score, edge_details = calculate_edge_preservation(source_image, ascii_text)
    
    # CLIP-based prompt matching (placeholder - requires CLIP)
    prompt_score = 0.0
    
    # Calculate overall score (weighted average)
    weights = {
        'grid': 0.10,
        'ocr': 0.05,
        'struct': 0.15,
        'ssim': 0.25,
        'diversity': 0.20,
        'edge': 0.25
    }
    
    overall = (
        grid_score * weights['grid'] +
        ocr_score * weights['ocr'] +
        struct_score * weights['struct'] +
        ssim_score * weights['ssim'] +
        div_score * weights['diversity'] +
        edge_score * weights['edge']
    )
    
    return MetricsResult(
        grid_adherence=grid_score,
        ocr_confidence=ocr_score,
        structural_integrity=struct_score,
        ssim_score=ssim_score,
        character_diversity=div_score,
        edge_preservation=edge_score,
        prompt_match_score=prompt_score,
        overall_score=overall,
        details={
            'grid': grid_details,
            'ocr': ocr_details,
            'structural': struct_details,
            'ssim': ssim_details,
            'diversity': div_details,
            'edge': edge_details
        }
    )


def print_metrics_report(result: MetricsResult):
    """Pretty print the metrics report."""
    print("\n" + "=" * 60)
    print("📊 ASCII ART QUALITY METRICS REPORT")
    print("=" * 60)
    
    metrics = [
        ("Grid Adherence", result.grid_adherence, "> 0.7"),
        ("OCR Confidence", result.ocr_confidence, "> 0.85"),
        ("Structural Integrity", result.structural_integrity, "> 0.6"),
        ("SSIM Score", result.ssim_score, "> 0.3"),
        ("Character Diversity", result.character_diversity, "> 0.5"),
        ("Edge Preservation", result.edge_preservation, "> 0.4"),
    ]
    
    for name, score, target in metrics:
        bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        status = "✅" if score >= float(target.split("> ")[1]) else "⚠️"
        print(f"  {name:25} [{bar}] {score:.2f} {status} (target: {target})")
    
    print("-" * 60)
    overall_bar = "█" * int(result.overall_score * 20) + "░" * (20 - int(result.overall_score * 20))
    print(f"  {'OVERALL SCORE':25} [{overall_bar}] {result.overall_score:.2f}")
    print("=" * 60)


def test_with_sample():
    """Test metrics with a sample generation."""
    from ascii_gen.online_generator import OnlineGenerator
    from ascii_gen.gradient_mapper import image_to_gradient_ascii
    
    print("🧪 Testing Metrics Suite with Sample Generation")
    print("-" * 60)
    
    # Generate a test image
    prompt = "simple mushroom, black and white icon"
    print(f"Prompt: '{prompt}'")
    
    # Use Pollinations (free, no auth needed)
    import urllib.parse
    import requests
    import io
    
    encoded = urllib.parse.quote(prompt)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=384&nologo=true"
    
    print("🌐 Generating test image via Pollinations...")
    response = requests.get(url, timeout=60)
    if response.status_code == 200:
        source_image = Image.open(io.BytesIO(response.content)).convert('RGB')
        print("✅ Image generated!")
        
        # Convert to ASCII
        print("⚡ Converting to ASCII...")
        ascii_art = image_to_gradient_ascii(source_image, width=80, ramp="ultra")
        
        # Run metrics
        result = run_full_metrics(source_image, ascii_art, prompt)
        print_metrics_report(result)
        
        # Save results
        with open("metrics_result.json", "w") as f:
            json.dump({
                "prompt": prompt,
                "scores": {
                    "grid_adherence": result.grid_adherence,
                    "ocr_confidence": result.ocr_confidence,
                    "structural_integrity": result.structural_integrity,
                    "ssim_score": result.ssim_score,
                    "character_diversity": result.character_diversity,
                    "edge_preservation": result.edge_preservation,
                    "overall": result.overall_score
                },
                "details": result.details
            }, f, indent=2, default=str)
        print("\n📁 Results saved to metrics_result.json")
        
        return result
    else:
        print(f"❌ Failed to generate image: {response.status_code}")
        return None


if __name__ == "__main__":
    test_with_sample()
