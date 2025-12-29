"""
Advanced Preprocessing Module for ASCII Art

Novel techniques from academic research:
1. Saliency-guided edge detection (focus on important regions)
2. Bilateral filtering (preserve edges, remove noise)
3. Multi-scale processing (capture both structure and detail)
4. Adaptive thresholding (handle varying complexity)
5. Contour simplification (reduce noise artifacts)

Based on:
- "Perception-Sensitive Structure Extraction" (IEEE)
- "Evaluating ML Approaches for ASCII Art" (Coumar & Kingston, 2025)
- OpenCV Spectral Residual Saliency (Hou & Zhang, 2007)
"""

import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for advanced preprocessing."""
    use_saliency: bool = True
    use_bilateral: bool = True
    use_multiscale: bool = False
    canny_low: int = 50
    canny_high: int = 150
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75
    bilateral_sigma_space: float = 75
    saliency_weight: float = 0.5  # How much to weight salient regions


def compute_saliency_map(image: np.ndarray) -> np.ndarray:
    """
    Compute saliency map using Spectral Residual method.
    
    This identifies the "important" regions of an image - where humans
    would naturally focus their attention.
    
    Args:
        image: Grayscale image (uint8)
        
    Returns:
        Saliency map (float32, normalized 0-1)
    """
    # Ensure grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    # Resize for faster processing
    scale = min(1.0, 256 / max(gray.shape))
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale)
    else:
        small = gray
    
    # Spectral Residual method
    # 1. Compute log amplitude spectrum
    float_img = small.astype(np.float32)
    dft = cv2.dft(float_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Get magnitude and phase
    magnitude, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
    log_magnitude = np.log(magnitude + 1e-10)
    
    # 2. Compute spectral residual
    mean_log_mag = cv2.blur(log_magnitude, (3, 3))
    spectral_residual = log_magnitude - mean_log_mag
    
    # 3. Reconstruct with spectral residual
    exp_sr = np.exp(spectral_residual)
    real = exp_sr * np.cos(phase)
    imag = exp_sr * np.sin(phase)
    
    dft_sr = np.zeros_like(dft_shift)
    dft_sr[:, :, 0] = real
    dft_sr[:, :, 1] = imag
    
    dft_ishift = np.fft.ifftshift(dft_sr)
    saliency_small = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    
    # 4. Take magnitude and square (makes edges sharper)
    saliency_small = np.abs(saliency_small) ** 2
    
    # 5. Gaussian smoothing
    saliency_small = cv2.GaussianBlur(saliency_small, (9, 9), 2.5)
    
    # Resize back to original size
    saliency = cv2.resize(saliency_small, (gray.shape[1], gray.shape[0]))
    
    # Normalize to 0-1
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-10)
    
    return saliency.astype(np.float32)


def bilateral_filter(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """
    Apply bilateral filter to reduce noise while preserving edges.
    
    This is crucial for ASCII art because it removes texture noise
    (like fur, fabric patterns) while keeping important structural edges.
    """
    return cv2.bilateralFilter(
        image,
        d=config.bilateral_d,
        sigmaColor=config.bilateral_sigma_color,
        sigmaSpace=config.bilateral_sigma_space,
    )


def saliency_weighted_canny(
    image: np.ndarray,
    saliency: np.ndarray,
    config: PreprocessingConfig,
) -> np.ndarray:
    """
    Apply Canny edge detection weighted by saliency.
    
    This gives more importance to edges in salient regions,
    helping to preserve important structures while suppressing
    noise in less important areas.
    """
    # Standard Canny edge detection
    edges = cv2.Canny(image, config.canny_low, config.canny_high)
    
    # Weight edges by saliency
    weighted = edges.astype(np.float32) * (
        config.saliency_weight + (1 - config.saliency_weight) * saliency
    )
    
    # Threshold back to binary
    weighted_edges = (weighted > 127).astype(np.uint8) * 255
    
    return weighted_edges


def multiscale_edges(image: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """
    Compute edges at multiple scales and combine.
    
    Large scale: Captures overall structure (body, main shapes)
    Medium scale: Captures details (limbs, features)
    Small scale: Fine details (eyes, small decorations)
    """
    scales = [(1.0, 0.5), (0.5, 0.3), (0.25, 0.2)]  # (scale, weight)
    combined = np.zeros_like(image, dtype=np.float32)
    
    for scale, weight in scales:
        # Resize
        h, w = image.shape[:2]
        new_size = (max(32, int(w * scale)), max(32, int(h * scale)))
        scaled = cv2.resize(image, new_size)
        
        # Edge detection at this scale
        edges = cv2.Canny(scaled, config.canny_low, config.canny_high)
        
        # Resize back
        edges_full = cv2.resize(edges, (w, h))
        
        # Accumulate
        combined += edges_full.astype(np.float32) * weight
    
    # Normalize and threshold
    combined = np.clip(combined, 0, 255).astype(np.uint8)
    _, result = cv2.threshold(combined, 64, 255, cv2.THRESH_BINARY)
    
    return result


def simplify_contours(edges: np.ndarray, epsilon_factor: float = 0.001) -> np.ndarray:
    """
    Simplify contours to reduce noise artifacts.
    
    Uses Douglas-Peucker algorithm to approximate contours with fewer points,
    resulting in cleaner lines in the ASCII output.
    """
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create output image
    result = np.zeros_like(edges)
    
    # Simplify and draw each contour
    for contour in contours:
        # Skip very small contours (noise)
        if cv2.arcLength(contour, True) < 20:
            continue
        
        # Approximate contour
        epsilon = epsilon_factor * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Draw
        cv2.drawContours(result, [approx], 0, 255, 1)
    
    return result


def preprocess_for_ascii(
    image: Image.Image,
    config: Optional[PreprocessingConfig] = None,
) -> np.ndarray:
    """
    Advanced preprocessing pipeline for ASCII art conversion.
    
    This implements novel techniques from research papers:
    1. Saliency detection to focus on important regions
    2. Bilateral filtering to reduce noise while preserving edges
    3. Multi-scale edge detection for structure + detail
    4. Contour simplification to reduce artifacts
    
    Args:
        image: PIL Image
        config: Preprocessing configuration
        
    Returns:
        Preprocessed edge image (uint8, 0-255)
    """
    if config is None:
        config = PreprocessingConfig()
    
    # Convert to numpy
    img_array = np.array(image)
    
    # Convert to grayscale if needed
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    
    # Step 1: Bilateral filtering (noise reduction, edge preservation)
    if config.use_bilateral:
        gray = bilateral_filter(gray, config)
    
    # Step 2: Compute saliency map
    if config.use_saliency:
        saliency = compute_saliency_map(gray)
        edges = saliency_weighted_canny(gray, saliency, config)
    else:
        edges = cv2.Canny(gray, config.canny_low, config.canny_high)
    
    # Step 3: Multi-scale processing (optional, slower)
    if config.use_multiscale:
        multiscale = multiscale_edges(gray, config)
        # Combine with saliency edges
        edges = cv2.bitwise_or(edges, multiscale)
    
    # Step 4: Contour simplification
    edges = simplify_contours(edges)
    
    return edges


# Quick comparison test
if __name__ == "__main__":
    from PIL import Image
    import os
    
    print("=" * 60)
    print("ADVANCED PREPROCESSING COMPARISON")
    print("=" * 60)
    
    # Load test image
    test_path = "outputs/house_and_tree.png"
    if os.path.exists(test_path):
        img = Image.open(test_path)
        
        # Standard preprocessing (current method)
        gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
        standard_edges = cv2.Canny(gray, 50, 150)
        
        # Advanced preprocessing (new method)
        advanced_edges = preprocess_for_ascii(img)
        
        # Save comparison
        cv2.imwrite("outputs/edges_standard.png", standard_edges)
        cv2.imwrite("outputs/edges_advanced.png", advanced_edges)
        
        print(f"Standard edges: {np.sum(standard_edges > 0)} edge pixels")
        print(f"Advanced edges: {np.sum(advanced_edges > 0)} edge pixels")
        print("Saved to outputs/edges_standard.png and outputs/edges_advanced.png")
    else:
        print(f"Test image not found: {test_path}")
