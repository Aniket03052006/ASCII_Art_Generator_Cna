"""
Image Preprocessing Utilities

Provides image preprocessing functions for ASCII art generation:
- Edge detection (Canny, Sobel)
- Contrast enhancement
- Resizing with aspect ratio preservation
- Binary thresholding
"""

from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image


def apply_canny_edge(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
) -> np.ndarray:
    """
    Apply Canny edge detection.
    
    Args:
        image: Grayscale image (0-255)
        low_threshold: Lower threshold for hysteresis
        high_threshold: Upper threshold for hysteresis
        
    Returns:
        Binary edge image (0 or 255)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return cv2.Canny(image, low_threshold, high_threshold)


def apply_sobel_edge(
    image: np.ndarray,
    ksize: int = 3,
    threshold: Optional[float] = None,
) -> np.ndarray:
    """
    Apply Sobel edge detection (gradient magnitude).
    
    Args:
        image: Grayscale image
        ksize: Kernel size for Sobel operator
        threshold: Optional threshold for binary output
        
    Returns:
        Edge magnitude image (or binary if threshold given)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize)
    
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8)
    
    if threshold is not None:
        _, magnitude = cv2.threshold(
            magnitude, 
            int(threshold * 255), 
            255, 
            cv2.THRESH_BINARY
        )
    
    return magnitude


def enhance_contrast(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """
    Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        image: Grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for equalization
        
    Returns:
        Contrast-enhanced image
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(image)


def resize_for_ascii(
    image: Image.Image,
    char_width: int = 80,
    char_height: Optional[int] = None,
    tile_size: Tuple[int, int] = (10, 16),
    maintain_aspect: bool = True,
) -> Image.Image:
    """
    Resize image to fit the target ASCII dimensions.
    
    Args:
        image: PIL Image to resize
        char_width: Target width in characters
        char_height: Target height in characters (auto if None)
        tile_size: Size of each character tile (width, height)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    tile_w, tile_h = tile_size
    target_width = char_width * tile_w
    
    if maintain_aspect:
        # Calculate height to maintain aspect ratio
        # Account for character aspect ratio (tiles are taller than wide)
        aspect_ratio = image.width / image.height
        char_aspect = tile_w / tile_h
        
        # Adjust for terminal display (characters are ~2x taller than wide visually)
        adjusted_height = int(target_width / aspect_ratio * char_aspect)
        
        # Round to multiple of tile height
        adjusted_height = (adjusted_height // tile_h) * tile_h
        
        if char_height is not None:
            target_height = min(char_height * tile_h, adjusted_height)
        else:
            target_height = adjusted_height
    else:
        target_height = (char_height or 40) * tile_h
    
    return image.resize((target_width, target_height), Image.Resampling.LANCZOS)


def apply_threshold(
    image: np.ndarray,
    threshold: int = 127,
    invert: bool = False,
) -> np.ndarray:
    """
    Apply binary threshold to image.
    
    Args:
        image: Grayscale image
        threshold: Threshold value (0-255)
        invert: Whether to invert the result
        
    Returns:
        Binary image (0 or 255)
    """
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    _, binary = cv2.threshold(image, threshold, 255, thresh_type)
    return binary


def apply_adaptive_threshold(
    image: np.ndarray,
    block_size: int = 11,
    c: int = 2,
    invert: bool = False,
) -> np.ndarray:
    """
    Apply adaptive thresholding for variable lighting conditions.
    
    Args:
        image: Grayscale image
        block_size: Size of neighborhood for threshold calculation
        c: Constant subtracted from mean
        invert: Whether to invert the result
        
    Returns:
        Binary image
    """
    thresh_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        block_size,
        c
    )


def preprocess_for_structure(
    image: Image.Image,
    char_width: int = 80,
    edge_method: str = "canny",
    enhance_before_edge: bool = True,
) -> Tuple[Image.Image, np.ndarray]:
    """
    Full preprocessing pipeline for structure-based ASCII art.
    
    Args:
        image: Input PIL Image
        char_width: Target width in characters
        edge_method: "canny" or "sobel"
        enhance_before_edge: Apply contrast enhancement before edges
        
    Returns:
        Tuple of (resized image, edge image array)
    """
    # Resize
    resized = resize_for_ascii(image, char_width=char_width)
    
    # Convert to grayscale array
    gray = np.array(resized.convert('L'))
    
    # Enhance contrast
    if enhance_before_edge:
        gray = enhance_contrast(gray)
    
    # Edge detection
    if edge_method == "canny":
        edges = apply_canny_edge(gray)
    elif edge_method == "sobel":
        edges = apply_sobel_edge(gray, threshold=0.3)
    else:
        edges = gray
    
    return resized, edges
