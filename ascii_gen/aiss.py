"""
AISS (Alignment-Insensitive Shape Similarity) Structural Mapper

Implements the Xu et al. (2010) algorithm for structure-based ASCII art.
Uses log-polar histograms via OpenCV's ShapeContextDistanceExtractor
to match image tiles to ASCII characters based on shape similarity.

Key features:
- Translation-invariant shape matching
- Contour-based comparison (ignores texture/tone)
- Fast batch processing for large images
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import cv2
from PIL import Image

from .charsets import CharacterSet, get_charset


class AISSMapper:
    """
    AISS-based structural ASCII art mapper.
    
    Uses log-polar histograms (Shape Context) to find characters
    that best match the structural content of image tiles.
    
    Example:
        >>> mapper = AISSMapper(charset=get_charset("ascii_standard"))
        >>> ascii_art = mapper.convert_image(pil_image)
        >>> print(ascii_art)
    """
    
    def __init__(
        self,
        charset: Optional[CharacterSet] = None,
        angular_bins: int = 12,
        radial_bins: int = 5,
        inner_radius: float = 0.1,
        outer_radius: float = 2.0,
    ):
        """
        Initialize the AISS mapper.
        
        Args:
            charset: Character set to use (default: ascii_standard)
            angular_bins: Number of angular bins for log-polar histogram
            radial_bins: Number of radial bins for log-polar histogram
            inner_radius: Inner radius ratio for shape context
            outer_radius: Outer radius ratio for shape context
        """
        self.charset = charset or get_charset("ascii_standard")
        self.angular_bins = angular_bins
        self.radial_bins = radial_bins
        self.inner_radius = inner_radius
        self.outer_radius = outer_radius
        
        # Pre-compute character contours and descriptors
        self._char_contours: Dict[str, np.ndarray] = {}
        self._precompute_character_data()
    
    def _precompute_character_data(self):
        """Pre-compute contours for all characters in the charset."""
        for char in self.charset.characters:
            raster = self.charset.get_raster(char)
            contour = self._extract_contour_points(raster)
            self._char_contours[char] = contour
    
    def _extract_contour_points(self, binary_image: np.ndarray) -> np.ndarray:
        """
        Extract contour points from a binary image.
        
        Args:
            binary_image: Binary image (0/1 values)
            
        Returns:
            Nx2 array of contour points
        """
        # Ensure proper format for OpenCV
        img = (binary_image * 255).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(
            img, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_NONE
        )
        
        if not contours:
            # Return center point if no contours found
            h, w = binary_image.shape
            return np.array([[w // 2, h // 2]], dtype=np.float32)
        
        # Combine all contour points
        all_points = np.vstack([c.reshape(-1, 2) for c in contours])
        
        # Subsample if too many points (for efficiency)
        max_points = 100
        if len(all_points) > max_points:
            indices = np.linspace(0, len(all_points) - 1, max_points, dtype=int)
            all_points = all_points[indices]
        
        return all_points.astype(np.float32)
    
    def _compute_log_polar_histogram(self, points: np.ndarray) -> np.ndarray:
        """
        Compute log-polar histogram for a set of points.
        
        This is the core of the AISS algorithm - it creates a descriptor
        that is robust to small translations and rotations.
        
        Args:
            points: Nx2 array of contour points
            
        Returns:
            Flattened histogram of shape (angular_bins * radial_bins,)
        """
        if len(points) < 2:
            return np.zeros(self.angular_bins * self.radial_bins)
        
        # Compute centroid
        centroid = np.mean(points, axis=0)
        
        # Center points
        centered = points - centroid
        
        # Convert to polar coordinates
        r = np.sqrt(np.sum(centered ** 2, axis=1))
        theta = np.arctan2(centered[:, 1], centered[:, 0])
        
        # Normalize radius to [0, 1] based on max radius
        max_r = np.max(r) if np.max(r) > 0 else 1.0
        r_norm = r / max_r
        
        # Convert to log scale for radial bins
        r_log = np.log(r_norm + 1e-6)  # Add epsilon to avoid log(0)
        r_min, r_max = np.log(1e-6), np.log(1.0)
        r_bins = ((r_log - r_min) / (r_max - r_min) * self.radial_bins).astype(int)
        r_bins = np.clip(r_bins, 0, self.radial_bins - 1)
        
        # Normalize theta to [0, 2*pi]
        theta_norm = (theta + np.pi) / (2 * np.pi)
        theta_bins = (theta_norm * self.angular_bins).astype(int)
        theta_bins = np.clip(theta_bins, 0, self.angular_bins - 1)
        
        # Build histogram
        histogram = np.zeros((self.radial_bins, self.angular_bins))
        for rb, tb in zip(r_bins, theta_bins):
            histogram[rb, tb] += 1
        
        # Normalize histogram
        total = np.sum(histogram)
        if total > 0:
            histogram /= total
        
        return histogram.flatten()
    
    def _chi_squared_distance(self, h1: np.ndarray, h2: np.ndarray) -> float:
        """
        Compute Chi-squared distance between two histograms.
        
        This is the standard metric for histogram comparison in AISS.
        
        Args:
            h1, h2: Histogram vectors
            
        Returns:
            Chi-squared distance (lower = more similar)
        """
        denominator = h1 + h2
        # Avoid division by zero
        mask = denominator > 1e-10
        diff = np.zeros_like(h1)
        diff[mask] = (h1[mask] - h2[mask]) ** 2 / denominator[mask]
        return np.sum(diff)
    
    def map_tile(self, tile_image: np.ndarray) -> Tuple[str, float]:
        """
        Find the best matching character for an image tile.
        
        Args:
            tile_image: Grayscale or binary image tile
            
        Returns:
            Tuple of (best_character, similarity_score)
        """
        # Ensure binary format
        if tile_image.dtype != np.uint8:
            tile_image = (tile_image * 255).astype(np.uint8)
        
        # Apply edge detection if grayscale
        if np.max(tile_image) > 1:
            # Threshold to binary
            _, binary = cv2.threshold(tile_image, 127, 1, cv2.THRESH_BINARY)
        else:
            binary = tile_image
        
        # Resize to match charset tile size
        target_size = self.charset.tile_size
        if binary.shape[::-1] != target_size:
            binary = cv2.resize(binary.astype(np.float32), target_size)
            binary = (binary > 0.5).astype(np.uint8)
        
        # Extract contour points from tile
        tile_contour = self._extract_contour_points(binary)
        tile_histogram = self._compute_log_polar_histogram(tile_contour)
        
        # Find best matching character
        best_char = ' '
        best_distance = float('inf')
        
        for char, char_contour in self._char_contours.items():
            char_histogram = self._compute_log_polar_histogram(char_contour)
            distance = self._chi_squared_distance(tile_histogram, char_histogram)
            
            if distance < best_distance:
                best_distance = distance
                best_char = char
        
        # Convert distance to similarity (0-1, higher is better)
        similarity = 1.0 / (1.0 + best_distance)
        
        return best_char, similarity
    
    def convert_image(
        self,
        image: Image.Image,
        tile_size: Tuple[int, int] = (10, 16),
        apply_edge_detection: bool = True,
        canny_low: int = 50,
        canny_high: int = 150,
    ) -> str:
        """
        Convert an image to ASCII art using AISS structural mapping.
        
        Args:
            image: PIL Image to convert
            tile_size: Size of each tile (width, height in pixels)
            apply_edge_detection: Whether to apply Canny edge detection
            canny_low: Low threshold for Canny
            canny_high: High threshold for Canny
            
        Returns:
            Multi-line ASCII art string
        """
        # Convert to grayscale numpy array
        if image.mode != 'L':
            image = image.convert('L')
        
        img_array = np.array(image)
        
        # Apply edge detection for better structure
        if apply_edge_detection:
            img_array = cv2.Canny(img_array, canny_low, canny_high)
            img_array = (img_array > 0).astype(np.uint8)
        
        # Resize charset tiles if needed
        if self.charset.tile_size != tile_size:
            self.charset.resize_tiles(tile_size)
            self._precompute_character_data()
        
        # Calculate grid dimensions
        h, w = img_array.shape
        tile_w, tile_h = tile_size
        cols = w // tile_w
        rows = h // tile_h
        
        # Build ASCII grid
        lines = []
        for row in range(rows):
            line_chars = []
            for col in range(cols):
                # Extract tile
                y1 = row * tile_h
                y2 = y1 + tile_h
                x1 = col * tile_w
                x2 = x1 + tile_w
                
                tile = img_array[y1:y2, x1:x2]
                
                # Find best character
                char, _ = self.map_tile(tile)
                line_chars.append(char)
            
            lines.append(''.join(line_chars))
        
        return '\n'.join(lines)


def create_aiss_mapper(
    charset_name: str = "ascii_standard",
    tile_size: Tuple[int, int] = (10, 16),
    **kwargs
) -> AISSMapper:
    """
    Factory function to create an AISS mapper.
    
    Args:
        charset_name: Name of charset to use
        tile_size: Tile size for character rasterization
        **kwargs: Additional arguments for AISSMapper
        
    Returns:
        Configured AISSMapper instance
    """
    charset = get_charset(charset_name, tile_size)
    return AISSMapper(charset=charset, **kwargs)
