"""
Character Set Definitions and Utilities

Provides multiple character sets for ASCII art generation:
- ASCII_STANDARD: 95 printable ASCII characters
- ASCII_DENSE: Subset optimized for tone-based rendering
- ANSI_BLOCKS: Extended block graphics (░▒▓█)
- ANSI_LINES: Box-drawing characters
- SHIFT_JIS: Japanese character subset for rich structure

Each character can be rasterized to a binary image for structural matching.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2


# ============================================================================
# CHARACTER SET DEFINITIONS
# ============================================================================

# Standard 95 printable ASCII (0x20-0x7E)
ASCII_STANDARD = (
    " !\"#$%&'()*+,-./0123456789:;<=>?"
    "@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_"
    "`abcdefghijklmnopqrstuvwxyz{|}~"
)

# Dense characters sorted by visual density (for tone-based fallback)
ASCII_DENSE = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# Structural characters - good for edges and lines
ASCII_STRUCTURAL = " .-_=+|/\\<>()[]{}#@"

# Heavy/Bold characters for clearer boundaries and high contrast
ASCII_HEAVY = " @#%8&WM$B0OQZEX"

# ANSI block graphics (extended ASCII / Unicode)
ANSI_BLOCKS = " ░▒▓█▄▀▌▐"

# ANSI line drawing
ANSI_LINES = "╔╗╚╝║═┌┐└┘│─├┤┬┴┼"

# Combined ANSI set
ANSI_FULL = ANSI_BLOCKS + ANSI_LINES

# Japanese Shift-JIS subset (commonly used in AA)
# Including hiragana, katakana, and structural symbols
SHIFT_JIS_SUBSET = (
    "　、。・ー「」"  # Punctuation
    "あいうえおかきくけこさしすせそたちつてとなにぬねの"  # Hiragana
    "はひふへほまみむめもやゆよらりるれろわをん"
    "アイウエオカキクケコサシスセソタチツテトナニヌネノ"  # Katakana
    "ハヒフヘホマミムメモヤユヨラリルレロワヲン"
    "人口大小中上下左右"  # Kanji subset
)


@dataclass
class CharacterSet:
    """
    A character set with pre-computed rasterizations for structural matching.
    
    Attributes:
        name: Identifier for the charset
        characters: String of all characters in the set
        tile_size: Size of rasterized character images (width, height)
        rasters: Dict mapping character -> binary numpy array
        densities: Dict mapping character -> visual density (0-1)
    """
    name: str
    characters: str
    tile_size: Tuple[int, int] = (10, 16)
    rasters: Dict[str, np.ndarray] = field(default_factory=dict)
    densities: Dict[str, float] = field(default_factory=dict)
    _font: Optional[ImageFont.FreeTypeFont] = field(default=None, repr=False)
    
    def __post_init__(self):
        """Initialize rasters and densities after creation."""
        if not self.rasters:
            self._rasterize_all()
        if not self.densities:
            self._compute_densities()
    
    def _get_font(self, size: int = 12) -> ImageFont.FreeTypeFont:
        """Get a monospace font for rendering."""
        if self._font is None:
            # Try common monospace fonts
            font_names = [
                "/System/Library/Fonts/Menlo.ttc",  # macOS
                "/System/Library/Fonts/Monaco.dfont",  # macOS fallback
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",  # Linux
                "Consolas",  # Windows
            ]
            for font_name in font_names:
                try:
                    self._font = ImageFont.truetype(font_name, size)
                    break
                except (OSError, IOError):
                    continue
            
            if self._font is None:
                # Fallback to default
                self._font = ImageFont.load_default()
        
        return self._font
    
    def _rasterize_char(self, char: str) -> np.ndarray:
        """
        Render a single character to a binary numpy array.
        
        Args:
            char: Single character to render
            
        Returns:
            Binary numpy array where 1 = foreground, 0 = background
        """
        width, height = self.tile_size
        
        # Create white background image
        img = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(img)
        
        # Get font
        font = self._get_font(height - 2)
        
        # Calculate centering
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2 - bbox[0]
        y = (height - text_height) // 2 - bbox[1]
        
        # Draw character in black
        draw.text((x, y), char, fill=0, font=font)
        
        # Convert to binary numpy array (invert so foreground=1)
        arr = np.array(img)
        binary = (arr < 128).astype(np.uint8)
        
        return binary
    
    def _rasterize_all(self):
        """Pre-compute rasters for all characters."""
        self.rasters = {}
        for char in self.characters:
            self.rasters[char] = self._rasterize_char(char)
    
    def _compute_densities(self):
        """Compute visual density (foreground ratio) for each character."""
        self.densities = {}
        for char, raster in self.rasters.items():
            total_pixels = raster.size
            foreground_pixels = np.sum(raster)
            self.densities[char] = foreground_pixels / total_pixels if total_pixels > 0 else 0.0
    
    def get_raster(self, char: str) -> np.ndarray:
        """Get the binary raster for a character."""
        return self.rasters.get(char, self.rasters.get(' ', np.zeros(self.tile_size[::-1], dtype=np.uint8)))
    
    def get_density(self, char: str) -> float:
        """Get the visual density of a character (0-1)."""
        return self.densities.get(char, 0.0)
    
    def get_chars_by_density(self) -> List[str]:
        """Get characters sorted by density (lightest to darkest)."""
        return sorted(self.characters, key=lambda c: self.densities.get(c, 0.0))
    
    def find_by_density(self, target_density: float) -> str:
        """Find the character closest to a target density."""
        best_char = ' '
        best_diff = float('inf')
        
        for char, density in self.densities.items():
            diff = abs(density - target_density)
            if diff < best_diff:
                best_diff = diff
                best_char = char
        
        return best_char
    
    def get_all_rasters_as_array(self) -> Tuple[np.ndarray, List[str]]:
        """
        Get all rasters as a stacked numpy array.
        
        Returns:
            Tuple of (N x H x W array of rasters, list of corresponding characters)
        """
        chars = list(self.characters)
        rasters = np.stack([self.rasters[c] for c in chars])
        return rasters, chars
    
    def resize_tiles(self, new_size: Tuple[int, int]):
        """Resize all character rasters to a new tile size."""
        old_rasters = self.rasters.copy()
        self.tile_size = new_size
        self.rasters = {}
        
        for char, raster in old_rasters.items():
            # Resize using OpenCV
            resized = cv2.resize(
                raster.astype(np.float32), 
                new_size, 
                interpolation=cv2.INTER_AREA
            )
            self.rasters[char] = (resized > 0.5).astype(np.uint8)
        
        self._compute_densities()


# ============================================================================
# CHARSET FACTORY
# ============================================================================

_CHARSET_REGISTRY: Dict[str, CharacterSet] = {}


def get_charset(
    name: str = "ascii_standard",
    tile_size: Tuple[int, int] = (10, 16)
) -> CharacterSet:
    """
    Get a character set by name.
    
    Available charsets:
        - ascii_standard: 95 printable ASCII characters
        - ascii_dense: Characters sorted by density
        - ascii_structural: Edge/line-friendly subset
        - ansi_blocks: Block graphics (░▒▓█)
        - ansi_lines: Box drawing characters
        - ansi_full: Combined ANSI blocks + lines
        - shift_jis: Japanese character subset
    
    Args:
        name: Name of the charset
        tile_size: Size for character rasterization
        
    Returns:
        CharacterSet instance
    """
    cache_key = f"{name}_{tile_size[0]}x{tile_size[1]}"
    
    if cache_key not in _CHARSET_REGISTRY:
        char_map = {
            "ascii_standard": ASCII_STANDARD,
            "ascii_dense": ASCII_DENSE,
            "ascii_structural": ASCII_STRUCTURAL,
            "ascii_heavy": ASCII_HEAVY,
            "ansi_blocks": ANSI_BLOCKS,
            "ansi_lines": ANSI_LINES,
            "ansi_full": ANSI_FULL,
            "shift_jis": SHIFT_JIS_SUBSET,
        }
        
        if name not in char_map:
            raise ValueError(f"Unknown charset: {name}. Available: {list(char_map.keys())}")
        
        _CHARSET_REGISTRY[cache_key] = CharacterSet(
            name=name,
            characters=char_map[name],
            tile_size=tile_size
        )
    
    return _CHARSET_REGISTRY[cache_key]


def list_charsets() -> List[str]:
    """List all available charset names."""
    return [
        "ascii_standard",
        "ascii_dense", 
        "ascii_structural",
        "ascii_heavy",
        "ansi_blocks",
        "ansi_lines",
        "ansi_full",
        "shift_jis",
    ]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def visualize_charset(charset: CharacterSet, cols: int = 16) -> Image.Image:
    """
    Create a visual grid showing all characters and their rasters.
    
    Args:
        charset: CharacterSet to visualize
        cols: Number of columns in the grid
        
    Returns:
        PIL Image showing the charset
    """
    chars = list(charset.characters)
    rows = (len(chars) + cols - 1) // cols
    
    tile_w, tile_h = charset.tile_size
    padding = 2
    cell_w = tile_w + padding * 2
    cell_h = tile_h + padding * 2
    
    img = Image.new('L', (cols * cell_w, rows * cell_h), color=200)
    
    for idx, char in enumerate(chars):
        row = idx // cols
        col = idx % cols
        
        x = col * cell_w + padding
        y = row * cell_h + padding
        
        # Get raster and paste
        raster = charset.get_raster(char)
        raster_img = Image.fromarray((1 - raster) * 255)  # Invert for display
        img.paste(raster_img, (x, y))
    
    return img
