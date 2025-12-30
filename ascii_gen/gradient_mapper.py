"""
Gradient-Based ASCII Mapper v1
High-quality ASCII art generation using brightness-to-character density mapping.

Key Features:
1. Extended character ramps (70+ characters sorted by density)
2. Floyd-Steinberg dithering for smooth gradients
3. Edge enhancement for crisp contours
4. Multiple preset ramps (detailed, standard, minimal)
5. Block-based averaging for cleaner output
"""

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
from typing import Optional, Tuple, List
from dataclasses import dataclass
import cv2


# =============================================================================
# CHARACTER RAMPS - Sorted by visual density (dark to light)
# =============================================================================

# Ultra-detailed ramp - 70 characters for maximum gradient smoothness
RAMP_ULTRA = "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "

# Detailed ramp - 40 characters for good gradients
RAMP_DETAILED = "@%#*+=-:. $@B%8&WM#oahkbdpwmZO0QCJYXzvunxrjft/|()1[]?-_~<>i!l;:,^`"

# Standard ramp - 16 characters (good balance)
RAMP_STANDARD = "@%#*+=-:. "

# Minimal ramp - 10 characters for stylized look
RAMP_MINIMAL = "@#S%?*+;:. "

# Block ramp - Uses block characters for density
RAMP_BLOCKS = "█▓▒░ "

# Structural ramp - Emphasizes edges and structure
RAMP_STRUCTURAL = "@#$%&8BMW*oahkbd|/\\(){}[]<>+=-~:;,.`' "

# Custom high-contrast ramp optimized for ASCII art
RAMP_CONTRAST = "@@@@@%%%%####****++++====----::::....    "


# Neat ramp - High contrast, structural characters only (user request)
RAMP_NEAT = " @#|/\\()<>[] "  # Prioritize space, solid blocks, and strong lines

@dataclass
class GradientConfig:
    """Configuration for gradient-based ASCII conversion."""
    ramp: str = RAMP_STANDARD
    width: int = 80                    # Output width in characters
    contrast: float = 1.5              # Contrast enhancement (1.0 = none)
    brightness: float = 1.0            # Brightness adjustment
    sharpness: float = 1.5             # Edge sharpening
    dither: bool = True                # Enable Floyd-Steinberg dithering
    invert: bool = False               # Invert brightness (for dark backgrounds)
    invert_ramp: bool = False          # Invert ramp mapping (white->dense instead of white->sparse)
    edge_enhance: bool = True          # Enhance edges before conversion
    gamma: float = 1.0                 # Gamma correction (< 1 = brighter midtones)
    block_size: Tuple[int, int] = (2, 4)  # Pixels per character (width, height)
    aspect_ratio: float = 0.5          # Character aspect ratio correction (height/width of char)


class GradientMapper:
    """
    High-quality ASCII art generator using brightness-to-character mapping.
    
    This mapper uses visual density of characters to create smooth gradients,
    producing more photorealistic ASCII art compared to edge-based methods.
    """
    
    def __init__(self, config: Optional[GradientConfig] = None):
        self.config = config or GradientConfig()
        self._precompute_ramp()
    
    def _precompute_ramp(self):
        """Precompute character lookup for fast conversion."""
        ramp = self.config.ramp
        self.ramp_length = len(ramp)
        self.char_array = np.array(list(ramp))
    
    def set_ramp(self, ramp: str):
        """Change the character ramp."""
        self.config.ramp = ramp
        self._precompute_ramp()
    
    def _preprocess(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to enhance image for ASCII conversion."""
        # Convert to grayscale if needed
        if image.mode != 'L':
            image = image.convert('L')
        
        # Apply contrast enhancement
        if self.config.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(self.config.contrast)
        
        # Apply brightness adjustment
        if self.config.brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(self.config.brightness)
        
        # Apply sharpening
        if self.config.sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(self.config.sharpness)
        
        # Apply gamma correction
        if self.config.gamma != 1.0:
            img_array = np.array(image, dtype=np.float32) / 255.0
            img_array = np.power(img_array, self.config.gamma)
            image = Image.fromarray((img_array * 255).astype(np.uint8))
        
        # Apply histogram equalization for better local contrast (emphasis)
        img_array = np.array(image)
        img_eq = cv2.equalizeHist(img_array)
        # Blend original with equalized for balanced emphasis
        img_array = cv2.addWeighted(img_array, 0.6, img_eq, 0.4, 0)
        image = Image.fromarray(img_array)
        
        # Apply edge enhancement
        if self.config.edge_enhance:
            # Blend original with edge-enhanced version
            edges = image.filter(ImageFilter.EDGE_ENHANCE_MORE)
            image = Image.blend(image, edges, alpha=0.3)
        
        # Invert if needed (for dark background display)
        if self.config.invert:
            image = ImageOps.invert(image)
        
        return image
    
    def _resize_for_ascii(self, image: Image.Image) -> Image.Image:
        """
        Resize image to match ASCII output dimensions.
        
        Uses aspect_ratio config to correct for character dimensions.
        Default 0.5 means chars are ~2x taller than wide.
        """
        target_width = self.config.width
        
        # Calculate output height using aspect ratio correction
        img_aspect = image.height / image.width
        char_aspect = self.config.aspect_ratio  # How much vertical compression
        
        # Output dimensions in characters
        out_height = int(target_width * img_aspect * char_aspect)
        
        # Calculate pixel dimensions for processing
        block_w, block_h = self.config.block_size
        new_width = target_width * block_w
        new_height = out_height * block_h
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def _apply_dithering(self, image: np.ndarray) -> np.ndarray:
        """
        Apply Floyd-Steinberg dithering for smoother gradients.
        
        This distributes quantization error to neighboring pixels,
        creating the illusion of more gray levels.
        """
        if not self.config.dither:
            return image
        
        img = image.astype(np.float32)
        height, width = img.shape
        
        # Number of levels in our ramp
        levels = self.ramp_length
        
        for y in range(height):
            for x in range(width):
                old_pixel = img[y, x]
                
                # Quantize to nearest ramp level
                new_pixel = round(old_pixel / 255 * (levels - 1)) * 255 / (levels - 1)
                img[y, x] = new_pixel
                
                # Calculate and distribute error
                error = old_pixel - new_pixel
                
                if x + 1 < width:
                    img[y, x + 1] += error * 7 / 16
                if y + 1 < height:
                    if x > 0:
                        img[y + 1, x - 1] += error * 3 / 16
                    img[y + 1, x] += error * 5 / 16
                    if x + 1 < width:
                        img[y + 1, x + 1] += error * 1 / 16
        
        return np.clip(img, 0, 255).astype(np.uint8)
    
    def _block_average(self, image: np.ndarray) -> np.ndarray:
        """
        Average pixels into blocks for character mapping.
        
        Each block will become one ASCII character.
        """
        block_w, block_h = self.config.block_size
        height, width = image.shape
        
        # Calculate output dimensions
        out_h = height // block_h
        out_w = width // block_w
        
        # Reshape into blocks and average
        blocks = image[:out_h * block_h, :out_w * block_w]
        blocks = blocks.reshape(out_h, block_h, out_w, block_w)
        averaged = blocks.mean(axis=(1, 3))
        
        return averaged.astype(np.uint8)
    
    def _map_brightness_to_char(self, brightness_map: np.ndarray) -> np.ndarray:
        """
        Map brightness values to ASCII characters.
        
        Default: Bright pixels (white) map to sparse chars (spaces).
                 Dark pixels (black) map to dense chars ($, @, etc).
        If invert_ramp: Inverted mapping (white->dense, black->sparse).
        """
        # Force near-white pixels to pure white (clean background)
        white_threshold = 220
        clean_brightness = np.where(brightness_map > white_threshold, 255, brightness_map)
        
        # Map brightness to ramp index
        if self.config.invert_ramp:
            # Inverted: HIGH brightness = LOW index = DARK char
            indices = ((255 - clean_brightness) / 255 * (self.ramp_length - 1)).astype(int)
        else:
            # Default: HIGH brightness = HIGH index = LIGHT char (space)
            indices = (clean_brightness / 255 * (self.ramp_length - 1)).astype(int)
        
        indices = np.clip(indices, 0, self.ramp_length - 1)
        
        # Map to characters
        char_map = self.char_array[indices]
        
        return char_map
    
    def convert(self, image: Image.Image) -> str:
        """
        Convert image to high-quality ASCII art.
        
        Args:
            image: PIL Image to convert
            
        Returns:
            ASCII art string
        """
        # Step 1: Preprocess
        processed = self._preprocess(image)
        
        # Step 2: Resize to ASCII dimensions
        resized = self._resize_for_ascii(processed)
        
        # Step 3: Convert to numpy array
        img_array = np.array(resized)
        
        # Step 4: Apply dithering (optional)
        if self.config.dither:
            img_array = self._apply_dithering(img_array)
        
        # Step 5: Average into blocks
        block_brightness = self._block_average(img_array)
        
        # Step 6: Map to characters
        char_map = self._map_brightness_to_char(block_brightness)
        
        # Step 7: Build output string
        lines = [''.join(row) for row in char_map]
        return '\n'.join(lines)
    
    def convert_with_edges(self, image: Image.Image, edge_weight: float = 0.5) -> str:
        """
        Convert with edge detection blended in for crisp contours.
        
        Args:
            image: PIL Image to convert
            edge_weight: How much to blend edges (0-1)
            
        Returns:
            ASCII art string with enhanced edges
        """
        # Convert to grayscale
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image.copy()
        
        # Detect edges with Canny
        img_array = np.array(gray)
        edges = cv2.Canny(img_array, 50, 150)
        
        # Dilate edges slightly for visibility
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Invert edges (edges should be dark/dense in ASCII)
        edges_inverted = 255 - edges
        
        # Blend with original (preprocessed) image
        processed = self._preprocess(gray)
        processed_array = np.array(processed)
        
        blended = (processed_array * (1 - edge_weight) + edges_inverted * edge_weight).astype(np.uint8)
        blended_img = Image.fromarray(blended)
        
        # Continue with normal conversion
        resized = self._resize_for_ascii(blended_img)
        img_array = np.array(resized)
        
        if self.config.dither:
            img_array = self._apply_dithering(img_array)
        
        block_brightness = self._block_average(img_array)
        char_map = self._map_brightness_to_char(block_brightness)
        
        lines = [''.join(row) for row in char_map]
        return '\n'.join(lines)


# =============================================================================
# Preset Configurations
# =============================================================================

def create_ultra_detailed_mapper(width: int = 100) -> GradientMapper:
    """Create mapper for ultra-detailed output (best for large displays)."""
    config = GradientConfig(
        ramp=RAMP_ULTRA,
        width=width,
        contrast=1.4,
        sharpness=1.3,
        dither=True,
        edge_enhance=True,
        block_size=(1, 2),  # Higher resolution
    )
    return GradientMapper(config)


def create_standard_mapper(width: int = 80) -> GradientMapper:
    """Create mapper for standard output (good balance)."""
    config = GradientConfig(
        ramp=RAMP_STANDARD,
        width=width,
        contrast=1.5,
        sharpness=1.5,
        dither=True,
        edge_enhance=True,
    )
    return GradientMapper(config)


def create_minimal_mapper(width: int = 60) -> GradientMapper:
    """Create mapper for minimal/stylized output."""
    config = GradientConfig(
        ramp=RAMP_MINIMAL,
        width=width,
        contrast=2.0,
        sharpness=2.0,
        dither=False,
        edge_enhance=True,
    )
    return GradientMapper(config)


def create_neat_mapper(width: int = 80) -> GradientMapper:
    """Create mapper for neat/structural output (user request)."""
    config = GradientConfig(
        ramp=RAMP_NEAT,
        width=width,
        contrast=2.5,          # High contrast for clean look
        sharpness=2.0,         # High sharpness for edges
        dither=False,          # No dithering for cleaner look
        edge_enhance=True,     
        invert=False
    )
    return GradientMapper(config)


def create_portrait_mapper(width: int = 120) -> GradientMapper:
    """Create mapper optimized for portraits (balanced contrast, high detail)."""
    config = GradientConfig(
        ramp=RAMP_ULTRA,       # Max gray levels for smooth shading
        width=width,
        contrast=1.2,          # Gentle contrast to keep nose/cheek details
        sharpness=1.3,         # Moderate sharpness
        dither=True,           # Dithering essential for skin tones
        edge_enhance=True,
    )
    return GradientMapper(config)


def create_block_mapper(width: int = 80) -> GradientMapper:
    """Create mapper using block characters."""
    config = GradientConfig(
        ramp=RAMP_BLOCKS,
        width=width,
        contrast=1.3,
        dither=True,
        edge_enhance=False,
    )
    return GradientMapper(config)


# =============================================================================
# Convenience Functions
# =============================================================================

def image_to_gradient_ascii(
    image: Image.Image,
    width: int = 80,
    ramp: str = "standard",
    with_edges: bool = True,
    edge_weight: float = 0.3,
    invert_ramp: bool = False,
) -> str:
    """
    Convert image to high-quality gradient ASCII art.
    
    Args:
        image: PIL Image
        width: Output width in characters
        ramp: Ramp preset ("ultra", "standard", "minimal", "blocks", "neat", "portrait")
        with_edges: Blend in edge detection for crisp contours
        edge_weight: How much to blend edges (0-1)
        invert_ramp: If True, inverts the brightness mapping (for dark BG style)
        
    Returns:
        ASCII art string
    """
    # Select ramp
    ramps = {
        "ultra": RAMP_ULTRA,
        "detailed": RAMP_DETAILED,
        "standard": RAMP_STANDARD,
        "minimal": RAMP_MINIMAL,
        "blocks": RAMP_BLOCKS,
        "structural": RAMP_STRUCTURAL,
        "neat": RAMP_NEAT,
        "portrait": RAMP_ULTRA, # Portrait uses Ultra ramp
    }
    selected_ramp = ramps.get(ramp, RAMP_STANDARD)
    
    # Defaults base on mode
    contrast = 1.5
    dither = True
    
    if ramp == "neat":
        contrast = 2.5
        dither = False
        if edge_weight < 0.5: edge_weight = 0.7
    elif ramp == "portrait":
        contrast = 1.2  # Lower contrast for portraits to keep details
        dither = True
        if edge_weight > 0.3: edge_weight = 0.2 # Lower edge weight for portraits
    
    # Create config
    config = GradientConfig(
        ramp=selected_ramp,
        width=width,
        contrast=contrast,
        sharpness=1.3,
        dither=dither,
        edge_enhance=True,
        invert_ramp=invert_ramp,
    )
    
    mapper = GradientMapper(config)
    
    if with_edges:
        return mapper.convert_with_edges(image, edge_weight)
    else:
        return mapper.convert(image)


# =============================================================================
# CLI Test
# =============================================================================

if __name__ == "__main__":
    from PIL import Image, ImageDraw
    
    print("=" * 80)
    print("GRADIENT MAPPER TEST")
    print("=" * 80)
    
    # Create test image (gradient circle)
    size = 400
    img = Image.new('L', (size, size), color=255)
    draw = ImageDraw.Draw(img)
    
    # Draw concentric circles with gradient
    for i in range(20):
        radius = size // 2 - i * 10
        if radius > 0:
            gray = int(255 * (i / 20))
            draw.ellipse(
                [size//2 - radius, size//2 - radius, size//2 + radius, size//2 + radius],
                outline=gray,
                width=8
            )
    
    # Test with different presets
    for preset_name, preset_fn in [
        ("Ultra Detailed", create_ultra_detailed_mapper),
        ("Standard", create_standard_mapper),
        ("Minimal", create_minimal_mapper),
    ]:
        print(f"\n--- {preset_name} ---")
        mapper = preset_fn(width=60)
        result = mapper.convert(img)
        print(result[:500])  # Show first part
        print(f"... ({len(result)} total chars)")
