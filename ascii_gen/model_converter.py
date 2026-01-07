"""
Model-Enhanced ASCII Converter
Uses trained ViT/ResNet models for character selection
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np
from ascii_gen.enhanced_mapper import get_enhanced_mapper


class ModelEnhancedConverter:
    """Converts images to ASCII using trained neural network models"""
    
    def __init__(self, model_path: str):
        """Initialize with a specific model (ViT or ResNet)"""
        self.mapper = get_enhanced_mapper(model_path=model_path)
        self.RAMP = " .:-=+*#%@"
    
    def convert_image(self, image: Image.Image, width: int = 80) -> str:
        """
        Convert PIL Image to ASCII art using the trained model.
        
        Args:
            image: PIL Image to convert
            width: Character width of output
            
        Returns:
            ASCII art string
        """
        # Calculate dimensions
        aspect_ratio = image.height / image.width
        height = int(width * aspect_ratio * 0.45)  # 0.45 compensates for character aspect ratio
        
        # Resize image to match character grid
        # Each character represents an 8x14 tile
        tile_width = 8
        tile_height = 14
        pixel_width = width * tile_width
        pixel_height = height * tile_height
        
        resized = image.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
        
        # Convert to grayscale for brightness calculation
        gray = resized.convert('L')
        gray_array = np.array(gray)
        
        # Generate ASCII art tile by tile
        ascii_lines = []
        
        for y in range(height):
            line = ""
            for x in range(width):
                # Extract tile
                x1 = x * tile_width
                y1 = y * tile_height
                x2 = x1 + tile_width
                y2 = y1 + tile_height
                
                # Get tile as PIL Image
                tile_img = resized.crop((x1, y1, x2, y2))
                
                # Calculate average brightness
                tile_gray = gray_array[y1:y2, x1:x2]
                brightness = np.mean(tile_gray)
                
                # Use enhanced char selection if model available
                if self.mapper.is_available():
                    char = self.mapper.enhanced_char_select(tile_img, brightness)
                else:
                    # Fallback to simple brightness mapping
                    idx = int((brightness / 255) * (len(self.RAMP) - 1))
                    char = self.RAMP[idx]
                
                line += char
            
            ascii_lines.append(line)
        
        return '\n'.join(ascii_lines)


def convert_with_model(image: Image.Image, model_path: str, width: int = 80) -> str:
    """
    Convenience function to convert an image with a specific model.
    
    Args:
        image: PIL Image to convert
        model_path: Path to model file (ViT or ResNet)
        width: Character width
        
    Returns:
        ASCII art string
    """
    converter = ModelEnhancedConverter(model_path)
    return converter.convert_image(image, width=width)
