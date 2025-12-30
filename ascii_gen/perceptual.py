"""
Perceptual ASCII Mapping Module
Implements structural similarity (SSIM) based mapping for high-fidelity ASCII art.
"Research Grade" implementation - slower but preserves structure better than brightness.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity as ssim
import cv2
from typing import List, Tuple, Dict, Optional
import time

# Use the extended structural charset
CHARSET_STRUCTURAL = " .'`^,:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"


class SSIMMapper:
    """
    Mappers pixels to characters by maximizing Structural Similarity (SSIM).
    This preserves local shapes (lines, curves, edges) much better than density.
    """
    
    def __init__(self, width: int = 80, tile_size: Tuple[int, int] = (10, 20), charset: Optional[str] = None):
        self.width = width
        self.tile_size = tile_size
        self.charset = charset if charset else CHARSET_STRUCTURAL  # Dynamic charset support
        self.char_rasters: Dict[str, np.ndarray] = {}
        self.char_densities: Dict[str, float] = {}
        
        # Pre-compute rasters
        self._rasterize_charset()
        
    def _rasterize_charset(self):
        """Render all characters to numpy arrays."""
        try:
            # Try to load a good monospace font
            font_paths = [
                "/System/Library/Fonts/Menlo.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/System/Library/Fonts/Monaco.dfont",
            ]
            font = None
            for p in font_paths:
                try:
                    font = ImageFont.truetype(p, 18)
                    break
                except:
                    continue
            
            if not font:
                font = ImageFont.load_default()
        except:
             font = ImageFont.load_default()
             
        w, h = self.tile_size
        
        for char in self.charset:
            img = Image.new('L', (w, h), color=0) # Black background
            draw = ImageDraw.Draw(img)
            
            # Center text
            bbox = draw.textbbox((0, 0), char, font=font)
            cw = bbox[2] - bbox[0]
            ch = bbox[3] - bbox[1]
            x = (w - cw) // 2
            y = (h - ch) // 2
            
            draw.text((x, y), char, fill=255, font=font) # White text
            
            # Store as normalized float array
            arr = np.array(img, dtype=np.float32) / 255.0
            self.char_rasters[char] = arr
            self.char_densities[char] = np.mean(arr)
            
        # Sort characters by density for optimization
        self.sorted_chars = sorted(self.charset, key=lambda c: self.char_densities[c])

    def convert_image(self, image: Image.Image) -> str:
        """
        Convert image using vectorized SSIM optimization.
        "Research Grade" speed: ~0.5s instead of 200s.
        """
        # 1. Resize image
        tw, th = self.tile_size
        target_w = self.width * tw
        aspect = image.height / image.width
        grid_h = int(self.width * aspect * (tw / th))
        target_h = grid_h * th
        
        img_resized = image.resize((target_w, target_h), Image.Resampling.LANCZOS).convert('L')
        img_arr = np.array(img_resized, dtype=np.float32) / 255.0
        img_arr = 1.0 - img_arr # Invert to match white-on-black rasters
        
        # 2. Prepare Chars Tensor
        chars = list(self.char_rasters.keys())
        rasters = np.array([self.char_rasters[c] for c in chars]) # (N, H, W)
        
        # Precompute char stats (N,)
        mu_x = rasters.mean(axis=(1, 2))
        sig_x_sq = rasters.var(axis=(1, 2))
        
        # Constants for SSIM
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        lines = []
        print(f"⚡ Fast SSIM Mapping {self.width}x{grid_h}...")
        
        # Optim: Process row by row
        for r in range(grid_h):
            # Extract row of tiles: (W, H_tile, W_tile)
            y1, y2 = r * th, (r + 1) * th
            row_slice = img_arr[y1:y2, :]
            
            # Split into tiles
            tiles = []
            for c in range(self.width):
                tiles.append(row_slice[:, c*tw:(c+1)*tw])
            tiles = np.array(tiles) # (Width, H_tile, W_tile)
            
            # Compute Tile Stats (Width,)
            mu_y = tiles.mean(axis=(1, 2))
            sig_y_sq = tiles.var(axis=(1, 2))
            
            # Vectorized Covariance: (Width, N)
            # Cov(X, Y) = E[XY] - E[X]E[Y]
            # We need E[XY] for every pair of (tile, char)
            # tiles: (W, H, W_tile), rasters: (N, H, W_tile)
            # Reshape for dot product:
            # tiles_flat: (W, H*W_tile)
            # rasters_flat: (N, H*W_tile)
            
            tiles_flat = tiles.reshape(self.width, -1)
            rasters_flat = rasters.reshape(len(chars), -1)
            
            # E[XY] via matrix multiplication: (W, N)
            mean_xy = (tiles_flat @ rasters_flat.T) / (tw * th)
            
            # Broadcast subtract means
            # mu_x: (N,), mu_y: (W,) -> (W, N)
            sigma_xy = mean_xy - np.outer(mu_y, mu_x)
            
            # Compute SSIM (W, N)
            # numerator1 = 2 * mu_x * mu_y + C1
            num1 = 2 * np.outer(mu_y, mu_x) + C1
            # numerator2 = 2 * sigma_xy + C2
            num2 = 2 * sigma_xy + C2
            # denom1 = mu_x^2 + mu_y^2 + C1
            den1 = np.outer(mu_y**2, np.ones(len(chars))) + np.outer(np.ones(self.width), mu_x**2) + C1
            # denom2 = sigma_x^2 + sigma_y^2 + C2
            den2 = np.outer(sig_y_sq, np.ones(len(chars))) + np.outer(np.ones(self.width), sig_x_sq) + C2
            
            ssim_map = (num1 * num2) / (den1 * den2)
            
            # Find best match for each tile in row
            best_indices = np.argmax(ssim_map, axis=1)
            row_chars = [chars[idx] for idx in best_indices]
            
            lines.append(''.join(row_chars))
            
        return '\n'.join(lines)


def create_ssim_mapper(width: int = 80, charset: Optional[str] = None) -> SSIMMapper:
    return SSIMMapper(width=width, charset=charset)
