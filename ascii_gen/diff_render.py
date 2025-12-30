"""
Differentiable ASCII Rendering Module (Experimental)
Optimizes character grid directly against CLIP loss using PyTorch.
This allows "generation from scratch" (Text-to-ASCII) without an intermediate image.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFont, ImageDraw
from typing import List, Tuple, Optional
from transformers import CLIPProcessor, CLIPModel

# Expanded charset for maximum visual Detail
CHARSET = " .'`^,:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

class DiffRenderer(nn.Module):
    def __init__(self, 
                 device: str = "cpu", 
                 font_path: str = "/System/Library/Fonts/Menlo.ttc",
                 char_size: Tuple[int, int] = (14, 28)):
        super().__init__()
        self.device = device
        self.char_size = char_size
        self.cw, self.ch = char_size
        self.chars = list(CHARSET)
        self.n_chars = len(self.chars)
        
        # Load CLIP (Local model required for gradients)
        print("⚡ Loading CLIP for Differentiable Rendering (this may download ~500MB)...")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model.eval() # Freeze CLIP weights
        
        # Precompute Font Rasters
        self.font_tensor = self._create_font_tensor(font_path).to(device) # (N, 1, H, W)

    def _create_font_tensor(self, font_path: str) -> torch.Tensor:
        """Rasterize all chars to a tensor."""
        try:
            font = ImageFont.truetype(font_path, 20)
        except:
            font = ImageFont.load_default()
            
        tensors = []
        for c in self.chars:
            img = Image.new('L', self.char_size, 0)
            draw = ImageDraw.Draw(img)
            # Center char
            bbox = draw.textbbox((0, 0), c, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            x = (self.cw - w) // 2
            y = (self.ch - h) // 2
            draw.text((x, y), c, fill=255, font=font)
            
            # Normalize to 0-1
            arr = np.array(img, dtype=np.float32) / 255.0
            tensors.append(arr)
            
        # Stack: (N, 1, H, W) for convolution convenience
        return torch.tensor(np.stack(tensors), dtype=torch.float32).unsqueeze(1)

    def forward(self, grid_logits: torch.Tensor) -> torch.Tensor:
        """
        Render generic logits to soft image.
        grid_logits: (Rows, Cols, N_Chars)
        Returns: (1, 1, H_pixels, W_pixels)
        """
        rows, cols, _ = grid_logits.shape
        
        # Softmax to get probabilities: (Rows, Cols, N)
        probs = F.softmax(grid_logits, dim=-1)
        
        # We need to construct the full image from tiles.
        # This acts like a "Transposed Convolution" or simply weighted sum of tiles.
        # Reshape probs to (Rows*Cols, N)
        probs_flat = probs.view(-1, self.n_chars)
        
        # Fonts: (N, 1, TH, TW)
        # We want: (Rows*Cols, 1, TH, TW) weighted sum
        # Helper: (Batch, N) x (N, H, W) -> (Batch, H, W)
        
        # Einsum is cleaner: b=batch(tiles), n=chars, h=tile_h, w=tile_w
        # This computes the expected tile image for each grid cell
        soft_tiles = torch.einsum('bn,nhw->bhw', probs_flat, self.font_tensor.squeeze(1)) # (Rows*Cols, TH, TW)
        
        # Reshape back to grid
        soft_tiles = soft_tiles.view(rows, cols, self.ch, self.cw)
        
        # Rearrange to image: (Rows, TH, Cols, TW) -> (Rows*TH, Cols*TW)
        soft_image = torch.cat([torch.cat([soft_tiles[r, c] for c in range(cols)], dim=1) for r in range(rows)], dim=0)
        
        return soft_image.unsqueeze(0).unsqueeze(0) # (1, 1, H, W)

    def optimize(self, prompt: str, width: int = 50, steps: int = 200) -> str:
        """
        Run optimization loop.
        """
        aspect = 0.55 # Char aspect ratio correction
        rows = int(width * aspect) # Approximate square canvas
        cols = width
        
        # Initialize logits (learnable parameters)
        # Start uniform roughly
        grid_logits = (torch.randn(rows, cols, self.n_chars, device=self.device) * 0.1).clone().detach().requires_grad_(True)
        
        optimizer = torch.optim.Adam([grid_logits], lr=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=0.01)
        
        # Get target embedding
        # Normalize prompt for consistency (CLIP handles it but good practice)
        inputs = self.clip_processor(text=[prompt], return_tensors="pt", padding=True)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs.to(self.device))
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
        print(f"🎨 Optimizing ASCII for '{prompt}' ({rows}x{cols})...")
        
        for i in range(steps):
            optimizer.zero_grad()
            
            # 1. Render Soft Image
            soft_image = self(grid_logits) # (1, 1, H, W)
            
            # 2. Augmentation / Resizing for CLIP
            # CLIP expects 224x224 RGB.
            # We have Grayscale. Replicate channels.
            img_224 = F.interpolate(soft_image, size=(224, 224), mode='bilinear')
            img_rgb = img_224.repeat(1, 3, 1, 1) # (1, 3, 224, 224)
            
            # Normalize (approximate CLIP stats)
            # CLIP uses mean=[0.481, ...], std=[0.268, ...]
            # We act on 0-1 range roughly
            img_norm = (img_rgb - 0.481) / 0.268
            
            # 3. Compute Image Embedding
            image_features = self.clip_model.get_image_features(pixel_values=img_norm)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # 4. Loss = 1 - CosineSimilarity
            loss = 1.0 - (text_features @ image_features.T).mean()
            
            # Add entropy loss to encourage sharp decisions (one char per cell)
            # Minimize entropy -> Maximize sum(p * log p)
            probs = F.softmax(grid_logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
            
            # Ramp up entropy constraint over time for cleaner final output
            entropy_weight = 0.01 * (i / steps)
            total_loss = loss + entropy_weight * entropy
            
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            
            if i % 20 == 0:
                print(f"   Step {i}/{steps} | Loss: {total_loss.item():.4f}")
                
        # Final Generation
        final_indices = torch.argmax(grid_logits, dim=-1) # (Rows, Cols)
        lines = []
        for r in range(rows):
            line = "".join([self.chars[idx] for idx in final_indices[r]])
            lines.append(line)
            
        return "\n".join(lines)

def run_diff_render_demo(prompt="a mushroom"):
    renderer = DiffRenderer()
    ascii_art = renderer.optimize(prompt)
    print("\n" + ascii_art)
    return ascii_art
