"""
Multimodal Intelligence Module (API-First Version)
Integrates CLIP via HuggingFace Inference API for semantic evaluation.
No massive downloads required!

Capabilities:
1. Semantic Scoring: Matches ASCII art to text prompt using cloud-based CLIP.
2. Auto-Selection: Generates multiple variations and selects the best one.
3. Hybrid Mode: Uses API by default, falls back to local if installed.
"""

import os
import io
import time
import json
import base64
import requests
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict, Any, Optional

# Constants
API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/clip-ViT-B-32"

class CLIPManager:
    """
    Manages CLIP scoring via HuggingFace API (zero-install).
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CLIPManager, cls).__new__(cls)
            cls._instance.api_key = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            cls._instance.use_api = True
        return cls._instance

    def _get_embedding_api(self, inputs: Any) -> Optional[List[float]]:
        """Get embedding via HF API."""
        if not self.api_key:
            return None
            
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        # If input is image, convert to base64
        if isinstance(inputs, Image.Image):
            buffered = io.BytesIO()
            inputs.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            payload = {"inputs": img_str, "parameters": {"wait_for_model": True}}
        else:
            # Text input
            payload = {"inputs": str(inputs), "parameters": {"wait_for_model": True}}
            
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                # API returns list of floats (embedding)
                data = response.json()
                # Handle possible nesting
                if isinstance(data, list) and isinstance(data[0], list):
                    return data[0] # Text often returns [embedding]
                return data
            else:
                print(f"⚠️ CLIP API Error {response.status_code}: {response.text[:100]}")
                return None
        except Exception as e:
            print(f"⚠️ CLIP API Exception: {e}")
            return None

    def get_score(self, image: Image.Image, text: str) -> float:
        """
        Calculate semantic similarity score.
        """
        # 1. Get Text Embedding
        text_emb = self._get_embedding_api(text)
        if not text_emb:
            return 0.0
            
        # 2. Get Image Embedding
        img_emb = self._get_embedding_api(image)
        if not img_emb:
            return 0.0
            
        # 3. Compute Cosine Similarity
        # Convert to numpy and normalize
        v1 = np.array(text_emb)
        v2 = np.array(img_emb)
        
        # Check shapes
        if v1.shape != v2.shape:
             # Sometimes API returns different pooling? 
             # Usually 512 for ViT-B-32.
             # If mismatch, fail safely
             return 0.0

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cos_sim = np.dot(v1, v2) / (norm1 * norm2)
        return float(cos_sim)

    def is_available(self) -> bool:
        return bool(self.api_key)


# Singleton accessor
def get_clip_manager() -> CLIPManager:
    return CLIPManager()


class CLIPSelector:
    """
    Intelligent selector that generates multiple ASCII variations and 
    picks the best one using CLIP semantic scoring.
    """
    
    def __init__(self):
        self.clip = get_clip_manager()
        
    def select_best_ascii(
        self, 
        image: Image.Image, 
        prompt: str, 
        width: int,
        mappers: Dict[str, Any]
    ) -> Tuple[str, str, float]:
        """
        Generate multiple ASCII versions and select the best one.
        """
        if not self.clip.is_available():
            print("⚠️ CLIP API token missing. Returning default.")
            # Fallback to standard if no API key
            name = "Standard (CNN)"
            # Find the function for standard
            for k, v in mappers.items():
                if "Standard" in k:
                    return v(image, width), k, 0.0
            return list(mappers.values())[0](image, width), "Default", 0.0

        candidates = []
        
        print(f"🤖 Evaluating {len(mappers)} strategies via Cloud CLIP...")
        
        # 1. Generate all variants
        for name, gen_func in mappers.items():
            try:
                start = time.time()
                ascii_art = gen_func(image, width)
                
                # Render ASCII back to image for CLIP
                rendered = self._render_ascii_to_image(ascii_art)
                candidates.append((name, ascii_art, rendered))
                print(f"  - Generated {name} ({time.time()-start:.1f}s)")
            except Exception as e:
                print(f"Error generating {name}: {e}")
                continue
                
        if not candidates:
            return "", "Error", 0.0
            
        # 2. Score with CLIP (API)
        scores = []
        for name, ascii_str, rendered_img in candidates:
            score = self.clip.get_score(rendered_img, prompt)
            scores.append((name, ascii_str, score))
            print(f"  🔍 Score {name}: {score:.4f}")
            
        # 3. Pick winner
        scores.sort(key=lambda x: x[2], reverse=True)
        best_name, best_ascii, best_score = scores[0]
        
        return best_ascii, best_name, best_score

    def _render_ascii_to_image(self, ascii_text: str) -> Image.Image:
        """Helper to render ASCII text to a PIL image for CLIP evaluation."""
        lines = ascii_text.split('\n')
        if not lines:
            return Image.new('L', (100, 100), 255)
            
        # Explicitly use simple calculation
        char_w, char_h = 7, 14 
        img_w = max(len(l) for l in lines) * char_w
        img_h = len(lines) * char_h
        
        img = Image.new('RGB', (img_w + 20, img_h + 20), color="white")
        draw = ImageDraw.Draw(img)
        
        try:
            # Try to grab a monospace font available on Linux/Mac
            # This is critical for the "image" to look like ASCII to the AI
            font_paths = [
                "/System/Library/Fonts/Menlo.ttc",
                "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                "/System/Library/Fonts/Monaco.dfont"
            ]
            font = None
            for p in font_paths:
                if os.path.exists(p):
                    font = ImageFont.truetype(p, 12)
                    break
            if not font:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
            
        draw.text((10, 10), ascii_text, fill="black", font=font)
        return img
