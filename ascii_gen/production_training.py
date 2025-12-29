"""
Production-Ready Enhanced Training

Key fixes from previous version:
1. Training data now matches inference pipeline (edge-style tiles)
2. Better character-to-tile matching based on edge density
3. Improved RF with more estimators and better features
"""

from typing import Optional, Tuple, List
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from torch.optim.lr_scheduler import CosineAnnealingLR
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.ensemble import RandomForestClassifier
    from skimage.feature import hog
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import cv2
from .charsets import CharacterSet, get_charset


class EdgeStyleDataGenerator:
    """
    Generate training data that matches edge-detection inference.
    
    Instead of training on clean font renders, we:
    1. Render characters
    2. Apply edge detection to match inference pipeline
    3. This aligns training and inference data distributions
    """
    
    FONTS = [
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.dfont",
        "/System/Library/Fonts/Courier.dfont",
    ]
    
    def __init__(self, characters: List[str], tile_size: Tuple[int, int] = (8, 14)):
        self.characters = characters
        self.tile_size = tile_size
        self.fonts = [f for f in self.FONTS if os.path.exists(f)]
        if not self.fonts:
            self.fonts = [None]
    
    def _render_and_edge(self, char: str, font_path: Optional[str]) -> np.ndarray:
        """Render character and apply edge detection like inference does."""
        w, h = self.tile_size
        
        # Render at 2x for better quality
        img = Image.new('L', (w * 2, h * 2), color=255)
        draw = ImageDraw.Draw(img)
        
        try:
            font = ImageFont.truetype(font_path, h * 2 - 4) if font_path else ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        bbox = draw.textbbox((0, 0), char, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (w * 2 - text_w) // 2
        y = (h * 2 - text_h) // 2
        draw.text((x, y), char, fill=0, font=font)
        
        # Convert to numpy and resize
        img_array = np.array(img)
        img_resized = cv2.resize(img_array, self.tile_size, interpolation=cv2.INTER_AREA)
        
        # Apply Canny edge detection (same as inference)
        edges = cv2.Canny(img_resized, 50, 150)
        
        return edges
    
    def _augment(self, img: np.ndarray, n: int = 15) -> List[np.ndarray]:
        """Apply augmentations that preserve edge structure."""
        results = [img.copy()]
        h, w = img.shape
        
        for _ in range(n):
            aug = img.copy().astype(np.float32)
            aug_type = np.random.randint(0, 5)
            
            if aug_type == 0:
                # Small translation
                dx, dy = np.random.randint(-1, 2), np.random.randint(-1, 2)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aug = cv2.warpAffine(aug.astype(np.uint8), M, (w, h))
            elif aug_type == 1:
                # Small rotation
                angle = np.random.uniform(-5, 5)
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
                aug = cv2.warpAffine(aug.astype(np.uint8), M, (w, h))
            elif aug_type == 2:
                # Dilation (thicken edges)
                kernel = np.ones((2, 2), np.uint8)
                aug = cv2.dilate(aug.astype(np.uint8), kernel, iterations=1)
            elif aug_type == 3:
                # Erosion (thin edges)
                kernel = np.ones((2, 2), np.uint8)
                aug = cv2.erode(aug.astype(np.uint8), kernel, iterations=1)
            elif aug_type == 4:
                # Add sparse noise
                noise_mask = np.random.random(aug.shape) < 0.05
                aug[noise_mask] = np.random.choice([0, 255], size=noise_mask.sum())
            
            results.append(aug.astype(np.float32))
        
        return results
    
    def generate(self, augments: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """Generate edge-style training dataset."""
        X, y = [], []
        
        for idx, char in enumerate(self.characters):
            for font in self.fonts:
                base = self._render_and_edge(char, font)
                for aug in self._augment(base, augments):
                    X.append(aug / 255.0)  # Normalize
                    y.append(idx)
        
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)


class ProductionCNN(nn.Module):
    """Optimized CNN for edge-based character classification."""
    
    def __init__(self, num_classes: int):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


class ProductionCNNMapper:
    """Production-ready CNN mapper with edge-aligned training."""
    
    def __init__(self, charset: str = "ascii_structural", tile_size: Tuple[int, int] = (8, 14)):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required")
        
        self.tile_size = tile_size
        self._charset = get_charset(charset, tile_size) if isinstance(charset, str) else charset
        self.char_to_idx = {c: i for i, c in enumerate(self._charset.characters)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        
        self.model = ProductionCNN(len(self._charset.characters)).to(self.device)
        self._trained = False
    
    def train(self, epochs: int = 80, augments: int = 25):
        """Train on edge-aligned data."""
        print(f"Training Production CNN ({sum(p.numel() for p in self.model.parameters()):,} params)...")
        
        gen = EdgeStyleDataGenerator(list(self._charset.characters), self.tile_size)
        X, y = gen.generate(augments)
        
        X_t = torch.FloatTensor(X[:, None]).to(self.device)
        y_t = torch.LongTensor(y).to(self.device)
        
        loader = DataLoader(TensorDataset(X_t, y_t), batch_size=64, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        sched = CosineAnnealingLR(opt, epochs)
        
        self.model.train()
        for ep in range(epochs):
            correct = total = 0
            for bx, by in loader:
                opt.zero_grad()
                loss = F.cross_entropy(self.model(bx), by)
                loss.backward()
                opt.step()
                correct += (self.model(bx).argmax(1) == by).sum().item()
                total += by.size(0)
            sched.step()
            if (ep + 1) % 20 == 0:
                print(f"  Epoch {ep+1}/{epochs}: Acc={100*correct/total:.1f}%")
        
        self._trained = True
        print(f"✅ Training complete!")
    
    def convert_image(self, image: Image.Image, tile_size=None, apply_edge_detection=True) -> str:
        if not self._trained:
            self.train()
        
        tile_size = tile_size or self.tile_size
        tw, th = tile_size
        
        img = np.array(image.convert('L'))
        if apply_edge_detection:
            img = cv2.Canny(img, 50, 150)
        
        h, w = img.shape
        lines = []
        self.model.eval()
        
        with torch.no_grad():
            for r in range(h // th):
                row = []
                for c in range(w // tw):
                    tile = img[r*th:(r+1)*th, c*tw:(c+1)*tw]
                    tile = cv2.resize(tile, self.tile_size) / 255.0
                    t = torch.FloatTensor(tile).unsqueeze(0).unsqueeze(0).to(self.device)
                    idx = self.model(t).argmax(1).item()
                    row.append(self.idx_to_char[idx])
                lines.append(''.join(row))
        
        return '\n'.join(lines)
    
    def save(self, path: str):
        torch.save({'model': self.model.state_dict(), 'chars': self.char_to_idx}, path)
    
    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model'])
        self.char_to_idx = ckpt['chars']
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self._trained = True


class ProductionRFMapper:
    """Enhanced Random Forest with better features."""
    
    def __init__(self, charset: str = "ascii_structural", tile_size: Tuple[int, int] = (8, 14)):
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("sklearn required")
        
        self.tile_size = tile_size
        self._charset = get_charset(charset, tile_size) if isinstance(charset, str) else charset
        self.char_to_idx = {c: i for i, c in enumerate(self._charset.characters)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        
        # More estimators for better accuracy
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=3,
            n_jobs=-1,
            random_state=42,
        )
        self._trained = False
    
    def _extract_features(self, img: np.ndarray) -> np.ndarray:
        """Extract HoG + density features."""
        img_resized = cv2.resize(img, self.tile_size)
        
        # HoG features
        hog_feat = hog(
            img_resized,
            orientations=8,
            pixels_per_cell=(4, 4),
            cells_per_block=(1, 1),
            feature_vector=True,
        )
        
        # Density features (edge presence in quadrants)
        h, w = img_resized.shape
        quadrants = [
            img_resized[:h//2, :w//2],
            img_resized[:h//2, w//2:],
            img_resized[h//2:, :w//2],
            img_resized[h//2:, w//2:],
        ]
        density = [np.mean(q > 0) for q in quadrants]
        
        return np.concatenate([hog_feat, density])
    
    def train(self, augments: int = 20):
        """Train on edge-aligned data."""
        print(f"Training Production RF (200 trees)...")
        
        gen = EdgeStyleDataGenerator(list(self._charset.characters), self.tile_size)
        X_img, y = gen.generate(augments)
        
        # Extract features
        X = np.array([self._extract_features((img * 255).astype(np.uint8)) for img in X_img])
        
        self.model.fit(X, y)
        acc = 100 * self.model.score(X, y)
        print(f"✅ RF Training complete! Accuracy: {acc:.1f}%")
        self._trained = True
    
    def convert_image(self, image: Image.Image, tile_size=None, apply_edge_detection=True) -> str:
        if not self._trained:
            self.train()
        
        tile_size = tile_size or self.tile_size
        tw, th = tile_size
        
        img = np.array(image.convert('L'))
        if apply_edge_detection:
            img = cv2.Canny(img, 50, 150)
        
        h, w = img.shape
        lines = []
        
        for r in range(h // th):
            row = []
            for c in range(w // tw):
                tile = img[r*th:(r+1)*th, c*tw:(c+1)*tw]
                feat = self._extract_features(tile).reshape(1, -1)
                idx = self.model.predict(feat)[0]
                row.append(self.idx_to_char[idx])
            lines.append(''.join(row))
        
        return '\n'.join(lines)
    
    def save(self, path: str):
        joblib.dump({'model': self.model, 'chars': self.char_to_idx}, path)
    
    def load(self, path: str):
        data = joblib.load(path)
        self.model = data['model']
        self.char_to_idx = data['chars']
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self._trained = True


def compare_all_mappers(image_path: str = "test_input.png"):
    """Compare all mappers on the same image."""
    from PIL import Image
    import time
    
    img = Image.open(image_path)
    
    print("=" * 60)
    print("PRODUCTION MAPPER COMPARISON")
    print("=" * 60)
    
    # Production CNN
    print("\n🧠 PRODUCTION CNN")
    print("-" * 40)
    cnn = ProductionCNNMapper()
    start = time.time()
    cnn.train(epochs=80, augments=25)
    result_cnn = cnn.convert_image(img, apply_edge_detection=True)
    cnn_time = time.time() - start
    print(result_cnn)
    print(f"Time: {cnn_time:.1f}s")
    cnn.save("models/production_cnn.pth")
    
    # Production RF
    print("\n🌲 PRODUCTION RF")
    print("-" * 40)
    rf = ProductionRFMapper()
    start = time.time()
    rf.train(augments=25)
    result_rf = rf.convert_image(img, apply_edge_detection=True)
    rf_time = time.time() - start
    print(result_rf)
    print(f"Time: {rf_time:.1f}s")
    rf.save("models/production_rf.joblib")
    
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    print(f"CNN: {cnn_time:.1f}s")
    print(f"RF:  {rf_time:.1f}s")


if __name__ == "__main__":
    compare_all_mappers()
