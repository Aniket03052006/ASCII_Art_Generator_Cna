"""
CNN-based ASCII Art Mapper

A modern PyTorch implementation inspired by DeepAA architecture.
Uses a lightweight CNN to classify image tiles into ASCII characters.

Key features:
- PyTorch-based (works with MPS on Apple Silicon)
- Pre-trains on character rasters with augmentation
- Higher quality than Random Forest for complex images
- Slower inference but better visual results
"""

from typing import Optional, Tuple, List
import numpy as np
from PIL import Image
import warnings

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. CNN mapper will not work.")

from .charsets import CharacterSet, get_charset


class ASCIIConvNet(nn.Module):
    """
    Lightweight CNN for character classification.
    
    Architecture inspired by DeepAA but modernized:
    - 3 conv layers with batch norm
    - Global average pooling
    - Single FC layer to character classes
    """
    
    def __init__(self, num_classes: int, input_size: Tuple[int, int] = (14, 8)):
        super().__init__()
        
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Global average pooling -> FC
        self.fc = nn.Linear(128, num_classes)
        
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Conv block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        # Conv block 3
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class CNNMapper:
    """
    CNN-based ASCII art mapper.
    
    Higher quality than Random Forest but slower.
    Best for detailed/complex images.
    
    Example:
        >>> mapper = CNNMapper(charset="ascii_standard")
        >>> mapper.train()
        >>> ascii_art = mapper.convert_image(image)
    """
    
    def __init__(
        self,
        charset: str = "ascii_standard",
        tile_size: Tuple[int, int] = (8, 14),
        device: str = "auto",
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for CNN mapper")
        
        self.tile_size = tile_size  # (width, height)
        self.input_size = (tile_size[1], tile_size[0])  # (H, W) for PyTorch
        
        # Load charset
        if isinstance(charset, str):
            self._charset = get_charset(charset, tile_size)
        else:
            self._charset = charset
        
        self.char_to_idx = {c: i for i, c in enumerate(self._charset.characters)}
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self.num_classes = len(self._charset.characters)
        
        # Device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Model
        self.model = ASCIIConvNet(self.num_classes, self.input_size)
        self.model.to(self.device)
        
        self._is_trained = False
    
    def _augment_batch(self, images: np.ndarray, n_augments: int = 5) -> np.ndarray:
        """Apply augmentation to training data."""
        augmented = [images]
        
        for _ in range(n_augments):
            # Random noise
            noisy = images + np.random.normal(0, 0.1, images.shape)
            noisy = np.clip(noisy, 0, 1)
            augmented.append(noisy)
            
            # Random brightness
            bright = images * np.random.uniform(0.8, 1.2)
            bright = np.clip(bright, 0, 1)
            augmented.append(bright)
        
        return np.concatenate(augmented, axis=0)
    
    def train(self, epochs: int = 30, batch_size: int = 64, verbose: bool = True):
        """Train the CNN on character rasters."""
        if verbose:
            print(f"Training CNN on {self.num_classes} characters...")
        
        # Prepare training data from character rasters
        X = []
        y = []
        
        for char in self._charset.characters:
            raster = self._charset.get_raster(char)
            # Normalize to [0, 1]
            raster_norm = raster.astype(np.float32)
            X.append(raster_norm)
            y.append(self.char_to_idx[char])
        
        X = np.array(X)
        y = np.array(y)
        
        # Augment: _augment_batch creates 1 + n_augments*2 copies
        n_augments = 5
        X_aug = self._augment_batch(X, n_augments=n_augments)
        num_copies = 1 + n_augments * 2  # 1 original + 5 noise + 5 brightness = 11
        y_aug = np.tile(y, num_copies)
        
        # Add channel dimension: (N, H, W) -> (N, 1, H, W)
        X_aug = X_aug[:, np.newaxis, :, :]
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_aug).to(self.device)
        y_tensor = torch.LongTensor(y_aug).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(batch_y).sum().item()
                total += batch_y.size(0)
            
            if verbose and (epoch + 1) % 10 == 0:
                acc = 100. * correct / total
                print(f"  Epoch {epoch+1}/{epochs}: Loss={total_loss/len(loader):.4f}, Acc={acc:.1f}%")
        
        self._is_trained = True
        if verbose:
            print(f"✅ CNN training complete! Final accuracy: {100.*correct/total:.1f}%")
    
    def _preprocess_tile(self, tile: np.ndarray) -> torch.Tensor:
        """Preprocess a tile for inference."""
        import cv2
        
        # Resize to expected size
        tile_resized = cv2.resize(tile, self.tile_size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        tile_norm = tile_resized.astype(np.float32) / 255.0
        
        # Add batch and channel dims
        tile_tensor = torch.FloatTensor(tile_norm).unsqueeze(0).unsqueeze(0)
        
        return tile_tensor.to(self.device)
    
    def predict_char(self, tile: np.ndarray) -> str:
        """Predict the best character for a tile."""
        if not self._is_trained:
            raise RuntimeError("Model not trained. Call train() first.")
        
        self.model.eval()
        with torch.no_grad():
            tile_tensor = self._preprocess_tile(tile)
            output = self.model(tile_tensor)
            _, predicted = output.max(1)
            idx = predicted.item()
        
        return self.idx_to_char[idx]
    
    def convert_image(
        self,
        image: Image.Image,
        tile_size: Optional[Tuple[int, int]] = None,
        apply_edge_detection: bool = True,
    ) -> str:
        """Convert an image to ASCII art."""
        import cv2
        
        if not self._is_trained:
            self.train()
        
        tile_size = tile_size or self.tile_size
        tile_w, tile_h = tile_size
        
        # Convert to grayscale
        if image.mode != 'L':
            image = image.convert('L')
        
        img_array = np.array(image)
        
        # Optional edge detection
        if apply_edge_detection:
            img_array = cv2.Canny(img_array, 50, 150)
        
        height, width = img_array.shape
        
        # Calculate grid dimensions
        cols = width // tile_w
        rows = height // tile_h
        
        # Process tiles
        lines = []
        self.model.eval()
        
        with torch.no_grad():
            for row in range(rows):
                line_chars = []
                for col in range(cols):
                    # Extract tile
                    y1, y2 = row * tile_h, (row + 1) * tile_h
                    x1, x2 = col * tile_w, (col + 1) * tile_w
                    tile = img_array[y1:y2, x1:x2]
                    
                    # Predict
                    char = self.predict_char(tile)
                    line_chars.append(char)
                
                lines.append(''.join(line_chars))
        
        return '\n'.join(lines)
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'model_state': self.model.state_dict(),
            'char_to_idx': self.char_to_idx,
            'tile_size': self.tile_size,
        }, path)
        print(f"✅ Model saved to {path}")
    
    def load_model(self, path: str):
        """Load pre-trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.char_to_idx = checkpoint['char_to_idx']
        self.idx_to_char = {i: c for c, i in self.char_to_idx.items()}
        self._is_trained = True
        print(f"✅ Model loaded from {path}")


def create_cnn_mapper(
    charset: str = "ascii_standard",
    tile_size: Tuple[int, int] = (8, 14),
    train: bool = True,
    model_path: Optional[str] = None,
) -> CNNMapper:
    """
    Factory function to create a CNN mapper.
    
    Args:
        charset: Character set name
        tile_size: Tile dimensions (width, height)
        train: Whether to train on creation
        model_path: Path to pre-trained model
        
    Returns:
        Configured CNNMapper
    """
    mapper = CNNMapper(charset=charset, tile_size=tile_size)
    
    if model_path:
        mapper.load_model(model_path)
    elif train:
        mapper.train()
    
    return mapper
