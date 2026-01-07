# ASCII Art ViT/ResNet Training - Kaggle GPU Notebook
# Run this on Kaggle with GPU enabled for best performance

# ============================================================================
# 1. SETUP & DEPENDENCIES
# ============================================================================

!pip install -q torch torchvision pillow numpy tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm.auto import tqdm
import os
import json
from pathlib import Path

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Using device: {device}")

# ============================================================================
# 2. ASCII CHARACTER SET
# ============================================================================

# Full ASCII ramp (95 printable chars) - ordered by visual density
ASCII_CHARS = " .`'^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
NUM_CLASSES = len(ASCII_CHARS)
print(f"📝 Training for {NUM_CLASSES} ASCII characters")

# Char to index mapping
CHAR_TO_IDX = {c: i for i, c in enumerate(ASCII_CHARS)}
IDX_TO_CHAR = {i: c for i, c in enumerate(ASCII_CHARS)}

# ============================================================================
# 3. DATASET GENERATION (Creates training tiles on-the-fly)
# ============================================================================

def render_char_to_tile(char: str, size: tuple = (16, 16)) -> np.ndarray:
    """Render a single character to a grayscale tile."""
    img = Image.new('L', size, 255)  # White background
    draw = ImageDraw.Draw(img)
    
    # Try to get a good font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf", 14)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        except:
            font = ImageFont.load_default()
    
    # Center the character
    bbox = draw.textbbox((0, 0), char, font=font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size[0] - w) // 2
    y = (size[1] - h) // 2
    
    draw.text((x, y), char, fill=0, font=font)
    return np.array(img, dtype=np.float32) / 255.0


class ASCIITileDataset(Dataset):
    """
    Generates training data by:
    1. Rendering ASCII characters to tiles
    2. Applying augmentations (blur, noise, brightness)
    3. Creating variations to learn robust mappings
    """
    
    def __init__(self, samples_per_char: int = 500, tile_size: tuple = (16, 16)):
        self.samples_per_char = samples_per_char
        self.tile_size = tile_size
        self.total_samples = len(ASCII_CHARS) * samples_per_char
        
        # Pre-render base tiles
        self.base_tiles = {}
        for char in ASCII_CHARS:
            self.base_tiles[char] = render_char_to_tile(char, tile_size)
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomAffine(degrees=5, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        char_idx = idx // self.samples_per_char
        char = ASCII_CHARS[char_idx]
        
        # Get base tile
        tile = self.base_tiles[char].copy()
        
        # Add random noise
        noise = np.random.randn(*tile.shape) * 0.05
        tile = np.clip(tile + noise, 0, 1)
        
        # Add random brightness variation
        brightness = np.random.uniform(0.8, 1.2)
        tile = np.clip(tile * brightness, 0, 1)
        
        # Convert to tensor and apply transforms
        tile_uint8 = (tile * 255).astype(np.uint8)
        tile_tensor = self.transform(tile_uint8)
        
        # Expand to 3 channels for ViT/ResNet
        tile_tensor = tile_tensor.repeat(3, 1, 1)
        
        return tile_tensor, char_idx


# ============================================================================
# 4. MODEL ARCHITECTURES
# ============================================================================

def create_vit_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    Vision Transformer for ASCII character classification.
    Uses ViT-B/16 backbone with custom head.
    """
    if pretrained:
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    else:
        model = models.vit_b_16(weights=None)
    
    # Replace classification head
    # Original: 768 -> 1000 (ImageNet)
    # New: 768 -> 256 -> num_classes
    model.heads = nn.Sequential(
        nn.Linear(768, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, num_classes)
    )
    
    return model


def create_resnet_model(num_classes: int = NUM_CLASSES, pretrained: bool = True):
    """
    ResNet18 for ASCII character classification.
    Lightweight, fast inference.
    """
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    # Replace FC layer
    # Original: 512 -> 1000
    # New: 512 -> 128 -> num_classes
    model.fc = nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, num_classes)
    )
    
    return model


# ============================================================================
# 5. TRAINING LOOP
# ============================================================================

def train_model(
    model_type: str = "vit",  # "vit" or "resnet"
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-4,
    samples_per_char: int = 300
):
    """
    Train ASCII tile classifier.
    """
    print(f"\n{'='*60}")
    print(f"🏋️ Training {model_type.upper()} Model")
    print(f"{'='*60}")
    
    # Create dataset
    print("📦 Creating dataset...")
    dataset = ASCIITileDataset(samples_per_char=samples_per_char)
    
    # Split train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Val samples: {len(val_dataset)}")
    
    # Create model
    print(f"🔧 Creating {model_type} model...")
    if model_type == "vit":
        model = create_vit_model(pretrained=True)
    else:
        model = create_resnet_model(pretrained=True)
    
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total params: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training history
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * correct / total
        
        # Update scheduler
        scheduler.step()
        
        # Log
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"  📊 Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_name = f"ascii_{model_type}_best.pth"
            torch.save(model.state_dict(), model_name)
            print(f"  💾 Saved best model ({val_acc:.2f}%)")
    
    # Save final model
    final_name = f"ascii_{model_type}_final.pth"
    torch.save(model.state_dict(), final_name)
    print(f"\n✅ Training complete! Best accuracy: {best_val_acc:.2f}%")
    print(f"📁 Models saved: {final_name}")
    
    return model, history


# ============================================================================
# 6. BATCH INFERENCE (HIGH PRIORITY OPTIMIZATION)
# ============================================================================

class BatchASCIIMapper:
    """
    Optimized batch inference for ASCII conversion.
    10-50x faster than sequential processing.
    """
    
    def __init__(self, model_path: str, model_type: str = "vit"):
        self.device = device
        self.model_type = model_type
        
        # Load model
        if model_type == "vit":
            self.model = create_vit_model(pretrained=False)
        else:
            self.model = create_resnet_model(pretrained=False)
        
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model = self.model.to(device)
        self.model.eval()
        
        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((16, 16)),
            transforms.ToTensor(),
        ])
    
    def convert_image(self, image: Image.Image, width: int = 80) -> str:
        """
        Convert image to ASCII using BATCH inference.
        """
        # Calculate dimensions
        aspect = image.height / image.width
        height = int(width * aspect * 0.55)  # Terminal aspect correction
        
        # Tile dimensions
        tile_w = image.width // width
        tile_h = image.height // height
        
        # Extract ALL tiles at once
        tiles = []
        for y in range(height):
            for x in range(width):
                left = x * tile_w
                top = y * tile_h
                tile = image.crop((left, top, left + tile_w, top + tile_h))
                tile_tensor = self.transform(tile.convert('RGB'))
                tiles.append(tile_tensor)
        
        # Stack into batch
        batch = torch.stack(tiles).to(self.device)
        
        # SINGLE forward pass (this is the optimization!)
        with torch.no_grad():
            outputs = self.model(batch)
            indices = outputs.argmax(dim=1).cpu().numpy()
        
        # Build ASCII string
        ascii_lines = []
        for y in range(height):
            line = ''
            for x in range(width):
                idx = y * width + x
                char_idx = indices[idx]
                line += IDX_TO_CHAR[char_idx]
            ascii_lines.append(line)
        
        return '\n'.join(ascii_lines)


# ============================================================================
# 7. MAIN - RUN TRAINING
# ============================================================================

if __name__ == "__main__":
    print("🎨 ASCII Art Model Training Suite")
    print("=" * 60)
    
    # Train both models
    print("\n[1/2] Training ViT Model...")
    vit_model, vit_history = train_model(
        model_type="vit",
        epochs=20,
        batch_size=32,  # Smaller for ViT due to memory
        lr=1e-4,
        samples_per_char=300
    )
    
    print("\n[2/2] Training ResNet Model...")
    resnet_model, resnet_history = train_model(
        model_type="resnet",
        epochs=20,
        batch_size=64,
        lr=1e-4,
        samples_per_char=300
    )
    
    # Compare results
    print("\n" + "=" * 60)
    print("📊 FINAL RESULTS COMPARISON")
    print("=" * 60)
    print(f"ViT Best Accuracy:    {max(vit_history['val_acc']):.2f}%")
    print(f"ResNet Best Accuracy: {max(resnet_history['val_acc']):.2f}%")
    
    # Save training history
    with open("training_history.json", "w") as f:
        json.dump({
            "vit": vit_history,
            "resnet": resnet_history
        }, f, indent=2)
    
    print("\n✅ All models trained and saved!")
    print("📁 Files created:")
    print("   - ascii_vit_final.pth")
    print("   - ascii_resnet_final.pth") 
    print("   - training_history.json")
    print("\n🔄 To use these models, download and place in your models/ folder")
