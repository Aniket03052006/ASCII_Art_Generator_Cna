# Pre-ASCII LoRA Training for FLUX.1 - Kaggle GPU Notebook
# ==============================================================================
# Implements the "Structural Alignment in Generative Diffusion" methodology
# for training a Pre-ASCII LoRA that generates grid-aligned text images.
#
# Requirements: Kaggle GPU (T4/P100) or Colab Pro (A100)
# Training time: ~2-3 hours for 500 images on A100
# ==============================================================================

# ============================================================================
# 1. SETUP & DEPENDENCIES
# ============================================================================

!pip install -q torch torchvision pillow numpy tqdm
!pip install -q diffusers transformers accelerate bitsandbytes
!pip install -q peft  # For LoRA training
!pip install -q gdown  # For Google Drive download

import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
import random
import json
import os
from tqdm.auto import tqdm
from dataclasses import dataclass
from typing import List, Tuple, Dict

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# 1.5 OPTION: DOWNLOAD 140K ASCII DATASET FROM GOOGLE DRIVE
# ============================================================================
# Instead of generating procedural data, you can use this pre-made 140K dataset

USE_PRETRAINED_DATASET = True  # Set to True to use the 140K dataset

if USE_PRETRAINED_DATASET:
    import gdown
    
    # 140K ASCII Art Dataset
    DATASET_URL = "https://drive.google.com/uc?id=162_fXfIvAq7AVfhOcs5yJzXgLFVW5VGz"
    DATASET_PATH = "ascii_dataset_140k.zip"
    
    print("📥 Downloading 140K ASCII Dataset from Google Drive...")
    gdown.download(DATASET_URL, DATASET_PATH, quiet=False)
    
    print("📦 Extracting dataset...")
    !unzip -qo {DATASET_PATH} -d ascii_lora_dataset
    
    print("✅ Dataset ready! Skipping procedural generation.")
    SKIP_GENERATION = True
else:
    SKIP_GENERATION = False

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🚀 Device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# 2. CONSTANTS - VAE STRIDE ALIGNMENT (CRITICAL!)
# ============================================================================
# FLUX.1 VAE has stride 16. We MUST align our grid to this.
# 1024 / 16 = 64 cells. Each cell is exactly 16x16 pixels.

CANVAS_SIZE = 1024      # Output resolution
GRID_DIM = 64           # 64x64 character grid
CELL_SIZE = 16          # Each character cell is 16x16 pixels
VAE_STRIDE = 16         # FLUX.1 VAE downsampling factor

# Verify alignment
assert CANVAS_SIZE == GRID_DIM * CELL_SIZE, "Grid must align with canvas!"
assert CELL_SIZE == VAE_STRIDE, "Cell size must match VAE stride!"

print(f"📐 Grid Configuration:")
print(f"   Canvas: {CANVAS_SIZE}x{CANVAS_SIZE}")
print(f"   Grid: {GRID_DIM}x{GRID_DIM}")
print(f"   Cell: {CELL_SIZE}x{CELL_SIZE} (VAE-aligned ✅)")

# ============================================================================
# 3. ASCII CHARACTER SETS FOR MAPS
# ============================================================================

# Different character sets teach topology, not just memorize '#'
CHARSETS = {
    "classic": {"wall": "#", "floor": ".", "door": "+", "corridor": ","},
    "solid": {"wall": "█", "floor": " ", "door": "▓", "corridor": "░"},
    "minimal": {"wall": "X", "floor": " ", "door": "O", "corridor": "-"},
    "box": {"wall": "■", "floor": "·", "door": "□", "corridor": "·"},
}

# ============================================================================
# 4. PROCEDURAL MAP GENERATORS
# ============================================================================

class CellularAutomataGenerator:
    """
    Generates organic cave-like structures.
    Teaches the model irregular but grid-aligned shapes.
    """
    
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
    
    def generate(self, wall_probability: float = 0.45, iterations: int = 5) -> np.ndarray:
        # Initialize random grid
        grid = np.random.random((self.height, self.width)) < wall_probability
        
        # Cellular automata smoothing
        for _ in range(iterations):
            new_grid = grid.copy()
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    # Count wall neighbors
                    neighbors = np.sum(grid[y-1:y+2, x-1:x+2]) - grid[y, x]
                    # Rule: if >= 5 neighbors are walls, become wall
                    new_grid[y, x] = neighbors >= 5
            grid = new_grid
        
        # Ensure border is walls
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
        
        return grid


class BSPDungeonGenerator:
    """
    Binary Space Partitioning for rectangular room dungeons.
    Teaches orthogonal structures and corridors.
    """
    
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
        self.min_room_size = 6
    
    def generate(self) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]]]:
        grid = np.ones((self.height, self.width), dtype=bool)  # All walls
        rooms = []
        
        # Recursive BSP
        self._split(grid, rooms, 1, 1, self.width - 2, self.height - 2, 0)
        
        # Connect rooms with corridors
        for i in range(len(rooms) - 1):
            self._connect_rooms(grid, rooms[i], rooms[i + 1])
        
        return grid, rooms
    
    def _split(self, grid, rooms, x, y, w, h, depth):
        # Base case: too deep or too small
        # Fix: Ensure partition is at least min_room_size + padding (2)
        if depth > 4 or w < self.min_room_size + 2 or h < self.min_room_size + 2:
            # Check iff we have space for a room
            if w >= self.min_room_size + 1 and h >= self.min_room_size + 1:
                try:
                    # Safe size calculation
                    max_w = min(w - 1, 12)
                    max_h = min(h - 1, 12)
                    
                    if max_w < self.min_room_size or max_h < self.min_room_size:
                         return # Too small for room

                    room_w = random.randint(self.min_room_size, max_w)
                    room_h = random.randint(self.min_room_size, max_h)
                    
                    # Safe position calculation
                    max_x = w - room_w
                    max_y = h - room_h
                    
                    if max_x <= 0 or max_y <= 0:
                         room_x, room_y = x, y
                    else:
                        room_x = x + random.randint(0, max_x)
                        room_y = y + random.randint(0, max_y)
                    
                    # Carve room (set to floor)
                    # Boundary safeguard
                    room_y2 = min(room_y+room_h, self.height-1)
                    room_x2 = min(room_x+room_w, self.width-1)
                    
                    grid[room_y:room_y2, room_x:room_x2] = False
                    rooms.append((room_x, room_y, room_w, room_h))
                except ValueError:
                    pass # Skip if random generation fails
            return
        
        # Split
        if w > h:
             # Vertical split
            split_min = self.min_room_size + 2
            split_max = w - split_min
            
            if split_max <= split_min:
                 return # Cannot split further
            
            split = x + random.randint(split_min, split_max)
            self._split(grid, rooms, x, y, split - x, h, depth + 1)
            self._split(grid, rooms, split, y, x + w - split, h, depth + 1)
        else:
            # Horizontal split
            split_min = self.min_room_size + 2
            split_max = h - split_min
            
            if split_max <= split_min:
                 return # Cannot split further

            split = y + random.randint(split_min, split_max)
            self._split(grid, rooms, x, y, w, split - y, depth + 1)
            self._split(grid, rooms, x, split, w, y + h - split, depth + 1)
    
    def _connect_rooms(self, grid, room1, room2):
        # Get room centers
        x1 = room1[0] + room1[2] // 2
        y1 = room1[1] + room1[3] // 2
        x2 = room2[0] + room2[2] // 2
        y2 = room2[1] + room2[3] // 2
        
        # L-shaped corridor
        if random.random() < 0.5:
            for x in range(min(x1, x2), max(x1, x2) + 1):
                grid[y1, x] = False
            for y in range(min(y1, y2), max(y1, y2) + 1):
                grid[y, x2] = False
        else:
            for y in range(min(y1, y2), max(y1, y2) + 1):
                grid[y, x1] = False
            for x in range(min(x1, x2), max(x1, x2) + 1):
                grid[y2, x] = False


class DrunkardWalkGenerator:
    """
    Random walk cave generation.
    Teaches winding, continuous structures.
    """
    
    def __init__(self, width: int = 64, height: int = 64):
        self.width = width
        self.height = height
    
    def generate(self, walk_length: int = 2000) -> np.ndarray:
        grid = np.ones((self.height, self.width), dtype=bool)
        
        # Start in center
        x, y = self.width // 2, self.height // 2
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for _ in range(walk_length):
            grid[y, x] = False  # Carve floor
            
            # Random direction
            dx, dy = random.choice(directions)
            nx, ny = x + dx, y + dy
            
            # Stay in bounds
            if 1 <= nx < self.width - 1 and 1 <= ny < self.height - 1:
                x, y = nx, ny
        
        # Border walls
        grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
        
        return grid


# ============================================================================
# 5. RENDERING PIPELINE (VAE-ALIGNED)
# ============================================================================

def get_square_font(size: int = 16):
    """
    Get a square monospace font for perfect grid alignment.
    Falls back to default if Square font not available.
    """
    font_paths = [
        "Square.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    
    for path in font_paths:
        try:
            return ImageFont.truetype(path, size)
        except:
            continue
    
    return ImageFont.load_default()


def render_map_to_image(
    grid: np.ndarray,
    charset: Dict[str, str],
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    fg_color: Tuple[int, int, int] = (255, 255, 255),
    supersampling: int = 2  # Render at 2x then downsample for anti-aliasing
) -> Image.Image:
    """
    Render a boolean grid to a VAE-aligned image.
    
    Args:
        grid: 64x64 boolean array (True = wall)
        charset: Character mapping
        bg_color: Background RGB
        fg_color: Foreground RGB
        supersampling: Render at Nx resolution for smoother edges
    """
    render_size = CANVAS_SIZE * supersampling
    cell_size = CELL_SIZE * supersampling
    
    # Create image
    img = Image.new('RGB', (render_size, render_size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # Load font
    font = get_square_font(cell_size)
    
    # Render each cell
    for y in range(GRID_DIM):
        for x in range(GRID_DIM):
            is_wall = grid[y, x]
            char = charset["wall"] if is_wall else charset["floor"]
            
            # Exact pixel position
            px = x * cell_size
            py = y * cell_size
            
            draw.text((px, py), char, font=font, fill=fg_color)
    
    # Downsample with LANCZOS for slight edge softening (helps VAE)
    if supersampling > 1:
        img = img.resize((CANVAS_SIZE, CANVAS_SIZE), Image.LANCZOS)
    
    return img


# ============================================================================
# 6. DATASET GENERATION
# ============================================================================

@dataclass
class DatasetConfig:
    output_dir: str = "ascii_lora_dataset"
    num_samples: int = 800
    generators: List[str] = None  # ["cellular", "bsp", "drunkard"]
    
    def __post_init__(self):
        if self.generators is None:
            self.generators = ["cellular", "bsp", "drunkard"]


def generate_caption(generator_type: str, charset_name: str, style: str) -> str:
    """
    Generate diverse, descriptive captions for LoRA training.
    Uses the trigger word 'ascii_structure_style'.
    """
    layouts = {
        "cellular": [
            "organic cave system with irregular walls",
            "natural cavern with winding passages",
            "eroded stone dungeon with organic shapes"
        ],
        "bsp": [
            "rectangular dungeon with connected rooms",
            "structured maze with orthogonal corridors",
            "classic roguelike dungeon layout"
        ],
        "drunkard": [
            "winding corridor system",
            "tortuous cave passages",
            "narrow meandering dungeon paths"
        ]
    }
    
    styles = {
        "white_on_black": "white ascii text on pure black background, high contrast",
        "black_on_white": "black ascii text on white paper background, inverted",
        "amber": "amber terminal text on dark background, CRT monitor style",
        "green": "green matrix-style text on black background, phosphor glow"
    }
    
    layout_desc = random.choice(layouts[generator_type])
    style_desc = styles.get(style, styles["white_on_black"])
    
    caption = f"ascii_structure_style, {layout_desc}, {style_desc}, 2d top down view, monospaced font, distinct characters, 64x64 grid"
    
    return caption


def generate_dataset(config: DatasetConfig):
    """
    Generate the complete training dataset.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    (output_dir / "images").mkdir(exist_ok=True)
    
    # Initialize generators
    generators = {
        "cellular": CellularAutomataGenerator(),
        "bsp": BSPDungeonGenerator(),
        "drunkard": DrunkardWalkGenerator()
    }
    
    # Color schemes
    color_schemes = {
        "white_on_black": ((0, 0, 0), (255, 255, 255)),
        "black_on_white": ((255, 255, 255), (0, 0, 0)),
        "amber": ((0, 0, 0), (255, 176, 0)),
        "green": ((0, 0, 0), (0, 255, 65)),
    }
    
    metadata = []
    
    print(f"🎨 Generating {config.num_samples} training samples...")
    
    for i in tqdm(range(config.num_samples)):
        # Random selections
        gen_type = random.choice(config.generators)
        charset_name = random.choice(list(CHARSETS.keys()))
        style = random.choice(list(color_schemes.keys()))
        
        # Generate map
        generator = generators[gen_type]
        if gen_type == "bsp":
            grid, _ = generator.generate()
        else:
            grid = generator.generate()
        
        # Render image
        bg_color, fg_color = color_schemes[style]
        img = render_map_to_image(
            grid, 
            CHARSETS[charset_name],
            bg_color, 
            fg_color,
            supersampling=2
        )
        
        # Generate caption
        caption = generate_caption(gen_type, charset_name, style)
        
        # Save
        img_path = output_dir / "images" / f"{i:06d}.png"
        txt_path = output_dir / "images" / f"{i:06d}.txt"
        
        img.save(img_path, "PNG")
        txt_path.write_text(caption)
        
        metadata.append({
            "image": str(img_path),
            "caption": caption,
            "generator": gen_type,
            "charset": charset_name,
            "style": style
        })
    
    # Save metadata
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Dataset generated: {output_dir}")
    print(f"   Images: {config.num_samples}")
    print(f"   Generators: {config.generators}")
    
    return output_dir


# ============================================================================
# 7. TRAINING CONFIGURATION (AI-Toolkit Compatible)
# ============================================================================

TRAINING_CONFIG = """
# FLUX.1 Pre-ASCII LoRA Configuration
# For use with ostris/ai-toolkit

model:
  name_or_path: "black-forest-labs/FLUX.1-dev"
  is_flux: true
  quantize: true  # Set false if using H100/A100 80GB

lora:
  linear: 64      # High rank for structural detail
  linear_alpha: 32 # Stabilized scaling (alpha/rank = 0.5)

dataset:
  folder_path: "/kaggle/working/ascii_lora_dataset/images"
  resolution: 1024
  center_crop: false  # CRITICAL: Never crop grid-aligned data
  caption_ext: "txt"
  batch_size: 1
  shuffle: true

train:
  steps: 3000
  gradient_accumulation_steps: 4
  learning_rate: 4e-4
  optimizer: "adamw8bit"
  lr_scheduler: "constant"
  mixed_precision: "bf16"
  gradient_checkpointing: true
  save_every: 500

triggers:
  - "ascii_structure_style"
"""


# ============================================================================
# 8. QUALITY METRICS (Grid Adherence, OCR, Structure)
# ============================================================================

def calculate_grid_adherence(image: Image.Image) -> float:
    """
    FFT analysis to detect 16px grid harmonics.
    High score = strong grid structure.
    """
    import numpy as np
    
    img_array = np.array(image.convert('L'), dtype=np.float32)
    
    # 2D FFT
    fft = np.fft.fft2(img_array)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    
    h, w = img_array.shape
    center_x, center_y = w // 2, h // 2
    
    # Expected frequency for 16px grid
    expected_freq = CANVAS_SIZE // CELL_SIZE  # 64
    
    # Sample harmonics
    harmonics = []
    for harmonic in [1, 2, 3]:
        freq_offset = expected_freq * harmonic
        if center_x + freq_offset < w:
            harmonics.append(magnitude[center_y, center_x + freq_offset])
        if center_y + freq_offset < h:
            harmonics.append(magnitude[center_y + freq_offset, center_x])
    
    if harmonics:
        harmonic_strength = np.mean(harmonics)
        noise_floor = np.median(magnitude)
        ratio = harmonic_strength / (noise_floor + 1e-6)
        return min(1.0, ratio / 10.0)
    
    return 0.0


def validate_structural_integrity(grid: np.ndarray) -> Dict:
    """
    Check topology: closed loops, connected regions.
    """
    from scipy import ndimage
    
    # Find floor regions
    floors = ~grid
    labeled, num_regions = ndimage.label(floors)
    
    # Count cells in each region
    region_sizes = ndimage.sum(floors, labeled, range(1, num_regions + 1))
    
    # Good dungeon has 1 large connected floor region
    if len(region_sizes) > 0:
        largest_region = max(region_sizes) if region_sizes.size > 0 else 0
        total_floor = np.sum(floors)
        connectivity = largest_region / total_floor if total_floor > 0 else 0
    else:
        connectivity = 0
    
    return {
        "num_regions": num_regions,
        "connectivity_score": connectivity,
        "floor_percentage": np.mean(floors) * 100
    }


# ============================================================================
# 9. MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("🎮 Pre-ASCII LoRA Dataset Generator for FLUX.1")
    print("=" * 60)
    print()
    
    # Generate dataset
    if 'SKIP_GENERATION' in globals() and SKIP_GENERATION:
        print("⏭️ Skipping procedural generation (Dataset already downloaded).")
        dataset_path = Path("ascii_lora_dataset")
    else:
        config = DatasetConfig(
            output_dir="ascii_lora_dataset",
            num_samples=800,
            generators=["cellular", "bsp", "drunkard"]
        )
        dataset_path = generate_dataset(config)
    
    # Save training config
    config_path = dataset_path / "train_config.yaml"
    config_path.write_text(TRAINING_CONFIG)
    print(f"📄 Training config saved: {config_path}")
    
    # Validate a few samples
    print("\n📊 Validating samples...")
    from pathlib import Path
    import random
    
    sample_images = list((dataset_path / "images").glob("*.png"))[:5]
    
    for img_path in sample_images:
        img = Image.open(img_path)
        grid_score = calculate_grid_adherence(img)
        print(f"   {img_path.name}: Grid Adherence = {grid_score:.2f}")
    
    print("\n" + "=" * 60)
    print("✅ Dataset generation complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Upload dataset to cloud GPU (A100 recommended)")
    print("2. Install ai-toolkit: git clone https://github.com/ostris/ai-toolkit")
    print("3. Run: python run.py train_config.yaml")
    print("4. Download ascii_structure_v1.safetensors")
    print("5. Use in ComfyUI with trigger: 'ascii_structure_style'")


# ============================================================================
# 10. RUN TRAINING (AI-TOOLKIT)
# ============================================================================
# Only run this if we are in the main execution block
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🏋️ STARTING TRAINING PIPELINE")
    print("=" * 60)
    
    import os
    import subprocess
    import sys

    # Clone AI-Toolkit if not present
    if not os.path.exists("ai-toolkit"):
        print("🚀 Cloning AI-Toolkit...")
        try:
            subprocess.run(["git", "clone", "https://github.com/ostris/ai-toolkit"], check=True)
            subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd="ai-toolkit", check=True)
            print("📦 Installing dependencies (this may take a minute)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "oyaml", "-r", "ai-toolkit/requirements.txt"], check=True)
        except Exception as e:
            print(f"❌ Error installing AI-Toolkit: {e}")
            print("try running: !git clone https://github.com/ostris/ai-toolkit && pip install -r ai-toolkit/requirements.txt")

    # Run Training
    print("\n🚀 LAUNCHING TRAINING...")
    print("   Config: ascii_lora_dataset/train_config.yaml")
    print("   (This will take 2-3 hours on T4 GPU)")
    
    try:
        # Run the training script
        # We use sys.executable to ensure we use the same python environment
        cmd = [sys.executable, "ai-toolkit/run.py", "ascii_lora_dataset/train_config.yaml"]
        subprocess.run(cmd, check=True)
        print("\n✅ Training Complete!")
        print("💾 Output saved to: ai-toolkit/output/")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("Check the logs above for details.")

    # ============================================================================
    # 11. EXPORT & DOWNLOAD
    # ============================================================================
    print("\n" + "=" * 60)
    print("📦 PREPARING DOWNLOAD")
    print("=" * 60)
    
    import shutil
    import glob
    
    # Find the safetensors file in the output directory
    # Usually in: ai-toolkit/output/[name]/[name].safetensors
    output_files = glob.glob("ai-toolkit/output/**/*.safetensors", recursive=True)
    
    if output_files:
        latest_file = max(output_files, key=os.path.getctime)
        dataset_name = "ascii_140k" if USE_PRETRAINED_DATASET else "ascii_procedural"
        final_filename = f"flux_lora_{dataset_name}.safetensors"
        
        print(f"✅ Found model: {latest_file}")
        
        # Move to root for easy download
        shutil.copy(latest_file, final_filename)
        
        print("\n🎉 SUCCESS! Your model is ready.")
        print(f"👉 FILE TO DOWNLOAD: {final_filename}")
        print("(Look in the 'Output' tab on the right sidebar)")
    else:
        print("⚠️ No .safetensors file found. Training might have failed.")
