"""
Pre-ASCII Dataset Generator for FLUX.1 LoRA Training.

Why this is superior to web scraping:
1. HARMONIC ALIGNMENT: Generates characters exactly 16x16 pixels.
   FLUX.1 VAE has a stride of 16 (or 8). 
   1024px / 16px = 64 cols.
   Web images often have fractional widths (e.g. 12px chars) causing VAE aliasing.

2. PURE SIGNAL: No JPG artifacts. Binary black/white (or controlled CRT colors).

3. TOPOLOGY: Algorithms teach specific structures (Caves vs Rooms).
"""

import os
import random
import numpy as np
from PIL import Image, ImageDraw

# =============================================================================
# 1. HARDCODED 16x16 BITMAPS (No external fonts needed)
# =============================================================================

# '#' - Wall (Dense Block)
CHAR_WALL = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0],
    [0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0],
    [0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0],
    [0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0],
    [0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0],
    [0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0],
    [0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0],
    [0,1,0,1,1,0,0,1,1,0,0,1,1,0,1,0],
    [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0],
    [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
], dtype=np.uint8)

# '.' - Floor (Small dot)
CHAR_FLOOR = np.zeros((16, 16), dtype=np.uint8)
CHAR_FLOOR[7:9, 7:9] = 1

# ' ' - Void
CHAR_VOID = np.zeros((16, 16), dtype=np.uint8)

# '+' - Door
CHAR_DOOR = np.zeros((16, 16), dtype=np.uint8)
CHAR_DOOR[2:14, 7:9] = 1 # Vertical
CHAR_DOOR[7:9, 2:14] = 1 # Horizontal


# =============================================================================
# 2. GENERATOR ALGORITHMS
# =============================================================================

class StructuralGenerator:
    def __init__(self, size=64):
        self.size = size  # 64x64 chars = 1024x1024 pixels
        
    def generate_caves(self, smoothing_steps=5):
        """Cellular Automata for organic caves."""
        # Random init (45% walls)
        grid = np.random.choice(['#', '.'], size=(self.size, self.size), p=[0.45, 0.55])
        
        for _ in range(smoothing_steps):
            new_grid = grid.copy()
            for y in range(1, self.size-1):
                for x in range(1, self.size-1):
                    # Count wall neighbors
                    neighbors = grid[y-1:y+2, x-1:x+2]
                    wall_count = np.sum(neighbors == '#')
                    
                    if wall_count >= 5:
                        new_grid[y, x] = '#'
                    else:
                        new_grid[y, x] = '.'
            grid = new_grid
            
        return grid, "organic cave system, cellular automata, chaotic layout"

    def generate_bsp_rooms(self):
        """Binary Space Partitioning for rectangular rooms."""
        grid = np.full((self.size, self.size), '#')
        
        rooms = []
        def split_space(x, y, w, h):
            if w < 20 or h < 20: # Leaf node (room)
                # Ensure space is big enough for a room
                if w < 6 or h < 6: return
                
                # Create room with margin
                # room width must be at least 3, max w-2
                max_rw = w - 2
                max_rh = h - 2
                
                if max_rw < 3 or max_rh < 3: return
                
                rw = random.randint(3, max_rw)
                rh = random.randint(3, max_rh)
                
                rx = x + random.randint(1, w - rw - 1)
                ry = y + random.randint(1, h - rh - 1)
                
                # Check bounds again just in case
                if rx+rw < self.size and ry+rh < self.size:
                    grid[ry:ry+rh, rx:rx+rw] = '.'
                    rooms.append((rx, ry, rw, rh))
                return

            # Split
            # Ensure split leaves enough space for both sides (min 10)
            if random.random() > 0.5 and w > 20: # Vertical split
                split = random.randint(w//3, 2*w//3)
                if split < 10 or (w-split) < 10: 
                    split = w // 2 # Forced center split if random fails
                    
                split_space(x, y, split, h)
                split_space(x+split, y, w-split, h)
            elif h > 20: # Horizontal split
                split = random.randint(h//3, 2*h//3)
                if split < 10 or (h-split) < 10:
                    split = h // 2
                    
                split_space(x, y, w, split)
                split_space(x, y+split, w, h-split)
            else:
                 # Cannot split further, treat as leaf
                 split_space(x, y, w, h)


        split_space(0, 0, self.size, self.size)
        
        # Connect rooms simply (center to center)
        for i in range(len(rooms)-1):
            r1 = rooms[i]
            r2 = rooms[i+1]
            c1 = (r1[0] + r1[2]//2, r1[1] + r1[3]//2)
            c2 = (r2[0] + r2[2]//2, r2[1] + r2[3]//2)
            
            # Draw L-shaped corridor
            # Horizontal (x1 -> x2) at y1
            x_start, x_end = sorted([c1[0], c2[0]])
            grid[c1[1], x_start:x_end+1] = '.'
            
            # Vertical (y1 -> y2) at x2
            y_start, y_end = sorted([c1[1], c2[1]])
            grid[y_start:y_end+1, c2[0]] = '.'
            
        return grid, "rectangular rooms, dungeon map, bsp tree structure"


# =============================================================================
# 3. RENDERER
# =============================================================================

def render_grid(grid, output_path):
    """Render 64x64 grid to 1024x1024 image using 16x16 bitmaps."""
    h, w = grid.shape
    img_h, img_w = h * 16, w * 16
    
    # Create canvas (Black bg)
    canvas = np.zeros((img_h, img_w), dtype=np.uint8)
    
    for y in range(h):
        for x in range(w):
            char = grid[y, x]
            bitmap = CHAR_VOID
            if char == '#': bitmap = CHAR_WALL
            elif char == '.': bitmap = CHAR_FLOOR
            elif char == '+': bitmap = CHAR_DOOR
            
            # Paste bitmap (Y, X) -> (Pixel Y, Pixel X)
            py, px = y*16, x*16
            canvas[py:py+16, px:px+16] = bitmap * 255
            
    # Convert to PIL
    image = Image.fromarray(canvas, mode='L')
    
    # Save
    image.save(output_path)
    print(f"Generated: {output_path}")

# =============================================================================
# MAIN EXECUTABLE
# =============================================================================

if __name__ == "__main__":
    output_dir = "ascii_training_data"
    os.makedirs(output_dir, exist_ok=True)
    
    gen = StructuralGenerator(size=64)
    
    print("🏭 Manufacturing Pre-ASCII Dataset...")
    
    # Generate 5 Caves
    for i in range(5):
        grid, style = gen.generate_caves()
        render_grid(grid, f"{output_dir}/cave_{i}.png")
        # Save caption
        with open(f"{output_dir}/cave_{i}.txt", "w") as f:
            f.write(f"ascii_structure_style, {style}, high contrast, 16px grid")
            
    # Generate 5 Dungeons
    for i in range(5):
        grid, style = gen.generate_bsp_rooms()
        render_grid(grid, f"{output_dir}/dungeon_{i}.png")
        with open(f"{output_dir}/dungeon_{i}.txt", "w") as f:
            f.write(f"ascii_structure_style, {style}, high contrast, 16px grid")

    print(f"✅ Created 10 sample images in '{output_dir}/'")
    print("   Resolution: 1024x1024")
    print("   Alignment: Perfect 16x16 pixel grid")
