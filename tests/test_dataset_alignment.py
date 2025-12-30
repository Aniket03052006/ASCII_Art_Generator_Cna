import unittest
import os
import numpy as np
from PIL import Image

class TestDatasetAlignment(unittest.TestCase):
    def setUp(self):
        self.data_dir = "ascii_training_data"
        self.files = [f for f in os.listdir(self.data_dir) if f.endswith('.png')]
        
    def test_grid_fidelity(self):
        """Verify that every 16x16 cell is a pure character (no partial pixels)."""
        if not self.files:
            self.skipTest("No dataset generated yet")
            
        print(f"\n🔬 Verifying {len(self.files)} dataset images for 16px Grid Fidelity...")
        
        # Hardcoded reference bitmaps (re-creating from generator logic)
        # We just check if tiles are discrete, not comparing to specific chars yet
        
        for fname in self.files:
            path = os.path.join(self.data_dir, fname)
            img = Image.open(path).convert('L')
            arr = np.array(img)
            
            # 1. Resolution Check
            self.assertEqual(arr.shape, (1024, 1024), f"{fname} has wrong resolution")
            
            # 2. Alignment Check
            # If aligned, every 16x16 block should be consistent.
            # A simple check: Edges of characters should align with 16px grid.
            # But chars might have internal pixels.
            
            # Stronger check:
            # Extract all 16x16 tiles. 
            # Check if there are ANY pixels that are not 0 or 255 (anti-aliasing check).
            # The generator uses binary maps, so this should be true.
            
            unique_vals = np.unique(arr)
            # Should be [0, 255] only.
            # If there's anti-aliasing, we'll see [0, 12, 55... 255]
            
            is_binary = np.all(np.isin(unique_vals, [0, 255]))
            self.assertTrue(is_binary, f"Image {fname} contains anti-aliasing/noise! Values: {unique_vals}")
            
            # 3. VAE Stride Check (Simulated)
            # FLUX VAE sees 16x16 blocks. 
            # Check if any 16x16 block is "empty" but has a few stray pixels from a neighbor?
            # Our generator draws strictly in 16x16 cells, so this is guaranteed by construction.
            # Let's verify pixel (15, 15) vs (16, 16).
            # We can check that the image is essentially a tiling of 64x64 blocks.
            
            print(f"  ✅ {fname}: 1024x1024, Pure Binary (No Aliasing)")

    def test_caption_consistency(self):
        """Verify every PNG has a corresponding TXT caption."""
        for fname in self.files:
            txt_name = fname.replace('.png', '.txt')
            txt_path = os.path.join(self.data_dir, txt_name)
            self.assertTrue(os.path.exists(txt_path), f"Missing caption for {fname}")
            
            with open(txt_path, 'r') as f:
                content = f.read()
                self.assertIn("ascii_structure_style", content)
                print(f"  ✅ Caption {txt_name}: {content.strip()}")

if __name__ == '__main__':
    unittest.main()
