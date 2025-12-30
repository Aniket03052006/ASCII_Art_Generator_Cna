
import unittest
import time
import numpy as np
from PIL import Image
from unittest.mock import MagicMock, patch
from ascii_gen.perceptual import SSIMMapper, create_ssim_mapper
from ascii_gen.gradient_mapper import create_portrait_mapper, RAMP_ULTRA
from ascii_gen.multimodal import CLIPSelector, CLIPManager

class TestInnovativeFeatures(unittest.TestCase):
    
    def setUp(self):
        # Create a synthetic image for testing
        self.test_img = Image.new('RGB', (600, 400), color='white')
        # Add some shapes
        from PIL import ImageDraw
        draw = ImageDraw.Draw(self.test_img)
        draw.rectangle([100, 100, 200, 200], fill='black')
        draw.ellipse([300, 100, 400, 200], fill='gray')
        
    def test_ssim_mapper_performance(self):
        """Benchmark the vectorized SSIM mapper."""
        print("\n⚡ Testing SSIM Mapper Performance...")
        width = 80
        mapper = create_ssim_mapper(width=width)
        
        start_time = time.time()
        ascii_art = mapper.convert_image(self.test_img)
        duration = time.time() - start_time
        
        print(f"  - SSIM Conversion (80 chars wide): {duration:.4f}s")
        print(f"  - Output length: {len(ascii_art)} chars")
        
        # Performance assertion: Should be fast (< 1.0s for this size)
        # Vectorized implementation is usually 0.1s - 0.4s
        self.assertLess(duration, 2.0, "SSIM Mapper is too slow! (Recall it uses vectorization)")
        self.assertGreater(len(ascii_art), 100)
        
    def test_portrait_mapper_config(self):
        """Verify Portrait mapper settings."""
        print("\n🎨 Testing Portrait Mapper Configuration...")
        mapper = create_portrait_mapper(width=150)
        config = mapper.config
        
        self.assertEqual(config.width, 150)
        self.assertEqual(config.ramp, RAMP_ULTRA)
        self.assertTrue(config.edge_enhance)
        # Check specific tuning for portraits
        self.assertLess(config.contrast, 1.4, "Portrait contrast should be gentle")
        
    @patch('ascii_gen.multimodal.CLIPManager.get_score')
    @patch('ascii_gen.multimodal.CLIPManager.is_available')
    def test_clip_selector_logic(self, mock_is_available, mock_get_score):
        """Test AI Auto-Select logic (Mocked CLIP)."""
        print("\n🤖 Testing CLIP Selector Logic...")
        mock_is_available.return_value = True
        
        # Mock scores: 
        # Strategy A: 0.2
        # Strategy B: 0.9 (Winner)
        # Strategy C: 0.5
        mock_get_score.side_effect = [0.2, 0.9, 0.5]
        
        selector = CLIPSelector()
        
        # Mock mappers
        mappers = {
            "Strategy A": lambda img, w: "AAA",
            "Strategy B": lambda img, w: "BBB", # Best
            "Strategy C": lambda img, w: "CCC",
        }
        
        best_ascii, best_name, score = selector.select_best_ascii(self.test_img, "test prompt", 80, mappers)
        
        print(f"  - Winner: {best_name} (Score: {score})")
        
        self.assertEqual(best_name, "Strategy B")
        self.assertEqual(best_ascii, "BBB")
        self.assertEqual(score, 0.9)

if __name__ == '__main__':
    unittest.main()
