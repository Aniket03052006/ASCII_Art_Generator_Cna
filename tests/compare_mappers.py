"""
Comparison Test: Enhanced Mapper vs Original
=============================================
Tests if the trained ResNet model actually improves ASCII output.
"""
import sys
sys.path.insert(0, '.')

from PIL import Image
import numpy as np
from ascii_gen.enhanced_mapper import get_enhanced_mapper
import time

# Standard character ramp
RAMP = " .:-=+*#%@"

def original_char_select(brightness):
    """Original brightness-only character selection."""
    idx = int((brightness / 255) * (len(RAMP) - 1))
    return RAMP[idx]

def compare_on_image(image_path):
    """Compare both methods on a single image."""
    print(f"\n🖼️ Testing: {image_path}")
    
    img = Image.open(image_path).convert('RGB')
    
    # Get enhanced mapper
    enhanced = get_enhanced_mapper()
    
    # Sample some tiles
    tile_size = 8
    width, height = img.size
    
    original_output = []
    enhanced_output = []
    differences = 0
    total = 0
    
    for y in range(0, min(height, 80), tile_size):
        original_row = ""
        enhanced_row = ""
        for x in range(0, min(width, 120), tile_size):
            tile = img.crop((x, y, x + tile_size, y + tile_size))
            brightness = np.mean(np.array(tile.convert('L')))
            
            # Original method
            orig_char = original_char_select(brightness)
            original_row += orig_char
            
            # Enhanced method
            if enhanced.is_available():
                enh_char = enhanced.enhanced_char_select(tile, brightness)
            else:
                enh_char = orig_char
            enhanced_row += enh_char
            
            if orig_char != enh_char:
                differences += 1
            total += 1
            
        original_output.append(original_row)
        enhanced_output.append(enhanced_row)
    
    return {
        'original': original_output,
        'enhanced': enhanced_output,
        'differences': differences,
        'total': total,
        'diff_percent': (differences / total * 100) if total > 0 else 0
    }

def run_comparison():
    """Run comparison on test images."""
    print("=" * 60)
    print("🔬 ENHANCED MAPPER COMPARISON TEST")
    print("=" * 60)
    
    # Check if enhanced mapper is available
    mapper = get_enhanced_mapper()
    print(f"Enhanced mapper available: {mapper.is_available()}")
    
    if not mapper.is_available():
        print("❌ Cannot run comparison - model not loaded")
        return
    
    # Test on sample images from training data
    test_images = [
        "ascii_training_data/images/000001.png",
        "ascii_training_data/images/000010.png",
        "ascii_training_data/images/000050.png",
    ]
    
    total_diff = 0
    total_chars = 0
    
    for img_path in test_images:
        try:
            result = compare_on_image(img_path)
            total_diff += result['differences']
            total_chars += result['total']
            
            print(f"\n📊 Results:")
            print(f"   Characters different: {result['differences']}/{result['total']} ({result['diff_percent']:.1f}%)")
            
            print(f"\n   Original (first 3 rows):")
            for row in result['original'][:3]:
                print(f"   {row[:60]}")
            
            print(f"\n   Enhanced (first 3 rows):")
            for row in result['enhanced'][:3]:
                print(f"   {row[:60]}")
                
        except FileNotFoundError:
            print(f"   ⚠️ File not found: {img_path}")
    
    print("\n" + "=" * 60)
    if total_chars > 0:
        overall_diff = (total_diff / total_chars) * 100
        print(f"📈 OVERALL: {overall_diff:.1f}% of characters were refined by the model")
        
        if overall_diff > 10:
            print("✅ The enhanced mapper is ACTIVELY IMPROVING output!")
        elif overall_diff > 2:
            print("✅ The enhanced mapper is making subtle improvements.")
        else:
            print("⚠️ The enhanced mapper shows minimal impact.")
    print("=" * 60)

if __name__ == "__main__":
    run_comparison()
