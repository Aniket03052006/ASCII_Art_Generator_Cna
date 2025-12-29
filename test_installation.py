#!/usr/bin/env python3
"""
Quick test script to verify installation and basic functionality.
Run after: pip install -r requirements.txt
"""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing module imports...")
    
    modules = [
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("numpy", "NumPy"),
        ("sklearn", "scikit-learn"),
        ("skimage", "scikit-image"),
        ("torch", "PyTorch"),
    ]
    
    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError as e:
            print(f"  ❌ {name}: {e}")
            all_ok = False
    
    if not all_ok:
        print("\n⚠️ Some dependencies missing. Run: pip install -r requirements.txt")
        return False
    
    print("\n✅ All dependencies installed!\n")
    return True


def test_ascii_gen():
    """Test the ascii_gen package."""
    print("Testing ascii_gen package...")
    
    try:
        from ascii_gen.charsets import get_charset, list_charsets
        print("  ✅ charsets module")
        
        charset = get_charset("ascii_structural")
        print(f"      Created charset with {len(charset.characters)} characters")
        print(f"      Sample raster shape: {charset.get_raster('|').shape}")
    except Exception as e:
        print(f"  ❌ charsets: {e}")
        return False
    
    try:
        from ascii_gen.aiss import AISSMapper
        print("  ✅ aiss module")
    except Exception as e:
        print(f"  ❌ aiss: {e}")
        return False
    
    try:
        from ascii_gen.random_forest import RandomForestMapper
        print("  ✅ random_forest module")
    except Exception as e:
        print(f"  ❌ random_forest: {e}")
        return False
    
    try:
        from ascii_gen.pipeline import PromptToASCII, image_to_ascii
        print("  ✅ pipeline module")
    except Exception as e:
        print(f"  ❌ pipeline: {e}")
        return False
    
    print("\n✅ All ascii_gen modules OK!\n")
    return True


def test_basic_conversion():
    """Test a simple image-to-ASCII conversion."""
    print("Testing basic conversion...")
    
    import numpy as np
    from PIL import Image
    
    # Create a simple test image with a circle
    img = np.ones((200, 300, 3), dtype=np.uint8) * 255
    import cv2
    cv2.circle(img, (150, 100), 50, (0, 0, 0), 2)
    test_image = Image.fromarray(img)
    
    from ascii_gen.pipeline import image_to_ascii
    
    print("  Converting test image (circle)...")
    result = image_to_ascii(
        test_image,
        mapper="random_forest",
        charset="ascii_structural",
        char_width=40,
    )
    
    print("  Result:")
    print("-" * 40)
    for line in result.text.split('\n')[:10]:  # Show first 10 lines
        print(f"  {line}")
    print("-" * 40)
    
    stats = result.get_stats()
    print(f"\n  Stats: {stats['width']}x{stats['height']} chars, {stats['unique_characters']} unique")
    
    print("\n✅ Basic conversion works!\n")
    return True


def main():
    print("=" * 50)
    print("ASCII Art Generator - Quick Test")
    print("=" * 50)
    print()
    
    if not test_imports():
        return 1
    
    if not test_ascii_gen():
        return 1
    
    if not test_basic_conversion():
        return 1
    
    print("=" * 50)
    print("✅ All tests passed! The library is ready to use.")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  1. Open notebooks/01_experimentation.ipynb in Jupyter")
    print("  2. Or use: from ascii_gen import image_to_ascii")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
