"""
Generate a synthetic test image and convert it to ASCII.
This serves as a smoke test for the pipeline without needing Stable Diffusion.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from ascii_gen import image_to_ascii

def create_synthetic_image(width=600, height=400):
    """Create a synthetic image with shapes and gradient."""
    # Create blank image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a gradient background
    for y in range(height):
        # Blue to Red vertical gradient
        r = int(255 * y / height)
        b = int(255 * (1 - y / height))
        img[y, :, 0] = b  # Blue
        img[y, :, 2] = r  # Red
        
    # Convert to PIL for drawing text and shapes
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Draw a large white circle
    draw.ellipse((50, 50, 200, 200), fill='white', outline='black')
    
    # Draw a yellow rectangle
    draw.rectangle((250, 50, 350, 150), fill='yellow', outline='black')
    
    # Draw a triangle
    draw.polygon([(450, 50), (400, 150), (500, 150)], fill='cyan', outline='black')
    
    # Draw some text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 40)
    except:
        font = ImageFont.load_default()
        
    draw.text((50, 250), "ASCII Gen", fill='white', font=font)
    draw.text((50, 300), "Smoke Test", fill='lime', font=font)
    
    return pil_img

def main():
    print("🎨 Generating synthetic test image...")
    image = create_synthetic_image()
    image.save("test_input.png")
    print("✅ Saved 'test_input.png'")
    
    print("\n🔄 Converting to ASCII (Random Forest)...")
    try:
        # Using a slightly larger width to capture detail
        result = image_to_ascii(
            image,
            mapper="random_forest",
            charset="ascii_standard", # Standard encoding
            char_width=80,
            apply_edge_detection=True
        )
        
        print("\n" + "="*80)
        print("ASCII ART RESULT")
        print("="*80)
        result.display()
        print("="*80)
        
        # Save output
        result.save("test_output.txt")
        print("\n✅ Saved 'test_output.txt'")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
