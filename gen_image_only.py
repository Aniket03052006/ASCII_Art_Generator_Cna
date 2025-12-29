"""
Generate synthetic image ONLY.
"""
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def create_synthetic_image(width=600, height=400):
    # Create blank image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a gradient background
    for y in range(height):
        r = int(255 * y / height)
        b = int(255 * (1 - y / height))
        img[y, :, 0] = b  # Blue
        img[y, :, 2] = r  # Red
        
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Draw shapes
    draw.ellipse((50, 50, 200, 200), fill='white', outline='black')
    draw.rectangle((250, 50, 350, 150), fill='yellow', outline='black')
    draw.polygon([(450, 50), (400, 150), (500, 150)], fill='cyan', outline='black')
    
    # Draw text
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 40)
    except:
        font = ImageFont.load_default()
        
    draw.text((50, 250), "ASCII Gen", fill='white', font=font)
    
    return pil_img

if __name__ == "__main__":
    print("Generating image...")
    img = create_synthetic_image()
    img.save("test_input.png")
    print("Done: test_input.png")
