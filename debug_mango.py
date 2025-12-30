
import sys
import os
sys.path.append(".")
from PIL import Image
from ascii_gen.gradient_mapper import image_to_gradient_ascii

# User's uploaded image path
IMG_PATH = "/Users/aniketmandal06/.gemini/antigravity/brain/f2372832-caa4-4283-8c34-af0ba9ef1625/uploaded_image_1767107655602.png"
OUT_PATH = "outputs/debug_mango_output.txt"

def test_mango():
    if not os.path.exists(IMG_PATH):
        print(f"Image not found at {IMG_PATH}")
        return

    img = Image.open(IMG_PATH)
    
    # 1. Standard Conversion (What the user likely saw)
    ascii_art = image_to_gradient_ascii(
        img, 
        width=80, 
        ramp="standard", 
        with_edges=True,
        edge_weight=0.3
    )
    
    with open(OUT_PATH, "w") as f:
        f.write("--- STANDARD OUTPUT ---\n")
        f.write(ascii_art)
        f.write("\n\n")

    print(f"Saved debug output to {OUT_PATH}")

if __name__ == "__main__":
    test_mango()
