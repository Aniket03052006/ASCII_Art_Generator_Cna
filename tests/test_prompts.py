"""
Comprehensive Prompt Testing Script

Tests the full pipeline with multiple detailed prompts:
1. Generate image with FLUX.1 Schnell
2. Convert to ASCII with Production CNN

NOTE: First run downloads ~30GB of model files!
"""

from PIL import Image
import time
import os

# Detailed prompts for testing
PROMPTS = [
    # Simple geometric shapes
    "A simple black and white line drawing of a house with a triangular roof",
    
    # Nature
    "High contrast silhouette of a single pine tree against white background",
    
    # Abstract
    "Simple geometric pattern with circles and straight lines, black on white",
    
    # Character
    "Minimalist line art of a cat face, simple outlines",
]

def test_with_prompts():
    """Test the full prompt-to-ASCII pipeline."""
    from ascii_gen import PromptToASCII
    from ascii_gen.production_training import ProductionCNNMapper
    
    print("=" * 70)
    print("TESTING PROMPT-TO-ASCII WITH FLUX.1 SCHNELL")
    print("⚠️  First run will download ~30GB of model files!")
    print("=" * 70)
    
    # Initialize pipeline with FLUX
    pipeline = PromptToASCII(
        mapper="random_forest",  # Using RF for faster demo
        sd_model="flux-schnell",
        charset="ascii_structural",
    )
    
    # Also load production CNN for comparison
    cnn = ProductionCNNMapper()
    try:
        cnn.load("models/production_cnn.pth")
        has_cnn = True
    except:
        print("No pre-trained CNN found, training...")
        cnn.train(epochs=50)
        has_cnn = True
    
    os.makedirs("outputs", exist_ok=True)
    
    for i, prompt in enumerate(PROMPTS, 1):
        print(f"\n{'=' * 70}")
        print(f"PROMPT {i}: {prompt[:60]}...")
        print("=" * 70)
        
        start = time.time()
        
        # Generate with pipeline
        result = pipeline.generate(
            prompt=prompt,
            char_width=60,
            seed=42 + i,
        )
        
        gen_time = time.time() - start
        
        print(f"\n🎨 RANDOM FOREST OUTPUT ({gen_time:.1f}s):")
        print("-" * 50)
        result.display()
        
        # Save source image
        if result.source_image:
            result.source_image.save(f"outputs/prompt_{i}_image.png")
        
        # Also convert with CNN
        if has_cnn and result.source_image:
            cnn_result = cnn.convert_image(result.source_image)
            print(f"\n🧠 PRODUCTION CNN OUTPUT:")
            print("-" * 50)
            print(cnn_result)
        
        # Save ASCII outputs
        result.save(f"outputs/prompt_{i}_rf.txt")
        with open(f"outputs/prompt_{i}_cnn.txt", "w") as f:
            f.write(cnn_result if has_cnn else "N/A")
        
        print(f"\n✅ Saved to outputs/prompt_{i}_*.png/txt")

def test_on_images_only():
    """Test on sample images without FLUX (faster demo)."""
    from ascii_gen.production_training import ProductionCNNMapper, ProductionRFMapper
    
    print("=" * 70)
    print("TESTING ON SAMPLE IMAGES (NO FLUX REQUIRED)")
    print("=" * 70)
    
    # Generate some test patterns
    import numpy as np
    from PIL import Image, ImageDraw
    
    test_images = []
    
    # 1. Circle
    img = Image.new('L', (400, 280), color=255)
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 40, 300, 240], outline=0, width=3)
    test_images.append(("Circle", img))
    
    # 2. Triangle
    img = Image.new('L', (400, 280), color=255)
    draw = ImageDraw.Draw(img)
    draw.polygon([(200, 40), (100, 240), (300, 240)], outline=0, width=3)
    test_images.append(("Triangle", img))
    
    # 3. House shape
    img = Image.new('L', (400, 280), color=255)
    draw = ImageDraw.Draw(img)
    draw.polygon([(200, 40), (100, 120), (300, 120)], outline=0, width=3)  # Roof
    draw.rectangle([120, 120, 280, 240], outline=0, width=3)  # House
    draw.rectangle([180, 180, 220, 240], outline=0, width=3)  # Door
    test_images.append(("House", img))
    
    # 4. Star
    img = Image.new('L', (400, 280), color=255)
    draw = ImageDraw.Draw(img)
    points = []
    for i in range(5):
        angle = i * 144 - 90
        import math
        x = 200 + 100 * math.cos(math.radians(angle))
        y = 140 + 100 * math.sin(math.radians(angle))
        points.append((x, y))
    draw.polygon(points, outline=0, width=3)
    test_images.append(("Star", img))
    
    # Load mappers
    cnn = ProductionCNNMapper()
    rf = ProductionRFMapper()
    
    try:
        cnn.load("models/production_cnn.pth")
        rf.load("models/production_rf.joblib")
    except:
        print("Training mappers (first run)...")
        cnn.train(epochs=50)
        rf.train()
    
    os.makedirs("outputs", exist_ok=True)
    
    for name, img in test_images:
        print(f"\n{'=' * 70}")
        print(f"SHAPE: {name}")
        print("=" * 70)
        
        img.save(f"outputs/{name.lower()}_input.png")
        
        print("\n🧠 CNN OUTPUT:")
        print("-" * 50)
        cnn_result = cnn.convert_image(img)
        print(cnn_result)
        
        print("\n🌲 RF OUTPUT:")
        print("-" * 50)
        rf_result = rf.convert_image(img)
        print(rf_result)
        
        with open(f"outputs/{name.lower()}_cnn.txt", "w") as f:
            f.write(cnn_result)
        with open(f"outputs/{name.lower()}_rf.txt", "w") as f:
            f.write(rf_result)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--flux":
        # Full test with FLUX generation (downloads 30GB)
        test_with_prompts()
    else:
        # Quick test on generated shapes (no download)
        test_on_images_only()
