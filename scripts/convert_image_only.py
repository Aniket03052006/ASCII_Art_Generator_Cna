"""
Convert image to ASCII ONLY.
"""
from PIL import Image
from ascii_gen.pipeline import image_to_ascii

if __name__ == "__main__":
    print("Loading image...")
    image = Image.open("test_input.png")
    
    print("Converting to ASCII...")
    # Use fewer trees for speed in this test
    # Note: image_to_ascii uses default RF params, so we'll configure manual pipeline
    
    from ascii_gen import PromptToASCII
    pipeline = PromptToASCII(
        mapper="random_forest",
        charset="ascii_standard", # Standard encoding
        tile_size=(10, 16),
        auto_train_rf=True
    )
    
    # Hack to speed up training for this test
    if pipeline._rf_mapper:
        pipeline._rf_mapper.n_estimators = 10
    
    result = pipeline.from_image(image, char_width=60)
    
    print("\nASCII OUTPUT:")
    print("=" * 60)
    result.display()
    print("=" * 60)
    
    result.save("test_output.txt")
    print("\n✅ Saved to test_output.txt")
