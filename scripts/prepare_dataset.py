"""
Dataset Preparation Script for Flux LoRA Training
==================================================
Loads the locally downloaded mrzjy/ascii_art_generation_140k dataset
and renders ASCII art to images.

Prerequisites:
    huggingface-cli download --repo-type dataset mrzjy/ascii_art_generation_140k --local-dir ./ascii_hf_raw

Usage:
    python scripts/prepare_dataset.py --output ascii_training_data --limit 5000
"""
import os
import argparse
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

def render_ascii_to_image(ascii_text, font_size=14, bg_color="white", text_color="black"):
    """
    Renders ASCII text to a PIL Image.
    High contrast for optimal Flux training.
    """
    lines = ascii_text.split('\n')
    if not lines or all(len(l.strip()) == 0 for l in lines):
        return None
    
    # Try to load a monospace font
    font = None
    font_names = ["Menlo.ttc", "Consolas.ttf", "Courier New.ttf", "LiberationMono-Regular.ttf", "DejaVuSansMono.ttf"]
    for name in font_names:
        try:
            font = ImageFont.truetype(name, font_size)
            break
        except:
            continue
    if not font:
        font = ImageFont.load_default()

    # Calculate size using font metrics
    try:
        bbox = font.getbbox("A")
        char_w = bbox[2] - bbox[0]
        char_h = bbox[3] - bbox[1] + 2
    except:
        char_w, char_h = 8, 14
    
    max_line_chars = max(len(line) for line in lines) if lines else 0
    img_width = max_line_chars * char_w + 20
    img_height = len(lines) * char_h + 20
    
    # Skip images that are too large or small
    if img_width > 2048 or img_height > 2048 or img_width < 50 or img_height < 50:
        return None

    img = Image.new('RGB', (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    y = 10
    for line in lines:
        draw.text((10, y), line, font=font, fill=text_color)
        y += char_h
        
    return img

def extract_ascii_from_conversations(conversations):
    """
    Extracts the ASCII art from a conversations list.
    The assistant's response contains the ASCII art.
    """
    for turn in conversations:
        if turn.get('role') == 'assistant':
            return turn.get('content', '')
    return ''

def extract_prompt_from_conversations(conversations):
    """
    Extracts the user prompt from a conversations list.
    """
    for turn in conversations:
        if turn.get('role') == 'user':
            return turn.get('content', '')
    return ''

def process_dataset(input_dir, output_dir, limit=None):
    print(f"Loading dataset from {input_dir}...")
    try:
        ds = load_from_disk(input_dir)
        print(f"✅ Dataset loaded successfully! Total samples: {len(ds)}")
        print(f"   Columns: {ds.column_names}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Metadata file for training
    metadata_path = os.path.join(output_dir, "metadata.csv")
    with open(metadata_path, "w") as meta:
        meta.write("file_name,text\n")
    
    valid_count = 0
    skipped_count = 0
    
    print("Rendering images locally (No API Usage)...")
    
    total = min(limit, len(ds)) if limit else len(ds)
    
    for i in tqdm(range(total)):
        item = ds[i]
        
        # Extract from conversations format
        conversations = item.get('conversations', [])
        ascii_art = extract_ascii_from_conversations(conversations)
        prompt = extract_prompt_from_conversations(conversations)
            
        if not ascii_art or len(ascii_art) < 20:
            skipped_count += 1
            continue
            
        # Render ASCII to image
        img = render_ascii_to_image(ascii_art)
        
        if img:
            filename = f"{valid_count:06d}.png"
            img.save(os.path.join(images_dir, filename))
            
            # Write metadata
            clean_prompt = prompt.replace("\n", " ").replace('"', "'").strip()
            if not clean_prompt:
                clean_prompt = "ASCII art"
            with open(metadata_path, "a") as meta:
                meta.write(f"images/{filename},\"{clean_prompt}\"\n")
                
            valid_count += 1
        else:
            skipped_count += 1

    print(f"\n✅ Successfully prepared {valid_count} images in {output_dir}")
    print(f"⚠️  Skipped {skipped_count} items (too large/small or empty)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare ASCII dataset for Flux LoRA training")
    parser.add_argument("--input", type=str, default="ascii_hf_raw/train", help="Path to downloaded dataset")
    parser.add_argument("--output", type=str, default="ascii_training_data_local", help="Output directory")
    parser.add_argument("--limit", type=int, default=1000, help="Limit number of examples")
    args = parser.parse_args()
    
    process_dataset(args.input, args.output, args.limit)
