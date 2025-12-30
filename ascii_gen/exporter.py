
from PIL import Image, ImageDraw, ImageFont
import os

def render_ascii_to_image(ascii_text: str, font_size: int = 18, bg_color: str = "white", text_color: str = "black") -> str:
    """
    Renders an ASCII string to a PNG image.
    Returns the path to the saved image.
    """
    lines = ascii_text.splitlines()
    if not lines:
        return None

    # Load Font (Monospace is crucial)
    font_path = "/System/Library/Fonts/Menlo.ttc"
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        # Fallback to default if not found
        font = ImageFont.load_default()

    # Calculate Dimensions
    # We measure 'A' to get consistent char dims
    left, top, right, bottom = font.getbbox("A")
    char_width = right - left
    char_height = (bottom - top) + 2 # Add padding
    
    # Calculate Image Size
    max_line_len = max(len(line) for line in lines)
    img_width = max_line_len * char_width + 40 # Padding
    img_height = len(lines) * char_height + 40
    
    # Create Image
    image = Image.new("RGB", (img_width, img_height), color=bg_color)
    draw = ImageDraw.Draw(image)
    
    # Draw Text
    y_text = 20
    for line in lines:
        draw.text((20, y_text), line, font=font, fill=text_color)
        y_text += char_height
        
    # Save
    output_path = os.path.abspath("outputs/ascii_export.png")
    os.makedirs("outputs", exist_ok=True)
    image.save(output_path)
    
    return output_path
