# API Documentation

## Core Classes

### PromptToASCII

The main pipeline class for converting prompts to ASCII art.

```python
from ascii_gen import PromptToASCII

pipeline = PromptToASCII(
    mapper="random_forest",  # Options: "aiss", "random_forest", "cnn", "both"
    charset="ascii_structural",  # Character set to use
    tile_size=(8, 14),  # Tile dimensions (width, height)
    sd_model="flux-schnell",  # Image generation model
)

# Generate from prompt
result = pipeline.generate(
    prompt="a mountain landscape",
    char_width=60,
    seed=42,
)
result.display()

# Convert from image
from PIL import Image
img = Image.open("photo.jpg")
result = pipeline.from_image(img, char_width=80)
result.save("output.txt")
```

### OnlineGenerator

Generate images using HuggingFace's free API (no local download needed).

```python
from ascii_gen.online_generator import OnlineGenerator

gen = OnlineGenerator(api_key="hf_your_token")
image = gen.generate(
    prompt="a cat on the moon",
    width=512,
    height=384,
    seed=42,
)
```

### ProductionCNNMapper

High-quality CNN-based mapper with edge-aligned training.

```python
from ascii_gen.production_training import ProductionCNNMapper

mapper = ProductionCNNMapper(charset="ascii_structural")
mapper.train(epochs=80, augments=25)  # ~80% accuracy
mapper.save("models/production_cnn.pth")

# Later, load and use
mapper.load("models/production_cnn.pth")
ascii_art = mapper.convert_image(image, apply_edge_detection=True)
```

### ProductionRFMapper

Random Forest mapper with HoG features.

```python
from ascii_gen.production_training import ProductionRFMapper

mapper = ProductionRFMapper(charset="ascii_structural")
mapper.train(augments=25)
ascii_art = mapper.convert_image(image)
```

## Character Sets

Available character sets:

| Name | Characters | Best For |
|------|------------|----------|
| `ascii_standard` | All 95 printable ASCII | General use |
| `ascii_structural` | `\|/-\_=[]<>(){}#@.` | Line art |
| `ascii_blocks` | Block elements | Dense shading |
| `ascii_minimal` | `. -=+#@` | Simple output |

```python
from ascii_gen import get_charset

charset = get_charset("ascii_structural", tile_size=(8, 14))
print(charset.characters)
```

## Web Interface

Launch the Gradio web UI:

```bash
export HF_TOKEN="hf_your_token"
python web/app.py
```

Opens at http://localhost:7860

## CLI Usage

```bash
# Interactive mode
python scripts/cli.py

# Single prompt
python scripts/cli.py "a sunset over mountains"

# With options
python scripts/cli.py --width 80 --seed 42 --mapper rf "your prompt"
```
