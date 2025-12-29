# Prompt-to-ASCII Art Generator

A complete Python library for converting text prompts to ASCII art using Stable Diffusion, CLIP, and structural mapping algorithms.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **🎨 Dual Mapping Algorithms**: AISS (Log-Polar) + Random Forest for structure-based conversion
- **🚀 Fast Generation**: SDXL-Turbo for 4-step image synthesis
- **💻 Apple Silicon Optimized**: MPS acceleration for M1/M2/M3/M4 Macs
- **📊 Multiple Charsets**: ASCII, ANSI blocks, Shift-JIS support
- **📓 Interactive Notebooks**: Jupyter experimentation environment

## 🚀 Quick Start

### Installation

```bash
# Clone and install
cd ASCII_Gen
pip install -r requirements.txt
```

### Basic Usage

```python
from ascii_gen import PromptToASCII, image_to_ascii

# From existing image
result = image_to_ascii("path/to/image.jpg", char_width=80)
result.display()

# From text prompt (requires SDXL-Turbo download)
pipeline = PromptToASCII(mapper="random_forest")
result = pipeline.generate("cyberpunk cityscape at night")
result.display()
result.save("output.html")
```

### Compare Algorithms

```python
pipeline = PromptToASCII(mapper="both")
aiss_result, rf_result = pipeline.from_image("image.jpg", return_comparison=True)

# View quality metrics
comparison = pipeline.compare_results(aiss_result, rf_result)
print(comparison)
```

## 📁 Project Structure

```
ASCII_Gen/
├── ascii_gen/
│   ├── __init__.py          # Package exports
│   ├── charsets.py           # Character set definitions
│   ├── aiss.py               # AISS log-polar algorithm
│   ├── random_forest.py      # ML character classifier
│   ├── generator.py          # Stable Diffusion pipeline
│   ├── pipeline.py           # End-to-end orchestration
│   ├── preprocessing.py      # Image utilities
│   ├── metrics.py            # Quality assessment
│   └── result.py             # Output container
├── notebooks/
│   └── 01_experimentation.ipynb
├── examples/
│   └── sample_outputs/
├── requirements.txt
└── README.md
```

## 🎯 Algorithms

### AISS (Alignment-Insensitive Shape Similarity)
Based on Xu et al. (2010), uses log-polar histograms to match image tiles to character shapes. Translation-invariant and structure-focused.

### Random Forest
Based on 2025 research (Coumar & Kingston), uses HoG features with ensemble classification. Achieves comparable SSIM to CNNs with 10x faster inference.

## 📊 Character Sets

| Name | Characters | Best For |
|------|------------|----------|
| `ascii_standard` | 95 printable | Universal compatibility |
| `ascii_structural` | `.-_\|/\\<>[]#@` | Line art, edges |
| `ansi_blocks` | `░▒▓█` | Dense fills |
| `shift_jis` | Japanese | Rich structural variety |

## 🛠️ Requirements

- Python 3.9+
- PyTorch with MPS support (for M-series Macs)
- ~6GB disk space for SDXL-Turbo model

## 📚 References

- Xu, X., Zhang, L., & Wong, T. T. (2010). *Structure-based ASCII Art*. SIGGRAPH.
- Coumar, S., & Kingston, Z. (2025). *Evaluating ML Approaches for ASCII Art Generation*.
- HuggingFace Diffusers: https://huggingface.co/docs/diffusers

## 📄 License

MIT License
