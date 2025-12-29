# ASCII Art Generator

<p align="center">
  <img src="outputs/iteration3_combined.png" alt="Cat on Chair Demo" width="400">
</p>

Transform text prompts and images into stunning ASCII art using AI and computer vision.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![FLUX.1](https://img.shields.io/badge/Model-FLUX.1--schnell-purple)](https://huggingface.co/black-forest-labs/FLUX.1-schnell)

---

## 🎯 Approach

This project implements a **multi-stage pipeline** for converting text prompts to ASCII art:

```
Text Prompt → [Prompt Engineering] → [Image Generation] → [Preprocessing] → [Character Mapping] → ASCII Art
```

### Pipeline Stages

| Stage | Method | Research Basis |
|-------|--------|----------------|
| **Prompt Engineering** | Industrial-standard templates | Google Prompt Engineering (2024) |
| **Image Generation** | FLUX.1 Schnell (4-step diffusion) | Black Forest Labs |
| **Preprocessing** | Saliency + Bilateral + Edge | Spectral Residual (Hou & Zhang, 2007) |
| **Character Mapping** | Production CNN (243K params) | DeepAA-inspired architecture |

---

## 🔬 Research Papers & Techniques

### 1. Image Generation
**Model**: [FLUX.1 Schnell](https://huggingface.co/black-forest-labs/FLUX.1-schnell)
- 4-step inference (fastest in class)
- Apache 2.0 license (commercial use allowed)
- Via HuggingFace Inference API (free tier)

### 2. Prompt Engineering
Based on **Google's Prompt Engineering Best Practices (2024)**:
- Be specific, use positive instructions
- Structure prompts with clear components
- Category-based template selection
- A/B testing framework for optimization

**Templates implemented**:
- `CHARACTER` - For subjects doing actions
- `MULTI_OBJECT` - For multiple distinct items
- `SCENE` - For landscapes and environments
- `DETAILED` - Default high-quality template
- `MINIMAL` - Fast generation

### 3. Saliency Detection
**Paper**: "Saliency Detection: A Spectral Residual Approach" (Hou & Zhang, CVPR 2007)

Focuses on "important" regions of the image:
```python
# Spectral Residual method
log_amplitude → spectral_residual → saliency_map
```

### 4. Perception-Sensitive Structure Extraction
**Paper**: "Structure-based ASCII Art" (ResearchGate / IEEE)

Key insight: Standard edge detection treats all edges equally. We weight edges by saliency to preserve important structures while suppressing texture noise.

### 5. ASCII Art ML Evaluation
**Paper**: "Evaluating Machine Learning Approaches for ASCII Art Generation" (Coumar & Kingston, 2025)

Key findings:
- Hybrid methods (classical + ML) outperform pure CNNs
- Classical ML achieves similar accuracy with fewer resources
- Diffusion models produce the most visually appealing results

### 6. AISS (Alignment-Insensitive Shape Similarity)
**Paper**: SIGGRAPH / TU Wien

Structure-based ASCII art requires tolerance for misalignment. AISS metric accounts for differences in position, orientation, and scaling.

---

## 🏗️ Architecture

```
ascii_gen/
├── pipeline.py              # Main PromptToASCII class
├── prompt_engineering.py    # Industrial-standard prompt templates
├── online_generator.py      # HuggingFace API client (FLUX.1)
├── advanced_preprocessing.py # Saliency + bilateral + contours
├── production_training.py   # CNN/RF mapper training
├── cnn_mapper.py           # Neural network character classifier
├── random_forest.py        # Random forest character classifier
├── aiss.py                 # Log-polar histogram similarity
└── charsets.py             # ASCII character sets
```

---

## 🧠 Models

### Production CNN Mapper
- **Architecture**: 3-layer CNN (32→64→128 channels)
- **Parameters**: 243K
- **Training**: Edge-aligned tiles with augmentation
- **Accuracy**: ~79%

### Random Forest Mapper
- **Features**: HoG + edge density
- **Estimators**: 200 trees
- **Accuracy**: ~79%

### AISS Mapper
- **Method**: Log-polar histogram matching
- **Reference**: Original structure-based ASCII art paper

---

## 🚀 Quick Start

### Web Interface
```bash
export HF_TOKEN="your_huggingface_token"
python web/app.py
# Open http://localhost:7860
```

### Python API
```python
from ascii_gen.online_generator import OnlineGenerator
from ascii_gen.production_training import ProductionCNNMapper
from ascii_gen.advanced_preprocessing import preprocess_for_ascii

# Generate image
gen = OnlineGenerator(api_key="hf_...")
image = gen.generate("a cat sitting on a chair")

# Convert to ASCII
cnn = ProductionCNNMapper()
cnn.load("models/production_cnn.pth")
edges = preprocess_for_ascii(image)
ascii_art = cnn.convert_image(Image.fromarray(edges))
print(ascii_art)
```

---

## 📊 A/B Testing Results

Tested on "cat sitting on a chair":

| Template | Description | Clarity |
|----------|-------------|---------|
| CHARACTER | Cartoon mascot style | ⭐⭐⭐ |
| MULTI | Separated icons | ⭐⭐ |
| COMBINED | Explicit body parts | ⭐⭐⭐⭐ |

**Best Practice**: Describe specific body parts/features for best results:
```
"cat clearly visible with ears head body tail,
 chair clearly visible with seat back and legs"
```

---

## 📁 Project Structure

```
ASCII_Gen/
├── ascii_gen/           # Core library
├── web/                 # Gradio web interface
├── tests/               # Test suite
├── scripts/             # CLI tools
├── notebooks/           # Jupyter notebooks
├── models/              # Trained weights
├── outputs/             # Generated ASCII art
├── docs/                # Documentation
├── pyproject.toml       # Modern Python packaging
├── Makefile             # Development commands
└── README.md            # This file
```

---

## 📚 References

1. Hou, X., & Zhang, L. (2007). Saliency Detection: A Spectral Residual Approach. CVPR.
2. Coumar, S., & Kingston, Z. (2025). Evaluating Machine Learning Approaches for ASCII Art Generation. arXiv.
3. Xu, X., et al. (2010). Structure-based ASCII Art. SIGGRAPH.
4. Google. (2024). Prompt Engineering Best Practices.
5. Black Forest Labs. FLUX.1-schnell. HuggingFace.

---

## 📜 License

MIT License - see [LICENSE](LICENSE)

---

## 🙏 Acknowledgments

- [Black Forest Labs](https://huggingface.co/black-forest-labs) for FLUX.1 Schnell
- [HuggingFace](https://huggingface.co) for free inference API
- OpenCV community for saliency detection
- DeepAA research for CNN architecture inspiration
