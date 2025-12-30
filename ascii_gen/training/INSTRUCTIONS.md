# 🎓 Training Kit: Pre-ASCII Structural LoRA

This kit contains everything needed to train a **FLUX.1-dev LoRA** that understands "Pre-ASCII" structural grids. This model is critical for generating clean, 16px-aligned layouts (caves, dungeons, text) that convert perfectly to ASCII characters.

## 🎯 Objective
Train the model to generate **1024x1024 black-and-white grid images** that align perfectly with a 16x16 pixel grid. This overcomes the "diffusion bias" where base models generate soft, anti-aliased edges that look messy when converted to ASCII.

---

## 🛠️ Prerequisites

### Hardware
- **GPU**: NVIDIA GPU with 24GB+ VRAM (e.g., RTX 3090, 4090, A100).
- **RAM**: 32GB+ System RAM.

### Software
- **Trainer**: [Kohya_ss](https://github.com/bmaltais/kohya_ss) or [SimpleTuner](https://github.com/bghira/SimpleTuner) (Standard SDXL/Flux training functionality).
- **Base Model**: `black-forest-labs/FLUX.1-dev` (Requires HuggingFace Access Token).

---

## 🚀 Step-by-Step Instructions

### 1. Generate the Dataset
We use a procedural generator to create "Perfect" training data. This ensures the model learns the exact pixel-perfect alignment we need.

1.  Navigate to the project root:
    ```bash
    cd /path/to/ASCII_Gen
    ```

2.  Run the generator script:
    ```bash
    python ascii_gen/training/dataset_generator.py
    ```

3.  **Verify Output**:
    - Check the folder `ascii_training_data`.
    - You should see 50+ images (`.png`) and text files (`.txt`).
    - Images should be 1024x1024, black and white, looking like retro dungeon maps.

### 2. Configure Training environment
If using `SimpleTuner` or a cloud pod (RunPod/Lambda), use the provided config as a baseline.

**Key Hyperparameters (Critical):**
- **Resolution**: `1024,1024` (Must be square and multiple of 16).
- **Rank (Linear)**: `64` or `128` (We need high rank to learn rigid structures; standard 16 is too low).
- **Alpha**: `32` (Rank / 2).
- **Learning Rate**: `4e-4` (Aggressive learning for structure).
- **Optimizer**: `adamw8bit` (Saves VRAM).
- **Trigger Word**: `ascii_structure_style`.

### 3. Run Training (Kohya Example)
If using `sd-scripts` (Kohya command line):

```bash
accelerate launch --num_cpu_threads_per_process=2 "train_network.py" \
  --pretrained_model_name_or_path="black-forest-labs/FLUX.1-dev" \
  --dataset_config="config/dataset_config.toml" \
  --output_dir="output" \
  --output_name="flux_ascii_structure_v1" \
  --learning_rate=4e-4 \
  --network_dim=64 \
  --network_alpha=32 \
  --mixed_precision="bf16" \
  --save_model_as="safetensors" \
  --max_train_steps=3000
```

### 4. Verification Check
After training, generate an image with this prompt to verify success:
> "ascii_structure_style, systematic dungeon map, infinite corridor, white on black"

**Success Criteria:**
1.  Is the background purely black (`#000000`)?
2.  Are the lines crisp white (`#FFFFFF`) with NO gray anti-aliasing?
3.  Do the walls line up with a 16x16 pixel grid overlay?

---

## 📦 File Reference
- **Generator**: `ascii_gen/training/dataset_generator.py`
- **Config Reference**: `config/train_flux_ascii.yaml`
