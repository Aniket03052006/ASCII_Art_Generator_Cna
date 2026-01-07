"""
Fast ViT vs ResNet Model Comparison
Uses gradient conversion for ASCII (fast) and evaluates models by feature quality
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import numpy as np
import requests
import io
import urllib.parse
from datetime import datetime


# Selected stress tests (subset for speed)
STRESS_TESTS = [
    ("Simple Cat", "cat sitting on a chair", 3),
    ("Ambiguity", "bat on a trunk next to a mouse", 8),
    ("Spatial", "tiny cat on top of large chair", 7),
    ("House", "simple house with roof and door", 4),
    ("Tree", "tree with trunk and branches", 5),
]


def generate_line_art_image(prompt: str, seed: int = 42) -> Image.Image:
    """Generate line art image using Pollinations"""
    enhanced = f"simple line drawing diagram, {prompt}, black lines on white background, high contrast, minimal detail"
    print(f"  📝 Prompt: '{enhanced[:60]}...'")
    
    encoded = urllib.parse.quote(enhanced)
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=384&nologo=true&model=turbo&seed={seed}"
    
    try:
        resp = requests.get(url, timeout=90)
        if resp.status_code == 200:
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            print(f"  ✅ Image generated ({img.size})")
            return img
        else:
            print(f"  ❌ Error {resp.status_code}")
            return None
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None


def convert_to_ascii(image: Image.Image, width: int = 80) -> str:
    """Fast gradient-based ASCII conversion"""
    from ascii_gen.gradient_mapper import image_to_gradient_ascii
    return image_to_gradient_ascii(image, width=width, ramp="standard", with_edges=True, edge_weight=0.4)


def evaluate_model_features(image: Image.Image, model_name: str, model_path: Path) -> dict:
    """Evaluate model's feature extraction quality on full image"""
    from ascii_gen.enhanced_mapper import get_enhanced_mapper
    
    print(f"  🧠 Loading {model_name}...")
    mapper = get_enhanced_mapper(model_path=str(model_path))
    
    if not mapper.is_available():
        return {"available": False, "model": model_name}
    
    # Extract features from full image (not tiles - much faster!)
    features = mapper.extract_features(image)
    
    if features is None:
        return {"available": False, "model": model_name}
    
    # Compute quality metrics from features
    feature_stats = {
        "available": True,
        "model": model_name,
        "mean": float(np.mean(features)),
        "std": float(np.std(features)),
        "magnitude": float(np.mean(np.abs(features))),
        "max": float(np.max(features)),
        "min": float(np.min(features)),
        "sparsity": float(np.sum(np.abs(features) < 0.01) / len(features)),  # % near-zero
    }
    
    print(f"  ✅ {model_name}: mean={feature_stats['mean']:.3f}, std={feature_stats['std']:.3f}, mag={feature_stats['magnitude']:.3f}")
    
    return feature_stats


def run_comparison():
    """Run the fast model comparison"""
    print("="*80)
    print("FAST VIT vs RESNET MODEL COMPARISON")
    print("Using gradient conversion + full-image feature evaluation")
    print("="*80)
    
    models_dir = Path(__file__).parent.parent / "models"
    output_dir = Path(__file__).parent.parent / "outputs" / "model_comparison_fast"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    for name, prompt, severity in STRESS_TESTS:
        print(f"\n{'='*80}")
        print(f"TEST: {name} (Severity: {severity}/10)")
        print(f"PROMPT: '{prompt}'")
        print("="*80)
        
        # Generate image
        print("  🎨 Generating line art...")
        image = generate_line_art_image(prompt)
        if image is None:
            results.append({"test": name, "success": False})
            continue
        
        # Save source image
        test_dir = output_dir / name.replace(" ", "_")
        test_dir.mkdir(exist_ok=True)
        image.save(test_dir / "source.png")
        
        # Convert to ASCII (fast gradient method)
        print("  🔤 Converting to ASCII...")
        ascii_art = convert_to_ascii(image)
        
        with open(test_dir / "ascii_output.txt", "w") as f:
            f.write(f"Test: {name}\n")
            f.write(f"Prompt: {prompt}\n")
            f.write("="*80 + "\n\n")
            f.write(ascii_art)
        
        # Evaluate models
        print("  📊 Evaluating models...")
        vit_stats = evaluate_model_features(image, "ViT", models_dir / "ascii_vit_final.pth")
        resnet_stats = evaluate_model_features(image, "ResNet", models_dir / "ascii_resnet18_final.pth")
        
        results.append({
            "test": name,
            "prompt": prompt,
            "success": True,
            "vit": vit_stats,
            "resnet": resnet_stats,
            "output_dir": str(test_dir)
        })
        
        print(f"\n✅ Test complete!")
    
    # Generate summary report
    print("\n" + "="*80)
    print("MODEL COMPARISON SUMMARY")
    print("="*80)
    
    vit_scores = []
    resnet_scores = []
    
    for r in results:
        if r["success"]:
            if r["vit"]["available"]:
                vit_scores.append(r["vit"]["magnitude"])
            if r["resnet"]["available"]:
                resnet_scores.append(r["resnet"]["magnitude"])
    
    if vit_scores:
        print(f"\nViT Average Feature Magnitude: {np.mean(vit_scores):.4f}")
    if resnet_scores:
        print(f"ResNet Average Feature Magnitude: {np.mean(resnet_scores):.4f}")
    
    # Determine winner
    if vit_scores and resnet_scores:
        vit_avg = np.mean(vit_scores)
        resnet_avg = np.mean(resnet_scores)
        
        if vit_avg > resnet_avg:
            print("\n🏆 ViT produces STRONGER features (potentially better detail capture)")
        else:
            print("\n🏆 ResNet produces STRONGER features (potentially better structure)")
        
        print(f"\n💡 Note: Higher magnitude ≠ always better. Check ASCII outputs!")
    
    # Save report
    report_path = output_dir / "summary_report.txt"
    with open(report_path, "w") as f:
        f.write("VIT vs RESNET COMPARISON REPORT\n")
        f.write("="*60 + "\n\n")
        
        for r in results:
            f.write(f"\n{r['test']}: {'✅' if r['success'] else '❌'}\n")
            if r['success']:
                f.write(f"  Prompt: {r['prompt']}\n")
                if r['vit']['available']:
                    f.write(f"  ViT:    mag={r['vit']['magnitude']:.4f}, std={r['vit']['std']:.4f}\n")
                if r['resnet']['available']:
                    f.write(f"  ResNet: mag={r['resnet']['magnitude']:.4f}, std={r['resnet']['std']:.4f}\n")
    
    print(f"\n📁 Results saved to: {output_dir}")
    print(f"📊 Report: {report_path}")
    
    return results


if __name__ == "__main__":
    run_comparison()
