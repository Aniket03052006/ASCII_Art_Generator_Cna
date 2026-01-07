"""
ViT vs ResNet Model Comparison Test
Uses Pollinations API for fast image generation
Runs stress tests to compare model quality
"""

import sys
import os
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from datetime import datetime

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import requests
import io
import urllib.parse


@dataclass
class StressTest:
    """Simplified stress test case"""
    name: str
    prompt: str
    severity: int


# Select 10 representative stress tests
SELECTED_TESTS = [
    StressTest("Context Ambiguity", "bat on a trunk next to a mouse under a light", 8),
    StressTest("Pitcher on Mound", "pitcher on the mound", 7),
    StressTest("Simple Cat", "cat sitting on a chair", 3),
    StressTest("Kitchen Sink", "cat on chair with lamp, table, and window", 8),
    StressTest("Nested Complexity", "cat on a chair that's on a table", 7),
    StressTest("Spatial Stress", "tiny cat on top of large chair behind tall table", 9),
    StressTest("Abstract Simple", "simple house with roof and door", 4),
    StressTest("Action Request", "cat jumping over a chair", 6),
    StressTest("Scale Mix", "ant next to elephant (simplified)", 8),
    StressTest("Tree Scene", "a simple tree with trunk and leafy branches", 5),
]


class ModelComparator:
    """Compare ViT vs ResNet ASCII mappers"""
    
    def __init__(self, output_dir: str = "outputs/model_comparison"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Verify models exist
        models_dir = Path(__file__).parent.parent / "models"
        vit_path = models_dir / "ascii_vit_final.pth"
        resnet_path = models_dir / "ascii_resnet18_final.pth"
        
        print("🔄 Checking for models...")
        if vit_path.exists():
            print(f"  ✅ ViT model found: {vit_path}")
        else:
            print(f"  ⚠️  ViT model not found: {vit_path}")
        
        if resnet_path.exists():
            print(f"  ✅ ResNet model found: {resnet_path}")
        else:
            print(f"  ⚠️  ResNet model not found: {resnet_path}")
        
        print()
    
    def generate_image_pollinations(self, prompt: str, seed: int = 42) -> Image.Image:
        """Generate image using Pollinations API with line art prompt engineering"""
        print(f"  🎨 Generating image with Pollinations...")
        
        # Check if it's a portrait/face request
        is_portrait = any(word in prompt.lower() for word in ["face", "portrait", "person", "man", "woman", "child"])
        
        if is_portrait:
            # For portraits: request with shading but still clean
            enhanced_prompt = f"clean portrait sketch with shading, {prompt}, high contrast, simple clean lines"
        else:
            # For everything else: request line diagram style
            enhanced_prompt = f"simple line drawing diagram, {prompt}, clean black lines on white background, high contrast, no shading, minimal detail"
        
        print(f"  📝 Enhanced prompt: '{enhanced_prompt[:80]}...'")
        
        # Use Pollinations Turbo for speed
        encoded = urllib.parse.quote(enhanced_prompt)
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
    
    def convert_with_model(self, image: Image.Image, model_name: str, model_path: str, width: int = 80) -> str:
        """Convert image to ASCII using specified model"""
        print(f"  🔤 Converting with {model_name}...")
        
        try:
            # Use the model-enhanced converter
            from ascii_gen.model_converter import convert_with_model
            ascii_art = convert_with_model(image, model_path=model_path, width=width)
            
            print(f"  ✅ Converted ({len(ascii_art.split(chr(10)))} lines)")
            return ascii_art
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simple gradient
            from ascii_gen.gradient_mapper import image_to_gradient_ascii
            ascii_art = image_to_gradient_ascii(image, width=width, ramp="standard")
            return ascii_art
    
    def run_single_test(self, test: StressTest) -> Dict:
        """Run a single stress test with both models"""
        print(f"\n{'='*80}")
        print(f"TEST: {test.name}")
        print(f"PROMPT: '{test.prompt}'")
        print(f"SEVERITY: {test.severity}/10")
        print(f"{'='*80}")
        
        # Generate image
        image = self.generate_image_pollinations(test.prompt)
        if image is None:
            return {"success": False, "error": "Image generation failed"}
        
        # Get model paths
        models_dir = Path(__file__).parent.parent / "models"
        vit_path = str(models_dir / "ascii_vit_final.pth")
        resnet_path = str(models_dir / "ascii_resnet18_final.pth")
        
        # Convert with both models
        vit_ascii = self.convert_with_model(image, "ViT", vit_path)
        resnet_ascii = self.convert_with_model(image, "ResNet", resnet_path)
        
        # Save outputs
        test_dir = self.output_dir / f"{test.name.replace(' ', '_')}"
        test_dir.mkdir(exist_ok=True)
        
        # Save image
        image.save(test_dir / "source_image.png")
        
        # Save ASCII outputs
        with open(test_dir / "vit_output.txt", "w") as f:
            f.write(f"Model: ViT Vision Transformer\n")
            f.write(f"Prompt: {test.prompt}\n")
            f.write(f"{'='*80}\n\n")
            f.write(vit_ascii)
        
        with open(test_dir / "resnet_output.txt", "w") as f:
            f.write(f"Model: ResNet18\n")
            f.write(f"Prompt: {test.prompt}\n")
            f.write(f"{'='*80}\n\n")
            f.write(resnet_ascii)
        
        # Create side-by-side comparison
        comparison = self.create_comparison_text(test, vit_ascii, resnet_ascii)
        with open(test_dir / "comparison.txt", "w") as f:
            f.write(comparison)
        
        print(f"\n✅ Test complete! Outputs saved to: {test_dir}")
        
        return {
            "success": True,
            "test": test,
            "vit_ascii": vit_ascii,
            "resnet_ascii": resnet_ascii,
            "image": image,
            "output_dir": test_dir
        }
    
    def create_comparison_text(self, test: StressTest, vit: str, resnet: str) -> str:
        """Create side-by-side comparison text"""
        vit_lines = vit.split('\n')
        resnet_lines = resnet.split('\n')
        
        max_lines = max(len(vit_lines), len(resnet_lines))
        max_width = max(len(line) for line in vit_lines) if vit_lines else 80
        
        comparison = []
        comparison.append(f"{'='*160}")
        comparison.append(f"PROMPT: {test.prompt}")
        comparison.append(f"{'='*160}")
        comparison.append(f"{'ViT Vision Transformer':^{max_width+5}} | {'ResNet18':^{max_width+5}}")
        comparison.append(f"{'-'*max_width:^{max_width+5}} | {'-'*max_width:^{max_width+5}}")
        
        for i in range(max_lines):
            vit_line = vit_lines[i] if i < len(vit_lines) else ""
            resnet_line = resnet_lines[i] if i < len(resnet_lines) else ""
            comparison.append(f"{vit_line:<{max_width+5}} | {resnet_line:<{max_width+5}}")
        
        comparison.append(f"{'='*160}")
        return '\n'.join(comparison)
    
    def run_all_tests(self) -> Dict:
        """Run all stress tests"""
        print("\n" + "="*80)
        print("STARTING VIT vs RESNET MODEL COMPARISON")
        print(f"Total Tests: {len(SELECTED_TESTS)}")
        print("Image Generator: Pollinations Turbo")
        print("="*80)
        
        results = []
        successful = 0
        
        for test in SELECTED_TESTS:
            result = self.run_single_test(test)
            results.append(result)
            if result["success"]:
                successful += 1
        
        # Generate summary report
        self.generate_summary_report(results, successful)
        
        return {
            "total": len(SELECTED_TESTS),
            "successful": successful,
            "results": results
        }
    
    def generate_summary_report(self, results: List[Dict], successful: int):
        """Generate summary report"""
        report_path = self.output_dir / f"summary_{self.timestamp}.txt"
        
        with open(report_path, "w") as f:
            f.write("="*80 + "\n")
            f.write("VIT vs RESNET MODEL COMPARISON - SUMMARY REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {self.timestamp}\n")
            f.write(f"Total Tests: {len(results)}\n")
            f.write(f"Successful: {successful}/{len(results)}\n")
            f.write(f"Image Generator: Pollinations Turbo\n\n")
            
            f.write("="*80 + "\n")
            f.write("TEST RESULTS\n")
            f.write("="*80 + "\n\n")
            
            for i, result in enumerate(results, 1):
                if result["success"]:
                    test = result["test"]
                    f.write(f"{i}. {test.name} (Severity: {test.severity}/10)\n")
                    f.write(f"   Prompt: '{test.prompt}'\n")
                    f.write(f"   Output: {result['output_dir']}\n")
                    f.write(f"   ✅ SUCCESS\n\n")
                else:
                    f.write(f"{i}. FAILED: {result.get('error', 'Unknown error')}\n\n")
            
            f.write("="*80 + "\n")
            f.write("ANALYSIS GUIDE\n")
            f.write("="*80 + "\n\n")
            f.write("To evaluate which model is better, look for:\n\n")
            f.write("1. STRUCTURAL ACCURACY\n")
            f.write("   - Does the ASCII preserve the main shapes/structure?\n")
            f.write("   - Are edges and boundaries clear?\n\n")
            f.write("2. CHARACTER SELECTION\n")
            f.write("   - Are characters appropriate for the brightness?\n")
            f.write("   - Is there good contrast between light/dark areas?\n\n")
            f.write("3. DETAIL PRESERVATION\n")
            f.write("   - Are important details visible?\n")
            f.write("   - Is there good use of character variety?\n\n")
            f.write("4. OVERALL READABILITY\n")
            f.write("   - Can you recognize the subject?\n")
            f.write("   - Does it look aesthetically pleasing?\n\n")
            f.write("Compare the outputs in each test's comparison.txt file.\n")
        
        print(f"\n📊 Summary report saved to: {report_path}")
        print(f"\n📁 All outputs in: {self.output_dir}")


def main():
    """Run the comparison tests"""
    comparator = ModelComparator()
    summary = comparator.run_all_tests()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE!")
    print("="*80)
    print(f"✅ Successful: {summary['successful']}/{summary['total']}")
    print(f"📁 Results saved to: {comparator.output_dir}")
    print("\nNext steps:")
    print("1. Review the comparison.txt files in each test folder")
    print("2. Check the summary report for overview")
    print("3. Determine which model performs better overall")
    print("="*80)


if __name__ == "__main__":
    main()
