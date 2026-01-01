"""
Stress Test Suite for ASCII Art Generator
==========================================
Tests the LLM rewriter with challenging prompts.
"""
import sys
sys.path.insert(0, '.')

from ascii_gen.llm_rewriter import LLMPromptRewriter

# Initialize rewriter
rewriter = LLMPromptRewriter()

# Test prompts
STRESS_TESTS = [
    {
        "name": "Ambiguity Stress Test",
        "prompt": "bat on a trunk next to a mouse",
        "check": ["bat", "trunk", "mouse"],  # All 3 must appear
        "issue": "Tests 3 ambiguous terms (bat=animal/sports, trunk=tree/elephant, mouse=animal/computer)"
    },
    {
        "name": "Complexity Bomb",
        "prompt": "cat sitting on a wooden chair in a cozy living room with a fireplace, bookshelf, lamp, rug, and window showing sunset",
        "check": ["cat", "chair"],  # At minimum these must appear
        "issue": "Tests complexity detection (7+ objects)"
    },
    {
        "name": "Impossible Request",
        "prompt": "photorealistic 3D rotating rainbow-colored holographic cat with flowing fur dancing in slow motion",
        "check": ["cat"],
        "issue": "Tests impossible constraints (3D, rotation, color, motion)"
    },
    {
        "name": "Vague Mystery",
        "prompt": "thing on stuff",
        "check": [],  # Should request clarification or provide default
        "issue": "Zero useful information - tests fallback behavior"
    },
    {
        "name": "Abstract Challenge",  
        "prompt": "freedom",
        "check": [],  # Should convert to symbolic representation
        "issue": "Tests abstract concept handling"
    },
    {
        "name": "Spatial Nightmare",
        "prompt": "small cat sitting on the back of a large chair next to a tall table",
        "check": ["cat", "chair", "table"],  # All 3 must appear
        "issue": "Tests spatial relationships and relative sizing"
    },
]

def run_stress_tests():
    print("=" * 70)
    print("🧪 ASCII ART STRESS TEST SUITE")
    print("=" * 70)
    
    results = []
    
    for i, test in enumerate(STRESS_TESTS, 1):
        print(f"\n{'='*70}")
        print(f"TEST {i}: {test['name']}")
        print(f"{'='*70}")
        print(f"📝 INPUT: \"{test['prompt']}\"")
        print(f"⚠️  ISSUE: {test['issue']}")
        
        # Run through rewriter
        result = rewriter.rewrite(test['prompt'])
        
        if result:
            print(f"\n✅ REWRITTEN PROMPT:")
            print(f"   {result.rewritten[:200]}...")
            
            print(f"\n📊 ANALYSIS:")
            print(f"   Classification: {result.classification}")
            print(f"   Complexity: {result.complexity_score:.2f}")
            print(f"   Simplification Applied: {result.simplification_applied}")
            
            # Check if required elements are preserved
            prompt_lower = result.rewritten.lower()
            missing = [elem for elem in test['check'] if elem not in prompt_lower]
            
            if missing:
                print(f"   ❌ MISSING ELEMENTS: {missing}")
                status = "FAIL"
            else:
                print(f"   ✅ All required elements present")
                status = "PASS"
            
            results.append({
                "test": test['name'],
                "status": status,
                "missing": missing,
                "prompt": result.rewritten[:100]
            })
        else:
            print(f"\n❌ REWRITER FAILED (returned None)")
            results.append({
                "test": test['name'],
                "status": "ERROR",
                "missing": test['check'],
                "prompt": ""
            })
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 STRESS TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')
    
    for r in results:
        emoji = "✅" if r['status'] == 'PASS' else "❌" if r['status'] == 'FAIL' else "⚠️"
        print(f"   {emoji} {r['test']}: {r['status']}")
        if r['missing']:
            print(f"      Missing: {r['missing']}")
    
    print(f"\n   TOTAL: {passed} passed, {failed} failed, {errors} errors")
    print("=" * 70)
    
    return results

if __name__ == "__main__":
    run_stress_tests()
