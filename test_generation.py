#!/usr/bin/env python
"""
Test script for the generation endpoint.
Run the server first with: uv run -m app.run_mae_app
"""

import requests
import json
import sys


def test_generation_api(
    prompt="The cat sat on the",
    layer=11,
    trainer=0,
    max_new_tokens=20,
    temperature=1.0,
    top_k_features=10,
    reasoning_effort="medium",
    base_url="http://localhost:7863"
):
    """Test the generation API endpoint."""
    
    print(f"Testing generation with prompt: '{prompt}'")
    print(f"Layer: {layer}, Trainer: {trainer}, Max tokens: {max_new_tokens}")
    print(f"Reasoning effort: {reasoning_effort}")
    print("-" * 60)
    
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "prompt": prompt,
                "layer": layer,
                "trainer": trainer,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_k_features": top_k_features,
                "reasoning_effort": reasoning_effort,
            },
            timeout=300  # 5 minute timeout
        )
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
        
        result = response.json()
        
        print(f"\n✅ Generation successful!")
        print(f"\nPrompt: {result['prompt_text']}")
        print(f"\nGenerated text: {result['generated_text']}")
        print(f"\nGenerated {len(result['generated_tokens'])} tokens")
        print(f"\n{'='*60}")
        print("Token-by-Token Features:")
        print(f"{'='*60}")
        
        for i, token_data in enumerate(result['features_per_token']):
            token_text = token_data['token_text'].replace('\n', '\\n').replace('\t', '\\t')
            print(f"\n[{i}] Token: '{token_text}'")
            print(f"    Top {min(5, len(token_data['top_features']))} features:")
            for feat in token_data['top_features'][:5]:
                print(f"      {feat['feature_id']:6d}: {feat['activation']:6.3f}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Connection error. Is the server running?")
        print("   Start it with: uv run -m app.run_mae_app")
        return False
    except requests.exceptions.Timeout:
        print("❌ Request timeout. Generation may take a while on first run.")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run test cases."""
    
    print("=" * 60)
    print("SAE Feature Generation API Test")
    print("=" * 60)
    print()
    
    # Test case 1: Simple completion
    print("\n" + "="*60)
    print("Test 1: Simple completion")
    print("="*60)
    success1 = test_generation_api(
        prompt="The cat sat on the",
        max_new_tokens=10,
        temperature=0.0,  # Greedy for reproducibility
    )
    
    # Test case 2: Different layer
    print("\n\n" + "="*60)
    print("Test 2: Different layer (layer 7)")
    print("="*60)
    success2 = test_generation_api(
        prompt="Once upon a time",
        layer=7,
        max_new_tokens=15,
        temperature=0.5,
    )
    
    # Test case 3: Different reasoning effort
    print("\n\n" + "="*60)
    print("Test 3: High reasoning effort")
    print("="*60)
    success3 = test_generation_api(
        prompt="In my opinion,",
        max_new_tokens=10,
        top_k_features=20,
        temperature=0.7,
        reasoning_effort="high",
    )
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    print(f"Test 1 (Simple completion):  {'✅ PASS' if success1 else '❌ FAIL'}")
    print(f"Test 2 (Different layer):    {'✅ PASS' if success2 else '❌ FAIL'}")
    print(f"Test 3 (High reasoning):     {'✅ PASS' if success3 else '❌ FAIL'}")
    
    all_passed = success1 and success2 and success3
    print(f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

