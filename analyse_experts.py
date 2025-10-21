"""
Track router logits at each layer and token using nnsight.

This script loads a model, runs it on a prompt, and extracts router logits
from each MoE layer at each token position.
"""

import torch
from nnsight import LanguageModel
from transformers import AutoTokenizer
import numpy as np
from pathlib import Path
import pickle

# Configuration
MODEL_PATH = "openai/gpt-oss-20b"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model_with_nnsight(model_path=MODEL_PATH):
    """Load model using nnsight for tracing."""
    print(f"Loading model with nnsight: {model_path}")
    
    # Load tokenizer separately
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model with nnsight
    model = LanguageModel(model_path, device_map="auto", dispatch=True)
    
    print(f"Model loaded successfully")
    return tokenizer, model


def extract_router_logits(model, tokenizer, prompt, max_new_tokens=50):
    """
    Extract router logits at each layer for each token during generation.
    
    Args:
        model: NNsight LanguageModel
        tokenizer: HuggingFace tokenizer
        prompt: Input text prompt
        max_new_tokens: Maximum tokens to generate
        
    Returns:
        dict with router logits, tokens, and metadata
    """
    print(f"\nPrompt: {prompt}")
    
    # Format prompt if needed (for chat models)
    try:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
            reasoning_effort="low"
        )
    except:
        # If chat template not available, use raw prompt
        formatted_prompt = prompt
    
    print(f"\nFormatted prompt: {formatted_prompt[:200]}...")
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(model.device)
    prompt_length = input_ids.shape[1]
    
    print(f"Prompt tokens: {prompt_length}")
    
    # Storage for router logits
    # Structure: {layer_idx: {token_position: logits}}
    router_logits_by_layer = {}
    
    # First, let's inspect the model structure to find MoE layers
    print("\n" + "="*60)
    print("Inspecting model structure for MoE layers...")
    print("="*60)
    
    with model.trace(formatted_prompt, max_new_tokens=max_new_tokens, scan=False) as tracer:
        # Access the model structure
        # For GPT-OSS, we expect transformer.h[i] to contain layers
        # And MoE layers should have a router/gate component
        
        print(f"\nModel type: {type(model.model)}")
        print(f"Model config: {model.model.config}")
        
        # Try to find MoE layers
        # Common patterns: layer.mlp.router, layer.mlp.gate, layer.block_sparse_moe.gate
        layers = model.model.transformer.h
        print(f"\nNumber of layers: {len(layers)}")
        
        # Collect router logits from each layer
        for layer_idx in range(len(layers)):
            layer = layers[layer_idx]
            
            # Check if this layer has MoE components
            # Common attributes: router, gate, block_sparse_moe
            try:
                # Try different possible locations for router
                if hasattr(layer.mlp, 'router'):
                    router = layer.mlp.router
                    print(f"Layer {layer_idx}: Found router at layer.mlp.router")
                elif hasattr(layer.mlp, 'gate'):
                    router = layer.mlp.gate
                    print(f"Layer {layer_idx}: Found gate at layer.mlp.gate")
                elif hasattr(layer, 'block_sparse_moe'):
                    if hasattr(layer.block_sparse_moe, 'gate'):
                        router = layer.block_sparse_moe.gate
                        print(f"Layer {layer_idx}: Found gate at layer.block_sparse_moe.gate")
                    elif hasattr(layer.block_sparse_moe, 'router'):
                        router = layer.block_sparse_moe.router
                        print(f"Layer {layer_idx}: Found router at layer.block_sparse_moe.router")
                    else:
                        continue
                else:
                    # Not an MoE layer, skip
                    continue
                
                # Save router output (logits)
                router_output = router.output.save()
                router_logits_by_layer[layer_idx] = router_output
                
            except AttributeError as e:
                # This layer doesn't have MoE, skip
                continue
        
        # Also save the generated tokens
        generated_output = model.generator.output.save()
    
    print(f"\nFound MoE layers: {list(router_logits_by_layer.keys())}")
    
    # Extract the saved values
    results = {
        'prompt': prompt,
        'formatted_prompt': formatted_prompt,
        'prompt_length': prompt_length,
        'generated_tokens': generated_output.value if hasattr(generated_output, 'value') else generated_output,
        'router_logits': {},
    }
    
    # Process router logits for each layer
    for layer_idx, router_output in router_logits_by_layer.items():
        # Get the actual tensor value
        logits = router_output.value if hasattr(router_output, 'value') else router_output
        
        if logits is not None:
            results['router_logits'][layer_idx] = logits
            print(f"\nLayer {layer_idx} router logits shape: {logits.shape if isinstance(logits, torch.Tensor) else 'N/A'}")
    
    # Decode generated text
    if isinstance(results['generated_tokens'], torch.Tensor):
        generated_text = tokenizer.decode(results['generated_tokens'][0])
        results['generated_text'] = generated_text
        print(f"\nGenerated text: {generated_text[:500]}...")
    
    return results


def analyze_router_logits(results, output_dir="router_analysis"):
    """
    Analyze and save router logits.
    
    Args:
        results: Dictionary from extract_router_logits
        output_dir: Directory to save results
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("ROUTER LOGITS ANALYSIS")
    print("="*60)
    
    router_logits = results['router_logits']
    
    if not router_logits:
        print("No router logits found! This might not be an MoE model.")
        return
    
    # Analyze each layer's router logits
    for layer_idx, logits in router_logits.items():
        if not isinstance(logits, torch.Tensor):
            print(f"\nLayer {layer_idx}: No logits (skipping)")
            continue
            
        print(f"\nLayer {layer_idx}:")
        print(f"  Shape: {logits.shape}")
        print(f"  Data type: {logits.dtype}")
        
        # If shape is [batch, seq_len, num_experts], analyze routing distribution
        if len(logits.shape) == 3:
            batch_size, seq_len, num_experts = logits.shape
            print(f"  Batch size: {batch_size}")
            print(f"  Sequence length: {seq_len}")
            print(f"  Number of experts: {num_experts}")
            
            # Convert to probabilities
            routing_probs = torch.softmax(logits, dim=-1)
            
            # Analyze which experts are most active
            top_experts = torch.argmax(routing_probs, dim=-1)  # [batch, seq_len]
            
            print(f"\n  Top expert per token position (first 20 tokens):")
            for pos in range(min(20, seq_len)):
                expert_id = top_experts[0, pos].item()
                prob = routing_probs[0, pos, expert_id].item()
                print(f"    Token {pos}: Expert {expert_id} (prob: {prob:.3f})")
            
            # Overall expert usage statistics
            expert_usage = torch.bincount(top_experts[0], minlength=num_experts)
            print(f"\n  Expert usage across all tokens:")
            for expert_id in range(num_experts):
                count = expert_usage[expert_id].item()
                pct = 100 * count / seq_len
                print(f"    Expert {expert_id}: {count}/{seq_len} tokens ({pct:.1f}%)")
    
    # Save results
    print(f"\n\nSaving results to {output_path}/...")
    
    # Save raw data
    with open(output_path / "router_logits.pkl", 'wb') as f:
        pickle.dump(results, f)
    print(f"  - router_logits.pkl: Raw router logits and metadata")
    
    # Save human-readable summary
    with open(output_path / "summary.txt", 'w') as f:
        f.write("Router Logits Analysis\n")
        f.write("="*60 + "\n\n")
        f.write(f"Prompt: {results['prompt']}\n\n")
        f.write(f"Prompt length: {results['prompt_length']} tokens\n")
        f.write(f"Generated text:\n{results.get('generated_text', 'N/A')}\n\n")
        f.write(f"MoE Layers found: {list(router_logits.keys())}\n\n")
        
        for layer_idx, logits in router_logits.items():
            if isinstance(logits, torch.Tensor):
                f.write(f"\nLayer {layer_idx}:\n")
                f.write(f"  Shape: {logits.shape}\n")
                
                if len(logits.shape) == 3:
                    routing_probs = torch.softmax(logits, dim=-1)
                    top_experts = torch.argmax(routing_probs, dim=-1)
                    f.write(f"  Top experts per token: {top_experts[0].tolist()}\n")
    
    print(f"  - summary.txt: Human-readable summary")
    print("\nAnalysis complete!")


def main():
    """Main function to run router logit tracking."""
    
    # Load model
    tokenizer, model = load_model_with_nnsight()
    
    # Test prompt
    prompt = "The quick brown fox jumps over the lazy dog."
    
    # Extract router logits
    results = extract_router_logits(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=30
    )
    
    # Analyze and save results
    analyze_router_logits(results)


if __name__ == "__main__":
    main()

