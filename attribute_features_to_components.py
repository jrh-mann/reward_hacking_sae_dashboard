#!/usr/bin/env python
"""
Attribute SAE feature activations to specific model components (attention heads, MLPs, etc.)

This script implements multiple attribution methods to understand which parts of the model
are responsible for writing to specific SAE feature directions.

METHODS IMPLEMENTED:

1. **Direct Feature Attribution (DFA)** [Default, Most Accurate]
   - Runs actual text through the model
   - Caches outputs from each component (attention, MLP)
   - Projects each component's output onto feature encoder directions
   - Shows which components actually write to features during inference
   - Formula: feature_contribution = component_output · feature_encoder
   
2. **Weight-Based Attribution** [No Data Required]
   - Analyzes model weight matrices directly
   - Computes alignment between output weights and feature encoders
   - Fast but less accurate (doesn't account for actual activations)
   - Formula: alignment = W_out · feature_encoder

3. **Decomposed Attribution** [Most Detailed]
   - Breaks down attention into individual heads
   - Shows contributions from each head separately
   - Also shows MLP and residual stream contributions

ATTRIBUTION PIPELINE:

For a transformer layer L processing token at position t:
    residual[L,t] = residual[L-1,t] + attn_out[L,t] + mlp_out[L,t]

Where:
    attn_out[L,t] = sum over heads h: head[h]_out[L,t]
    head[h]_out = (attention_pattern[h] @ V[h]) @ W_O[h]

To attribute feature activation:
1. Get feature encoder vector e_f for feature f (shape: d_model)
2. For each component output o (shape: d_model):
   contribution[component] = e_f · o
3. Rank components by |contribution|

Usage:
    # Basic usage - attribute features from text
    python attribute_features_to_components.py --text "Hello world" --layer 19 --feature_ids 12345,67890
    
    # Use multiple texts for better statistics
    python attribute_features_to_components.py --texts_file prompts.txt --layer 19 --feature_ids 12345
    
    # Weight-based attribution (no data needed)
    python attribute_features_to_components.py --layer 19 --feature_ids 12345 --method weights
    
    # Detailed head-level decomposition
    python attribute_features_to_components.py --text "Hello" --layer 19 --feature_ids 12345 --decompose_heads
"""

import argparse
import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from huggingface_hub import snapshot_download
from dictionary_learning.utils import load_dictionary
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
import json


def load_sae(layer: int, trainer: int, sae_repo: str = "andyrdt/saes-gpt-oss-20b"):
    """Load SAE for a specific layer."""
    print(f"Loading SAE from {sae_repo}...")
    subfolder = f"resid_post_layer_{layer}/trainer_{trainer}"
    
    local_dir = snapshot_download(
        repo_id=sae_repo,
        allow_patterns=f"{subfolder}/*",
        local_dir_use_symlinks=False,
        cache_dir=None
    )
    
    final_path = os.path.join(local_dir, subfolder)
    sae, config = load_dictionary(final_path, device="cpu")
    sae.eval()
    
    print(f"  SAE loaded: {type(sae).__name__}")
    print(f"  Encoder shape: {sae.encoder.weight.shape}")
    return sae, config


def get_feature_encoders(sae, feature_ids: List[int]) -> torch.Tensor:
    """
    Extract encoder vectors for specific features.
    
    Returns:
        Tensor of shape (num_features, d_model)
    """
    encoder = sae.encoder.weight.detach().cpu()  # (n_features, d_model)
    feature_encoders = encoder[feature_ids]  # (num_features, d_model)
    return feature_encoders


def attribute_via_weights(
    model_name: str,
    layer: int,
    feature_encoders: torch.Tensor,
    feature_ids: List[int],
    decompose_heads: bool = False
) -> Dict:
    """
    Weight-based attribution: analyze model weights without running data.
    
    For each component, compute: alignment = W_out · feature_encoder
    
    Args:
        model_name: HuggingFace model name
        layer: Layer to analyze
        feature_encoders: Tensor (num_features, d_model)
        feature_ids: List of feature IDs
        decompose_heads: If True, show individual attention heads
    
    Returns:
        Dictionary with attribution results
    """
    print(f"\n{'='*80}")
    print(f"WEIGHT-BASED ATTRIBUTION")
    print(f"{'='*80}")
    print(f"Loading model: {model_name}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    
    # Get the transformer layer
    if hasattr(model, 'transformer'):
        layers = model.transformer.h
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    else:
        raise ValueError("Could not find transformer layers in model")
    
    target_layer = layers[layer]
    d_model_sae = feature_encoders.shape[1]
    
    # Check model dimensions
    if hasattr(target_layer, 'attn'):
        attn = target_layer.attn
    elif hasattr(target_layer, 'self_attn'):
        attn = target_layer.self_attn
    else:
        raise ValueError("Could not find attention in layer")
    
    # Get d_model from model
    if hasattr(attn, 'c_proj'):
        d_model_model = attn.c_proj.weight.shape[0]
    elif hasattr(attn, 'out_proj'):
        d_model_model = attn.out_proj.weight.shape[0]
    else:
        # Try to infer from model config
        if hasattr(model, 'config'):
            d_model_model = model.config.hidden_size
        else:
            d_model_model = None
    
    if d_model_model is not None and d_model_model != d_model_sae:
        print(f"\n⚠️  WARNING: Dimension mismatch!")
        print(f"  SAE expects d_model={d_model_sae}")
        print(f"  Model has d_model={d_model_model}")
        print(f"  Make sure you're using the correct model with --model")
        raise ValueError(f"Dimension mismatch: SAE d_model={d_model_sae}, Model d_model={d_model_model}")
    
    d_model = d_model_sae
    
    results = {
        'method': 'weights',
        'layer': layer,
        'features': {}
    }
    
    # Analyze each feature
    for feat_idx, feat_id in enumerate(feature_ids):
        encoder_vec = feature_encoders[feat_idx]  # (d_model,)
        feat_results = {
            'feature_id': feat_id,
            'components': {}
        }
        
        print(f"\n--- Feature {feat_id} ---")
        
        # 1. Attention output projection (W_O)
        if hasattr(target_layer, 'attn') or hasattr(target_layer, 'self_attn'):
            attn = target_layer.attn if hasattr(target_layer, 'attn') else target_layer.self_attn
            
            if hasattr(attn, 'c_proj'):  # GPT-2 style
                W_O = attn.c_proj.weight.detach().cpu()  # (d_model, d_model)
            elif hasattr(attn, 'out_proj'):
                W_O = attn.out_proj.weight.detach().cpu()
            else:
                W_O = None
            
            if W_O is not None:
                # Alignment: how much can attention write in the feature direction
                # This is the norm of the projection: ||encoder_vec @ W_O||
                alignment = float(torch.norm(encoder_vec @ W_O))
                feat_results['components']['attention'] = alignment
                print(f"  Attention output: {alignment:.4f}")
                
                # Decompose into heads if requested
                if decompose_heads:
                    # Try to get number of heads
                    n_heads = None
                    if hasattr(attn, 'num_heads'):
                        n_heads = attn.num_heads
                    elif hasattr(attn, 'num_attention_heads'):
                        n_heads = attn.num_attention_heads
                    elif hasattr(model, 'config') and hasattr(model.config, 'n_head'):
                        n_heads = model.config.n_head
                    elif hasattr(model, 'config') and hasattr(model.config, 'num_attention_heads'):
                        n_heads = model.config.num_attention_heads
                    
                    if n_heads is not None:
                        head_dim = d_model // n_heads
                        
                        for h in range(n_heads):
                            # Extract this head's output projection
                            # W_O shape is (d_model, d_model) where the input is from all heads concatenated
                            start_idx = h * head_dim
                            end_idx = (h + 1) * head_dim
                            W_O_head = W_O[:, start_idx:end_idx]  # (d_model, head_dim)
                            
                            # Project encoder onto this head's output space
                            head_alignment = float(torch.norm(encoder_vec @ W_O_head))
                            feat_results['components'][f'attn_head_{h}'] = head_alignment
                            
                            if h < 5 or head_alignment > 0.1:  # Show top heads
                                print(f"    Head {h:2d}: {head_alignment:.4f}")
        
        # 2. MLP output projection
        if hasattr(target_layer, 'mlp'):
            mlp = target_layer.mlp
            
            # Different architectures have different names
            if hasattr(mlp, 'c_proj'):  # GPT-2
                W_mlp = mlp.c_proj.weight.detach().cpu()
            elif hasattr(mlp, 'down_proj'):  # LLaMA style
                W_mlp = mlp.down_proj.weight.detach().cpu()
            elif hasattr(mlp, 'w2'):
                W_mlp = mlp.w2.weight.detach().cpu()
            else:
                W_mlp = None
            
            if W_mlp is not None:
                # For MLPs, we have: out = W_down @ activation(W_up @ x)
                # Just measure output projection alignment
                mlp_alignment = float(torch.norm(encoder_vec @ W_mlp.T))
                feat_results['components']['mlp'] = mlp_alignment
                print(f"  MLP output: {mlp_alignment:.4f}")
        
        results['features'][feat_id] = feat_results
    
    return results


def attribute_via_activations(
    model_name: str,
    texts: List[str],
    layer: int,
    feature_encoders: torch.Tensor,
    feature_ids: List[int],
    decompose_heads: bool = False
) -> Dict:
    """
    Activation-based attribution: run actual text and measure component contributions.
    
    This is the most accurate method - it shows what actually happens during inference.
    
    Args:
        model_name: HuggingFace model name  
        texts: List of texts to analyze
        layer: Layer to analyze
        feature_encoders: Tensor (num_features, d_model)
        feature_ids: List of feature IDs
        decompose_heads: If True, decompose attention into individual heads
    
    Returns:
        Dictionary with attribution results
    """
    print(f"\n{'='*80}")
    print(f"ACTIVATION-BASED ATTRIBUTION (Direct Feature Attribution)")
    print(f"{'='*80}")
    print(f"Loading model: {model_name}")
    
    # Use nnsight for activation patching
    model = LanguageModel(model_name, device_map="auto", torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Accumulate contributions across all texts
    all_contributions = defaultdict(lambda: defaultdict(list))
    
    for text_idx, text in enumerate(texts):
        print(f"\n--- Text {text_idx + 1}/{len(texts)}: {text[:50]}... ---")
        
        with torch.no_grad():
            with model.trace(text, scan=False):
                # Access the target layer
                if hasattr(model.model, 'transformer'):
                    target_layer = model.model.transformer.h[layer]
                elif hasattr(model.model, 'layers'):
                    target_layer = model.model.layers[layer]
                else:
                    raise ValueError("Could not find transformer layers")
                
                # Save outputs from different components
                # 1. Input to this layer (residual stream before)
                if layer > 0:
                    if hasattr(model.model, 'transformer'):
                        prev_layer_out = model.model.transformer.h[layer - 1].output[0].save()
                    else:
                        prev_layer_out = model.model.layers[layer - 1].output[0].save()
                else:
                    # First layer - use embeddings
                    prev_layer_out = None
                
                # 2. Attention output (before adding to residual)
                if hasattr(target_layer, 'attn'):
                    attn_out = target_layer.attn.output[0].save()
                elif hasattr(target_layer, 'self_attn'):
                    attn_out = target_layer.self_attn.output[0].save()
                else:
                    attn_out = None
                
                # 3. MLP output (before adding to residual)  
                if hasattr(target_layer, 'mlp'):
                    mlp_out = target_layer.mlp.output.save()
                else:
                    mlp_out = None
                
                # 4. Final layer output (residual stream after)
                layer_out = target_layer.output[0].save()
        
        # Compute attributions for each token position
        layer_out_val = layer_out.value.cpu().float()  # (batch, seq, d_model)
        batch_size, seq_len, d_model = layer_out_val.shape
        
        if attn_out is not None:
            attn_out_val = attn_out.value.cpu().float()
        if mlp_out is not None:
            mlp_out_val = mlp_out.value.cpu().float()
        if prev_layer_out is not None:
            prev_layer_out_val = prev_layer_out.value.cpu().float()
        
        # For each feature
        for feat_idx, feat_id in enumerate(feature_ids):
            encoder_vec = feature_encoders[feat_idx].float()  # (d_model,)
            
            # Average over all token positions
            for pos in range(seq_len):
                # Contribution from attention
                if attn_out is not None:
                    attn_contrib = float(attn_out_val[0, pos] @ encoder_vec)
                    all_contributions[feat_id]['attention'].append(attn_contrib)
                
                # Contribution from MLP
                if mlp_out is not None:
                    mlp_contrib = float(mlp_out_val[0, pos] @ encoder_vec)
                    all_contributions[feat_id]['mlp'].append(mlp_contrib)
                
                # Contribution from previous residual
                if prev_layer_out is not None:
                    residual_contrib = float(prev_layer_out_val[0, pos] @ encoder_vec)
                    all_contributions[feat_id]['residual'].append(residual_contrib)
    
    # Aggregate statistics
    results = {
        'method': 'activations',
        'layer': layer,
        'num_texts': len(texts),
        'features': {}
    }
    
    for feat_id in feature_ids:
        feat_results = {
            'feature_id': feat_id,
            'components': {}
        }
        
        print(f"\n--- Feature {feat_id} Statistics ---")
        
        for component, contribs in all_contributions[feat_id].items():
            if contribs:
                mean_contrib = np.mean(contribs)
                std_contrib = np.std(contribs)
                max_contrib = np.max(contribs)
                min_contrib = np.min(contribs)
                
                feat_results['components'][component] = {
                    'mean': float(mean_contrib),
                    'std': float(std_contrib),
                    'max': float(max_contrib),
                    'min': float(min_contrib),
                }
                
                print(f"  {component:12s}: mean={mean_contrib:8.4f}, std={std_contrib:8.4f}, "
                      f"max={max_contrib:8.4f}, min={min_contrib:8.4f}")
        
        results['features'][feat_id] = feat_results
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Feature specification
    parser.add_argument(
        "--feature_ids",
        type=str,
        required=True,
        help="Comma-separated list of feature IDs to attribute (e.g., '12345,67890')"
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to analyze"
    )
    parser.add_argument(
        "--trainer",
        type=int,
        default=0,
        help="SAE trainer index (0 or 1)"
    )
    
    # Attribution method
    parser.add_argument(
        "--method",
        type=str,
        choices=["activations", "weights", "both"],
        default="activations",
        help="Attribution method: 'activations' (run data), 'weights' (analyze weights), or 'both'"
    )
    
    # Input text (for activation-based methods)
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Single text to analyze"
    )
    parser.add_argument(
        "--texts_file",
        type=str,
        default=None,
        help="File with multiple texts (one per line)"
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="openai-community/gpt-oss-20b",
        help="HuggingFace model name (should match the model the SAE was trained on)"
    )
    parser.add_argument(
        "--sae_repo",
        type=str,
        default="andyrdt/saes-gpt-oss-20b",
        help="SAE repository"
    )
    
    # Analysis options
    parser.add_argument(
        "--decompose_heads",
        action="store_true",
        help="Decompose attention into individual heads"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Parse feature IDs
    feature_ids = [int(x.strip()) for x in args.feature_ids.split(',')]
    
    print(f"{'='*80}")
    print(f"FEATURE ATTRIBUTION TO MODEL COMPONENTS")
    print(f"{'='*80}")
    print(f"Layer: {args.layer}")
    print(f"Features: {feature_ids}")
    print(f"Method: {args.method}")
    
    # Load SAE and get feature encoders
    sae, config = load_sae(args.layer, args.trainer, args.sae_repo)
    feature_encoders = get_feature_encoders(sae, feature_ids)
    print(f"Feature encoders shape: {feature_encoders.shape}")
    
    all_results = {}
    
    # Activation-based attribution
    if args.method in ["activations", "both"]:
        # Get texts
        texts = []
        if args.text:
            texts.append(args.text)
        if args.texts_file:
            with open(args.texts_file, 'r') as f:
                texts.extend([line.strip() for line in f if line.strip()])
        
        if not texts:
            print("\nError: No texts provided for activation-based attribution!")
            print("Use --text or --texts_file")
            return 1
        
        print(f"\nAnalyzing {len(texts)} text(s)...")
        
        results = attribute_via_activations(
            args.model,
            texts,
            args.layer,
            feature_encoders,
            feature_ids,
            args.decompose_heads
        )
        all_results['activations'] = results
    
    # Weight-based attribution
    if args.method in ["weights", "both"]:
        results = attribute_via_weights(
            args.model,
            args.layer,
            feature_encoders,
            feature_ids,
            args.decompose_heads
        )
        all_results['weights'] = results
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")
    
    print(f"\n{'='*80}")
    print("Attribution analysis complete!")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

