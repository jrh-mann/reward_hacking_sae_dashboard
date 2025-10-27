#!/usr/bin/env python
"""
Find SAE features most similar to steering vectors at each layer.

Usage:
    python find_similar_features.py --steering_vectors_path steering_vectors.pt
    
Note: This script downloads SAE weights from HuggingFace (andyrdt/saes-gpt-oss-20b).
The weights are separate from the MAE cache which only contains pre-computed examples.
"""
import argparse
import os
import sys
import torch
from typing import List, Tuple
from huggingface_hub import snapshot_download
from dictionary_learning.utils import load_dictionary


def load_steering_vectors(path: str) -> torch.Tensor:
    """Load steering vectors from .pt file."""
    steering = torch.load(path, map_location="cpu")
    if isinstance(steering, dict):
        # If it's a dict, try to extract the tensor
        if "steering_vectors" in steering:
            steering = steering["steering_vectors"]
        else:
            # Take the first tensor value
            steering = next(iter(steering.values()))
    return steering


def load_sae_decoder(layer: int, trainer: int, sae_repo: str = "andyrdt/saes-gpt-oss-20b") -> torch.Tensor:
    """
    Load SAE decoder weights for a specific layer and trainer.
    
    Downloads from HuggingFace and uses dictionary_learning to load the SAE.
    Returns the decoder weight matrix with shape (d_model, n_features).
    """
    print(f"  Loading SAE weights from HuggingFace ({sae_repo})...")
    subfolder = f"resid_post_layer_{layer}/trainer_{trainer}"
    
    # Download SAE weights (uses HF cache, won't re-download if already present)
    local_dir = snapshot_download(
        repo_id=sae_repo,
        allow_patterns=f"{subfolder}/*",
        local_dir_use_symlinks=False,
        cache_dir=None  # Uses default ~/.cache/huggingface/hub/
    )
    
    final_path = os.path.join(local_dir, subfolder)
    sae, config = load_dictionary(final_path, device="cpu")
    
    # Extract decoder weights
    # The BatchTopKSAE has a decoder which is a nn.Linear module
    if hasattr(sae, 'decoder') and hasattr(sae.decoder, 'weight'):
        W_dec = sae.decoder.weight.detach().cpu()
        # decoder.weight has shape (d_model, n_features)
        # Each column represents a feature's direction in model space
    elif hasattr(sae, 'W_dec'):
        W_dec = sae.W_dec.detach().cpu()
    else:
        raise AttributeError(f"Could not find decoder weights in SAE. Available attributes: {dir(sae)}")
    
    print(f"  SAE decoder shape: {W_dec.shape} (d_model x n_features)")
    return W_dec


def compute_cosine_similarity(vec: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """
    Compute cosine similarity between a vector and all columns of a matrix.
    
    Args:
        vec: shape (d_model,)
        matrix: shape (d_model, n_features)
    
    Returns:
        similarities: shape (n_features,)
    """
    # Normalize vector
    vec_norm = vec / (vec.norm() + 1e-8)
    
    # Normalize each feature (column)
    matrix_norm = matrix / (matrix.norm(dim=0, keepdim=True) + 1e-8)
    
    # Compute dot product
    similarities = vec_norm @ matrix_norm
    
    return similarities


def find_top_k_features(
    steering_vector: torch.Tensor,
    decoder: torch.Tensor,
    k: int = 10
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar features to the steering vector.
    
    Returns:
        List of (feature_id, similarity_score) tuples
    """
    similarities = compute_cosine_similarity(steering_vector, decoder)
    top_k_indices = torch.topk(similarities, k=min(k, len(similarities)))
    
    results = [
        (int(idx), float(sim)) 
        for idx, sim in zip(top_k_indices.indices, top_k_indices.values)
    ]
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Find SAE features most similar to steering vectors"
    )
    parser.add_argument(
        "--steering_vectors_path",
        type=str,
        required=True,
        help="Path to steering_vectors.pt file (shape: [24, 2880])"
    )
    parser.add_argument(
        "--sae_repo",
        type=str,
        default="andyrdt/saes-gpt-oss-20b",
        help="HuggingFace repo containing SAE weights"
    )
    parser.add_argument(
        "--trainer",
        type=int,
        default=0,
        help="Trainer index (0=k64 or 1=k128)"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top similar features to report per layer"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="3,7,11,15,19,23",
        help="Comma-separated list of layers to analyze"
    )
    parser.add_argument(
        "--mae_cache_dir",
        type=str,
        default=".mae_cache",
        help="Directory where MAE data is cached (for dashboard URLs)"
    )
    
    args = parser.parse_args()
    
    # Load steering vectors
    print(f"Loading steering vectors from {args.steering_vectors_path}...")
    steering_vectors = load_steering_vectors(args.steering_vectors_path)
    print(f"Steering vectors shape: {steering_vectors.shape}")
    
    if steering_vectors.shape[0] != 24:
        print(f"Warning: Expected 24 layers, got {steering_vectors.shape[0]}")
    
    layers = [int(x.strip()) for x in args.layers.split(",")]
    
    print(f"\nAnalyzing layers: {layers}")
    print(f"SAE repo: {args.sae_repo}")
    print(f"Trainer: {args.trainer}")
    print(f"Top-K: {args.top_k}\n")
    print("=" * 80)
    
    # Process each layer
    for layer_idx in layers:
        print(f"\nLayer {layer_idx}")
        print("-" * 80)
        
        try:
            # Get steering vector for this layer
            if layer_idx >= steering_vectors.shape[0]:
                print(f"Warning: Layer {layer_idx} not in steering vectors (max: {steering_vectors.shape[0]-1})")
                continue
            
            steering_vec = steering_vectors[layer_idx]
            print(f"Steering vector shape: {steering_vec.shape}")
            
            # Load SAE decoder for this layer
            decoder = load_sae_decoder(layer_idx, args.trainer, args.sae_repo)
            
            # Verify dimensions match
            # decoder should be (d_model, n_features) and steering_vec should be (d_model,)
            if decoder.shape[0] != steering_vec.shape[0]:
                print(f"  Error: Dimension mismatch!")
                print(f"    Decoder shape: {decoder.shape} (expected: [{steering_vec.shape[0]}, n_features])")
                print(f"    Steering vector shape: {steering_vec.shape}")
                continue
            
            # Find top-k similar features
            print(f"  Computing cosine similarities with {decoder.shape[1]} features...")
            top_features = find_top_k_features(steering_vec, decoder, k=args.top_k)
            
            print(f"\n  Top {args.top_k} most similar features:")
            for rank, (feat_id, similarity) in enumerate(top_features, 1):
                print(f"    {rank:2d}. Feature {feat_id:5d}: similarity = {similarity:7.4f}")
            
            # Provide URL to view features
            feat_ids = ",".join(str(f[0]) for f in top_features[:5])
            print(f"\n  View top 5 in dashboard:")
            print(f"    http://localhost:7863/?model=gpt&layer={layer_idx}&trainer={args.trainer}&fids={feat_ids}")
            
        except Exception as e:
            print(f"  Error processing layer {layer_idx}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

