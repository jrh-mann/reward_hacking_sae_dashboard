#!/usr/bin/env python
"""
Decompose a steering vector into a sparse combination of SAE features.

This script finds which combination of n SAE feature directions best reconstructs
your steering vector using:
1. SAE encoder (optimal sparse decomposition)
2. Greedy search (interpretable iterative approach)
3. Top-k individual features (baseline)

Usage:
    python decompose_steering_vector.py --steering_vectors_path steering_vectors.pt --layer 19
"""
import argparse
import os
import sys
import torch
import numpy as np
from typing import List, Tuple, Dict
from huggingface_hub import snapshot_download
from dictionary_learning.utils import load_dictionary
import matplotlib.pyplot as plt


def load_steering_vectors(path: str) -> torch.Tensor:
    """Load steering vectors from .pt file."""
    steering = torch.load(path, map_location="cpu")
    if isinstance(steering, dict):
        if "steering_vectors" in steering:
            steering = steering["steering_vectors"]
        else:
            steering = next(iter(steering.values()))
    return steering


def load_sae(layer: int, trainer: int, sae_repo: str = "andyrdt/saes-gpt-oss-20b"):
    """Load the full SAE (encoder + decoder) for a specific layer and trainer."""
    print(f"  Loading SAE from HuggingFace ({sae_repo})...")
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
    return sae, config


def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    """Compute cosine similarity between two vectors."""
    vec1_norm = vec1 / (vec1.norm() + 1e-8)
    vec2_norm = vec2 / (vec2.norm() + 1e-8)
    return float(vec1_norm @ vec2_norm)


def compute_progressive_metrics(
    steering_vec: torch.Tensor,
    features: List[Tuple[int, float]],
    decoder: torch.Tensor,
    method_name: str = "method"
) -> Tuple[List[float], List[float]]:
    """
    Compute cosine similarity and L2 error as features are added progressively.
    
    Args:
        steering_vec: Original steering vector
        features: List of (feature_id, weight/activation) tuples
        decoder: SAE decoder weights
        method_name: Name of the method (for normalization handling)
    
    Returns:
        - List of cosine similarities (one per top-k)
        - List of L2 reconstruction errors (one per top-k)
    """
    cos_sims = []
    l2_errors = []
    
    reconstruction = torch.zeros_like(steering_vec)
    
    for i, (feat_id, weight) in enumerate(features):
        # Add this feature to the reconstruction
        if method_name == "topk":
            # Top-k uses normalized features
            feature_vec = decoder[:, feat_id] / (decoder[:, feat_id].norm() + 1e-8)
            reconstruction += feature_vec
        else:
            # Other methods use weighted features
            feature_vec = decoder[:, feat_id]
            reconstruction += weight * feature_vec
        
        # Compute metrics
        cos_sim = cosine_similarity(steering_vec, reconstruction)
        l2_error = (steering_vec - reconstruction).norm().item()
        
        cos_sims.append(cos_sim)
        l2_errors.append(l2_error)
    
    return cos_sims, l2_errors


def plot_progressive_metrics(
    results_dict: Dict[str, Tuple[List[float], List[float]]],
    output_path: str = "progressive_reconstruction.png"
):
    """
    Plot cosine similarity and L2 error as features are added progressively.
    
    Args:
        results_dict: Dictionary mapping method name to (cos_sims, l2_errors) tuple
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    colors = {
        'encoder': '#2E86AB',
        'greedy': '#A23B72',
        'topk': '#F18F01'
    }
    
    # Plot cosine similarity
    for method_name, (cos_sims, l2_errors) in results_dict.items():
        x = list(range(1, len(cos_sims) + 1))
        color = colors.get(method_name, '#333333')
        ax1.plot(x, cos_sims, marker='o', label=method_name.capitalize(), 
                 color=color, linewidth=2, markersize=4)
    
    ax1.set_xlabel('Number of Top Features', fontsize=12)
    ax1.set_ylabel('Cosine Similarity', fontsize=12)
    ax1.set_title('Cosine Similarity vs. Number of Features', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot L2 reconstruction error
    for method_name, (cos_sims, l2_errors) in results_dict.items():
        x = list(range(1, len(l2_errors) + 1))
        color = colors.get(method_name, '#333333')
        ax2.plot(x, l2_errors, marker='o', label=method_name.capitalize(),
                 color=color, linewidth=2, markersize=4)
    
    ax2.set_xlabel('Number of Top Features', fontsize=12)
    ax2.set_ylabel('L2 Reconstruction Error', fontsize=12)
    ax2.set_title('L2 Error vs. Number of Features', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  Graph saved to: {output_path}")
    plt.close()


def decompose_with_sae_encoder(
    steering_vec: torch.Tensor,
    sae,
    top_n: int = 20
) -> Tuple[List[Tuple[int, float]], torch.Tensor, float]:
    """
    Use the SAE encoder to decompose the steering vector.
    
    This is the most principled approach - the SAE is trained to decompose
    activations into sparse combinations of features.
    
    Returns:
        - List of (feature_id, activation) tuples for top-n features
        - Reconstructed vector
        - Cosine similarity with original
    """
    print(f"\n  Method 1: SAE Encoder Decomposition")
    print(f"  " + "-" * 60)
    
    with torch.no_grad():
        # Add batch dimension
        steering_batch = steering_vec.unsqueeze(0)
        
        # Encode: get sparse feature activations
        feature_acts = sae.encode(steering_batch).squeeze(0)
        
        # Decode: reconstruct from sparse features
        reconstructed = sae.decode(feature_acts.unsqueeze(0)).squeeze(0)
    
    # Get top-n active features
    nonzero_mask = feature_acts != 0
    nonzero_indices = torch.where(nonzero_mask)[0]
    nonzero_values = feature_acts[nonzero_mask]
    
    # Sort by absolute activation value
    sorted_indices = torch.argsort(torch.abs(nonzero_values), descending=True)
    
    top_features = [
        (int(nonzero_indices[idx]), float(nonzero_values[idx]))
        for idx in sorted_indices[:top_n]
    ]
    
    cos_sim = cosine_similarity(steering_vec, reconstructed)
    
    print(f"  Total active features: {len(nonzero_indices)}")
    print(f"  Cosine similarity: {cos_sim:.4f}")
    print(f"  L2 reconstruction error: {(steering_vec - reconstructed).norm():.4f}")
    
    return top_features, reconstructed, cos_sim


def decompose_greedy(
    steering_vec: torch.Tensor,
    decoder: torch.Tensor,
    max_features: int = 20,
    min_improvement: float = 0.001
) -> Tuple[List[Tuple[int, float]], torch.Tensor, float]:
    """
    Greedy search: iteratively add features that most improve cosine similarity.
    
    This is more interpretable but less optimal than the SAE encoder.
    
    Returns:
        - List of (feature_id, weight) tuples
        - Reconstructed vector
        - Cosine similarity with original
    """
    print(f"\n  Method 2: Greedy Feature Selection")
    print(f"  " + "-" * 60)
    
    selected_features = []
    selected_indices = set()
    current_reconstruction = torch.zeros_like(steering_vec)
    current_cos_sim = 0.0
    
    n_features = decoder.shape[1]
    
    for step in range(max_features):
        best_feature = None
        best_cos_sim = current_cos_sim
        best_reconstruction = current_reconstruction
        
        # Try adding each feature (that hasn't been selected)
        for feat_idx in range(n_features):
            if feat_idx in selected_indices:
                continue
            
            # Try adding this feature (with optimal weight via projection)
            feature_vec = decoder[:, feat_idx]
            
            # Optimal weight is the projection coefficient
            weight = float(steering_vec @ feature_vec / (feature_vec @ feature_vec + 1e-8))
            
            # Compute reconstruction with this feature added
            trial_reconstruction = current_reconstruction + weight * feature_vec
            trial_cos_sim = cosine_similarity(steering_vec, trial_reconstruction)
            
            if trial_cos_sim > best_cos_sim:
                best_cos_sim = trial_cos_sim
                best_feature = (feat_idx, weight)
                best_reconstruction = trial_reconstruction
        
        # Check if we improved enough
        improvement = best_cos_sim - current_cos_sim
        if improvement < min_improvement:
            print(f"  Stopped at {step} features (improvement < {min_improvement})")
            break
        
        if best_feature is None:
            print(f"  No more features improve the reconstruction")
            break
        
        # Add the best feature
        selected_features.append(best_feature)
        selected_indices.add(best_feature[0])
        current_reconstruction = best_reconstruction
        current_cos_sim = best_cos_sim
        
        if (step + 1) % 5 == 0:
            print(f"  Step {step + 1}: cos_sim = {current_cos_sim:.4f}")
    
    print(f"  Final features selected: {len(selected_features)}")
    print(f"  Final cosine similarity: {current_cos_sim:.4f}")
    
    return selected_features, current_reconstruction, current_cos_sim


def decompose_top_k(
    steering_vec: torch.Tensor,
    decoder: torch.Tensor,
    k: int = 20
) -> Tuple[List[Tuple[int, float]], torch.Tensor, float]:
    """
    Baseline: just use top-k most similar individual features (equal weighted sum).
    
    Returns:
        - List of (feature_id, similarity) tuples
        - Sum of top-k feature vectors
        - Cosine similarity with original
    """
    print(f"\n  Method 3: Top-K Most Similar Features (Baseline)")
    print(f"  " + "-" * 60)
    
    # Compute cosine similarities
    steering_norm = steering_vec / (steering_vec.norm() + 1e-8)
    decoder_norm = decoder / (decoder.norm(dim=0, keepdim=True) + 1e-8)
    similarities = steering_norm @ decoder_norm
    
    # Get top-k
    top_k_values, top_k_indices = torch.topk(similarities, k=k)
    
    # Sum top-k feature vectors (normalized)
    reconstruction = torch.zeros_like(steering_vec)
    for idx in top_k_indices:
        reconstruction += decoder[:, idx] / decoder[:, idx].norm()
    
    cos_sim = cosine_similarity(steering_vec, reconstruction)
    
    top_features = [
        (int(idx), float(sim))
        for idx, sim in zip(top_k_indices, top_k_values)
    ]
    
    print(f"  Top {k} features selected")
    print(f"  Cosine similarity of sum: {cos_sim:.4f}")
    
    return top_features, reconstruction, cos_sim


def main():
    parser = argparse.ArgumentParser(
        description="Decompose steering vector into SAE feature combinations"
    )
    parser.add_argument(
        "--steering_vectors_path",
        type=str,
        required=True,
        help="Path to steering_vectors.pt file"
    )
    parser.add_argument(
        "--layer",
        type=int,
        required=True,
        help="Layer to analyze (e.g., 19)"
    )
    parser.add_argument(
        "--trainer",
        type=int,
        default=0,
        help="Trainer index (0=k64 or 1=k128)"
    )
    parser.add_argument(
        "--top_n",
        type=int,
        default=20,
        help="Number of top features to display"
    )
    parser.add_argument(
        "--sae_repo",
        type=str,
        default="andyrdt/saes-gpt-oss-20b",
        help="HuggingFace repo containing SAE weights"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["encoder", "greedy", "topk", "all"],
        default="all",
        help="Decomposition method: 'encoder' (most principled), 'greedy' (best reconstruction), 'topk' (baseline), 'all' (compare all)"
    )
    parser.add_argument(
        "--output_graph",
        type=str,
        default=None,
        help="Output path for progressive reconstruction graph (default: progressive_reconstruction_layerN_trainerM.png)"
    )
    
    args = parser.parse_args()
    
    # Load steering vectors
    print(f"Loading steering vectors from {args.steering_vectors_path}...")
    steering_vectors = load_steering_vectors(args.steering_vectors_path)
    print(f"Steering vectors shape: {steering_vectors.shape}")
    
    if args.layer >= steering_vectors.shape[0]:
        print(f"Error: Layer {args.layer} not in steering vectors (max: {steering_vectors.shape[0]-1})")
        return 1
    
    steering_vec = steering_vectors[args.layer]
    print(f"Steering vector for layer {args.layer}: shape {steering_vec.shape}")
    print(f"Steering vector norm: {steering_vec.norm():.4f}")
    
    # Load SAE
    print(f"\nLoading SAE for layer {args.layer}, trainer {args.trainer}...")
    sae, config = load_sae(args.layer, args.trainer, args.sae_repo)
    
    # Get decoder for methods that need it
    decoder = sae.decoder.weight.detach().cpu()
    print(f"  Decoder shape: {decoder.shape} (d_model x n_features)")
    
    print("\n" + "=" * 80)
    print(f"DECOMPOSING STEERING VECTOR - LAYER {args.layer}")
    print("=" * 80)
    
    results = {}
    
    # Method 1: SAE Encoder (most principled)
    if args.method in ["encoder", "all"]:
        features, reconstruction, cos_sim = decompose_with_sae_encoder(
            steering_vec, sae, top_n=args.top_n
        )
        results["encoder"] = (features, reconstruction, cos_sim)
        
        print(f"\n  Top {min(args.top_n, len(features))} features:")
        for rank, (feat_id, activation) in enumerate(features[:args.top_n], 1):
            print(f"    {rank:2d}. Feature {feat_id:6d}: activation = {activation:8.4f}")
    
    # Method 2: Greedy search
    if args.method in ["greedy", "all"]:
        features, reconstruction, cos_sim = decompose_greedy(
            steering_vec, decoder, max_features=args.top_n
        )
        results["greedy"] = (features, reconstruction, cos_sim)
        
        print(f"\n  Selected features and weights:")
        for rank, (feat_id, weight) in enumerate(features, 1):
            print(f"    {rank:2d}. Feature {feat_id:6d}: weight = {weight:8.4f}")
    
    # Method 3: Top-k baseline
    if args.method in ["topk", "all"]:
        features, reconstruction, cos_sim = decompose_top_k(
            steering_vec, decoder, k=args.top_n
        )
        results["topk"] = (features, reconstruction, cos_sim)
        
        print(f"\n  Top {args.top_n} most similar features:")
        for rank, (feat_id, similarity) in enumerate(features[:10], 1):
            print(f"    {rank:2d}. Feature {feat_id:6d}: similarity = {similarity:7.4f}")
    
    # Summary comparison
    if args.method == "all":
        print("\n" + "=" * 80)
        print("SUMMARY COMPARISON")
        print("=" * 80)
        
        for method_name, (features, reconstruction, cos_sim) in results.items():
            l2_error = (steering_vec - reconstruction).norm()
            print(f"\n{method_name.upper()}:")
            print(f"  Cosine similarity: {cos_sim:.4f}")
            print(f"  L2 error: {l2_error:.4f}")
            print(f"  Features used: {len(features)}")
    
    # Compute progressive metrics and generate plot
    if results:
        print("\n" + "=" * 80)
        print("GENERATING PROGRESSIVE RECONSTRUCTION GRAPH")
        print("=" * 80)
        
        progressive_results = {}
        for method_name, (features, reconstruction, cos_sim) in results.items():
            print(f"\n  Computing progressive metrics for {method_name}...")
            cos_sims, l2_errors = compute_progressive_metrics(
                steering_vec, features, decoder, method_name
            )
            progressive_results[method_name] = (cos_sims, l2_errors)
        
        # Create and save the plot
        if args.output_graph:
            output_filename = args.output_graph
        else:
            output_filename = f"progressive_reconstruction_layer{args.layer}_trainer{args.trainer}.png"
        plot_progressive_metrics(progressive_results, output_filename)
    
    # Generate dashboard URLs
    print("\n" + "=" * 80)
    print("VIEW IN DASHBOARD")
    print("=" * 80)
    
    if "encoder" in results:
        features = results["encoder"][0]
        feat_ids = ",".join(str(f[0]) for f in features[:10])
        print(f"\nTop 10 features (SAE encoder):")
        print(f"  http://localhost:7863/?model=gpt&layer={args.layer}&trainer={args.trainer}&fids={feat_ids}")
    
    if "greedy" in results:
        features = results["greedy"][0]
        feat_ids = ",".join(str(f[0]) for f in features[:10])
        print(f"\nTop 10 features (Greedy):")
        print(f"  http://localhost:7863/?model=gpt&layer={args.layer}&trainer={args.trainer}&fids={feat_ids}")
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

