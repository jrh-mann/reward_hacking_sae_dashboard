#!/usr/bin/env python
"""
Visualize and analyze reward hacking feature analysis results.
"""

import pickle
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict


OUTPUT_DIR = Path("reward_hacking_analysis")


def load_results():
    """Load analysis results."""
    with open(OUTPUT_DIR / "all_generations.pkl", 'rb') as f:
        all_generations = pickle.load(f)
    
    with open(OUTPUT_DIR / "analysis_results.pkl", 'rb') as f:
        analysis_results = pickle.load(f)
    
    return all_generations, analysis_results


def print_examples(all_generations: List[Dict], num_examples: int = 3):
    """Print example generations from each category."""
    
    rh_examples = [g for g in all_generations if g['is_reward_hacking']]
    non_rh_examples = [g for g in all_generations if not g['is_reward_hacking']]
    
    print(f"\n{'='*80}")
    print(f"REWARD HACKING EXAMPLES ({len(rh_examples)} total)")
    print(f"{'='*80}")
    
    for i, example in enumerate(rh_examples[:num_examples], 1):
        print(f"\n--- Example {i} (Task: {example['task_name']}) ---")
        text = example['generated_text']
        # Truncate for display
        if len(text) > 500:
            text = text[:500] + "..."
        print(text)
        print()
    
    print(f"\n{'='*80}")
    print(f"NON-REWARD HACKING EXAMPLES ({len(non_rh_examples)} total)")
    print(f"{'='*80}")
    
    for i, example in enumerate(non_rh_examples[:num_examples], 1):
        print(f"\n--- Example {i} (Task: {example['task_name']}) ---")
        text = example['generated_text']
        # Truncate for display
        if len(text) > 500:
            text = text[:500] + "..."
        print(text)
        print()


def plot_feature_differences(analysis_results: Dict, top_k: int = 20):
    """Plot top feature differences."""
    
    feature_diff = analysis_results['feature_diff']
    
    # Get top overrepresented
    top_indices = np.argsort(feature_diff)[-top_k:][::-1]
    top_values = feature_diff[top_indices]
    
    # Get top underrepresented
    bottom_indices = np.argsort(feature_diff)[:top_k]
    bottom_values = feature_diff[bottom_indices]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot overrepresented
    ax1.barh(range(top_k), top_values, color='red', alpha=0.7)
    ax1.set_yticks(range(top_k))
    ax1.set_yticklabels([f"Feature {idx}" for idx in top_indices])
    ax1.set_xlabel('Activation Difference (RH - Non-RH)')
    ax1.set_title(f'Top {top_k} Features Overrepresented in Reward Hacking')
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Plot underrepresented
    ax2.barh(range(top_k), bottom_values, color='blue', alpha=0.7)
    ax2.set_yticks(range(top_k))
    ax2.set_yticklabels([f"Feature {idx}" for idx in bottom_indices])
    ax2.set_xlabel('Activation Difference (RH - Non-RH)')
    ax2.set_title(f'Top {top_k} Features Underrepresented in Reward Hacking')
    ax2.grid(axis='x', alpha=0.3)
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_differences.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot: {OUTPUT_DIR / 'feature_differences.png'}")
    plt.close()


def plot_distribution_comparison(all_generations: List[Dict], top_features: List[int]):
    """Plot distribution of feature activations for top features."""
    
    rh_gens = [g for g in all_generations if g['is_reward_hacking']]
    non_rh_gens = [g for g in all_generations if not g['is_reward_hacking']]
    
    num_features_to_plot = min(6, len(top_features))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feat_idx in enumerate(top_features[:num_features_to_plot]):
        ax = axes[i]
        
        # Get activations for this feature
        rh_acts = [g['features']['mean'][feat_idx].item() for g in rh_gens]
        non_rh_acts = [g['features']['mean'][feat_idx].item() for g in non_rh_gens]
        
        # Plot histograms
        ax.hist(non_rh_acts, bins=20, alpha=0.5, label='Non-RH', color='blue')
        ax.hist(rh_acts, bins=20, alpha=0.5, label='RH', color='red')
        ax.set_xlabel('Mean Activation')
        ax.set_ylabel('Count')
        ax.set_title(f'Feature {feat_idx}')
        ax.legend()
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "activation_distributions.png", dpi=300, bbox_inches='tight')
    print(f"Saved plot: {OUTPUT_DIR / 'activation_distributions.png'}")
    plt.close()


def analyze_by_task(all_generations: List[Dict]):
    """Analyze reward hacking rate by task."""
    
    tasks = {}
    for gen in all_generations:
        task_name = gen['task_name']
        if task_name not in tasks:
            tasks[task_name] = {'total': 0, 'reward_hacking': 0}
        tasks[task_name]['total'] += 1
        if gen['is_reward_hacking']:
            tasks[task_name]['reward_hacking'] += 1
    
    # Calculate rates
    task_data = []
    for task_name, counts in tasks.items():
        rate = counts['reward_hacking'] / counts['total'] if counts['total'] > 0 else 0
        task_data.append({
            'task': task_name,
            'rh_count': counts['reward_hacking'],
            'total': counts['total'],
            'rh_rate': rate,
        })
    
    # Sort by rate
    task_data = sorted(task_data, key=lambda x: x['rh_rate'], reverse=True)
    
    # Print table
    print(f"\n{'='*80}")
    print(f"REWARD HACKING RATE BY TASK")
    print(f"{'='*80}")
    print(f"{'Task':<30} {'RH Count':<10} {'Total':<10} {'RH Rate':<10}")
    print(f"{'-'*80}")
    
    for data in task_data:
        print(f"{data['task']:<30} {data['rh_count']:<10} {data['total']:<10} {data['rh_rate']:<10.2%}")
    
    # Plot
    if len(task_data) > 0:
        fig, ax = plt.subplots(figsize=(12, max(6, len(task_data) * 0.3)))
        
        task_names = [d['task'] for d in task_data]
        rates = [d['rh_rate'] for d in task_data]
        
        ax.barh(range(len(task_names)), rates, color='orange', alpha=0.7)
        ax.set_yticks(range(len(task_names)))
        ax.set_yticklabels(task_names, fontsize=8)
        ax.set_xlabel('Reward Hacking Rate')
        ax.set_title('Reward Hacking Rate by Task')
        ax.grid(axis='x', alpha=0.3)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "rh_rate_by_task.png", dpi=300, bbox_inches='tight')
        print(f"\nSaved plot: {OUTPUT_DIR / 'rh_rate_by_task.png'}")
        plt.close()
    
    return task_data


def create_feature_report(analysis_results: Dict, output_file: str = "feature_report.md"):
    """Create a markdown report of top features."""
    
    top_indices = analysis_results['top_overrepresented']['indices']
    top_values = analysis_results['top_overrepresented']['values']
    
    bottom_indices = analysis_results['top_underrepresented']['indices']
    bottom_values = analysis_results['top_underrepresented']['values']
    
    rh_avg = analysis_results['rh_feature_avg']
    non_rh_avg = analysis_results['non_rh_feature_avg']
    
    with open(OUTPUT_DIR / output_file, 'w') as f:
        f.write("# Reward Hacking Feature Analysis Report\n\n")
        
        f.write(f"## Summary\n\n")
        f.write(f"- **Reward Hacking Samples**: {analysis_results['num_reward_hacking']}\n")
        f.write(f"- **Non-Reward Hacking Samples**: {analysis_results['num_non_reward_hacking']}\n")
        f.write(f"- **Total Features Analyzed**: {len(rh_avg)}\n\n")
        
        f.write("## Top Features Associated with Reward Hacking\n\n")
        f.write("These features are significantly more active in reward hacking samples:\n\n")
        f.write("| Rank | Feature ID | Difference | RH Avg | Non-RH Avg | Explorer Link |\n")
        f.write("|------|------------|------------|--------|------------|---------------|\n")
        
        for rank, (idx, diff) in enumerate(zip(top_indices[:20], top_values[:20]), 1):
            rh_val = rh_avg[idx]
            non_rh_val = non_rh_avg[idx]
            # Link assumes server running on localhost:7863
            link = f"http://localhost:7863/?model=gpt&layer=11&trainer=0&fids={idx}"
            f.write(f"| {rank} | {idx} | {diff:.4f} | {rh_val:.4f} | {non_rh_val:.4f} | [View]({link}) |\n")
        
        f.write("\n## Features Less Active in Reward Hacking\n\n")
        f.write("These features are significantly less active in reward hacking samples:\n\n")
        f.write("| Rank | Feature ID | Difference | RH Avg | Non-RH Avg | Explorer Link |\n")
        f.write("|------|------------|------------|--------|------------|---------------|\n")
        
        for rank, (idx, diff) in enumerate(zip(bottom_indices[:20], bottom_values[:20]), 1):
            rh_val = rh_avg[idx]
            non_rh_val = non_rh_avg[idx]
            link = f"http://localhost:7863/?model=gpt&layer=11&trainer=0&fids={idx}"
            f.write(f"| {rank} | {idx} | {diff:.4f} | {rh_val:.4f} | {non_rh_val:.4f} | [View]({link}) |\n")
    
    print(f"\nSaved report: {OUTPUT_DIR / output_file}")


def main():
    """Main visualization pipeline."""
    
    print("Loading results...")
    all_generations, analysis_results = load_results()
    
    print("\nGenerating visualizations...")
    
    # Plot feature differences
    plot_feature_differences(analysis_results, top_k=20)
    
    # Plot distributions for top features
    top_features = analysis_results['top_overrepresented']['indices'][:6]
    plot_distribution_comparison(all_generations, top_features)
    
    # Analyze by task
    task_data = analyze_by_task(all_generations)
    
    # Create feature report
    create_feature_report(analysis_results)
    
    # Print examples
    print_examples(all_generations, num_examples=2)
    
    print(f"\n{'='*80}")
    print("Visualization complete!")
    print(f"{'='*80}")
    print(f"\nOutputs saved to: {OUTPUT_DIR}")
    print("  - feature_differences.png")
    print("  - activation_distributions.png")
    print("  - rh_rate_by_task.png")
    print("  - feature_report.md")


if __name__ == "__main__":
    main()

