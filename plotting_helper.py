#!/usr/bin/env python3
"""
Plotting Helper Script
Creates publication-quality plots for your report
"""

import json
from pathlib import Path
import numpy as np


def plot_single_experiment(experiment_name, output_file=None):
    """
    Plot loss curves for a single experiment
    
    Args:
        experiment_name (str): Name of experiment folder
        output_file (str): Where to save PDF (default: results/plots/{name}.pdf)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        return
    
    metrics_file = Path(f"experiments/{experiment_name}/metrics.json")
    
    if not metrics_file.exists():
        print(f"ERROR: No metrics found for {experiment_name}")
        return
    
    # Load metrics
    with open(metrics_file, 'r') as f:
        metrics = json.load(f)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(metrics['epochs'], metrics['losses'], 'o-', 
            linewidth=2.5, markersize=6, color='#1f77b4')
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title(f'Training Progress: {experiment_name}', 
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add final loss annotation
    final_loss = metrics['losses'][-1]
    ax.text(0.98, 0.05, f'Final Loss: {final_loss:.4f}',
           transform=ax.transAxes, ha='right', va='bottom',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
           fontsize=10)
    
    plt.tight_layout()
    
    # Save
    if output_file is None:
        output_file = f"results/plots/{experiment_name}.pdf"
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, format='pdf')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def plot_comparison(experiment_names, title="Experiment Comparison", 
                   output_file="results/plots/comparison.pdf"):
    """
    Compare multiple experiments on one plot
    
    Args:
        experiment_names (list): List of experiment folder names
        title (str): Plot title
        output_file (str): Where to save PDF
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for idx, exp_name in enumerate(experiment_names):
        metrics_file = Path(f"experiments/{exp_name}/metrics.json")
        
        if not metrics_file.exists():
            print(f"WARNING: No metrics for {exp_name}")
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        color = colors[idx % len(colors)]
        ax.plot(metrics['epochs'], metrics['losses'], 'o-',
               label=exp_name, linewidth=2.5, markersize=5, color=color)
    
    ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, format='pdf')
    print(f"✓ Saved: {output_file}")
    
    plt.close()


def create_table_for_report(experiment_names, output_file="results/metrics_table.txt"):
    """
    Create a Markdown table for your report
    
    Args:
        experiment_names (list): List of experiment names
        output_file (str): Where to save table
    """
    table = "| Experiment | Final Loss | Perplexity | Epochs |\n"
    table += "|---|---|---|---|\n"
    
    results = []
    
    for exp_name in experiment_names:
        metrics_file = Path(f"experiments/{exp_name}/metrics.json")
        
        if not metrics_file.exists():
            print(f"WARNING: No metrics for {exp_name}")
            continue
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        if metrics['losses']:
            final_loss = metrics['losses'][-1]
            perplexity = np.exp(final_loss)
            epochs = len(metrics['epochs'])
            
            table += f"| {exp_name} | {final_loss:.4f} | {perplexity:.2f} | {epochs} |\n"
            results.append((exp_name, final_loss, perplexity, epochs))
    
    # Save table
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        f.write(table)
    
    print(f"✓ Saved: {output_file}")
    print("\nTable for your report:")
    print(table)


# Example usage
if __name__ == "__main__":
    print("Plotting Helper Script")
    print("="*60)
    print("\nUsage examples:")
    print("\n1. Plot a single experiment:")
    print(">>> from plotting_helper import plot_single_experiment")
    print(">>> plot_single_experiment('baseline')")
    
    print("\n2. Compare multiple experiments:")
    print(">>> from plotting_helper import plot_comparison")
    print(">>> plot_comparison(['baseline', 'context_32', 'context_64'])")
    
    print("\n3. Create results table:")
    print(">>> from plotting_helper import create_table_for_report")
    print(">>> create_table_for_report(['baseline', 'context_32', 'context_64'])")
    
    print("\n" + "="*60)
    print("After running these, your plots will be in results/plots/")
    print("Use them in your report with:")
    print("  ![Figure 1: Training Loss](results/plots/baseline.pdf)")
