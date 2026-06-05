#!/usr/bin/env python3
"""
Training & Experimentation Helper Script
Helps you track metrics, create visualizations, and organize results
"""

import json
import os
import numpy as np
from pathlib import Path


class ExperimentTracker:
    """Track metrics and organize results for your experiments"""
    
    def __init__(self, experiment_name):
        """
        Initialize tracker for an experiment
        
        Args:
            experiment_name (str): Name of your experiment (e.g., 'baseline', 'context_32')
        """
        self.experiment_name = experiment_name
        self.exp_dir = Path(f"experiments/{experiment_name}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics = {
            'experiment': experiment_name,
            'epochs': [],
            'losses': [],
            'samples': {},
            'config': {}
        }
    
    def save_config(self, config_dict):
        """Save hyperparameter configuration"""
        self.metrics['config'] = config_dict
        print(f"✓ Saved config for {self.experiment_name}")
    
    def log_epoch(self, epoch, loss, sample_text=None):
        """
        Log results from one epoch
        
        Args:
            epoch (int): Epoch number
            loss (float): Training loss
            sample_text (str): Generated text sample (optional)
        """
        self.metrics['epochs'].append(epoch)
        self.metrics['losses'].append(float(loss))
        
        if sample_text:
            self.metrics['samples'][f'epoch_{epoch}'] = sample_text
            # Also save to individual file
            with open(self.exp_dir / f'epoch_{epoch}_sample.txt', 'w') as f:
                f.write(sample_text)
        
        print(f"✓ Epoch {epoch}: Loss={loss:.4f}")
    
    def save_metrics(self):
        """Save metrics to JSON file"""
        filepath = self.exp_dir / 'metrics.json'
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Saved metrics to {filepath}")
    
    def get_perplexity(self):
        """Calculate perplexity from final loss"""
        if self.metrics['losses']:
            final_loss = self.metrics['losses'][-1]
            perplexity = np.exp(final_loss)
            return perplexity
        return None
    
    def print_summary(self):
        """Print experiment summary"""
        print(f"\n{'='*60}")
        print(f"EXPERIMENT: {self.experiment_name}")
        print(f"{'='*60}")
        print(f"Epochs trained: {len(self.metrics['epochs'])}")
        if self.metrics['losses']:
            print(f"Initial loss: {self.metrics['losses'][0]:.4f}")
            print(f"Final loss: {self.metrics['losses'][-1]:.4f}")
            print(f"Final perplexity: {self.get_perplexity():.2f}")
        print(f"Config: {self.metrics['config']}")
        print(f"{'='*60}\n")


def create_summary_table(experiments_dir="experiments"):
    """
    Create a summary table of all experiments
    
    Args:
        experiments_dir (str): Directory containing experiment folders
    """
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Experiment':<20} {'Final Loss':<15} {'Perplexity':<15} {'Epochs':<10}")
    print("-"*80)
    
    results = []
    
    for exp_folder in Path(experiments_dir).iterdir():
        if exp_folder.is_dir():
            metrics_file = exp_folder / 'metrics.json'
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                
                if metrics['losses']:
                    final_loss = metrics['losses'][-1]
                    perplexity = np.exp(final_loss)
                    epochs = len(metrics['epochs'])
                    
                    print(f"{exp_folder.name:<20} {final_loss:<15.4f} {perplexity:<15.2f} {epochs:<10}")
                    results.append({
                        'name': exp_folder.name,
                        'loss': final_loss,
                        'perplexity': perplexity,
                        'epochs': epochs
                    })
    
    print("="*80 + "\n")
    return results


def generate_report_metrics():
    """
    Generate a CSV file suitable for including in your report
    """
    results = create_summary_table()
    
    if results:
        # Sort by loss
        results.sort(key=lambda x: x['loss'])
        
        # Create CSV
        csv_content = "Experiment,Final Loss,Perplexity,Epochs\n"
        for r in results:
            csv_content += f"{r['name']},{r['loss']:.4f},{r['perplexity']:.2f},{r['epochs']}\n"
        
        # Save
        with open('results/metrics_summary.csv', 'w') as f:
            f.write(csv_content)
        
        print("✓ Saved: results/metrics_summary.csv")
        print("\nYou can include this in your report as:")
        print("```csv")
        print(csv_content)
        print("```")


# Example usage in your training script:
if __name__ == "__main__":
    print("ExperimentTracker Helper Script")
    print("="*60)
    print("\nUsage in your train_gpt.py:")
    print("""
    from experiment_tracker import ExperimentTracker
    
    # At start of training:
    tracker = ExperimentTracker('baseline')
    tracker.save_config({
        'emb_dim': 64,
        'num_layers': 2,
        'block_size': 32,
        'learning_rate': 0.001
    })
    
    # In epoch loop:
    for epoch in range(epochs):
        # ... training code ...
        tracker.log_epoch(epoch + 1, avg_loss, generated_text)
    
    # At end:
    tracker.save_metrics()
    tracker.print_summary()
    """)
    
    print("\nTo view all experiments:")
    print(">>> from experiment_tracker import create_summary_table")
    print(">>> create_summary_table()")
    
    print("\nTo generate report metrics:")
    print(">>> from experiment_tracker import generate_report_metrics")
    print(">>> generate_report_metrics()")
