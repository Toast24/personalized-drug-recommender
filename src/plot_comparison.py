"""
Generate comparison plots for model performance analysis.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List

def plot_metric_comparison(metrics: Dict[str, Dict[str, float]],
                         selected_metrics: List[str] = None,
                         title: str = 'Model Performance Comparison',
                         save_path: str = None):
    """Create a bar plot comparing different metrics across models."""
    if selected_metrics is None:
        selected_metrics = ['f1', 'auroc', 'ndcg@10']
    
    # Prepare data for plotting
    models = list(metrics.keys())
    metric_values = {metric: [metrics[model][metric] for model in models] 
                    for metric in selected_metrics}
    
    # Set up the plot with style
    sns.set_style("whitegrid")
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.8 / len(selected_metrics)
    
    # Color scheme
    colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
    
    # Plot bars for each metric
    for i, (metric, values) in enumerate(metric_values.items()):
        plt.bar(x + (i - len(selected_metrics)/2 + 0.5) * width, 
               values, width, label=metric.upper(), color=colors[i],
               alpha=0.8, edgecolor='black', linewidth=1)
    
    # Customize plot
    plt.ylabel('Score', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.xticks(x, models, fontsize=10)
    plt.legend(fontsize=10, frameon=True)
    plt.grid(True, axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on top of bars
    for i, metric in enumerate(selected_metrics):
        for j, v in enumerate(metric_values[metric]):
            plt.text(j + (i - len(selected_metrics)/2 + 0.5) * width, v,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_convergence_comparison(models_epochs: Dict[str, int],
                              title: str = 'Model Convergence Speed',
                              save_path: str = None):
    """Create a horizontal bar plot showing epochs until convergence."""
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 5))
    
    models = list(models_epochs.keys())
    epochs = list(models_epochs.values())
    
    # Color scheme
    colors = ['#3498db' if e < 15 else '#e74c3c' for e in epochs]
    
    # Create horizontal bar plot
    bars = plt.barh(models, epochs, color=colors, alpha=0.8,
                   edgecolor='black', linewidth=1)
    
    plt.xlabel('Epochs until convergence', fontsize=12)
    plt.title(title, fontsize=14, pad=20)
    plt.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, i,
                f'{int(width)} epochs',
                ha='left', va='center', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_training_curves():
    """Create training curves comparison."""
    pass  # To be implemented if needed

def generate_model_comparison_plots(save_dir: str = 'models/plots'):
    """Generate and save all comparison plots."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Load results
    results = {}
    model_epochs = {}
    
    result_files = {
        'MLP': 'evaluation_results_2000.json',
        'Transformer': 'evaluation_results_transformer.json',
        'MF': 'evaluation_results_mf.json'
    }
    
    # Load evaluation results
    for model_name, filename in result_files.items():
        filepath = os.path.join('models', filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results[model_name] = json.load(f)
    
    # Known convergence epochs from our runs
    model_epochs = {
        'MLP': 36,
        'Transformer': 13,
        'MF': 10
    }
    
    # Generate metric comparison plot
    plot_metric_comparison(
        results,
        selected_metrics=['f1', 'auroc', 'ndcg@10'],
        title='Model Performance Comparison',
        save_path=os.path.join(save_dir, 'metric_comparison.png')
    )
    
    # Generate convergence comparison plot
    plot_convergence_comparison(
        model_epochs,
        title='Model Convergence Speed',
        save_path=os.path.join(save_dir, 'convergence_comparison.png')
    )

    print("Generated plots:")
    print(f"1. Model Performance Comparison: {os.path.join(save_dir, 'metric_comparison.png')}")
    print(f"2. Convergence Speed: {os.path.join(save_dir, 'convergence_comparison.png')}")

if __name__ == '__main__':
    generate_model_comparison_plots()