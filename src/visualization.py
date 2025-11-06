"""
Visualization utilities for the drug recommender model.
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
    
    # Set up the plot
    plt.figure(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.8 / len(selected_metrics)
    
    # Plot bars for each metric
    for i, (metric, values) in enumerate(metric_values.items()):
        offset = width * i - width * len(selected_metrics)/2 + width/2
        plt.bar(x + offset, values, width, label=metric.upper())
    
    # Customize plot
    plt.ylabel('Score')
    plt.title(title)
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for i, metric in enumerate(selected_metrics):
        offset = width * i - width * len(selected_metrics)/2 + width/2
        for j, v in enumerate(metric_values[metric]):
            plt.text(j + offset, v, f'{v:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_convergence_comparison(models_epochs: Dict[str, int],
                              title: str = 'Model Convergence Comparison',
                              save_path: str = None):
    """Create a horizontal bar plot showing epochs until convergence."""
    plt.figure(figsize=(10, 5))
    models = list(models_epochs.keys())
    epochs = list(models_epochs.values())
    
    # Create horizontal bar plot
    bars = plt.barh(models, epochs)
    
    plt.xlabel('Epochs until convergence')
    plt.title(title)
    plt.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{int(width)} epochs',
                ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str
):
    """Plot confusion matrix."""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true.flatten(), y_pred.flatten())
    
    # Create heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Negative', 'Positive'],
        yticklabels=['Negative', 'Positive']
    )
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig(save_path)
    plt.close()

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
    
    for model_name, filename in result_files.items():
        filepath = os.path.join('models', filename)
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                results[model_name] = json.load(f)
    
    # Known convergence epochs from our runs
    model_epochs = {
        'MLP': 19,
        'Transformer': 14,
        'MF': 9
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

def plot_roc_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str
):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, auc
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_true.flatten(), y_pred.flatten())
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # diagonal line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_prediction_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    protein_names: List[str],
    save_path: str,
    max_drugs: int = 20
):
    """Plot heatmap of predictions vs true values for top drugs."""
    plt.figure(figsize=(15, 8))
    
    # Select subset of drugs with most interactions
    n_interactions = y_true.sum(axis=1)
    top_drug_indices = np.argsort(n_interactions)[::-1][:max_drugs]
    
    # Create heatmap data
    heatmap_data = np.column_stack([
        y_true[top_drug_indices],
        y_pred[top_drug_indices]
    ])
    
    # Plot heatmap
    sns.heatmap(
        heatmap_data,
        cmap='coolwarm',
        center=0.5,
        vmin=0,
        vmax=1,
        xticklabels=['True', 'Predicted'],
        yticklabels=[f'Drug {i+1}' for i in range(len(top_drug_indices))]
    )
    plt.title('True vs Predicted Interactions Heatmap')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()