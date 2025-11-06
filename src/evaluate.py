"""
Evaluate the trained drug recommender model using various metrics.
"""
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
import logging
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from model import InteractionModel

def load_data(data_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """Load processed ChEMBL data and fingerprints."""
    fps = np.load(os.path.join(data_dir, 'fingerprints.npy'))
    activity_df = pd.read_csv(os.path.join(data_dir, 'bioactivity.csv'))
    
    # Create interaction matrix
    n_drugs = len(fps)
    n_proteins = activity_df['target_chembl_id'].nunique()
    interactions = np.zeros((n_drugs, n_proteins), dtype=np.float32)
    
    # Map IDs to indices
    drug_to_idx = {id_: idx for idx, id_ in enumerate(activity_df['molecule_chembl_id'].unique())}
    protein_to_idx = {id_: idx for idx, id_ in enumerate(activity_df['target_chembl_id'].unique())}
    
    # Fill interaction matrix
    for _, row in activity_df.iterrows():
        drug_idx = drug_to_idx[row['molecule_chembl_id']]
        protein_idx = protein_to_idx[row['target_chembl_id']]
        interactions[drug_idx, protein_idx] = 1.0
    
    return fps, interactions, drug_to_idx, protein_to_idx

def find_optimal_threshold(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Find optimal classification threshold using F1 score."""
    thresholds = np.linspace(0.01, 0.99, 50)
    best_f1 = 0
    best_threshold = 0.5
    
    # Filter out NaN values and convert to binary format
    mask = ~np.isnan(y_true)
    if not mask.any():
        return best_threshold
        
    y_true_valid = (y_true[mask] > 0.5).astype(float)  # Convert to binary
    y_pred_valid = y_pred[mask]
    
    for threshold in thresholds:
        y_pred_binary = (y_pred_valid >= threshold).astype(float)
        precision = precision_score(y_true_valid, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_valid, y_pred_binary, zero_division=0)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold

def compute_ranking_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute ranking-based metrics (MRR, MAP)."""
    mrr_scores = []
    ap_scores = []
    
    for i in range(y_true.shape[0]):
        # Handle missing values by masking
        mask = ~np.isnan(y_true[i])
        if mask.sum() == 0:
            continue
        true_masked = y_true[i][mask]
        pred_masked = y_pred[i][mask]
        true_targets = np.where(true_masked > 0)[0]
        if len(true_targets) > 0:
            # Sort predictions in descending order (on masked values)
            sorted_indices = np.argsort(pred_masked)[::-1]

            # MRR - reciprocal rank of first relevant item
            first_rel_rank = np.min([np.where(sorted_indices == t)[0][0] for t in true_targets]) + 1
            mrr_scores.append(1.0 / first_rel_rank)

            # MAP - mean precision at each relevant item
            precisions = []
            n_correct = 0
            for j, idx in enumerate(sorted_indices, 1):
                if idx in true_targets:
                    n_correct += 1
                    precisions.append(n_correct / j)
            if precisions:
                ap_scores.append(np.mean(precisions))
    
    return {
        'mrr': np.mean(mrr_scores) if mrr_scores else float('nan'),
        'map': np.mean(ap_scores) if ap_scores else float('nan')
    }

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float = None) -> Dict[str, float]:
    """Compute various evaluation metrics."""
    if threshold is None:
        threshold = find_optimal_threshold(y_true, y_pred)
    
    # Filter out NaN values and convert to binary format
    mask = ~np.isnan(y_true)
    if not mask.any():
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'auroc': 0.0, 'aupr': 0.0}
        
    y_true_valid = (y_true[mask] > 0.5).astype(float)  # Convert to binary
    y_pred_valid = y_pred[mask]
    
    # Binary predictions
    y_pred_binary = (y_pred_valid >= threshold).astype(float)
    
    # Basic metrics
    precision = precision_score(y_true_valid, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_valid, y_pred_binary, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
    
    # ROC AUC
    try:
        auroc = roc_auc_score(y_true_valid, y_pred_valid)
    except:
        auroc = float('nan')
    
    # Average Precision (area under precision-recall curve)
    try:
        aupr = average_precision_score(y_true_valid, y_pred_valid)
    except:
        aupr = float('nan')
    
    # NDCG@k for each protein (handle NaNs -> compute only where ground truth exists)
    ndcg_scores = []
    k = 10
    for i in range(y_true.shape[1]):
        true_relevance = y_true[:, i]
        predicted_scores = y_pred[:, i]
        # mask out missing ground-truth entries
        mask = ~np.isnan(true_relevance)
        if mask.sum() == 0:
            continue
        true_rel_valid = (true_relevance[mask] > 0.5).astype(float)
        pred_valid = predicted_scores[mask]
        if true_rel_valid.sum() > 0:  # Only compute NDCG if there are positive examples
            ranking = np.argsort(pred_valid)[::-1][:k]
            ndcg = ndcg_at_k(true_rel_valid[ranking], k)
            ndcg_scores.append(ndcg)
    
    ndcg_10 = np.mean(ndcg_scores) if ndcg_scores else float('nan')
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auroc': auroc,
        'aupr': aupr,
        'ndcg@10': ndcg_10
    }

def ndcg_at_k(rel: np.ndarray, k: int) -> float:
    """Compute NDCG@k for a single list of relevance scores."""
    dcg = 0.0
    idcg = 0.0
    rel = np.asarray(rel)[:k]
    # If rel contains non-binary values, keep them as-is; rel should be non-NaN here.
    for i, r in enumerate(rel, 1):
        dcg += r / np.log2(i + 1)

    ideal_rel = np.sort(rel)[::-1]
    for i, r in enumerate(ideal_rel, 1):
        idcg += r / np.log2(i + 1)

    return dcg / (idcg + 1e-10)

def plot_prediction_distribution(y_true: np.ndarray, y_pred: np.ndarray, threshold: float, save_path: str):
    """Plot distribution of predictions for positive and negative examples."""
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred[y_true == 0].flatten(), bins=50, alpha=0.5, label='Negative Examples', density=True)
    plt.hist(y_pred[y_true == 1].flatten(), bins=50, alpha=0.5, label='Positive Examples', density=True)
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.3f}')
    plt.xlabel('Prediction Score')
    plt.ylabel('Density')
    plt.title('Distribution of Prediction Scores')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_precision_recall_curve(y_true: np.ndarray, y_pred: np.ndarray, save_path: str):
    """Plot precision-recall curve."""
    from sklearn.metrics import precision_recall_curve
    precision, recall, thresholds = precision_recall_curve(y_true.flatten(), y_pred.flatten())
    
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load data
    data_dir = 'data/chembl'
    fps, interactions, drug_to_idx, protein_to_idx = load_data(data_dir)
    logger.info(f"Loaded {len(fps)} drugs with {interactions.shape[1]} targets")
    
    # Setup train/test split ensuring positive examples in both splits
    pos_indices = np.where(interactions.sum(axis=1) > 0)[0]
    neg_indices = np.where(interactions.sum(axis=1) == 0)[0]
    
    # Split positive examples
    train_pos, test_pos = train_test_split(
        pos_indices,
        test_size=0.2,
        random_state=42
    )
    
    # Split negative examples
    train_neg, test_neg = train_test_split(
        neg_indices,
        test_size=0.2,
        random_state=42
    )
    
    # Combine indices
    train_idx = np.concatenate([train_pos, train_neg])
    test_idx = np.concatenate([test_pos, test_neg])
    
    # Shuffle the combined indices
    np.random.shuffle(train_idx)
    np.random.shuffle(test_idx)
    
    # Verify splits
    n_pos_train = interactions[train_idx].sum()
    n_pos_test = interactions[test_idx].sum()
    logger.info(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
    logger.info(f"Train positives: {n_pos_train}, Test positives: {n_pos_test}")
    
    X_train = fps[train_idx]
    X_test = fps[test_idx]
    y_train = interactions[train_idx]
    y_test = interactions[test_idx]
    
    # Load model checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load('models/drug_recommender.pt', map_location=device)
    
    # Create model with same architecture
    config = checkpoint['config']
    model = InteractionModel(
        patient_dim=interactions.shape[1],
        drug_dim=fps.shape[1],
        latent_dim=config['hyperparameters']['latent_dim']
    ).to(device)
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get predictions
    with torch.no_grad():
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        dummy_protein = torch.zeros(len(X_test), interactions.shape[1]).to(device)
        pred, _, _ = model(dummy_protein, X_test_tensor)
        y_pred = torch.sigmoid(pred).cpu().numpy()
    
    # Find optimal threshold and compute metrics
    threshold = find_optimal_threshold(y_test, y_pred)
    metrics = compute_metrics(y_test, y_pred, threshold)
    
    # Add ranking metrics
    ranking_metrics = compute_ranking_metrics(y_test, y_pred)
    metrics.update(ranking_metrics)
    
    # Add threshold to metrics
    metrics['threshold'] = threshold
    
    # Print results
    logger.info("\nTest Set Metrics:")
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    
    # Generate visualizations
    os.makedirs('models/plots', exist_ok=True)

    # Basic plots
    plot_prediction_distribution(
        y_test, y_pred, threshold,
        'models/plots/prediction_distribution.png'
    )
    plot_precision_recall_curve(
        y_test, y_pred,
        'models/plots/precision_recall_curve.png'
    )

    # Additional plots
    from visualization import (
        plot_confusion_matrix,
        plot_roc_curve,
        plot_prediction_heatmap
    )

    # Convert predictions to binary using optimal threshold
    y_pred_binary = (y_pred >= threshold).astype(float)

    plot_confusion_matrix(
        y_test, y_pred_binary,
        'models/plots/confusion_matrix.png'
    )
    plot_roc_curve(
        y_test, y_pred,
        'models/plots/roc_curve.png'
    )
    plot_prediction_heatmap(
        y_test, y_pred,
        [f"Target {i+1}" for i in range(y_test.shape[1])],
        'models/plots/prediction_heatmap.png'
    )

    # Plot learning curves if available
    if 'train_losses' in checkpoint:
        plt.figure(figsize=(10, 6))
        plt.plot(checkpoint['train_losses'], label='Training Loss')
        plt.plot(checkpoint['val_losses'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        plt.savefig('models/plots/learning_curves.png')
        plt.close()

    # Save results
    results = {
        'metrics': metrics,
        'config': config,
        'dataset_info': {
            'n_drugs': int(len(fps)),
            'n_targets': int(interactions.shape[1]),
            'n_interactions': int(float(interactions.sum())),
            'sparsity': float(1 - (interactions.sum() / (len(fps) * interactions.shape[1])))
        }
    }
    
    results_path = os.path.join('models', 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved detailed results to {results_path}")
    
    # Print example predictions
    idx_to_protein = {v: k for k, v in protein_to_idx.items()}
    
    logger.info("\nExample Predictions:")
    for i in range(min(5, len(X_test))):
        true_targets = np.where(y_test[i] > 0)[0]
        pred_scores = y_pred[i]
        top_k = 5
        
        # Get top predicted targets
        top_pred_idx = np.argsort(pred_scores)[::-1][:top_k]
        
        logger.info(f"\nDrug {i+1}:")
        logger.info("True targets:")
        for idx in true_targets:
            logger.info(f"- {idx_to_protein[idx]}")
        
        logger.info("\nTop predicted targets:")
        for idx in top_pred_idx:
            logger.info(f"- {idx_to_protein[idx]}: {pred_scores[idx]:.4f}")

if __name__ == '__main__':
    main()