"""
Hyperparameter optimization using Optuna.
Performs k-fold cross-validation to find optimal hyperparameters.

Usage:
    python -m src.train_hpo --n-trials 100 --study-name drug_rec_v1
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import optuna
from pathlib import Path

from sklearn.model_selection import KFold
import json
import yaml
import logging

from model import InteractionModel
# Import evaluation metrics from sklearn
from sklearn.metrics import mean_squared_error, precision_score, ndcg_score



def objective(trial, data_config: dict, device: str = 'cpu', n_folds: int = 5):
    """Optuna objective function using k-fold CV."""
    # Load data
    fps = np.load(data_config['fingerprints_path'])  # Drug fingerprints
    activity_df = pd.read_csv(data_config['activity_path'])
    
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
    
    # Create splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    scores = []
    
    # Hyperparameters
    params = {
        'latent_dim': trial.suggest_int('latent_dim', 32, 256),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
        'use_mlp': True  # Fixed for this version
    }
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(interactions)):
        # Create model and optimizer
        model = InteractionModel(
            patient_dim=n_proteins,  # Use protein space as patient features
            drug_dim=fps.shape[1],   # Fingerprint dimension
            latent_dim=params['latent_dim'],
            mlp=params['use_mlp']
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        
        # Convert data to tensors
        X_train = torch.FloatTensor(interactions[train_idx]).to(device)
        Q_train = torch.FloatTensor(fps[train_idx]).to(device)
        X_val = torch.FloatTensor(interactions[val_idx]).to(device)
        Q_val = torch.FloatTensor(fps[val_idx]).to(device)
        
        # Train
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(50):  # Max epochs per fold
            model.train()
            # Forward pass
            # Split data into batches
            n_samples = len(X_train)
            batch_indices = torch.randperm(n_samples)
            batch_size = params['batch_size']
            
            total_loss = 0
            for idx in range(0, n_samples, batch_size):
                batch_idx = batch_indices[idx:min(idx + batch_size, n_samples)]
                x_batch = X_train[batch_idx]
                q_batch = Q_train[batch_idx]
                
                # Forward pass
                batch_pred, _, _ = model(x_batch, q_batch)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(batch_pred, x_batch)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_idx)
            
            avg_train_loss = total_loss / n_samples
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_pred, _, _ = model(X_val, Q_val)
                val_loss = torch.nn.functional.binary_cross_entropy_with_logits(val_pred, X_val).item()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        scores.append(best_val_loss)
    
    mean_loss = np.mean(scores)
    trial.set_user_attr('cv_scores', scores)
    return mean_loss



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/hpo.yaml')
    parser.add_argument('--data-dir', type=str, default='data/chembl')
    parser.add_argument('--output-dir', type=str, default='models')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up data config
    data_config = {
        'fingerprints_path': os.path.join(args.data_dir, 'fingerprints.npy'),
        'activity_path': os.path.join(args.data_dir, 'bioactivity.csv')
    }
    
    # Choose device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create study
    study = optuna.create_study(
        direction='minimize',
        storage=f'sqlite:///{os.path.join(args.output_dir, "hpo.db")}',
        load_if_exists=True
    )
    
    # Run optimization
    logger.info("Starting hyperparameter optimization...")
    study.optimize(
        lambda trial: objective(trial, data_config, device, n_folds=config['training']['n_folds']),
        n_trials=config['optuna']['n_trials'],
        show_progress_bar=True
    )
    
    # Save results
    logger.info("Best trial:")
    logger.info(f"  Value: {study.best_trial.value}")
    logger.info("  Params:")
    for key, value in study.best_trial.params.items():
        logger.info(f"    {key}: {value}")
    
    # Save best model config
    model_config = {
        'hyperparameters': study.best_trial.params,
        'performance': {
            'validation_loss': float(study.best_trial.value),
            'cv_scores': study.best_trial.user_attrs['cv_scores']
        },
        'training': {
            'n_folds': config['training']['n_folds'],
            'early_stopping_patience': 5
        }
    }
    
    config_path = os.path.join(args.output_dir, 'best_model_config.json')
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=2)
    logger.info(f"Saved model config to {config_path}")

if __name__ == '__main__':
    main()