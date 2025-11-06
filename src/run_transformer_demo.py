"""
Run the transformer-based drug recommender model on 2000 synthetic patients.
"""
import os
import sys
import numpy as np
import torch
import logging
from sklearn.model_selection import train_test_split
import json

from data import generate_synthetic_patients, build_interaction_dataset
from model_transformer import TransformerDrugRecommender
from eval import rmse, precision_at_k_matrix, ndcg_at_k
from evaluate import find_optimal_threshold, compute_metrics, compute_ranking_metrics


def main(n_patients=2000, epochs=50, d_model=256, num_heads=8, num_layers=3, dropout=0.1, lr=1e-4):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load drug fingerprints
    fps = np.load(os.path.join('data', 'chembl', 'fingerprints.npy'))
    logger.info(f"Loaded {fps.shape[0]} drugs with fingerprint dim {fps.shape[1]}")

    # Generate synthetic patients
    patients = generate_synthetic_patients(n_patients=n_patients)
    meta, Q, R = build_interaction_dataset(patients, [f"drug_{i}" for i in range(len(fps))], fps,
                                           observed_fraction=0.15, noise=0.1, latent_dim=d_model, seed=42)

    X = meta['patient_features']  # (n_p, patient_dim)
    mask = ~np.isnan(R)

    # Train/test split on patients
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    # Create validation set from training data
    train_idx, val_idx = train_test_split(train_idx, test_size=0.1, random_state=42)

    X_train = X[train_idx]
    R_train = R[train_idx]
    mask_train = mask[train_idx]

    X_val = X[val_idx]
    R_val = R[val_idx]
    mask_val = mask[val_idx]

    X_test = X[test_idx]
    R_test = R[test_idx]
    mask_test = mask[test_idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerDrugRecommender(
        patient_dim=X.shape[1], 
        drug_dim=Q.shape[1],
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='min', factor=0.5, patience=5
    )

    n_d = Q.shape[0]
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    logger.info(f"Training transformer model on {len(train_idx)} patients and {n_d} drugs for {epochs} epochs")
    logger.info(f"Model config: d_model={d_model}, heads={num_heads}, layers={num_layers}")

    for epoch in range(epochs):
        model.train()
        
        # Training step
        patients_t = torch.from_numpy(X_train).to(device)
        drugs_t = torch.from_numpy(Q).to(device)
        R_t = torch.from_numpy(np.nan_to_num(R_train, nan=0.0)).to(device)
        mask_t = torch.from_numpy(mask_train.astype(np.float32)).to(device)

        # Process all patient-drug pairs
        P_rep = patients_t.unsqueeze(1).expand(-1, n_d, -1).reshape(-1, X_train.shape[1])
        D_rep = drugs_t.unsqueeze(0).expand(len(X_train), -1, -1).reshape(-1, Q.shape[1])

        preds, _, _ = model(P_rep.float(), D_rep.float())
        preds = preds.view(len(X_train), n_d)

        loss = ((preds - R_t) ** 2 * mask_t).sum() / (mask_t.sum() + 1e-9)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
        opt.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            patients_val = torch.from_numpy(X_val).to(device)
            R_val_t = torch.from_numpy(np.nan_to_num(R_val, nan=0.0)).to(device)
            mask_val_t = torch.from_numpy(mask_val.astype(np.float32)).to(device)

            P_val_rep = patients_val.unsqueeze(1).expand(-1, n_d, -1).reshape(-1, X_val.shape[1])
            D_val_rep = drugs_t.unsqueeze(0).expand(len(X_val), -1, -1).reshape(-1, Q.shape[1])

            val_preds, _, _ = model(P_val_rep.float(), D_val_rep.float())
            val_preds = val_preds.view(len(X_val), n_d)
            val_loss = ((val_preds - R_val_t) ** 2 * mask_val_t).sum() / (mask_val_t.sum() + 1e-9)

        scheduler.step(val_loss)

        logger.info(f"Epoch {epoch+1}/{epochs} train_loss={loss.item():.4f} val_loss={val_loss.item():.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'train_loss': loss.item(),
                'val_loss': val_loss.item(),
            }, os.path.join('models', 'drug_recommender_transformer_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model for evaluation
    best_ckpt = torch.load(os.path.join('models', 'drug_recommender_transformer_best.pt'))
    model.load_state_dict(best_ckpt['model_state_dict'])

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        patients_t = torch.from_numpy(X_test).to(device)
        drugs_t = torch.from_numpy(Q).to(device)

        P_rep = patients_t.unsqueeze(1).expand(-1, n_d, -1).reshape(-1, X_test.shape[1])
        D_rep = drugs_t.unsqueeze(0).expand(len(X_test), -1, -1).reshape(-1, Q.shape[1])

        preds, _, _ = model(P_rep.float(), D_rep.float())
        preds = preds.view(len(X_test), n_d).cpu().numpy()

    # Compute metrics
    threshold = find_optimal_threshold(R_test, preds)
    metrics = compute_metrics(R_test, preds, threshold)
    ranking = compute_ranking_metrics(R_test, preds)
    metrics.update(ranking)
    metrics['threshold'] = threshold

    logger.info("Evaluation metrics on test set:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    # Save metrics
    os.makedirs('models', exist_ok=True)
    with open('models/evaluation_results_transformer.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info('Saved checkpoint and metrics to models/')


if __name__ == '__main__':
    main()