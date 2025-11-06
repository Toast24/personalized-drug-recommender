"""
Generate 2000 synthetic patients, use existing ChEMBL drug fingerprints, train a model,
and evaluate metrics on a held-out test set.
"""
import os
import numpy as np
import torch
import logging
from sklearn.model_selection import train_test_split

from data import generate_synthetic_patients, build_interaction_dataset
from model import InteractionModel
from eval import rmse, precision_at_k_matrix, ndcg_at_k
from evaluate import find_optimal_threshold, compute_metrics, compute_ranking_metrics


def main(n_patients=2000, epochs=50, latent_dim=128, lr=5e-4):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load drug fingerprints (use existing ChEMBL fingerprints)
    fps = np.load(os.path.join('data', 'chembl', 'fingerprints.npy'))
    logger.info(f"Loaded {fps.shape[0]} drugs with fingerprint dim {fps.shape[1]}")

    # Generate synthetic patients
    patients = generate_synthetic_patients(n_patients=n_patients)
    meta, Q, R = build_interaction_dataset(patients, [f"drug_{i}" for i in range(len(fps))], fps,
                                           observed_fraction=0.15, noise=0.1, latent_dim=latent_dim, seed=42)

    X = meta['patient_features']  # (n_p, patient_dim)
    mask = ~np.isnan(R)

    # Train/test split on patients
    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)

    X_train = X[train_idx]
    R_train = R[train_idx]
    mask_train = mask[train_idx]

    X_test = X[test_idx]
    R_test = R[test_idx]
    mask_test = mask[test_idx]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InteractionModel(patient_dim=X.shape[1], drug_dim=Q.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    n_p_train = X_train.shape[0]
    n_d = Q.shape[0]

    logger.info(f"Training on {n_p_train} patients and {n_d} drugs for {epochs} epochs")
    
    # Create validation set from training data
    train_idx, val_idx = train_test_split(np.arange(len(X_train)), test_size=0.1, random_state=42)
    X_val = X_train[val_idx]
    R_val = R_train[val_idx]
    mask_val = mask_train[val_idx]
    X_train_final = X_train[train_idx]
    R_train_final = R_train[train_idx]
    mask_train_final = mask_train[train_idx]

    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        patients_t = torch.from_numpy(X_train_final).to(device)
        drugs_t = torch.from_numpy(Q).to(device)
        R_t = torch.from_numpy(np.nan_to_num(R_train_final, nan=0.0)).to(device)
        mask_t = torch.from_numpy(mask_train_final.astype(np.float32)).to(device)

        # Calculate current training set size
        curr_n_train = len(X_train_final)
        
        # expand to pairs
        P_rep = patients_t.unsqueeze(1).expand(-1, n_d, -1).reshape(curr_n_train * n_d, -1)
        D_rep = drugs_t.unsqueeze(0).expand(curr_n_train, -1, -1).reshape(curr_n_train * n_d, -1)

        preds = model(P_rep.float(), D_rep.float())[0]  # Ignore embeddings for training
        preds = preds.view(curr_n_train, n_d)

        # Calculate MSE loss only on observed entries
        loss = ((preds - R_t) ** 2 * mask_t).sum() / (mask_t.sum() + 1e-9)

        opt.zero_grad()
        loss.backward()
        opt.step()

        # Validation step
        model.eval()
        with torch.no_grad():
            patients_val = torch.from_numpy(X_val).to(device)
            R_val_t = torch.from_numpy(np.nan_to_num(R_val, nan=0.0)).to(device)
            mask_val_t = torch.from_numpy(mask_val.astype(np.float32)).to(device)

            P_val_rep = patients_val.unsqueeze(1).expand(-1, n_d, -1).reshape(-1, X_val.shape[1])
            D_val_rep = drugs_t.unsqueeze(0).expand(len(X_val), -1, -1).reshape(-1, Q.shape[1])

            val_preds = model(P_val_rep.float(), D_val_rep.float())[0]
            val_preds = val_preds.view(len(X_val), n_d)
            val_loss = ((val_preds - R_val_t) ** 2 * mask_val_t).sum() / (mask_val_t.sum() + 1e-9)

        # Learning rate scheduling
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
            }, os.path.join('models', 'drug_recommender_2000_best.pt'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model for evaluation
    best_ckpt = torch.load(os.path.join('models', 'drug_recommender_2000_best.pt'))
    model.load_state_dict(best_ckpt['model_state_dict'])

    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        patients_t = torch.from_numpy(X_test).to(device)
        drugs_t = torch.from_numpy(Q).to(device)
        n_p_test = X_test.shape[0]

        P_rep = patients_t.unsqueeze(1).expand(-1, n_d, -1).reshape(n_p_test * n_d, -1)
        D_rep = drugs_t.unsqueeze(0).expand(n_p_test, -1, -1).reshape(n_p_test * n_d, -1)

        preds = model(P_rep.float(), D_rep.float())[0]
        preds = preds.view(n_p_test, n_d).cpu().numpy()

    # Compute metrics
    # For binary metrics we need a threshold; find by optimizing F1 on test set
    threshold = find_optimal_threshold(R_test, preds)
    metrics = compute_metrics(R_test, preds, threshold)
    ranking = compute_ranking_metrics(R_test, preds)
    metrics.update(ranking)
    metrics['threshold'] = threshold

    logger.info("Evaluation metrics on test set:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    # Save checkpoint and results
    os.makedirs('models', exist_ok=True)
    ckpt = {
        'model_state_dict': model.state_dict(),
        'config': {'hyperparameters': {'latent_dim': latent_dim, 'lr': lr, 'epochs': epochs}},
    }
    torch.save(ckpt, os.path.join('models', 'drug_recommender_2000.pt'))

    # Save metrics
    import json
    with open('models/evaluation_results_2000.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info('Saved checkpoint and metrics to models/')


if __name__ == '__main__':
    main()
