"""
Run a Matrix Factorization (with side features) baseline on synthetic patients.
"""
import os
import numpy as np
import torch
import logging
from sklearn.model_selection import train_test_split
import json

from data import generate_synthetic_patients, build_interaction_dataset
from model_mf import MFWithSideFeatures
from evaluate import find_optimal_threshold, compute_metrics, compute_ranking_metrics


def main(n_patients=2000, epochs=50, latent_dim=64, lr=1e-3, run_name='mf', batch_size=128, weight_decay=0.0):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    fps = np.load(os.path.join('data', 'chembl', 'fingerprints.npy'))
    logger.info(f"Loaded {fps.shape[0]} drugs with fingerprint dim {fps.shape[1]}")

    patients = generate_synthetic_patients(n_patients=n_patients)
    meta, Q, R = build_interaction_dataset(patients, [f"drug_{i}" for i in range(len(fps))], fps,
                                           observed_fraction=0.15, noise=0.1, latent_dim=latent_dim, seed=42)

    X = meta['patient_features']
    mask = ~np.isnan(R)

    idx = np.arange(len(X))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, random_state=42)
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
    model = MFWithSideFeatures(patient_dim=X.shape[1], drug_dim=Q.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5)

    n_d = Q.shape[0]
    best_val_loss = float('inf')
    patience = 8
    patience_counter = 0

    logger.info(f"Training MF baseline on {len(train_idx)} patients and {n_d} drugs for {epochs} epochs")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        drugs_t = torch.from_numpy(Q).to(device)

        # iterate over patient batches
        for start in range(0, len(X_train), batch_size):
            end = min(start + batch_size, len(X_train))
            batch_patients = X_train[start:end]
            batch_R = R_train[start:end]
            batch_mask = mask_train[start:end]

            patients_t = torch.from_numpy(batch_patients).to(device)
            R_t = torch.from_numpy(np.nan_to_num(batch_R, nan=0.0)).to(device)
            mask_t = torch.from_numpy(batch_mask.astype(np.float32)).to(device)

            # expand pairs for this batch
            P_rep = patients_t.unsqueeze(1).expand(-1, n_d, -1).reshape(-1, patients_t.shape[1])
            D_rep = drugs_t.unsqueeze(0).expand(len(batch_patients), -1, -1).reshape(-1, Q.shape[1])

            preds_batch, _, _ = model(P_rep.float(), D_rep.float())
            preds_batch = preds_batch.view(len(batch_patients), n_d)

            loss_batch = ((preds_batch - R_t) ** 2 * mask_t).sum() / (mask_t.sum() + 1e-9)

            opt.zero_grad()
            loss_batch.backward()
            opt.step()

            epoch_loss += loss_batch.item()
            n_batches += 1

        # average epoch loss
        loss = epoch_loss / max(1, n_batches)

        # validation
        model.eval()
        with torch.no_grad():
            patients_val = torch.from_numpy(X_val).to(device)
            R_val_t = torch.from_numpy(np.nan_to_num(R_val, nan=0.0)).to(device)
            mask_val_t = torch.from_numpy(mask_val.astype(np.float32)).to(device)

            # compute validation in batches to limit memory use
            val_epoch_loss = 0.0
            val_batches = 0
            for start in range(0, len(X_val), batch_size):
                end = min(start + batch_size, len(X_val))
                b_patients = patients_val[start:end]
                b_R = R_val_t[start:end]
                b_mask = mask_val_t[start:end]

                P_val_rep = b_patients.unsqueeze(1).expand(-1, n_d, -1).reshape(-1, b_patients.shape[1])
                D_val_rep = drugs_t.unsqueeze(0).expand(len(b_patients), -1, -1).reshape(-1, Q.shape[1])

                val_preds_batch, _, _ = model(P_val_rep.float(), D_val_rep.float())
                val_preds_batch = val_preds_batch.view(len(b_patients), n_d)
                val_loss_batch = ((val_preds_batch - b_R) ** 2 * b_mask).sum() / (b_mask.sum() + 1e-9)
                val_epoch_loss += val_loss_batch.item()
                val_batches += 1

            val_loss = val_epoch_loss / max(1, val_batches)

            scheduler.step(val_loss)

            logger.info(f"Epoch {epoch+1}/{epochs} train_loss={loss:.4f} val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs('models', exist_ok=True)
                best_model_path = os.path.join('models', f'drug_recommender_{run_name}_best.pt')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict()}, best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model
    best_ckpt = torch.load(os.path.join('models', f'drug_recommender_{run_name}_best.pt'))
    model.load_state_dict(best_ckpt['model_state_dict'])

    # test evaluation
    model.eval()
    with torch.no_grad():
        patients_t = torch.from_numpy(X_test).to(device)
        drugs_t = torch.from_numpy(Q).to(device)

        P_rep = patients_t.unsqueeze(1).expand(-1, n_d, -1).reshape(-1, X_test.shape[1])
        D_rep = drugs_t.unsqueeze(0).expand(len(X_test), -1, -1).reshape(-1, Q.shape[1])

        preds, _, _ = model(P_rep.float(), D_rep.float())
        preds = preds.view(len(X_test), n_d).cpu().numpy()

    threshold = find_optimal_threshold(R_test, preds)
    metrics = compute_metrics(R_test, preds, threshold)
    ranking = compute_ranking_metrics(R_test, preds)
    metrics.update(ranking)
    metrics['threshold'] = threshold

    logger.info("Evaluation metrics on test set:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    metrics_path = os.path.join('models', f'evaluation_results_{run_name}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info('Saved checkpoint and metrics to models/')


if __name__ == '__main__':
    main()
