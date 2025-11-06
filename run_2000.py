"""
Generate 2000 synthetic patients, use existing ChEMBL drug fingerprints, train a model,
and evaluate metrics on a held-out test set.
"""
import os
import numpy as np
import torch
import logging
from sklearn.model_selection import train_test_split

import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import data
import model
import eval
import evaluate


def main(n_patients=2000, epochs=20, latent_dim=64, lr=1e-3):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load drug fingerprints (use existing ChEMBL fingerprints)
    fps = np.load(os.path.join('data', 'chembl', 'fingerprints.npy'))
    logger.info(f"Loaded {fps.shape[0]} drugs with fingerprint dim {fps.shape[1]}")

    # Generate synthetic patients
    patients = data.generate_synthetic_patients(n_patients=n_patients)
    meta, Q, R = data.build_interaction_dataset(patients, [f"drug_{i}" for i in range(len(fps))], fps,
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
    model_instance = model.InteractionModel(patient_dim=X.shape[1], drug_dim=Q.shape[1], latent_dim=latent_dim).to(device)
    opt = torch.optim.Adam(model_instance.parameters(), lr=lr)

    n_p_train = X_train.shape[0]
    n_d = Q.shape[0]

    logger.info(f"Training on {n_p_train} patients and {n_d} drugs for {epochs} epochs")

    for epoch in range(epochs):
        model_instance.train()
        patients_t = torch.from_numpy(X_train).to(device)
        drugs_t = torch.from_numpy(Q).to(device)
        R_t = torch.from_numpy(np.nan_to_num(R_train, nan=0.0)).to(device)
        mask_t = torch.from_numpy(mask_train.astype(np.float32)).to(device)

        # expand to pairs
        P_rep = patients_t.unsqueeze(1).expand(-1, n_d, -1).reshape(n_p_train * n_d, -1)
        D_rep = drugs_t.unsqueeze(0).expand(n_p_train, -1, -1).reshape(n_p_train * n_d, -1)

        preds, _, _ = model_instance(P_rep.float(), D_rep.float())
        preds = preds.view(n_p_train, n_d)

        loss = ((preds - R_t) ** 2 * mask_t).sum() / (mask_t.sum() + 1e-9)

        opt.zero_grad()
        loss.backward()
        opt.step()

        logger.info(f"Epoch {epoch+1}/{epochs} loss={loss.item():.4f}")

    # Evaluate on test set
    model_instance.eval()
    with torch.no_grad():
        patients_t = torch.from_numpy(X_test).to(device)
        drugs_t = torch.from_numpy(Q).to(device)
        n_p_test = X_test.shape[0]

        P_rep = patients_t.unsqueeze(1).expand(-1, n_d, -1).reshape(n_p_test * n_d, -1)
        D_rep = drugs_t.unsqueeze(0).expand(n_p_test, -1, -1).reshape(n_p_test * n_d, -1)

        preds, _, _ = model_instance(P_rep.float(), D_rep.float())
        preds = preds.view(n_p_test, n_d).cpu().numpy()

    # Compute metrics
    # For binary metrics we need a threshold; find by optimizing F1 on test set
    threshold = evaluate.find_optimal_threshold(R_test, preds)
    metrics = evaluate.compute_metrics(R_test, preds, threshold)
    ranking = evaluate.compute_ranking_metrics(R_test, preds)
    metrics.update(ranking)
    metrics['threshold'] = threshold

    logger.info("Evaluation metrics on test set:")
    for k, v in metrics.items():
        logger.info(f"{k}: {v}")

    # Save checkpoint and results
    os.makedirs('models', exist_ok=True)
    ckpt = {
        'model_state_dict': model_instance.state_dict(),
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