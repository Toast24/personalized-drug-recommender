"""
Simple explainability demo: compute feature importance for patient features using a permutation approach or SHAP if available.
"""
import numpy as np
import argparse
from .data import generate_synthetic_patients, make_demo_drugs, build_interaction_dataset
from .model import create_model  # use factory function that handles torch imports

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False


def _run_model_eval(model, X, Q, R, mask, device='cpu'):
    """Helper that imports torch only when needed."""
    import torch
    model.eval()
    with torch.no_grad():
        patients = torch.from_numpy(X).to(device).float()
        drugs = torch.from_numpy(Q).to(device).float()
        n_p = patients.shape[0]
        n_d = drugs.shape[0]
        P_rep = patients.unsqueeze(1).expand(-1, n_d, -1).reshape(n_p * n_d, -1)
        D_rep = drugs.unsqueeze(0).expand(n_p, -1, -1).reshape(n_p * n_d, -1)
        preds, _, _ = model(P_rep, D_rep)
        preds = preds.view(n_p, n_d).cpu().numpy()
    return preds


def permutation_importance(model, X, Q, R, mask, device='cpu'):
    # compute baseline RMSE
    preds = _run_model_eval(model, X, Q, R, mask, device)
    from .eval import rmse
    base = rmse(R, preds)
    importances = []
    for col in range(X.shape[1]):
        Xp = X.copy()
        np.random.shuffle(Xp[:, col])
        preds = _run_model_eval(model, Xp, Q, R, mask, device)
        score = rmse(R, preds)
        importances.append(score - base)
    return np.array(importances), base


def run_demo(args):
    # prepare data
    patients = generate_synthetic_patients(n_patients=args.n_patients)
    drug_names, drug_fps = make_demo_drugs(nbits=args.fingerprint_dim)
    meta, Q, R = build_interaction_dataset(patients, drug_names, drug_fps,
                                          observed_fraction=args.observed_fraction,
                                          noise=args.noise, seed=args.seed)
    X = meta['patient_features']
    mask = ~np.isnan(R)

    # initialize model (imports torch internally)
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(patient_dim=X.shape[1], drug_dim=Q.shape[1], 
                        latent_dim=args.latent_dim, mlp=args.mlp).to(device)
    # quick train for demo
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    from .train import train_epoch
    train_epoch(model, optimizer, X, Q, R, mask, device=device)

    if SHAP_AVAILABLE:
        print('Running SHAP KernelExplainer (this may be slow)...')
        # create a small explainer on patient features for a single drug index
        import shap
        background = X[:50]
        def predict_fn(x_array):
            import torch
            # produce predicted score for drug 0
            model.eval()
            with torch.no_grad():
                patients = torch.from_numpy(x_array).to(device).float()
                drug = torch.from_numpy(Q[0:1]).to(device).float()
                P_rep = patients
                D_rep = drug.expand(patients.shape[0], -1)
                preds, _, _ = model(P_rep, D_rep)
                return preds.cpu().numpy()
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(X[:100])
        print('SHAP values shape:', np.array(shap_values).shape)
    else:
        print('SHAP not available; running permutation importance')
        importances, base = permutation_importance(model, X, Q, R, mask, device=device)
        print('Base RMSE:', base)
        print('Permutation importance (higher = more important):')
        print(importances)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['demo'], default='demo')
    parser.add_argument('--n-patients', type=int, default=200)
    parser.add_argument('--fingerprint-dim', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--observed-fraction', type=float, default=0.12)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--mlp', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.mode == 'demo':
        run_demo(args)
    else:
        raise NotImplementedError()
