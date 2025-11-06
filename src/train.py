"""
Training script for demo mode and hooks to plug real datasets.
Run with: python -m src.train --mode demo
"""
import argparse
import numpy as np
from tqdm import trange

# Import torch with a clear, actionable error message on Windows DLL init failures (WinError 1114)
try:
    import torch
    from torch import optim
except OSError as e:
    # Provide focused remediation steps for the common WinError 1114 / DLL init issue
    msg = f"""
Failed to import PyTorch due to a DLL initialization error: {e}

Common, practical fixes:

1) If you do NOT need GPU/CUDA support, install the CPU-only build of PyTorch (simplest fix):

   # PowerShell example
   pip uninstall -y torch torchvision torchaudio; 
   pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

2) If you DO want GPU/CUDA support, use conda and make sure your NVIDIA drivers and CUDA version match PyTorch's CUDA build:

   # conda (recommended for GPU-enabled installs on Windows)
   conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

3) Other things to check:
   - Ensure the Microsoft Visual C++ Redistributable is installed.
   - Update your GPU drivers (NVIDIA/AMD) to a recent version.
   - On some Windows versions, "Hardware-accelerated GPU scheduling" can cause DLL init failures; try toggling it off in Settings -> Graphics settings.

If you want, I can modify the project to avoid importing torch at module-import time so non-PyTorch parts can run even when torch import fails.
"""
    print(msg)
    raise

from .data import generate_synthetic_patients, make_demo_drugs, build_interaction_dataset
from .model import InteractionModel
from .eval import rmse, precision_at_k_matrix, ndcg_at_k


def train_epoch(model, opt, patients_feat, drug_feat, R_obs, mask, device='cpu'):
    model.train()
    losses = []
    patients = torch.from_numpy(patients_feat).to(device)
    drugs = torch.from_numpy(drug_feat).to(device)
    R = torch.from_numpy(np.nan_to_num(R_obs, nan=0.0)).to(device)
    mask_t = torch.from_numpy(mask.astype(np.float32)).to(device)

    opt.zero_grad()
    # compute scores for all pairs in batch (small demo)
    n_p = patients.shape[0]
    n_d = drugs.shape[0]
    # expand
    P_rep = patients.unsqueeze(1).expand(-1, n_d, -1).reshape(n_p * n_d, -1)
    D_rep = drugs.unsqueeze(0).expand(n_p, -1, -1).reshape(n_p * n_d, -1)
    preds, _, _ = model(P_rep.float(), D_rep.float())
    preds = preds.view(n_p, n_d)
    loss = ((preds - R) ** 2 * mask_t).sum() / (mask_t.sum() + 1e-9)
    loss.backward()
    opt.step()
    losses.append(float(loss.detach().cpu().numpy()))
    return float(np.mean(losses)), preds.detach().cpu().numpy()


def run_demo(args):
    patients = generate_synthetic_patients(n_patients=args.n_patients)
    drug_names, drug_fps = make_demo_drugs(nbits=args.fingerprint_dim)
    meta, Q, R = build_interaction_dataset(patients, drug_names, drug_fps,
                                          observed_fraction=args.observed_fraction,
                                          noise=args.noise, seed=args.seed)
    X = meta['patient_features']
    mask = ~np.isnan(R)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = InteractionModel(patient_dim=X.shape[1], drug_dim=Q.shape[1], latent_dim=args.latent_dim, mlp=args.mlp).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        loss, preds = train_epoch(model, opt, X, Q, R, mask, device=device)
        print(f"Epoch {epoch+1}/{args.epochs}  loss={loss:.4f}")

    # eval
    rm = rmse(R, preds)
    p5 = precision_at_k_matrix(R, preds, k=5)
    n5 = ndcg_at_k(R, preds, k=5)
    print(f"RMSE: {rm:.4f}, Precision@5: {p5:.4f}, NDCG@5: {n5:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['demo'], default='demo')
    parser.add_argument('--n-patients', type=int, default=200)
    parser.add_argument('--fingerprint-dim', type=int, default=256)
    parser.add_argument('--latent-dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--observed-fraction', type=float, default=0.12)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--mlp', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.mode == 'demo':
        run_demo(args)
    else:
        raise NotImplementedError('Only demo mode is implemented in this prototype')
