"""
Data utilities: synthetic patient generation, demo drug list (SMILES), fingerprinting (RDKit optional), and dataset creation.
"""
import random
import numpy as np
import pandas as pd
from typing import List, Tuple

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False


DEMO_SMILES = {
    # common drugs with simple SMILES (examples)
    'aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
    'lisinopril': 'CC(C)(C)C(N)C(=O)N1CCCC1C(=O)O',
    'metformin': 'CN(C)C(=N)N=C(N)N',
    'amlodipine': 'CC1=CC(=C(C=C1)Cl)N2C(=O)C=C(C2=O)C(C)C',
    'atorvastatin': 'CC(C)C1=CC=C(C=C1)C(C)C(=O)O',
}


def compute_morgan_fp(smiles: str, nbits: int = 2048) -> np.ndarray:
    """Compute Morgan fingerprint vector for a SMILES string. Fallback to random vector if RDKit missing."""
    if RDKit_AVAILABLE:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.random.RandomState(hash(smiles) % (2 ** 32)).randint(0, 2, size=(nbits,)).astype(np.float32)
        arr = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
        arr = np.asarray(arr, dtype=np.float32)
        return arr
    else:
        # deterministic pseudorandom fingerprint based on SMILES
        rs = np.random.RandomState(abs(hash(smiles)) % (2 ** 32))
        return rs.rand(nbits).astype(np.float32)


def make_demo_drugs(nbits: int = 512) -> Tuple[List[str], np.ndarray]:
    names = list(DEMO_SMILES.keys())
    fps = [compute_morgan_fp(s, nbits=nbits) for s in DEMO_SMILES.values()]
    return names, np.vstack(fps)


def generate_synthetic_patients(n_patients: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    ages = rng.randint(18, 90, size=n_patients)
    sex = rng.randint(0, 2, size=n_patients)  # 0 female, 1 male
    bmi = rng.normal(27, 6, size=n_patients).clip(15, 50)
    # some lab values
    sbp = rng.normal(130, 15, size=n_patients)
    creatinine = rng.normal(1.0, 0.3, size=n_patients).clip(0.3, 5.0)
    # binary comorbidities: diabetes, htn, cad
    diabetes = rng.binomial(1, 0.2, size=n_patients)
    htn = rng.binomial(1, 0.3, size=n_patients)
    cad = rng.binomial(1, 0.1, size=n_patients)

    df = pd.DataFrame({
        'age': ages,
        'sex': sex,
        'bmi': bmi,
        'sbp': sbp,
        'creatinine': creatinine,
        'diabetes': diabetes,
        'htn': htn,
        'cad': cad,
    })
    return df


def build_interaction_dataset(patients: pd.DataFrame, drug_names: List[str], drug_fps: np.ndarray,
                              observed_fraction: float = 0.15, noise: float = 0.1,
                              latent_dim: int = 32, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build patient-drug interaction matrices for demo.
    Returns X (patient features), Q (drug features), R (observed response matrix with NaN for missing).
    R simulated by generating latent patient and drug factors and computing dot product + noise.
    """
    rng = np.random.RandomState(seed)
    n_p = len(patients)
    n_d = len(drug_names)

    # basic patient feature matrix (normalized)
    X = patients.copy()
    numeric_cols = ['age', 'bmi', 'sbp', 'creatinine']
    X_num = (X[numeric_cols].values - X[numeric_cols].values.mean(axis=0)) / (X[numeric_cols].values.std(axis=0) + 1e-6)
    X_bin = X[['sex', 'diabetes', 'htn', 'cad']].values
    X_feat = np.hstack([X_num, X_bin])

    # drug features: use provided fingerprints (possibly high dim)
    Q = drug_fps.astype(np.float32)

    # Simulate latent factors
    P_lat = rng.normal(scale=1.0, size=(n_p, latent_dim)).astype(np.float32)
    Q_lat = rng.normal(scale=1.0, size=(n_d, latent_dim)).astype(np.float32)

    # Bias terms derived from simple rules: e.g., ACE inhibitors better for htn
    bias = np.zeros((n_p, n_d), dtype=np.float32)
    # example: drugs named 'lisinopril' -> positive effect for htn
    for j, name in enumerate(drug_names):
        if 'lisinopril' in name:
            bias[:, j] += 0.5 * patients['htn'].values
        if 'metformin' in name:
            bias[:, j] += 0.4 * patients['diabetes'].values

    R_true = P_lat.dot(Q_lat.T) + bias
    R_true = (R_true - R_true.mean()) / (R_true.std() + 1e-6)
    R_obs = R_true + rng.normal(scale=noise, size=R_true.shape)

    # Mask observed interactions
    mask = rng.rand(*R_obs.shape) < observed_fraction
    R = np.where(mask, R_obs, np.nan).astype(np.float32)

    metadata = {
        'patient_features': X_feat.astype(np.float32),
        'patient_df': patients,
    }
    return metadata, Q, R
