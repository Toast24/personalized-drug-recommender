"""Functions for building interaction datasets."""
import numpy as np

def build_interaction_dataset(patients, drug_ids, drug_features, observed_fraction=0.2,
                            noise=0.1, latent_dim=16, seed=42):
    """Build synthetic interaction dataset between patients and drugs.
    
    Args:
        patients: Patient features array (n_patients, patient_dim)
        drug_ids: List of drug IDs 
        drug_features: Drug features array (n_drugs, drug_dim)
        observed_fraction: Fraction of interactions to observe
        noise: Amount of noise to add to interactions
        latent_dim: Dimension of latent space for generating interactions
        seed: Random seed
        
    Returns:
        meta: Dict with patient and drug features
        Q: Drug features array
        R: Interaction matrix with some entries masked as nan
    """
    rng = np.random.RandomState(seed)
    
    n_patients = patients.shape[0]
    n_drugs = len(drug_ids)
    
    # Project patients and drugs into latent space
    P_latent = rng.normal(0, 1, size=(n_patients, latent_dim))
    Q_latent = rng.normal(0, 1, size=(n_drugs, latent_dim))
    
    # Generate interaction matrix
    R = P_latent @ Q_latent.T + rng.normal(0, noise, size=(n_patients, n_drugs))
    R = 1 / (1 + np.exp(-R))  # Sigmoid to get probabilities
    
    # Mask some entries as unobserved
    mask = rng.random(size=R.shape) < observed_fraction
    R[~mask] = np.nan
    
    meta = {
        'patient_features': patients,
        'drug_features': drug_features,
        'drug_ids': drug_ids
    }
    
    return meta, drug_features, R