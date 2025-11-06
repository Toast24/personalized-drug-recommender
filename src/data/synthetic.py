"""Functions for generating synthetic patient data."""
import numpy as np

def generate_synthetic_patients(n_patients=100, feature_dim=16, seed=42):
    """Generate synthetic patient features.
    
    Args:
        n_patients: Number of patients to generate
        feature_dim: Dimension of patient features
        seed: Random seed for reproducibility
        
    Returns:
        ndarray of shape (n_patients, feature_dim)
    """
    rng = np.random.RandomState(seed)
    # Generate random patient features - could be demographic, clinical, genetic etc.
    features = rng.normal(0, 1, size=(n_patients, feature_dim))
    return features