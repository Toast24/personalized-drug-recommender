"""Matrix Factorization (MF) style model that uses side features.

This model projects patient features and drug fingerprints into a shared latent
space and predicts interaction scores via a dot product followed by a sigmoid.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MFWithSideFeatures(nn.Module):
    def __init__(self, patient_dim: int, drug_dim: int, latent_dim: int = 64, hidden: int = 128):
        super().__init__()
        # Project patient features to latent vector
        self.patient_proj = nn.Sequential(
            nn.Linear(patient_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, latent_dim),
        )
        # Project drug fingerprints to latent vector
        self.drug_proj = nn.Sequential(
            nn.Linear(drug_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, latent_dim),
        )
        # Optional small MLP on concatenated latent vectors
        self.pred = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(latent_dim, 1)
        )

    def forward(self, patient_features, drug_features):
        # patient_features: (batch, patient_dim)
        # drug_features: (batch, drug_dim)
        p_lat = self.patient_proj(patient_features)
        d_lat = self.drug_proj(drug_features)
        # Option 1: dot product
        dot = (p_lat * d_lat).sum(dim=1, keepdim=True)  # (batch, 1)
        # Option 2: MLP on concat
        cat = torch.cat([p_lat, d_lat], dim=1)
        mlp_out = self.pred(cat)
        # Combine both signals (learned) - here simple average then sigmoid
        out = 0.5 * dot + 0.5 * mlp_out
        out = torch.sigmoid(out).squeeze(-1)
        return out, p_lat, d_lat
