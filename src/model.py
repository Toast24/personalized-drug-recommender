"""PyTorch model definitions for patient-drug interaction."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DrugEncoder(nn.Module):
    """Drug encoder that maps molecular fingerprints to latent space."""
    def __init__(self, input_dim: int, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class InteractionModel(nn.Module):
    """Main model that predicts drug-target interactions."""
    def __init__(self, patient_dim: int, drug_dim: int, latent_dim: int = 128, mlp: bool = True):
        super().__init__()
        self.drug_enc = DrugEncoder(drug_dim, latent_dim)
        self.protein_mlp = nn.Sequential(
            nn.Linear(patient_dim, latent_dim * 4),
            nn.BatchNorm1d(latent_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
        )
        
        self.interaction_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 4),
            nn.BatchNorm1d(latent_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(latent_dim * 4, latent_dim * 2),
            nn.BatchNorm1d(latent_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, 1),  # Output a single interaction score
        )
        self.mlp = mlp

    def forward(self, x, drug_x):
        # x: patient features (batch, patient_dim)
        # drug_x: drug fingerprints (batch, drug_dim)
        drug_emb = self.drug_enc(drug_x)  # (batch, latent_dim)
        protein_emb = self.protein_mlp(x)  # (batch, latent_dim)
        
        # Combine embeddings
        combined = torch.cat([drug_emb, protein_emb], dim=1)  # (batch, latent_dim*2)
        
        # Predict interaction score
        out = torch.sigmoid(self.interaction_mlp(combined)[:, 0])  # (batch,)
        
        return out, drug_emb, protein_emb