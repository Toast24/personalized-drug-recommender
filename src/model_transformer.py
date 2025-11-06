"""Alternative model architectures for drug recommendation."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        
        # Linear transformations
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to V
        out = torch.matmul(attention, V)
        
        # Concatenate heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear layer
        out = self.W_o(out)
        
        return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention
        attention_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x


class TransformerDrugRecommender(nn.Module):
    def __init__(self, patient_dim, drug_dim, d_model=256, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        # Project inputs to d_model dimension
        self.patient_embedding = nn.Sequential(
            nn.Linear(patient_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.drug_embedding = nn.Sequential(
            nn.Linear(drug_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.interaction_predictor = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
    def forward(self, patient_features, drug_features):
        # Embed inputs
        patient_emb = self.patient_embedding(patient_features)  # (batch, d_model)
        drug_emb = self.drug_embedding(drug_features)  # (batch, d_model)
        
        # Process patient features through transformer
        patient_encoded = patient_emb.unsqueeze(1)  # Add sequence dimension
        for layer in self.transformer_layers:
            patient_encoded = layer(patient_encoded)
        
        # Process drug features through transformer
        drug_encoded = drug_emb.unsqueeze(1)
        for layer in self.transformer_layers:
            drug_encoded = layer(drug_encoded)
        
        # Combine patient and drug representations
        patient_final = patient_encoded.squeeze(1)
        drug_final = drug_encoded.squeeze(1)
        combined = torch.cat([patient_final, drug_final], dim=1)
        
        # Predict interaction
        out = torch.sigmoid(self.interaction_predictor(combined))
        
        return out.squeeze(-1), patient_final, drug_final