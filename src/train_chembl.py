"""
Train the drug recommender model with the best hyperparameters on ChEMBL data.
"""
import os
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
import logging

from model import InteractionModel

def load_data(data_dir: str):
    """Load processed ChEMBL data and fingerprints."""
    fps = np.load(os.path.join(data_dir, 'fingerprints.npy'))
    activity_df = pd.read_csv(os.path.join(data_dir, 'bioactivity.csv'))
    
    # Create interaction matrix
    n_drugs = len(fps)
    n_proteins = activity_df['target_chembl_id'].nunique()
    interactions = np.zeros((n_drugs, n_proteins), dtype=np.float32)
    
    # Map IDs to indices
    drug_to_idx = {id_: idx for idx, id_ in enumerate(activity_df['molecule_chembl_id'].unique())}
    protein_to_idx = {id_: idx for idx, id_ in enumerate(activity_df['target_chembl_id'].unique())}
    
    # Fill interaction matrix
    for _, row in activity_df.iterrows():
        drug_idx = drug_to_idx[row['molecule_chembl_id']]
        protein_idx = protein_to_idx[row['target_chembl_id']]
        interactions[drug_idx, protein_idx] = 1.0
    
    return fps, interactions, drug_to_idx, protein_to_idx

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load data
    data_dir = 'data/chembl'
    fps, interactions, drug_to_idx, protein_to_idx = load_data(data_dir)
    logger.info(f"Loaded {len(fps)} drugs with {interactions.shape[1]} targets")
    
    # Load best hyperparameters
    with open('models/best_model_config.json') as f:
        config = json.load(f)
    
    hyperparams = config['hyperparameters']
    logger.info("Best hyperparameters:")
    for k, v in hyperparams.items():
        logger.info(f"  {k}: {v}")
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InteractionModel(
        patient_dim=interactions.shape[1],  # Number of protein targets
        drug_dim=fps.shape[1],             # Fingerprint dimension
        latent_dim=hyperparams['latent_dim']
    ).to(device)
    
    # Training data
    X = torch.FloatTensor(interactions).to(device)
    Q = torch.FloatTensor(fps).to(device)
    
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['learning_rate'])
    batch_size = hyperparams['batch_size']
    n_epochs = 50
    
    logger.info("Starting training...")
    for epoch in range(n_epochs):
        model.train()
        
        # Create batches
        n_samples = len(X)
        batch_indices = torch.randperm(n_samples)
        total_loss = 0
        
        for idx in range(0, n_samples, batch_size):
            batch_idx = batch_indices[idx:min(idx + batch_size, n_samples)]
            x_batch = X[batch_idx]
            q_batch = Q[batch_idx]
            
            # Forward pass
            pred, _, _ = model(x_batch, q_batch)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, x_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_idx)
        
        avg_loss = total_loss / n_samples
        logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.4f}")
    
    # Save model
    model_path = os.path.join('models', 'drug_recommender.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'drug_to_idx': drug_to_idx,
        'protein_to_idx': protein_to_idx,
        'config': config
    }, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Create a function to predict for new drugs
    def predict(smiles: str, fp: np.ndarray) -> dict:
        model.eval()
        with torch.no_grad():
            # Convert fingerprint to tensor
            fp_tensor = torch.FloatTensor(fp).unsqueeze(0).to(device)  # Add batch dimension
            
            # Use zero vector as protein profile (predict all interactions)
            x_tensor = torch.zeros(1, interactions.shape[1], device=device)
            
            # Get predictions
            pred, _, _ = model(x_tensor, fp_tensor)
            pred_proba = torch.sigmoid(pred).cpu().numpy()[0]
            
            # Convert to protein targets
            idx_to_protein = {v: k for k, v in protein_to_idx.items()}
            predictions = {idx_to_protein[i]: float(p) for i, p in enumerate(pred_proba)}
            
            # Sort by probability
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            return {
                'smiles': smiles,
                'predictions': [{'target_id': t, 'probability': p} for t, p in sorted_preds]
            }
    
    # Save example usage
    usage_path = os.path.join('models', 'example_usage.py')
    usage_code = '''"""
Example usage of the trained drug recommender model.
"""
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def load_model(model_path: str) -> callable:
    """Load the model and return a prediction function."""
    # Load model
    checkpoint = torch.load(model_path)
    
    # Reconstruct model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from model import InteractionModel
    config = checkpoint['config']
    model = InteractionModel(
        patient_dim=len(checkpoint['protein_to_idx']),
        drug_dim=2048,  # Morgan fingerprint dimension
        latent_dim=config['hyperparameters']['latent_dim']
    ).to(device)
    
    # Load state
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create prediction function
    idx_to_protein = {v: k for k, v in checkpoint['protein_to_idx'].items()}
    
    def predict(smiles: str) -> dict:
        """Predict protein targets for a drug SMILES string."""
        # Generate Morgan fingerprint
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
            
        fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
        
        with torch.no_grad():
            # Convert fingerprint to tensor
            fp_tensor = torch.FloatTensor(fp).unsqueeze(0).to(device)
            
            # Use zero vector as protein profile (predict all interactions)
            x_tensor = torch.zeros(1, len(idx_to_protein), device=device)
            
            # Get predictions
            pred, _, _ = model(x_tensor, fp_tensor)
            pred_proba = torch.sigmoid(pred).cpu().numpy()[0]
            
            # Convert to protein targets
            predictions = {idx_to_protein[i]: float(p) for i, p in enumerate(pred_proba)}
            
            # Sort by probability
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            return {
                'smiles': smiles,
                'predictions': [{'target_id': t, 'probability': p} for t, p in sorted_preds]
            }
    
    return predict

# Example usage
if __name__ == '__main__':
    # Load model
    model_path = 'models/drug_recommender.pt'
    predict_fn = load_model(model_path)
    
    # Example SMILES
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    
    # Get predictions
    result = predict_fn(smiles)
    
    print(f"Predictions for {result['smiles']}:")
    for i, pred in enumerate(result['predictions'][:5], 1):
        print(f"{i}. {pred['target_id']}: {pred['probability']:.4f}")
'''
    
    with open(usage_path, 'w') as f:
        f.write(usage_code)
    logger.info(f"Saved example usage to {usage_path}")

if __name__ == '__main__':
    main()