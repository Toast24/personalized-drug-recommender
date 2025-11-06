"""
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
