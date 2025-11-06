"""
Demo: Model Interpretability with SHAP
"""
import os
import numpy as np
import torch
import shap
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from model import InteractionModel
from data import generate_synthetic_patients, build_interaction_dataset

def load_model_and_data(n_patients=100):
    # Generate synthetic data
    print("Generating synthetic patient data...")
    patients = generate_synthetic_patients(n_patients=n_patients)
    
    # Load drug fingerprints
    print("Loading drug fingerprints...")
    fps = np.load(os.path.join('data', 'chembl', 'fingerprints.npy'))
    
    # Build interaction dataset
    meta, Q, R = build_interaction_dataset(
        patients,
        drug_ids=[f"drug_{i}" for i in range(len(fps))],
        drug_features=fps,
        observed_fraction=0.15,
        noise=0.1
    )
    
    # Load trained model
    print("Loading trained model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InteractionModel(
        patient_dim=meta['patient_features'].shape[1],
        drug_dim=Q.shape[1],
        latent_dim=128
    ).to(device)
    
    ckpt = torch.load('models/drug_recommender_2000.pt', map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    return model, meta['patient_features'], Q, R

def explain_predictions(model, patient_features, drug_features, feature_names=None):
    """Generate SHAP explanations for model predictions."""
    
    # Create background dataset for SHAP
    background_data = torch.from_numpy(patient_features[:100]).float()
    
    # Create explainer
    explainer = shap.DeepExplainer(model, background_data)
    
    # Get SHAP values for test patients
    test_data = torch.from_numpy(patient_features[:10]).float()
    shap_values = explainer.shap_values(test_data)
    
    # Plot SHAP summary
    if feature_names is None:
        feature_names = [f"Feature_{i}" for i in range(patient_features.shape[1])]
    
    plt.figure(figsize=(12, 6))
    shap.summary_plot(shap_values[0], test_data.numpy(), 
                     feature_names=feature_names,
                     show=False)
    plt.tight_layout()
    plt.savefig('demos/output/shap_summary.png')
    plt.close()
    
    return shap_values

def analyze_drug_similarity(model, drug_features):
    """Analyze drug similarity in the learned embedding space."""
    device = next(model.parameters()).device
    drug_t = torch.from_numpy(drug_features).float().to(device)
    
    # Get drug embeddings
    with torch.no_grad():
        drug_embeddings = model.drug_encoder(drug_t).cpu().numpy()
    
    # Calculate similarity matrix
    similarity = drug_embeddings @ drug_embeddings.T
    
    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity, cmap='YlOrRd')
    plt.colorbar()
    plt.title('Drug Similarity in Embedding Space')
    plt.savefig('demos/output/drug_embedding_similarity.png')
    plt.close()
    
    return drug_embeddings

def main():
    # Create output directory
    os.makedirs('demos/output', exist_ok=True)
    
    # Load model and data
    model, patient_features, drug_features, interactions = load_model_and_data()
    
    print("\nGenerating SHAP explanations...")
    feature_names = [f"Clinical_Feature_{i}" for i in range(patient_features.shape[1])]
    shap_values = explain_predictions(model, patient_features, drug_features, feature_names)
    
    print("\nAnalyzing drug similarities in embedding space...")
    drug_embeddings = analyze_drug_similarity(model, drug_features)
    
    print("\nInterpretability analysis complete. Outputs saved in demos/output/")
    print("- SHAP summary plot: shap_summary.png")
    print("- Drug embedding similarity: drug_embedding_similarity.png")

if __name__ == '__main__':
    main()