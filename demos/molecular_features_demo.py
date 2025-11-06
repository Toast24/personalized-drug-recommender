"""
Demo: Molecular Feature Generation using RDKit
"""
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import PandasTools

def generate_molecular_features():
    # Example SMILES for common drugs
    drugs = {
        'Aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)[C@H](C)C(=O)O',
        'Paracetamol': 'CC(=O)NC1=CC=C(O)C=C1',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'
    }
    
    # Create output directory
    os.makedirs('demos/output', exist_ok=True)
    
    # Generate and visualize molecular features
    features = []
    for name, smiles in drugs.items():
        mol = Chem.MolFromSmiles(smiles)
        
        # Generate 2D depiction
        img = Draw.MolToImage(mol)
        img.save(f'demos/output/{name.lower()}_2d.png')
        
        # Calculate Morgan fingerprints
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        
        # Calculate basic descriptors
        mw = Descriptors.ExactMolWt(mol)
        logp = Descriptors.MolLogP(mol)
        tpsa = Descriptors.TPSA(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        
        features.append({
            'Name': name,
            'SMILES': smiles,
            'MolWeight': mw,
            'LogP': logp,
            'TPSA': tpsa,
            'H-Bond-Donors': hbd,
            'H-Bond-Acceptors': hba,
            'Fingerprint': np.array(fp)
        })
        
        print(f"\nAnalyzed {name}:")
        print(f"Molecular Weight: {mw:.2f}")
        print(f"LogP: {logp:.2f}")
        print(f"TPSA: {tpsa:.2f}")
        print(f"H-Bond Donors: {hbd}")
        print(f"H-Bond Acceptors: {hba}")
    
    return features

if __name__ == '__main__':
    print("Generating molecular features for example drugs...")
    features = generate_molecular_features()
    print("\nGenerated 2D structures and molecular descriptors in demos/output/")