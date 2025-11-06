"""
RDKit utilities for molecular processing in drug recommender models.
"""
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

def generate_morgan_fingerprints(smiles_list: List[str], radius: int = 2, nBits: int = 2048) -> np.ndarray:
    """Generate Morgan fingerprints for a list of SMILES strings."""
    fingerprints = []
    valid_mols = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
            fingerprints.append(np.array(fp))
            valid_mols.append(mol)
    
    return np.vstack(fingerprints), valid_mols

def calculate_molecular_descriptors(mol) -> Dict[str, float]:
    """Calculate basic molecular descriptors using RDKit."""
    from rdkit.Chem import Descriptors
    
    return {
        'MolWt': Descriptors.ExactMolWt(mol),
        'LogP': Descriptors.MolLogP(mol),
        'TPSA': Descriptors.TPSA(mol),
        'HBD': Descriptors.NumHDonors(mol),
        'HBA': Descriptors.NumHAcceptors(mol),
        'RotBonds': Descriptors.NumRotatableBonds(mol)
    }

def process_drug_data(smiles_list: List[str]) -> Tuple[np.ndarray, List[Dict[str, float]]]:
    """Process drug SMILES to generate fingerprints and descriptors."""
    # Generate fingerprints
    fingerprints, valid_mols = generate_morgan_fingerprints(smiles_list)
    
    # Calculate descriptors
    descriptors = [calculate_molecular_descriptors(mol) for mol in valid_mols]
    
    return fingerprints, descriptors

def visualize_molecule(mol, filename: str):
    """Generate and save a 2D depiction of a molecule."""
    img = Draw.MolToImage(mol)
    img.save(filename)

def save_processed_data(output_dir: str, fingerprints: np.ndarray, descriptors: List[Dict[str, float]]):
    """Save processed molecular data."""
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, 'fingerprints.npy'), fingerprints)
    
    # Convert descriptors to DataFrame and save
    import pandas as pd
    pd.DataFrame(descriptors).to_csv(os.path.join(output_dir, 'descriptors.csv'), index=False)