"""
Generate molecular fingerprints for ChEMBL drugs.
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

def generate_fingerprints(smiles_list: list, n_bits: int = 2048) -> np.ndarray:
    """Generate Morgan fingerprints for a list of SMILES strings."""
    if not RDKIT_AVAILABLE:
        print("Warning: RDKit not available, using fallback fingerprints")
        return np.random.RandomState(42).randn(len(smiles_list), n_bits).astype(np.float32)
    
    fps = []
    for smiles in tqdm(smiles_list, desc="Generating fingerprints"):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Fallback for invalid SMILES
            fp = np.zeros(n_bits, dtype=np.float32)
        else:
            fp = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits))
        fps.append(fp)
    return np.vstack(fps).astype(np.float32)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-csv', type=str, default='data/chembl/approved_drugs.csv')
    parser.add_argument('--output-dir', type=str, default='data/chembl')
    parser.add_argument('--n-bits', type=int, default=2048)
    args = parser.parse_args()
    
    # Load drugs
    drugs_df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(drugs_df)} drugs")
    
    # Generate fingerprints
    fps = generate_fingerprints(drugs_df['canonical_smiles'].tolist(), args.n_bits)
    
    # Save
    output_path = Path(args.output_dir) / 'fingerprints.npy'
    np.save(output_path, fps)
    print(f"Saved fingerprints to {output_path}")
    print(f"Fingerprint shape: {fps.shape}")

if __name__ == '__main__':
    main()