"""
ChEMBL data download and processing script.
Fetches drug structures and bioactivity data from ChEMBL.

Usage:
    python -m src.data.chembl_download --output-dir data/chembl
"""
import os
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import requests
import json
from pathlib import Path
from tqdm import tqdm

# ChEMBL API endpoints
CHEMBL_API_BASE = "https://www.ebi.ac.uk/chembl/api/data"
CHEMBL_DOWNLOAD_BASE = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest"

def download_file(url: str, output_path: Path, desc: str = None) -> None:
    """Download a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 8192
    
    with open(output_path, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
    ) as pbar:
        for data in response.iter_content(block_size):
            f.write(data)
            pbar.update(len(data))

def fetch_compounds_by_approval(output_dir: Path) -> pd.DataFrame:
    """Fetch approved drugs from ChEMBL."""
    approved_url = f"{CHEMBL_API_BASE}/molecule?max_phase=4&format=json"
    response = requests.get(approved_url)
    data = response.json()
    
    compounds = []
    for mol in tqdm(data['molecules'], desc="Fetching approved drugs"):
        compound = {
            'molecule_chembl_id': mol['molecule_chembl_id'],
            'pref_name': mol.get('pref_name', ''),
            'max_phase': mol['max_phase'],
            'molecule_type': mol['molecule_type'],
            'canonical_smiles': mol.get('molecule_structures', {}).get('canonical_smiles', None),
        }
        compounds.append(compound)
    
    df = pd.DataFrame(compounds)
    df = df[df['canonical_smiles'].notna()]  # Keep only compounds with structures
    
    output_path = output_dir / 'approved_drugs.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} approved drugs to {output_path}")
    return df

def fetch_bioactivity_data(molecule_ids: List[str], output_dir: Path) -> pd.DataFrame:
    """Fetch bioactivity data for the given molecules."""
    activities = []
    chunk_size = 100  # Process in chunks to avoid API limits
    
    for i in tqdm(range(0, len(molecule_ids), chunk_size), desc="Fetching bioactivity"):
        chunk = molecule_ids[i:i + chunk_size]
        ids_str = ','.join(chunk)
        url = f"{CHEMBL_API_BASE}/activity?molecule_chembl_id__in={ids_str}&format=json"
        response = requests.get(url)
        data = response.json()
        
        for act in data['activities']:
            if act.get('standard_value') and act.get('standard_type'):
                activity = {
                    'molecule_chembl_id': act['molecule_chembl_id'],
                    'target_chembl_id': act.get('target_chembl_id', ''),
                    'standard_type': act['standard_type'],
                    'standard_value': act['standard_value'],
                    'standard_units': act.get('standard_units', ''),
                    'target_organism': act.get('target_organism', ''),
                }
                activities.append(activity)
    
    df = pd.DataFrame(activities)
    output_path = output_dir / 'bioactivity.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} bioactivity records to {output_path}")
    return df

def process_chembl_data(output_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Main function to download and process ChEMBL data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Get approved drugs
    drugs_df = fetch_compounds_by_approval(output_dir)
    
    # 2. Get bioactivity data for these drugs
    bio_df = fetch_bioactivity_data(drugs_df['molecule_chembl_id'].tolist(), output_dir)
    
    return drugs_df, bio_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=str, default='data/chembl',
                      help='Directory to save ChEMBL data')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    drugs_df, bio_df = process_chembl_data(output_dir)
    
    # Print summary
    print("\nDataset Summary:")
    print(f"Number of drugs: {len(drugs_df)}")
    print(f"Number of bioactivity records: {len(bio_df)}")
    print(f"\nMost common activity types:")
    print(bio_df['standard_type'].value_counts().head())

if __name__ == '__main__':
    main()