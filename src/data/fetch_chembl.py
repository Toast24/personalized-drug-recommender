"""
Download and process ChEMBL dataset for drug-target interaction prediction.
"""
import os
from chembl_webresource_client.new_client import new_client
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import List, Tuple, Dict
import logging
from tqdm import tqdm
import pickle

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def get_target_data(min_activities: int = 50) -> pd.DataFrame:
    """
    Get protein target data from ChEMBL.
    Only include targets with sufficient bioactivity data.
    """
    import time
    from requests.exceptions import RequestException
    
    max_retries = 3
    retry_delay = 5  # seconds
    target = new_client.target
    
    # Get human protein targets
    for attempt in range(max_retries):
        try:
            targets = list(target.filter(
                target_type="SINGLE PROTEIN",
                organism="Homo sapiens"
            ).only(['target_chembl_id', 'pref_name', 'target_type'])[:1000])
            break
        except RequestException as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(retry_delay)
    
    targets_df = pd.DataFrame(targets)
    logger = logging.getLogger(__name__)
    logger.info(f"Found {len(targets_df)} human protein targets")
    
    # Get activity counts for each target
    activity = new_client.activity
    all_activities = []
    
    # Process targets in batches to avoid timeouts
    batch_size = 50
    for i in range(0, len(targets_df), batch_size):
        batch_targets = targets_df['target_chembl_id'].iloc[i:i+batch_size].tolist()
        logger.info(f"Fetching activities for targets {i+1}-{min(i+batch_size, len(targets_df))}")
        
        for attempt in range(max_retries):
            try:
                batch_activities = list(activity.filter(
                    target_chembl_id__in=batch_targets,
                    standard_type="IC50",
                    standard_relation="=",
                   ).only(['target_chembl_id', 'molecule_chembl_id', 'standard_value'])[:10000])
                all_activities.extend(batch_activities)
                break
            except RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(retry_delay)
    
    activities_df = pd.DataFrame(all_activities)
    logger.info(f"Found {len(activities_df)} bioactivities")
    
    # Filter targets by activity count
    target_counts = activities_df['target_chembl_id'].value_counts()
    valid_targets = target_counts[target_counts >= min_activities].index
    
    result_df = targets_df[targets_df['target_chembl_id'].isin(valid_targets)]
    logger.info(f"Found {len(result_df)} targets with at least {min_activities} activities")
    
    return result_df

def get_drug_data(target_ids: List[str], activity_threshold: float = 1000) -> pd.DataFrame:
    """
    Get drug bioactivity data from ChEMBL.
    Only include compounds with IC50 values below threshold (in nM).
    """
    import time
    from requests.exceptions import RequestException
    
    max_retries = 3
    retry_delay = 5  # seconds
    activity = new_client.activity
    logger = logging.getLogger(__name__)
    
    all_activities = []
    batch_size = 20  # Process targets in smaller batches
    
    for i in range(0, len(target_ids), batch_size):
        batch_targets = target_ids[i:i+batch_size]
        logger.info(f"Fetching bioactivities for targets {i+1}-{min(i+batch_size, len(target_ids))}")
        
        for attempt in range(max_retries):
            try:
                batch_activities = list(activity.filter(
                    target_chembl_id__in=batch_targets,
                    standard_type="IC50",
                    standard_relation="=",
                    standard_value__lt=activity_threshold
                ).only([
                    'molecule_chembl_id', 'target_chembl_id',
                    'standard_value', 'standard_units'
                ]).limit(10000))
                
                all_activities.extend(batch_activities)
                logger.info(f"Found {len(batch_activities)} activities for current batch")
                break
            except RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    activities_df = pd.DataFrame(all_activities)
    logger.info(f"Total bioactivities found: {len(activities_df)}")
    
    return activities_df

def get_molecule_structures(molecule_ids: List[str]) -> pd.DataFrame:
    """Get molecular structure data from ChEMBL."""
    import time
    from requests.exceptions import RequestException
    
    max_retries = 3
    retry_delay = 5  # seconds
    molecule = new_client.molecule
    logger = logging.getLogger(__name__)
    
    all_molecules = []
    batch_size = 50  # Process molecules in batches
    
    for i in range(0, len(molecule_ids), batch_size):
        batch_molecules = molecule_ids[i:i+batch_size]
        logger.info(f"Fetching structures for molecules {i+1}-{min(i+batch_size, len(molecule_ids))}")
        
        for attempt in range(max_retries):
            try:
                batch_data = list(molecule.filter(
                    molecule_chembl_id__in=batch_molecules
                ).only(['molecule_chembl_id', 'molecule_structures']).limit(10000))
                
                all_molecules.extend(batch_data)
                logger.info(f"Found {len(batch_data)} structures for current batch")
                break
            except RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
    
    molecules_df = pd.DataFrame(all_molecules)
    logger.info(f"Total structures found: {len(molecules_df)}")
    
    return molecules_df

def generate_fingerprints(smiles_list: List[str], radius: int = 2, nBits: int = 2048) -> np.ndarray:
    """Generate Morgan fingerprints for molecules."""
    fingerprints = []
    valid_indices = []
    
    for idx, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits)
                fingerprints.append(np.array(fp))
                valid_indices.append(idx)
        except:
            continue
    
    return np.stack(fingerprints), valid_indices

def main():
    logger = setup_logging()
    os.makedirs('data/chembl', exist_ok=True)
    
    # Get protein targets
    logger.info("Fetching protein targets from ChEMBL...")
    targets_df = get_target_data(min_activities=100)
    logger.info(f"Found {len(targets_df)} protein targets")
    
    # Get bioactivity data
    logger.info("Fetching bioactivity data...")
    activities_df = get_drug_data(
        targets_df['target_chembl_id'].tolist(),
        activity_threshold=1000  # 1µM threshold
    )
    logger.info(f"Found {len(activities_df)} bioactivities")
    
    # Get molecular structures
    logger.info("Fetching molecular structures...")
    molecules_df = get_molecule_structures(
        activities_df['molecule_chembl_id'].unique()
    )
    
    # Extract SMILES and generate fingerprints
    logger.info("Generating molecular fingerprints...")
    smiles_list = [
        mol['canonical_smiles'] 
        for mol in molecules_df['molecule_structures'] 
        if mol is not None and 'canonical_smiles' in mol
    ]
    fingerprints, valid_indices = generate_fingerprints(smiles_list)
    
    # Filter activities to match valid molecules
    valid_molecules = molecules_df.iloc[valid_indices]
    activities_df = activities_df[
        activities_df['molecule_chembl_id'].isin(valid_molecules['molecule_chembl_id'])
    ]
    
    # Save processed data
    logger.info("Saving processed data...")
    np.save('data/chembl/fingerprints.npy', fingerprints)
    activities_df.to_csv('data/chembl/bioactivity.csv', index=False)
    targets_df.to_csv('data/chembl/targets.csv', index=False)
    
    # Save mapping dictionaries
    mappings = {
        'molecule_to_idx': {
            id_: idx for idx, id_ in enumerate(valid_molecules['molecule_chembl_id'])
        },
        'target_to_idx': {
            id_: idx for idx, id_ in enumerate(targets_df['target_chembl_id'])
        }
    }
    with open('data/chembl/mappings.pkl', 'wb') as f:
        pickle.dump(mappings, f)
    
    logger.info(f"Successfully processed {len(fingerprints)} drugs and {len(targets_df)} targets")

if __name__ == '__main__':
    main()