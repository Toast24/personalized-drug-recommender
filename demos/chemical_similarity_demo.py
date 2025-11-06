"""
Demo: Chemical Similarity and Clustering
"""
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.ML.Cluster import Butina
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_similarity_matrix(smiles_list):
    # Convert SMILES to molecules and generate fingerprints
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024) for mol in mols]
    
    # Calculate similarity matrix
    n = len(fps)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarity_matrix[i,j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
    
    return similarity_matrix

def cluster_molecules(similarity_matrix, cutoff=0.5):
    # Convert similarity matrix to distance matrix
    distances = 1 - similarity_matrix
    
    # Get distance list in the format required by Butina
    dist_list = []
    n = len(similarity_matrix)
    for i in range(1,n):
        for j in range(i):
            dist_list.append(distances[i][j])
    
    # Cluster using Butina algorithm
    clusters = Butina.ClusterData(dist_list, n, cutoff, isDistData=True)
    return clusters

def visualize_similarity(similarity_matrix, names, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=names, yticklabels=names)
    plt.title('Chemical Similarity Matrix (Tanimoto)')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Example drug SMILES
    drugs = {
        'Aspirin': 'CC(=O)OC1=CC=CC=C1C(=O)O',
        'Ibuprofen': 'CC(C)CC1=CC=C(C=C1)[C@H](C)C(=O)O',
        'Paracetamol': 'CC(=O)NC1=CC=C(O)C=C1',
        'Caffeine': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',
        'Morphine': 'CN1CC[C@]23[C@H]4C=C[C@@H]([C@H]2O)O[C@H]3[C@H](O)C=C4C1',
        'Penicillin': 'CC1(C(N2C(S1)[C@@H](C2=O)NC(=O)CC3=CC=CC=C3)C(=O)O)C'
    }
    
    names = list(drugs.keys())
    smiles_list = list(drugs.values())
    
    # Calculate similarity matrix
    print("Calculating chemical similarities...")
    similarity_matrix = calculate_similarity_matrix(smiles_list)
    
    # Create visualization directory
    os.makedirs('demos/output', exist_ok=True)
    
    # Visualize similarity matrix
    print("Generating similarity heatmap...")
    visualize_similarity(similarity_matrix, names, 'demos/output/similarity_matrix.png')
    
    # Cluster molecules
    print("\nClustering molecules...")
    clusters = cluster_molecules(similarity_matrix, cutoff=0.5)
    
    print("\nMolecule clusters (Tanimoto similarity > 0.5):")
    for i, cluster in enumerate(clusters):
        print(f"\nCluster {i+1}:")
        for idx in cluster:
            print(f"  - {names[idx]}")

if __name__ == '__main__':
    main()