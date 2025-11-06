# Code Guide: Personalized Drug Recommender

This guide documents the core files and functionality of the drug recommender project. Files are organized by their role in the pipeline.

## Core Files Used in Final Workflow

### Model Implementations
- `src/model.py` — MLP-style patient-drug interaction model
  - Classes: DrugEncoder, InteractionModel
  - Key feature: learns joint embeddings for patients and drugs
  - Used by: run_large_demo.py

- `src/model_transformer.py` — Transformer-based architecture
  - Classes: MultiHeadAttention, TransformerBlock, TransformerDrugRecommender
  - Features: self-attention layers for patient/drug interactions
  - Used by: run_transformer_demo.py

- `src/model_mf.py` — Matrix Factorization with side features
  - Classes: MFWithSideFeatures
  - Features: learns latent factors plus side feature integration
  - Used by: run_mf_demo.py, tune_mf.py
  - Best performer in final evaluations

### Training Runners
- `src/run_large_demo.py` — Runner for MLP model
  - Features: trains on 2000 synthetic patients
  - Includes: early stopping, validation splits
  - Outputs: saves checkpoint and evaluation metrics

- `src/run_transformer_demo.py` — Runner for transformer model
  - Similar to run_large_demo.py but with transformer-specific settings
  - Adds: gradient clipping, specialized learning rate

- `src/run_mf_demo.py` — Runner for MF baseline
  - Features: batched training/validation
  - Configurable: batch_size, weight_decay
  - Used by the tuning script

- `src/tune_mf.py` — Hyperparameter tuning for MF
  - Grid search over: latent_dim, lr, weight_decay, batch_size
  - Saves: tuning summary to models/tuning_results_mf.json

### Data Generation & Processing
- `src/data/synthetic.py` — Synthetic patient generator
  - Function: generate_synthetic_patients()
  - Creates: patient feature vectors
  - Used by all demo runners

- `src/data/interactions.py` — Interaction dataset builder
  - Function: build_interaction_dataset()
  - Creates: interaction matrix R with NaN masking
  - Used by all demo runners

- `src/data/fetch_chembl.py` — ChEMBL data fetcher
  - Downloads: bioactivity data and drug info
  - Used to create: fingerprints.npy

- `src/data/generate_fingerprints.py` — Fingerprint generator
  - Creates: drug fingerprint features
  - Output: data/chembl/fingerprints.npy

### Evaluation
- `src/evaluate.py` — Evaluation metrics computation
  - Functions: compute_metrics(), find_optimal_threshold()
  - Features: NaN-safe metric computation
  - Metrics: precision, recall, F1, AUROC, NDCG@k, MRR, MAP

## Files Not Used in Final Pipeline
These files were used in development but not in the final workflow:
- `src/eval.py` — Older evaluation code (superseded by evaluate.py)
- `src/explain.py` — SHAP-based explainability (prototype)
- `src/train.py` — Early training script (replaced by specific runners)
- `src/train_chembl.py` — ChEMBL training prototype
- `src/train_hpo.py` — Early hyperparameter optimization
- `src/visualization.py` — Plotting utilities (optional)
- `src/data.py` — Old data loading (split into synthetic.py/interactions.py)

## Directory Structure
```
src/
├── data/               # Data generation and processing
│   ├── synthetic.py    # Patient generation
│   ├── interactions.py # Interaction matrix creation
│   └── fetch_chembl.py # ChEMBL data fetching
├── model.py           # MLP model
├── model_transformer.py # Transformer model
├── model_mf.py        # Matrix Factorization model
├── run_*_demo.py      # Model-specific runners
├── tune_mf.py         # MF hyperparameter tuning
└── evaluate.py        # Evaluation metrics
```

## How to Use Each Component

### Generate Synthetic Dataset
```python
from src.data.synthetic import generate_synthetic_patients
from src.data.interactions import build_interaction_dataset
import numpy as np

# Load drug features
drug_features = np.load('data/chembl/fingerprints.npy')

# Generate synthetic patients
patients = generate_synthetic_patients(n_patients=2000, feature_dim=16)

# Build interaction dataset
meta, Q, R = build_interaction_dataset(
    patients=patients,
    drug_features=drug_features,
    observed_fraction=0.8
)
```

### Train Models
Each model has its own runner with appropriate hyperparameters:

MLP model:
```python
python src/run_large_demo.py --n_patients 2000 --epochs 50
```

Transformer model:
```python
python src/run_transformer_demo.py --n_patients 2000 --epochs 50
```

Matrix Factorization:
```python
python src/run_mf_demo.py --n_patients 2000 --latent_dim 16 --lr 0.0005 --weight_decay 1e-5 --batch_size 32 --epochs 50
```

### Evaluate Models
Evaluation metrics are computed automatically by the runners using `src/evaluate.py`. The metrics include:
- Binary classification: precision, recall, F1, AUROC, AUPR
- Ranking: NDCG@10, MRR, MAP

Results are saved as JSON files in the `models/` directory.

## Implementation Notes

### Model Architectures
1. MLP (InteractionModel)
   - Separate encoders for patients and drugs
   - Concatenation and MLP head for interaction prediction
   
2. Transformer
   - MultiHeadAttention for drug-patient interactions
   - Multiple transformer blocks
   - Final MLP prediction head

3. Matrix Factorization
   - Patient and drug latent factors
   - Side feature integration
   - Optional MLP on concatenated latents

### Training Pipeline
1. Data generation:
   - Generate synthetic patients
   - Load drug fingerprints
   - Create interaction matrix with masking

2. Model training:
   - Split data into train/val/test
   - Train with early stopping
   - Save best checkpoint

3. Evaluation:
   - Compute metrics on test set
   - Save results to JSON
   - Optional: generate plots

### Best Practices
- Always use NaN masking in evaluation
- Use early stopping for all models
- Save both best and final checkpoints
- Log hyperparameters with results