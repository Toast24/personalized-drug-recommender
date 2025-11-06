"""Small hyperparameter tuner for MF baseline.

This script runs a small grid search over latent_dim and learning rate by
calling `run_mf_demo.main` for each config. It collects the evaluation JSON
output and writes a summary `models/tuning_results_mf.json`.
"""
import os
import sys
import json
from itertools import product

# Ensure src is on path so we can import run_mf_demo
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from run_mf_demo import main as run_mf_main


def load_metrics(run_name):
    path = os.path.join('models', f'evaluation_results_{run_name}.json')
    if not os.path.exists(path):
        return None
    with open(path, 'r') as f:
        return json.load(f)


def main():
    # Expanded grid (moderate size). epochs set to full runs.
    latent_dims = [16, 32, 64]
    lrs = [5e-4, 1e-4]
    weight_decays = [0.0, 1e-5]
    batch_sizes = [32, 128]
    epochs = 50  # longer training for final tuning

    results = []

    for ld, lr, wd, bs in product(latent_dims, lrs, weight_decays, batch_sizes):
        run_name = f"mf_ld{ld}_lr{lr}_wd{wd}_bs{bs}".replace('.', 'p')
        print(f"Running config: latent_dim={ld}, lr={lr}, weight_decay={wd}, batch_size={bs}, run_name={run_name}")
        # Run training (this will save metrics to models/evaluation_results_{run_name}.json)
        run_mf_main(n_patients=2000, epochs=epochs, latent_dim=ld, lr=lr, run_name=run_name, batch_size=bs, weight_decay=wd)

        metrics = load_metrics(run_name)
        if metrics is None:
            print(f"No metrics found for {run_name}")
            continue
        metrics['config'] = {'latent_dim': ld, 'lr': lr, 'weight_decay': wd, 'batch_size': bs, 'epochs': epochs, 'run_name': run_name}
        results.append(metrics)

    os.makedirs('models', exist_ok=True)
    out_path = os.path.join('models', 'tuning_results_mf.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print best by f1
    best = None
    for r in results:
        if best is None or r.get('f1', 0) > best.get('f1', 0):
            best = r
    print('\nTuning complete. Best config:')
    print(json.dumps(best, indent=2))


if __name__ == '__main__':
    main()
