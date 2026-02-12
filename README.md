# Defense-Mechanism-Baselines
# FL Defense Baselines (Exploratory)

This repository contains lightweight implementations of baseline defense mechanisms for Federated Learning.
The goal is to understand FL simulation mechanics, reproduce core ideas from the literature, and provide a sandbox
for future work on adaptive multi-criteria reputation learning.

## Implemented Baselines
- **FedAvg**: standard averaging aggregation
- **FLTrust**: trusted root dataset → cosine alignment trust + norm scaling
- **FoolsGold (simplified)**: penalizes overly-similar client updates via cosine similarity
- **Attacks (toy)**: random update injection, label flipping (work-in-progress)

## Quickstart
```bash
pip install -r requirements.txt
python experiments/run_fltrust.py
python experiments/run_foolsgold_vs_fedavg.py
python experiments/run_fltrust_vs_fedavg.py
