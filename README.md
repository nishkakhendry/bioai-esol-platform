# bioai-esol-platform

Learning hands-on how to take an applied ML research model to production

## Goal of this Project

Main Focus = Taking research into production.

- The aim is NOT research specialisation in molecular ML; it is to gain familiarity with an established benchmark to better understand the field, and practice making well thought-out design decisions when taking the model to production

Note: This project is documented as an open-source-style guide, walking readers step by step through the process of taking research skills toward production along with me via this README. It explains how to run various software tools used in context of this project, but does not provide in-depth tutorials on the tools themselves, as they are widely adopted and well documented on their respective websites. My learning of how to use these tools was guided by this well-curated free online course: [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main)

**Roadmap**

Current progress highlighted in _italics_

(Dataset + model) → (Experiment tracking + hyperprameter sweep) → _(Analyse metrics to choose best model)_ → (Model inference API + deployment) → (Monitoring model performance with Evidently AI + Grafana) → (CI and Testing)

## Task

Regression Task: Predicting log solubility in mols per litre

### Dataset

ESOL (Estimated SOLubility): Small molecular dataset introduced by Delaney (2004) for predicting the aqueous solubility of small organic molecules. Imported from `torch_geometric` using code in `data/dataset.py`.

**Rationale for Choosing ESOL**

Gaining familiarity with molecular ML benchmarking while keeping the task simple and the dataset small.

- Small dataset (~1100 molecules) → trains fast
- Real-world relevance of solubility in drug discovery (drugs need to dissolve in bodily fluids)
- Common benchmark - good starting point + regression good for simulating monitoring/distribution shifts later on in project
- Want familiarity with: Basic molecular ML (molecules + Graph NN)

## Set up

Clone the repository

```
git clone https://github.com/nishkakhendry/bioai-esol-platform.git
cd bioai-esol-platform
```

Create conda environment & install dependencies

- Note: Python 3.10 chosen for stability with libraries like rdkit

```
conda create -n besol python=3.10
conda activate besol
```

The source code is structured as an installable Python package using a pyproject.toml file to provide standardized dependency management, reproducible builds, and clean installation via modern Python packaging tools.

```
pip install -e .
```

_Baseline Configuration_: `configs/config.yaml`

```
seed: 42

data:
  root: "data"
  dataset_name: "ESOL"
  train_ratio: 0.8
  val_ratio: 0.1

model:
  hidden_dim: 64
  num_layers: 2

training:
  batch_size: 32
  lr: 0.001
  epochs: 50

```

- Hidden dimension (64): sufficient representational capacity without overfitting on ~1100 samples.
- Two GCN layers: captures local chemical structure while avoiding graph oversmoothing.
- Batch size (32): balances gradient stability and generalisation on small datasets.
- Learning rate (1e-3): stable Adam default for small GNNs.
- Epochs (50): enough to observe convergence without excessive overfitting.

## Training the ESOL Model

Update `configs/config.yaml` with desired parameters

Run:

```
python train.py
```

## Hyperparameter Sweep with MLflow

Hyperparameter sweep using MLflow for experiment tracking in MLflow + automatic retraining and model promotion policy in MLflow model registry covered in: `documentation/hyperpara-sweeps_metric-analysis.md`

Outcome of this step = "champion" model ready for production deployment in MLflow model registry

## Model Inference via API + Deployment
