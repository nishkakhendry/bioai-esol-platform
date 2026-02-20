# bioai-esol-platform

Learning hands-on how to take an applied ML research model to production

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
pip install -r requirements.txt
```

_Baseline Configuration_: `config.yaml`

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

- Hidden dimension (64): sufficient representational capacity without overfitting on ~1k samples.
- Two GCN layers: captures local chemical structure while avoiding graph oversmoothing.
- Batch size (32): balances gradient stability and generalisation on small datasets.
- Learning rate (1e-3): stable Adam default for small GNNs.
- Epochs (50): enough to observe convergence without excessive overfitting.
