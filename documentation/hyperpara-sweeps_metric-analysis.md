# Hyperparameter Sweeps

This markdown file documents hyperparameter sweeps and analysis using MLflow for experiment tracking, and uses the MLflow Model Registry to promote the best-performing model to the Production stage.

## MLflow

### Set up

- Log in to AWS and create S3 for storing artifacts. Placeholder to be replaced by your bucket name = `<s3_bucket_name>`
- MLflow should be installed already during the repository set-up step
- To run a local MLflow server which stores artifacts in your new S3 bucket, run:

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3:`<s3_bucket_name>`
```

Note: You may need to run the `aws configure` command to properly interact with AWS S3 ([documentation here](https://docs.aws.amazon.com/cli/latest/reference/configure/))

Access MLflow UI at `http://localhost:5000`

**Rationale for this Design**

_Why not run MLflow on EC2?_

- Running MLflow on EC2 would incur continuous compute and storage costs, even when idle, which is unnecessary for my single-developer research workflow. A local server helped me avoid infrastructure overhead while preventing accidental consumption of AWS credits.

_Why local MLflow server + S3 artifacts?_

- The local MLflow server enables fast, low-cost experiment tracking and UI access, while storing artifacts in S3 provides durable, scalable, and production-aligned cloud storage. This hybrid setup decouples metadata from artifact storage in an effort to mirror real-world MLOps architectures without requiring full cloud deployment.

_Why --backend-store-uri sqlite:///mlflow.db?_

- Using SQLite as the backend store provides a lightweight, zero-maintenance database for tracking experiment metadata, making it ideal for my single-developer setting. It eliminates the need to provision and manage PostgreSQL on EC2. and keeps the setup lightweight while remaining compatible with future migration to a managed production database.

## Hyperparameters and Chosen Ranges

`hidden_dims = [32, 64, 128]`:
Tests increasing model capacity from lightweight (32) to moderately expressive (128). ESOL is small, so this range checks whether performance is capacity-limited without risking severe overfitting.

`lrs = [1e-2, 1e-3, 3e-3]`: Covers a typical effective range for Adam; 1e-3 is a strong default, 3e-3 tests slightly more aggressive updates, and 1e-2 checks whether faster convergence is possible without instability.

`batch_sizes = [16, 32]`: Small batches suit small datasets like ESOL and introduce beneficial gradient noise; 32 improves stability while 16 can improve generalisation.

`dropout = [0.0, 0.2, 0.5]`: Evaluates no regularisation (0.0), mild regularisation (0.2), and strong regularisation (0.5) to control overfitting in a small-data regime.

`weight_decay = [0, 1e-5, 1e-4]`: Tests no L2 regularisation vs light regularisation; small values are appropriate since strong weight decay can underfit small molecular datasets.

Slight modification to train.py (Appendix 1) to perform grid search over these values

# Analysing Runs with MLflow UI: Choosing the Best Model

Selection Criteria: Balance performance and stability

- The run with the lowest final `val_rmse` exhibits a larger generalisation gap (~+0.15) and a higher `best_val_rmse`, indicating potential instability or overfitting despite strong final performance.
- The run with the lowest `best_val_rmse` (popular-finch-805) achieves:
  - The best peak validation performance
  - A significantly smaller generalisation gap
  - Comparable `R²` only ~0.02 below the absolute maximum (higher `R²` captures more structure in the variation of solubility)
- The model with the absolute lowest generalisation gap was not selected because validation performance (RMSE) is prioritised over gap minimisation, provided the gap remains reasonably small.

MLflow UI containing logged metric values and plots used for comparisons, and to find epoch corresponding to `best_val_rmse` in selected run.

**Other observations from top-performing runs (lowest validation RMSE):**

- Larger model capacities (hidden_dim 128, occasionally 64) consistently perform better = performance improves with increased representational power within this range.
- Moderate learning rates (typically 3e-3) yield stronger performance = slightly more aggressive optimisation improves convergence without instability.
- Low to mild regularisation (dropout 0.0 or 0.2) is preferred, while strong dropout (0.5) tends to underfit in this small-data regime.
- Smaller batch size (16) seems to generalises better = randomness introduced by small batches can improve generalisation by preventing overfitting to very specific training patterns
- Light or no weight decay performs similarly, with no strong evidence that heavier L2 regularisation improves validation performance on this dataset.

# Promoting the Best Model to Production Stage

MLflow Model Registry provides structured versioning and lifecycle management of trained models, enabling controlled promotion from experimentation to production with full reproducibility. It is relevant because it bridges research and deployment by tracking model versions, metadata, and performance while supporting safe, governed model updates.

## Staging and Aliases in MLflow Model Registry

| Traditional Stage | Old Meaning                   | Modern Alias Equivalent          | User For                                                                                                             |
| ----------------- | ----------------------------- | -------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| **None**          | Newly registered, unevaluated | No alias assigned                | Freshly trained models, Models awaiting evaluation, Grid search candidates                                           |
| **Staging**       | Candidate under validation    | `candidate` or `staging` alias   | Offline evaluation, A/B testing, Shadow deployments, Manual QA review                                                |
| **Production**    | Live serving model            | `champion` or `production` alias | The model currently powering inference, Stable, validated, reproducible version, Model tied to monitoring and alerts |
| **Archived**      | Retired version               | No alias (or `archived` tag)     | Old production models, Failed candidates, Historical reproducibility                                                 |

**Why aliases > stages?**

MLflow aliases replace rigid lifecycle stages with flexible named pointers such as `champion`, `candidate`, `baseline` and `shadow`. Multiple aliases can coexist, enabling champion–challenger workflows, shadow deployments, and controlled promotion without hard state transitions.

## Model Promotion Policy

Implemented in `src/bioai_esol/mlops/promote_model.py`.
Run from `mlops` directory using:

```
python promote_model.py
```

Flow of this script:
Selects a candidate using logic described below -> Extracts hyperparameters from the chosen MLflow run -> Retrains using same training logic and data split with early stopping to optimal step -> Logs the retrained model as a new run -> Registers it as `candidate` -> Promotes it to `champion` if criteria are satisfied

The policy should:

1. Prevent accidental promotion of unstable models, and allow rollbacks
2. Be metric-driven (not subjective)
3. Be reproducible
4. Require minimal manual overhead (e.g., UI clicks)
5. Work for a single developer

**Registration Rule**

A model is eligible for registry **only if**:

- Training completed without error
- `best_val_rmse` is logged
- `test_rmse`, `test_mae`, `test_r2` are logged
- Generalisation gap is within acceptable bounds

If these conditions are not met → do not register.

**Candidate Assignment Rule**

Among all completed runs in an experiment, find the model satisfying:

- `best_val_rmse` is the lowest observed so far
- AND `generalisation_gap` < threshold
- AND `val_r2` within 5% of best observed

This prevents: Selecting unstable minima, Selecting high-variance runs, Selecting marginal improvements

Select the chosen model's hyperparameters as the "best" from the grid search. Retrain the model from scratch with early stopping using the same data split, then register it as a `candidate`. There is only one `candidate` at a time.

**Promotion to Champion Rule**

Promote `candidate` → `champion` only if:

- `test_rmse` improves over current champion by ≥ 1–2%
- AND no metric regression > tolerance (e.g., MAE worse by >5%)
- AND generalisation gap remains acceptable

This automatically “demotes” previous champion (since alias moves).

**Reproducibility Rule**

Before champion promotion, verify:

- Seed logged, Hyperparameters logged, Code version (git commit hash) logged, Dataset version fixed

If not → block promotion. This prevents silent irreproducibility.

→ All already being logged in MLflow runs

**Rollback Rule**

If new champion degrades in monitoring:

- Reassign `champion` to previous version, Tag failed version: `status="failed_in_monitoring"`

No deletion -- only alias reassignment.

# Appendix 1

Slightly modified main function in `train.py` for hyperparameter sweep with MLflow logging.

```
if __name__ == "__main__":
    base_config = load_config("configs/config.yaml")
    # original
    # run_experiment(base_config)

    # hyperparameter sweep
    run_tag = "grid_sweep"
    hidden_dims = [32, 64, 128]
    lrs = [1e-2, 1e-3, 3e-3]
    batch_sizes = [16, 32]
    dropout = [0.0, 0.2, 0.5]
    weight_decay = [0, 1e-5, 1e-4]

    for hidden_dim in hidden_dims:
        for lr in lrs:
            for batch_size in batch_sizes:
                for d in dropout:
                    for wd in weight_decay:

                        config = base_config.copy()

                        config["model"]["hidden_dim"] = hidden_dim
                        config["model"]["dropout"] = d
                        config["training"]["weight_decay"] = wd
                        config["training"]["lr"] = lr
                        config["training"]["batch_size"] = batch_size

                        print(f"Running experiment with hidden_dim={hidden_dim}, lr={lr}, batch_size={batch_size}, dropout={d}, weight_decay={wd}")
                        run_experiment(config, run_tag)
```
