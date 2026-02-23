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
