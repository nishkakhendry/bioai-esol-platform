import torch
import mlflow
from mlflow.tracking import MlflowClient
from torch_geometric.loader import DataLoader
import tqdm

from bioai_esol.utils.seed import set_seed
from bioai_esol.data.dataset import load_esol_dataset
from bioai_esol.data.scaffold_split import scaffold_split
from bioai_esol.models.gcn import GCN
from bioai_esol.training.trainer import train_one_epoch, evaluate


# ---------------------------
# Configuration
# ---------------------------

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "esol_prediction"
MODEL_NAME = "EsolPrediction"
SELECTION_TAG = "grid_sweep_200_epochs"

GAP_THRESHOLD = 0.20
R2_TOLERANCE = 0.05
IMPROVEMENT_MARGIN = 0.02
MAE_TOLERANCE = 0.05


# ---------------------------
# Candidate Selection
# ---------------------------

def select_candidate(client, experiment):
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' "
            f"and tags.selection_tag = '{SELECTION_TAG}'"
        ),
        output_format="list"
    )

    if not runs:
        return None

    best_val_r2_overall = max(
        r.data.metrics.get("val_r2", float("-inf")) for r in runs
    )

    sorted_runs = sorted(
        runs,
        key=lambda r: r.data.metrics.get("best_val_rmse", float("inf"))
    )

    for run in sorted_runs[:5]:
        gap = run.data.metrics.get("generalisation_gap")
        val_r2 = run.data.metrics.get("val_r2")

        if gap is None or val_r2 is None:
            continue

        if gap < GAP_THRESHOLD and val_r2 >= best_val_r2_overall * (1 - R2_TOLERANCE):
            return run

    return None


# ---------------------------
# Retrain Selected Model
# ---------------------------

def retrain_until_best_epoch(run):

    params = run.data.params
    best_epoch = int(run.data.metrics["best_epoch"])

    config = {k: float(v) if v.replace('.', '', 1).isdigit() else v
              for k, v in params.items()}

    set_seed(int(config["seed"]))

    dataset, smiles_list = load_esol_dataset(config["data_root"])

    train_idx, val_idx, test_idx = scaffold_split(
        dataset,
        smiles_list,
        float(config["data_train_ratio"]),
        float(config["data_val_ratio"]),
        int(config["seed"]),
    )

    train_loader = DataLoader(dataset[train_idx], batch_size=int(config["training_batch_size"]), shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=int(config["training_batch_size"]))
    test_loader = DataLoader(dataset[test_idx], batch_size=int(config["training_batch_size"]))


    model = GCN(
        input_dim=dataset.num_node_features,
        hidden_dim=int(config["model_hidden_dim"]),
        dropout=float(config["model_dropout"])
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training_lr"], weight_decay=config["training_weight_decay"])

    for epoch in tqdm.tqdm(range(best_epoch + 1)):
        train_one_epoch(model, train_loader, optimizer)
        val_rmse, val_mae, val_r2 = evaluate(model, val_loader)

    test_rmse, test_mae, test_r2 = evaluate(model, test_loader)

    return model, {
        "val_rmse": val_rmse.item(),
        "val_mae": val_mae.item(),
        "val_r2": val_r2.item(),
        "test_rmse": test_rmse.item(),
        "test_mae": test_mae.item(),
        "test_r2": test_r2.item(),
    }, best_epoch+1


# ---------------------------
# Promotion Logic
# ---------------------------

def promote_to_champion(client, version, metrics):

    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
        champion_run = mlflow.get_run(champion.run_id)

        champion_rmse = champion_run.data.metrics["test_rmse"]
        champion_mae = champion_run.data.metrics["test_mae"]

        improvement = (champion_rmse - metrics["test_rmse"]) / champion_rmse
        mae_regression = (metrics["test_mae"] - champion_mae) / champion_mae

        if improvement >= IMPROVEMENT_MARGIN and mae_regression <= MAE_TOLERANCE:
            client.set_registered_model_alias(
                name=MODEL_NAME,
                alias="champion",
                version=version
            )
            print("Promoted to champion.")
        else:
            print("Did not satisfy champion promotion criteria.")

    except Exception:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=version
        )
        print("No existing champion. Promoted automatically.")


# ---------------------------
# Main Pipeline
# ---------------------------

def main():

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError("Experiment not found.")

    candidate_run = select_candidate(client, experiment)

    if candidate_run is None:
        print("No suitable candidate found.")
        return

    print(f"Selected run {candidate_run.info.run_id}")

    with mlflow.start_run():

        model, metrics, epochs_trained = retrain_until_best_epoch(candidate_run)

        mlflow.set_tags({"selection_tag": "retrain_candidate"})
        mlflow.log_metrics(metrics)
        
        retrain_params = candidate_run.data.params.copy()
        retrain_params["training_epochs"] = epochs_trained
        mlflow.log_params(retrain_params)

        mlflow.pytorch.log_model(model, "model")

        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        result = mlflow.register_model(model_uri, MODEL_NAME)
        version = result.version

        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="candidate",
            version=version
        )

        print(f"Registered retrained model as candidate (v{version})")

        promote_to_champion(client, version, metrics)


if __name__ == "__main__":
    main()