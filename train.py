import torch
from torch_geometric.loader import DataLoader
import mlflow

from bioai_esol.utils.seed import set_seed
from bioai_esol.utils.config import load_config, flatten_dict
from bioai_esol.data.dataset import load_esol_dataset
from bioai_esol.data.scaffold_split import scaffold_split
from bioai_esol.models.gcn import GCN
from bioai_esol.training.trainer import train_one_epoch, evaluate


def run_experiment(config, run_tag):
    set_seed(config["seed"])

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("esol_prediction")
    mlflow.start_run()

    mlflow.log_params(config)
    mlflow.set_tags({
        "selection_tag": f"{run_tag}",
        "model": "gcn",
        "dataset": "ESOL",
        "split": "scaffold"
    })

    dataset, smiles_list = load_esol_dataset(config["data_root"])

    train_idx, val_idx, test_idx = scaffold_split(
        dataset,
        smiles_list,
        config["data_train_ratio"],
        config["data_val_ratio"],
        config["seed"],
    )

    train_loader = DataLoader(dataset[train_idx], batch_size=config["training_batch_size"], shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=config["training_batch_size"])
    test_loader = DataLoader(dataset[test_idx], batch_size=config["training_batch_size"])

    print(f"train, val, test: {len(dataset[train_idx])}, {len(dataset[val_idx])}, {len(dataset[test_idx])}")

    model = GCN(input_dim=dataset.num_node_features,
                hidden_dim=config["model_hidden_dim"],
                dropout=config["model_dropout"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training_lr"], weight_decay=config["training_weight_decay"])

    best_val_rmse, best_epoch = float("inf"), 0

    for epoch in range(config["training_epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer)    
        train_rmse = torch.sqrt(torch.tensor(train_loss))       # train loss is MSE, so take sqrt to get RMSE

        val_rmse, val_mae, val_r2 = evaluate(model, val_loader)

        if val_rmse.item() < best_val_rmse:
            best_epoch = epoch
            best_val_rmse = val_rmse.item()

        gap = val_rmse - train_rmse

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_rmse", train_rmse.item(), step=epoch)
        mlflow.log_metric("val_mae", val_mae.item(), step=epoch)
        mlflow.log_metric("val_r2", val_r2.item(), step=epoch)
        mlflow.log_metric("val_rmse", val_rmse.item(), step=epoch)
        mlflow.log_metric("generalisation_gap", gap, step=epoch)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.4f}")

    test_rmse, test_mae, test_r2 = evaluate(model, test_loader)
    mlflow.log_metric("test_rmse", test_rmse.item())
    mlflow.log_metric("test_mae", test_mae.item())
    mlflow.log_metric("test_r2", test_r2.item())
    mlflow.log_metric("best_val_rmse", best_val_rmse)
    mlflow.log_metric("best_epoch", best_epoch)

    mlflow.pytorch.log_model(model, "model")

    mlflow.end_run()


if __name__ == "__main__":
    base_config = load_config("configs/config.yaml")
    base_config = flatten_dict(base_config)
    # original
    # run_experiment(base_config, "test")

    # hyperparameter sweep
    run_tag = "grid_sweep_200_epochs"
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

                        config["model_hidden_dim"] = hidden_dim
                        config["model_dropout"] = d
                        config["training_weight_decay"] = wd
                        config["training_lr"] = lr
                        config["training_batch_size"] = batch_size

                        print(f"Running experiment with hidden_dim={hidden_dim}, lr={lr}, batch_size={batch_size}, dropout={d}, weight_decay={wd}")
                        run_experiment(config, run_tag)