import torch
from torch_geometric.loader import DataLoader
import mlflow

from bioai_esol.utils.seed import set_seed
from bioai_esol.utils.config import load_config
from bioai_esol.data.dataset import load_esol_dataset
from bioai_esol.data.scaffold_split import scaffold_split
from bioai_esol.models.gcn import GCN
from bioai_esol.training.trainer import train_one_epoch, evaluate


def main():
    config = load_config("configs/config.yaml")
    set_seed(config["seed"])

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("esol_prediction")
    mlflow.start_run()

    mlflow.log_params(config)

    dataset, smiles_list = load_esol_dataset(config["data"]["root"])

    train_idx, val_idx, test_idx = scaffold_split(
        dataset,
        smiles_list,
        config["data"]["train_ratio"],
        config["data"]["val_ratio"],
        config["seed"],
    )

    train_loader = DataLoader(dataset[train_idx], batch_size=config["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(dataset[val_idx], batch_size=config["training"]["batch_size"])
    test_loader = DataLoader(dataset[test_idx], batch_size=config["training"]["batch_size"])

    print(f"train, val, test: {len(dataset[train_idx])}, {len(dataset[val_idx])}, {len(dataset[test_idx])}")

    model = GCN(input_dim=dataset.num_node_features,
                hidden_dim=config["model"]["hidden_dim"])

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    for epoch in range(config["training"]["epochs"]):
        train_loss = train_one_epoch(model, train_loader, optimizer)
        val_rmse = evaluate(model, val_loader)

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_rmse", val_rmse.item(), step=epoch)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val RMSE={val_rmse:.4f}")

    test_rmse = evaluate(model, test_loader)
    mlflow.log_metric("test_rmse", test_rmse.item())

    mlflow.pytorch.log_model(model, "model")

    mlflow.end_run()


if __name__ == "__main__":
    main()