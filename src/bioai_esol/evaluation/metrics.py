import torch

def compute_rmse(preds, targets):
    """
    Compute Root Mean Squared Error.
    """
    return torch.sqrt(torch.mean((preds - targets) ** 2))