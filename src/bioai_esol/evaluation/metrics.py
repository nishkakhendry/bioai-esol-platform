import torch

def compute_rmse(preds, targets):
    """
    Compute Root Mean Squared Error.
    """
    return torch.sqrt(torch.mean((preds - targets) ** 2))

def compute_mae(preds, targets):
    """
    Compute Mean Absolute Error.
    """
    return torch.mean(torch.abs(preds - targets))

def compute_r2_score(preds, targets):
    """
    Compute R² Score.
    """
    ss_res = torch.sum((targets - preds) ** 2)
    ss_tot = torch.sum((targets - torch.mean(targets)) ** 2)

    # Avoid division by zero
    if ss_tot == 0:
        return torch.tensor(0.0)
    
    return 1 - ss_res / ss_tot