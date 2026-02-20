import torch
import torch.nn.functional as F
from bioai_esol.evaluation.metrics import compute_rmse, compute_mae, compute_r2_score

def train_one_epoch(model, loader, optimizer):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()       # at every loss.backward(), gradients are added to .grad fields of parameters. If you didn’t zero them out, gradients from previous *batches* would accumulate
        preds = model(batch.x, batch.edge_index, batch.batch)   # shape [batch_size]
        loss = F.mse_loss(preds, batch.y.view(-1))              # scalar: mean over shape [batch_size]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            p = model(batch.x, batch.edge_index, batch.batch)
            preds.append(p)
            targets.append(batch.y.view(-1))
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    return compute_rmse(preds, targets), compute_mae(preds, targets), compute_r2_score(preds, targets)