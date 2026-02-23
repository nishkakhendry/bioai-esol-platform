import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class GCN(torch.nn.Module):
    """
    Graph Convolutional Network for molecular property regression.
    """

    def __init__(self, input_dim, hidden_dim,dropout=0.0):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = x.float()  # Ensure node features are float for GCNConv

        # first GCN layer
        x = F.relu(self.conv1(x, edge_index))       # Conv → ReLU → Conv → ReLU = hierarchical nonlinear feature extraction over the graph
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Second GCN layer
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)

        # Graph-level pooling
        x = global_mean_pool(x, batch)          # [total_nodes_in_batch, hidden_dim] -> [num_graphs_in_batch, hidden_dim]: collapse node features → graph representation. Otherwise predicting per-node intead of per-graph + mean pooling size-invariant and stable for molecular graphs

        # final linear layer for regression
        return self.lin(x).view(-1)         # lin gives output [num_graphs, 1] regression output per graph, view(-1) flattens tensor to 1D to get shape [num_graphs] for loss calculation
    
    