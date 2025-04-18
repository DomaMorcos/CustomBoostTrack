import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class MOTGNN(nn.Module):
    def __init__(self, input_dim=260, hidden_dim=128, edge_dim=4, num_heads=4):
        super(MOTGNN, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, concat=True)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, heads=1, concat=False)
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        edge_scores = []
        for i, (src, dst) in enumerate(edge_index.t()):
            edge_feat = torch.cat([x[src], x[dst], edge_attr[i]], dim=-1)
            score = self.edge_mlp(edge_feat).sigmoid()
            edge_scores.append(score)
        return torch.stack(edge_scores) if edge_scores else torch.tensor([], dtype=torch.float32, device=x.device)