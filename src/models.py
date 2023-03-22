import torch
import torch.nn as nn

import torch_geometric as tg
import torch_geometric.nn as tgnn

class GCN(nn.Module):
    def __init__(self, indim, hidden, outdim, n_layers):
        super().__init__()
        self.l1 = tgnn.GCNConv(indim, hidden)

        self.layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.layers.append(
                tgnn.GCNConv(hidden, hidden)
            )
            self.layers.append(
                nn.ReLU()
            )

        self.out = tgnn.GCNConv(hidden, hidden)
        self.net = nn.Linear(hidden, outdim)
        

    def forward(self, x, edge_index, edge_attr=None, ptr=None):
        # x = self.l1(x, edge_index, edge_attr) if edge_attr is not None else  self.l1(x, edge_index)
        x = self.l1(x.float(), edge_index.long(), edge_attr.float())
        for layer in self.layers:
            if isinstance(layer, tgnn.GCNConv):
                x = layer(x.float(), edge_index.long(), edge_attr.float()) if edge_attr is not None else layer(x.float(), edge_index.long())
            else:
                x = layer(x)
        out = self.out(x.float(), edge_index.long(), edge_attr.float()) if edge_attr is not None else self.out(x.float(), edge_index.long())
        out = tgnn.pool.global_add_pool(out, ptr.long())
        out = self.net(out)
        return out

class GIN(nn.Module):
    def __init__(self, indim, hidden, outdim, n_layers):
        super().__init__()

    def forward(self, x, edge_index, edge_attr=None):
        pass

class GAT(nn.Module):
    def __init__(self, indim, hidden, outdim, n_layers):
        super().__init__()

    def forward(self, x, edge_index, edge_attr=None):
        pass

def build_gnn(layer_type):
    assert layer_type in ["GCN", "GIN", "GAT", "GPS"], "Model not found. Pick fron GCN, GIN, GAT, GPS."

    model_map = {
        "GCN": GCN,
        "GIN": GIN,
        "GAT": GAT
    }

    return model_map[layer_type]