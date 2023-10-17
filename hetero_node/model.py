import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor
from torch_geometric.nn import (Linear, GCNConv, SAGEConv, GATConv, GINConv, GATv2Conv,
    global_add_pool, global_mean_pool, global_max_pool)
from torch_geometric.utils import add_self_loops, negative_sampling, degree
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader

class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)
        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.spmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

def edgeidx2sparse(edge_index, num_nodes):
    return SparseTensor.from_edge_index(
    edge_index, sparse_sizes=(num_nodes, num_nodes)
    ).to(edge_index.device)

def creat_gnn_layer(name, first_channels, second_channels, heads):
    if name == "sage":
        layer = SAGEConv(first_channels, second_channels)
    elif name == "gcn":
        layer = GCNConv(first_channels, second_channels)
    elif name == "gin":
        layer = GINConv(Linear(first_channels, second_channels), train_eps=True)
    elif name == "gat":
        layer = GATConv(-1, second_channels, heads=heads)
    elif name == "gat2":
        layer = GATv2Conv(-1, second_channels, heads=heads)
    else:
        raise ValueError(name)
    return layer

def creat_activation_layer(activation):
    if activation is None:
        return nn.Identity()
    elif activation == "relu":
        return nn.ReLU()
    elif activation == "elu":
        return nn.ELU()
    elif activation == "prelu":
        return nn.PReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError("Unknown activation")

class GNN_Encoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, layer="gcn", num_layers=2, activation="relu"):
        super(GNN_Encoder, self).__init__()
        self.convs = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            first_channels = in_channels if i == 0 else hidden_channels
            second_channels = out_channels if i == num_layers - 1 else hidden_channels
            heads = 1 if i == num_layers - 1 or 'gat' not in layer else 4
            self.convs.append(creat_gnn_layer(layer, first_channels, second_channels, heads))
        self.activation = creat_activation_layer(activation)
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        edge_index = edgeidx2sparse(edge_index, x.size(0))
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.activation(x)
        x = self.convs[-1](x, edge_index)
        x = self.activation(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder, num_hidden, num_proj_hidden, tau= 0.5):
        super(Model, self).__init__()
        self.encoder = encoder
        self.tau = tau
        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)

    def forward(self, x, edge_index):
        return self.encoder(x, edge_index)

    def projection(self, z):
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))
        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1)  - refl_sim.diag() + between_sim.sum(1)))

    #------------- info_nce loss---------------
    def loss(self, z1, z2):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        return ret.mean()



def drop_feature(x, drop_prob):
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0
    return x
