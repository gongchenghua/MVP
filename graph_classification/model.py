import os.path as osp
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool, global_mean_pool, global_max_pool
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from utils import *
from sklearn.metrics import accuracy_score
import sys

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        for i in range(num_gc_layers):
            if i:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
            else:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
            conv = GINConv(nn)
            bn = torch.nn.BatchNorm1d(dim)
            self.convs.append(conv)
            self.bns.append(bn)

    def forward(self, x, aug_x, edge_index, aug_edge_index, id_mat, batch):
        con_x1s = []
        con_x1 = x
        for i in range(self.num_gc_layers):
            con_x1 = F.relu(self.convs[i](con_x1, edge_index))
            con_x1 = self.bns[i](con_x1)
            con_x1s.append(con_x1)
        con_x1pool = [global_add_pool(con_x1, batch) for con_x1 in con_x1s]
        con_x1 = torch.cat(con_x1pool, 1)

        con_x2s = []
        con_x2 = x
        for i in range(self.num_gc_layers):
            con_x2 = F.relu(self.convs[i](con_x2, aug_edge_index))
            con_x2 = self.bns[i](con_x2)
            con_x2s.append(con_x2)
        con_x2pool = [global_add_pool(con_x2, batch) for con_x2 in con_x2s]
        con_x2 = torch.cat(con_x2pool, 1)

        sem_x1s = []
        sem_x1 = x
        for i in range(self.num_gc_layers):
            sem_x1 = F.relu(self.convs[i](sem_x1, id_mat))
            sem_x1 = self.bns[i](sem_x1)
            sem_x1s.append(sem_x1)
        sem_x1pool = [global_add_pool(sem_x1, batch) for sem_x1 in sem_x1s]
        sem_x1 = torch.cat(sem_x1pool, 1)

        sem_x2s = []
        sem_x2 = aug_x
        for i in range(self.num_gc_layers):
            sem_x2 = F.relu(self.convs[i](sem_x2, id_mat))
            sem_x2 = self.bns[i](sem_x2)
            sem_x2s.append(sem_x2)
        sem_x2pool = [global_add_pool(sem_x2, batch) for sem_x2 in sem_x2s]
        sem_x2 = torch.cat(sem_x2pool, 1)

        return con_x1, con_x2, sem_x1, sem_x2

    def get_embeddings(self, loader):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        con_ret = []
        sem_ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data = data[0]
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                node_num, _ = data.x.size()
                id_mat = make_identity_matrix(node_num).to(device)
                con_x1, con_x2, sem_x1, sem_x2 = self.forward(x, x, edge_index, edge_index, id_mat, batch)
                con_ret.append(con_x1.cpu().numpy())
                sem_ret.append(sem_x1.cpu().numpy())
                y.append(data.y.cpu().numpy())
        con_ret = np.concatenate(con_ret, 0)
        sem_ret = np.concatenate(sem_ret, 0)
        y = np.concatenate(y, 0)
        return  con_ret, sem_ret, y

