import argparse
import math
import random
import os
import time
import numpy as np
from dataloader import load_data
from utils import *
from model import GCN
import torch
import scipy.sparse as sp
import torch.nn as nn
from torch.nn.parameter import Parameter
import networkx as nx
from scipy.sparse import csr_matrix
from torch_geometric.utils.convert import to_scipy_sparse_matrix
import csv
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import f1_score
import torch_geometric.transforms as T

class Model(nn.Module):
    def __init__(self, input, hidden, output, tau, drop):
        super(Model, self).__init__()
        self.tau = tau
        self.drop = drop
        self.fcs  = nn.ModuleList()
        self.fcs.append(GCN(input, 2 * hidden))
        self.fcs.append(GCN(2 * hidden, hidden))
        self.fcs.append(torch.nn.Linear(hidden, output))
        self.fcs.append(torch.nn.BatchNorm1d(output))
        self.fcs.append(torch.nn.Linear(output, hidden))
        self.params = list(self.fcs.parameters())
        self.activation = F.relu

    def forward(self, x1, x2, adj1, adj2, drop_edge_index):
        x1 = F.dropout(x1, p=self.drop, training=self.training)
        x2 = F.dropout(x2, p=self.drop, training=self.training)
        h1 = self.activation(self.fcs[0](x1, adj1))
        h1 = self.fcs[1](h1, adj1)
        h2 = self.activation(self.fcs[0](x2, adj1))
        h2 = self.fcs[1](h2, adj1)
        h3 = self.activation(self.fcs[0](x1, adj2))
        h3 = self.fcs[1](h3, adj2)
        h4 = self.activation(self.fcs[0](x1, drop_edge_index))
        h4 = self.fcs[1](h4, drop_edge_index)
        return h1, h2, h3, h4

    def projection(self, z):
        z = F.elu(self.fcs[3](self.fcs[2](z)))
        return self.fcs[4](z)

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
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def loss(self, z1, z2, mean=True):
        h1 = self.projection(z1)
        h2 = self.projection(z2)
        l1 = self.semi_loss(h1, h2)
        l2 = self.semi_loss(h2, h1)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
   
    @torch.no_grad()
    def get_semantic_emb(self, x1, adj1, adj2):
        h1 = self.activation(self.fcs[0](x1, adj1))
        h1 = self.fcs[1](h1, adj1)
        return h1 

    @torch.no_grad()
    def get_contextual_emb(self, x1, adj1, adj2):
        h3 = self.activation(self.fcs[0](x1, adj2))
        h3 = self.fcs[1](h3, adj2)
        return h3

def train(model, optimizer, x1, x2, adj1, adj2, drop_edge_index, args):
    model.train()
    optimizer.zero_grad()
    h1, h2, h3, h4 = model(x1, x2, adj1, adj2, drop_edge_index)
    loss1 = model.loss(h1, h2)   # semantic view
    loss2 = model.loss(h3, h4)   # contextual view
    loss = loss1 + args.alpha * loss2
    print(f"loss: {loss.item()}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def test(model, labels, epoch, x1, adj1, adj2, idx_train, idx_val, idx_test):
    model.eval()
    with torch.no_grad():
        representations = model.get_semantic_emb(x1, adj1, adj2)
        # representations = model.get_contextual_emb(x1, adj1, adj2)
        labels = labels.squeeze(1)
    result = LRE(representations, labels, idx_train, idx_val, idx_test)
    print(f"micro_f1: {result['micro_f1']}")
    print(f"macro_f1: {result['macro_f1']}")
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--delta', type=int, default=0,help='num of hops')
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--output_size', type=int, default=512)
    parser.add_argument('--feat_drop', type=float, default=0.2)
    parser.add_argument('--edge_drop', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='Wisconsin',help='texas/Wisconsin/cornell/chameleon')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--cur_split', type=int, default=1)
    args = parser.parse_args()
    set_seed(args.seed)

    print(args)
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    features, edges, num_classes, train_mask, val_mask, test_mask, labels, num_nodes, feat_dimension, data= load_data(args.dataset)
    edges, features = edges.to(device), features.to(device)

    adj = to_scipy_sparse_matrix(edges)
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized).to(device)
    for i in range(0,args.delta):
        print('-----in------')
        features = torch.matmul(adj_normalized,features)
        features = torch.matmul(adj_normalized,features)
    
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)
    print(f"num nodes {num_nodes} | num classes {num_classes} | num node feats {feat_dimension}")

    best_results = []
    for run in range(args.runs):
        cur_split = 0 if (train_mask.shape[1]==1) else (run % train_mask.shape[1])
        cur_split = args.cur_split
        idx_train = train_mask[:, cur_split]
        idx_val = val_mask[:, cur_split]
        idx_test = test_mask[:, cur_split]
        idx_train = np.where(idx_train == 1)[0]
        idx_val = np.where(idx_val == 1)[0]
        idx_test = np.where(idx_test == 1)[0]
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        model = Model(feat_dimension, args.hidden_size, args.output_size, args.tau, args.drop).to(device)
        print(model)

        optimizer = torch.optim.Adam([{'params':model.params, 'lr':args.lr, 'weight_decay':args.weight_decay}])
        all_results = []
        best_micro = 0
        best_macro = 0
        best_epoch = 0
        for epoch in range(args.epochs):
            x1, x2 = features, feat_drop(features, p=args.feat_drop)
            edge_index = edges
            adj1 = torch.eye(num_nodes).to(device)
            adj = to_scipy_sparse_matrix(edge_index=edges, num_nodes = num_nodes)
            adj2 = sparse_mx_to_torch_sparse_tensor(adj).to(device)
            drop_edge_index = edge_drop(edge_index, p=args.edge_drop)
            drop_edge_index = to_scipy_sparse_matrix(edge_index=drop_edge_index, num_nodes = num_nodes)
            drop_edge_index = sparse_mx_to_torch_sparse_tensor(drop_edge_index).to(device)

            loss= train(model, optimizer, x1, x2, adj1, adj2, drop_edge_index, args)
            print(f"run: {run}, epoch: {epoch}")
            result = test(model, labels, epoch, x1, adj1, adj2, idx_train, idx_val, idx_test)

            if result['micro_f1'] > best_micro:
                best_micro = result['micro_f1']
                best_macro = result['macro_f1']
                best_epoch = epoch
                #-------------save model-------------
                # save_name = args.dataset + '.pth'
                # torch.save(model.state_dict(), save_name)

        print('run:', run)
        print('best_epoch:', best_epoch)
        print('best_micro:', best_micro)
        
