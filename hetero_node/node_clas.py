import argparse
import math
import random
import os
import time
import numpy as np
from dataloader import load_data
from utils import *
from model import GCN
from prompt import *
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

def prompt_fusion(model,optimizer,con_emb,sem_emb,label,train_mask,val_mask,test_mask,label_num):
    criterion = torch.nn.NLLLoss()
    model.train()
    sem_con_emb = torch.cat((sem_emb,con_emb),dim = 1)
    emb = model(sem_con_emb)
    c_emb = get_centeremb(emb[train_mask],label[train_mask].unsqueeze(1),label_num)
    pred = torch.matmul(emb,c_emb.T)
    pred = F.log_softmax(pred,dim=1)
    loss = criterion(pred[train_mask], label[train_mask].squeeze())
    optimizer.zero_grad()
    loss.requires_grad_(True)
    loss.backward()
    optimizer.step()   
    train_acc = accuracy(pred[train_mask], label[train_mask])

    model.eval()
    val_acc = accuracy(pred[val_mask],label[val_mask])
    test_acc = accuracy(pred[test_mask],label[test_mask])
    return val_acc,test_acc
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--drop', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--hidden_size', type=int, default=512)
    parser.add_argument('--output_size', type=int, default=512)
    parser.add_argument('--feat_drop', type=float, default=0.2)
    parser.add_argument('--edge_drop', type=float, default=0.2)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--dataset', type=str, default='Wisconsin',help='Texas/Wisconsin/cornell/chameleon')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--runs', type=int, default=5)
    args = parser.parse_args()
    set_seed(args.seed)
    print(args)
    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    features, edges, num_classes, train_mask, val_mask, test_mask, labels, num_nodes, feat_dimension, data= load_data(args.dataset)
    edges, features = edges.to(device), features.to(device)
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(1)
    print(f"num nodes {num_nodes} | num classes {num_classes} | num node feats {feat_dimension}")

    model = Model(feat_dimension, args.hidden_size, args.output_size, args.tau, args.drop).to(device)
    save_name = args.dataset + '.pth'
    model.load_state_dict(torch.load(save_name))

    x1, x2 = features, feat_drop(features, p=args.feat_drop)
    edge_index = edges
    adj1 = torch.eye(num_nodes).to(device)
    adj = to_scipy_sparse_matrix(edge_index=edges, num_nodes = num_nodes)
    adj2 = sparse_mx_to_torch_sparse_tensor(adj).to(device)
    
    model.eval()
    labels = data.y.to(device)
    sem_emb = model.get_semantic_emb(x1, adj1, adj2)
    con_emb = model.get_contextual_emb(x1, adj1, adj2)
    sem_emb = F.normalize(sem_emb,p=2,dim=1)
    con_emb = F.normalize(con_emb,p=2,dim=1)
    accs = []
    for i in range(10):
        trainset,valset,testset = fewshot_split(nodenum=num_nodes,tasknum=10,trainshot=1,valshot=5,labelnum=5,labels=labels)
        count = i
        _trainset = torch.tensor(trainset[count])
        _valset = torch.tensor(valset[count])
        _testset = torch.tensor(testset[count])
        trainmask = index2mask(_trainset, num_nodes).to(device)
        valmask = index2mask(_valset, num_nodes).to(device)
        testmask = index2mask(_testset, num_nodes).to(device)

        # -------------prompt model initialize------------
        prompt_model = node_prompt_fusion(sem_emb.shape[1],args.alpha).to(device)
        prompt_model.reset_parameters()
        optimizer = torch.optim.AdamW(prompt_model.parameters(), lr=0.001, weight_decay=5e-5, amsgrad=True)
        optimizer.zero_grad()

        best_test_acc = -1
        best_val_acc = -1
        for epoch in range(1):
            val_acc,test_acc = prompt_fusion(prompt_model,optimizer,con_emb, sem_emb, labels,trainmask,valmask,testmask,5)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
        accs.append(best_test_acc)
    print(accs)
    print(np.mean(accs),np.std(accs))