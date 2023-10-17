import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import argparse
from utils import *
from aug import TUDataset_aug 
from torch_geometric.data import DataLoader
import sys
from torch import optim
from prompt import *
from model import Encoder
from model import *

class Model(nn.Module):

    def __init__(self, hidden_dim, num_gc_layers):
        super(Model, self).__init__()
        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), 
        nn.ELU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
  
    def forward(self, x, x_aug, edge_index, aug_edge_index, id_mat, batch):
        con_x1,con_x2,sem_x1,sem_x2 = self.encoder(x, x_aug, edge_index, aug_edge_index, id_mat, batch)
        con_x1 = self.proj_head(con_x1)
        con_x2 = self.proj_head(con_x2)
        sem_x1 = self.proj_head(sem_x1)
        sem_x2 = self.proj_head(sem_x2)
        return con_x1,con_x2,sem_x1,sem_x2
  
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def loss_cal(self, x, x_aug):
        T = 0.5
        f = lambda x: torch.exp(x / T)
        refl_sim = f(self.sim(x, x))
        between_sim = f(self.sim(x, x_aug))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())).mean()

def prompt_fusion(model,optimizer,con_emb,sem_emb,label,train_mask,val_mask,test_mask,label_num):
    criterion = torch.nn.NLLLoss()
    model.train()
    con_sem_emb = torch.cat((con_emb,sem_emb),dim=1)
    emb = model(con_sem_emb)
    c_emb = get_centeremb(emb[train_mask],label[train_mask].unsqueeze(1),label_num)
    pred = torch.matmul(emb,c_emb.T)
    pred = F.log_softmax(pred,dim=1)
    loss = criterion(pred[train_mask], label[train_mask].squeeze())
    optimizer.zero_grad()
    loss.requires_grad_(True)
    loss.backward(retain_graph=True)
    optimizer.step()   
    train_acc = accuracy(pred[train_mask], label[train_mask])
    model.eval()
    val_acc = accuracy(pred[val_mask],label[val_mask])
    test_acc = accuracy(pred[test_mask],label[test_mask])
    print(val_acc,test_acc)
    return val_acc,test_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='ENZYMES', help='Dataset')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=16)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--aplha', type=float, default=1.5)
    parser.add_argument('--act', action='store_true', default=False)
    parser.add_argument('--l2', action='store_true', default=False)
    args = parser.parse_args()
    set_seed(args.seed)
    path = osp.join('~/gch/public_data/pyg_data')
    dataset_eval = TUDataset_aug(path, name=args.dataset,use_node_attr=True, aug='none').shuffle()
    if args.dataset == 'PROTEINS':
        n = 1
    elif args.dataset == 'ENZYMES':
        n = 18
    else:
        n = 3
    print(dataset_eval.data)
    dataset_eval.data.x = dataset_eval.data.x[:,:n]
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size)
    dataset_num_features = dataset_eval.get_num_feature()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(args.hidden_dim, args.num_layers).to(device)

    save_name = args.dataset + '.pth'
    print(save_name)
    model.load_state_dict(torch.load(save_name))
    
    model.eval()
    con_emb, sem_emb, y = model.encoder.get_embeddings(dataloader_eval)
    num_nodes = con_emb.shape[0]
    con_emb = torch.tensor(con_emb).to(device)
    sem_emb = torch.tensor(sem_emb).to(device)
    if args.act:
        con_emb = torch.sigmoid(con_emb)
        sem_emb = torch.sigmoid(sem_emb)
    if args.l2:
        con_emb = F.normalize(con_emb,p=2,dim=1)
        sem_emb = F.normalize(sem_emb,p=2,dim=1)
    y = torch.tensor(y).to(device)
    acc = []
    trainset,valset,testset = fewshot_split(nodenum=num_nodes,tasknum=100,trainshot=5,valshot=5,labelnum=y.max()+1,labels=y)
    for i in range(args.epochs):
        count = i
        _trainset = torch.tensor(trainset[count])
        _valset = torch.tensor(valset[count])
        _testset = torch.tensor(testset[count])
        train_mask = index2mask(_trainset, num_nodes)
        val_mask = index2mask(_valset, num_nodes)
        test_mask = index2mask(_testset, num_nodes)

        prompt_model = node_prompt_fusion(con_emb.shape[1],lamb = args.aplha).to(device)
        prompt_model.reset_parameters()
        optimizer = torch.optim.Adam(prompt_model.parameters(), lr=0.005, weight_decay=3e-4)
        optimizer.zero_grad()
        best_test_acc = -1
        best_val_acc = -1
        for epoch in range(200):
            val_acc,test_acc = prompt_fusion(prompt_model,optimizer,con_emb, sem_emb, y,train_mask,val_mask,test_mask,y.max()+1)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
        acc.append(best_test_acc)
    print(acc)
    print(np.mean(acc),np.std(acc)) 
