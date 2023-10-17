import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from aug import TUDataset_aug as TUDataset
from torch_geometric.data import DataLoader
import sys
from torch import optim
import argparse
from model import Encoder
from evaluate_embedding import evaluate_embedding
from utils import *
import warnings
warnings.filterwarnings("ignore")

class Model(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers,tau):
        super(Model, self).__init__()
        self.embedding_dim = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.proj_head = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), 
        nn.ELU(inplace=True), nn.Linear(self.embedding_dim, self.embedding_dim))
        self.tau = tau
  
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
        T = self.tau
        f = lambda x: torch.exp(x / T)
        refl_sim = f(self.sim(x, x))
        between_sim = f(self.sim(x, x_aug))
        return -torch.log(between_sim.diag() / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag())).mean()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='PROTEINS', help='Dataset')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden-dim', type=int, default=16)
    parser.add_argument('--aug', type=str, default='none')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--feat_drop', type=float, default=0.2)
    parser.add_argument('--edge_drop', type=float, default=0.2)
    parser.add_argument('--aplha', type=float, default=0.5)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    args = parser.parse_args()
    set_seed(args.seed)

    path = osp.join('~/gch/public_data/pyg_data')
    dataset = TUDataset(path, name=args.dataset,use_node_attr=True, aug=args.aug).shuffle()
    dataset_eval = TUDataset(path, name=args.dataset,use_node_attr=True, aug='none').shuffle()
    if args.dataset == 'PROTEINS':
        n = 1
    elif args.dataset == 'ENZYMES':
        n = 18
    else:
        n = 3
    dataset.data.x = dataset.data.x[:,:n]
    dataset_eval.data.x = dataset_eval.data.x[:,:n]

    print(dataset_eval.data.x)
    dataset_num_features = dataset.get_num_feature()
    dataloader = DataLoader(dataset, batch_size=args.batch_size)
    dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(args.hidden_dim, args.num_layers, args.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    print('================')
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gin_layers: {}'.format(args.num_layers))
    print('================')

    model.eval()
    best_acc = 0
    for epoch in range(1, args.epochs+1):
        loss_all = 0
        model.train()
        for data in dataloader:
            data, data_aug = data
            optimizer.zero_grad()     
            node_num, _ = data.x.size()
            data = data.to(device)
            data_aug = data_aug.to(device)
            id_mat = make_identity_matrix(node_num).to(device)
            data_aug.x = feat_drop(data.x, p=args.feat_drop)
            data_aug.edge_index = edge_drop(data.edge_index, p=args.edge_drop)
            con_x1, con_x2, sem_x1, sem_x2 = model(data.x, data_aug.x, data.edge_index, data_aug.edge_index, id_mat, data.batch)
            loss_con = (model.loss_cal(con_x1, con_x2) + model.loss_cal(con_x2, con_x1))
            loss_sem = (model.loss_cal(sem_x1, sem_x2) + model.loss_cal(sem_x2, sem_x1))
            loss = loss_con + args.aplha * loss_sem
            print(loss)
            loss_all += loss.item() * data.num_graphs
            loss.backward()
            optimizer.step()
        print('Epoch {}, Loss {}'.format(epoch, loss_all / len(dataloader)))
        if epoch % args.log_interval == 0:
            model.eval()
            con_emb, sem_emb, y = model.encoder.get_embeddings(dataloader_eval)
            acc_val, acc = evaluate_embedding(con_emb, y)
            # if acc > best_acc:
            #     best_acc = acc 
            #     save_name = args.dataset + '.pth'
            #     torch.save(model.state_dict(), save_name)
    print(best_acc)
