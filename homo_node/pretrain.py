import argparse
import os.path as osp
import random
import yaml
from yaml import SafeLoader
from torch.utils.data import DataLoader
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from dataloader import load_data
from torch_geometric.utils import dropout_adj, to_scipy_sparse_matrix,contains_self_loops,add_self_loops
from utils import *
from model import Model, drop_feature, GNN_Encoder

def test(model, labels, epoch, data):
    best_micro = 0 
    for i in range(1):
        model.eval()
        with torch.no_grad():
            emb = model.encoder(data.x, data.edge_index)
        result = LRE(emb, labels, data.train_mask, data.val_mask, data.test_mask)
        if result['micro_f1'] > best_micro:
            best_micro = result['micro_f1']
        print('run:',i+1)
        print(f"micro_f1: {result['micro_f1']}")
        print(f"macro_f1: {result['macro_f1']}")
    return best_micro

def train(model, x, edge_index, identity, aplha):
    model.train()
    optimizer.zero_grad()
    # contextual view perturbation
    adj_1 = dropout_adj(edge_index, p=args.drop_edge_rate_1)[0]
    adj_2 = dropout_adj(edge_index, p=args.drop_edge_rate_2)[0]
    # semantic view peturbation
    x_1 = drop_feature(x, args.drop_fea_rate_1)
    x_2 = drop_feature(x, 0.)
    z1 = model(x, adj_1)
    z2 = model(x, adj_2)
    z3 = model(x_1, identity)
    z4 = model(x_2, identity)
    context_loss = model.loss(z1, z2)
    sem_loss = model.loss(z3,z4)
    loss = context_loss +  aplha * sem_loss
    loss = sem_loss
    loss.backward()
    optimizer.step()
    return loss.item()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora',help='Cora/CiteSeer/PubMed/dblp')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--aplha', type=float, default=1.)
    parser.add_argument('--pretrain_lr', type=float, default=0.001)
    parser.add_argument('--pretrain_wd', type=float, default=1e-5)
    parser.add_argument('--num_proj_hidden', type=int, default=128)
    parser.add_argument('--delta', type=int, default=2,help='num of hops')
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.1)
    parser.add_argument('--drop_fea_rate_1', type=float, default=0.4)
    parser.add_argument('--drop_fea_rate_2', type=float, default=0.)
    parser.add_argument('--activation', type=str, default='relu',help='prelu/relu/elu/tanh/sigmoid')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--tau', type=float, default=3)
    parser.add_argument("--layer", nargs="?", default="gcn", help="gcn/gat/sage")
    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers of encoder. (default: 1)')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
    parser.add_argument('--out_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(args.gpu_id)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features, edges, num_classes, train_mask, val_mask, test_mask, labels, num_nodes, num_features, data = load_data(args.dataset)
    edges, features = edges.to(device), features.to(device)
    
    adj = to_scipy_sparse_matrix(edges)
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized).to(device)
    for i in range(0,args.delta):
        features = torch.matmul(adj_normalized,features)
        features = torch.matmul(adj_normalized,features)

    data.to(device)
    encoder = GNN_Encoder(num_features, args.hidden_channels, 
    args.out_channels, layer=args.layer, num_layers=args.encoder_layers, activation=args.activation)
    model = Model(encoder, args.out_channels, args.num_proj_hidden, args.tau).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
    id_mat = make_identity_matrix(num_nodes).to(device)  
    best_micro = 0

    for epoch in range(1, args.epochs + 1):
        loss = train(model, features, edges, id_mat, args.aplha)
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f},')
        if epoch % 50 == 0:
            result = test(model, labels, epoch, data)
            if result > best_micro:
                best_micro = result
                save_name = args.dataset + 'classification.pth'
                torch.save(model.state_dict(), save_name)
    print(best_micro)

            


