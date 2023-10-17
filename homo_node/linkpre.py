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
from torch_geometric.datasets import CitationFull
from torch_geometric.utils import dropout_adj, to_scipy_sparse_matrix,contains_self_loops,add_self_loops
from utils import *
from model import Model, drop_feature, GNN_Encoder
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.utils import add_self_loops, negative_sampling
from prompt import *

def ce_loss(pos_out, neg_out):
    pos_loss = F.binary_cross_entropy(pos_out.sigmoid(), torch.ones_like(pos_out))
    neg_loss = F.binary_cross_entropy(neg_out.sigmoid(), torch.zeros_like(neg_out))
    return pos_loss + neg_loss

def edge_decoder(z, edge, sigmoid=True):
    x = z[edge[0]] * z[edge[1]]
    x = x.sum(-1)
    if sigmoid:
        return x.sigmoid()
    else:
        return x

def batch_predict(z, edges, batch_size=2 ** 16):
    preds = []
    for perm in DataLoader(range(edges.size(1)), batch_size):
        edge = edges[:, perm]
        preds += [edge_decoder(z, edge).squeeze().cpu()]
    pred = torch.cat(preds, dim=0)
    return pred

def test(z, pos_edge_index, neg_edge_index, batch_size=2**16):
    pos_pred = batch_predict(z, pos_edge_index)
    neg_pred = batch_predict(z, neg_edge_index)
    pred = torch.cat([pos_pred, neg_pred], dim=0)
    pos_y = pos_pred.new_ones(pos_pred.size(0))
    neg_y = neg_pred.new_zeros(neg_pred.size(0))
    y = torch.cat([pos_y, neg_y], dim=0)
    y, pred = y.detach().numpy(), pred.detach().numpy()
    return roc_auc_score(y, pred), average_precision_score(y, pred)

def random_negative_sampler(edge_index, num_nodes, num_neg_samples):
    neg_edges = torch.randint(0, num_nodes, size=(2, num_neg_samples)).to(edge_index)
    return neg_edges

def train_epoch(z, data, model, optimizer, batch_size=2 ** 16):
    edge_index = data.edge_index
    remaining_edges = edge_index
    masked_edges = getattr(data, "pos_edge_label_index", edge_index)
    aug_edge_index, _ = add_self_loops(edge_index)
    neg_edges = negative_sampling(aug_edge_index,num_nodes=data.num_nodes,num_neg_samples=masked_edges.view(2, -1).size(1),
    ).view_as(masked_edges)
    for perm in DataLoader(range(masked_edges.size(1)), batch_size=batch_size, shuffle=True):
        optimizer.zero_grad()
        out = model(z)
        batch_masked_edges = masked_edges[:, perm]
        batch_neg_edges = neg_edges[:, perm]
        pos_out = edge_decoder(out, batch_masked_edges, sigmoid=False)
        neg_out = edge_decoder(out, batch_neg_edges, sigmoid=False)
        loss = ce_loss(pos_out, neg_out)
        loss.backward(retain_graph=True)
        optimizer.step()
    valid_auc, valid_ap = test(out, splits['valid'].pos_edge_label_index, splits['valid'].neg_edge_label_index, batch_size=10000)
    test_auc, test_ap = test(out, splits['test'].pos_edge_label_index, splits['test'].neg_edge_label_index, batch_size=10000)
    results = {'AUC': (valid_auc, test_auc), 'AP': (valid_ap, test_ap)}
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='PubMed',help='Cora/CiteSeer/PubMed/dblp')
    parser.add_argument('--gpu_id', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--aplha', type=float, default=0.1)
    parser.add_argument('--pretrain_lr', type=float, default=0.01)
    parser.add_argument('--pretrain_wd', type=float, default=1e-5)
    parser.add_argument('--num_proj_hidden', type=int, default=128)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.1)
    parser.add_argument('--drop_fea_rate_1', type=float, default=0.4)
    parser.add_argument('--drop_fea_rate_2', type=float, default=0.)
    parser.add_argument('--activation', type=str, default='relu',help='prelu/relu/elu/tanh/sigmoid')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.5)
    parser.add_argument('--delta', type=int, default=2,help='num of hops')
    parser.add_argument("--layer", nargs="?", default="gcn", help="gcn/gat/sage")
    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers of encoder. (default: 1)')
    parser.add_argument('--hidden_channels', type=int, default=512, help='Channels of GNN encoder. (default: 128)')
    parser.add_argument('--out_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
    args = parser.parse_args()
    print(args)
    torch.cuda.set_device(args.gpu_id)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features, edges, num_classes, train_mask, val_mask, test_mask, labels, num_nodes, num_features, data = load_data(args.dataset)
    train_data, val_data, test_data = T.RandomLinkSplit(num_val=0.05, num_test=0.1,
                                                    is_undirected=True,
                                                    split_labels=True,
                                                    add_negative_train_samples=True)(data)
    splits = dict(train=train_data, valid=val_data, test=test_data)
    features = splits['train'].x.to(device)
    edges = splits['train'].edge_index.to(device)
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

    # save_name = args.dataset + 'lp.pth'
    save_name = args.dataset + 'classification.pth'
    encoder = GNN_Encoder(num_features, args.hidden_channels, 
    args.out_channels, layer=args.layer, num_layers=args.encoder_layers, activation=args.activation)
    model = Model(encoder, args.out_channels, args.num_proj_hidden, args.tau).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.pretrain_lr, weight_decay=args.pretrain_wd)
    id_mat = make_identity_matrix(num_nodes).to(device)
    model.load_state_dict(torch.load(save_name))
    model.eval()
    sem_emb = model.encoder(features, id_mat)
    con_emb = model.encoder(features, edges)
    z = torch.cat((con_emb,sem_emb),dim=1)  
    best_micro = 0    
    prompt = prompt_linear(z.shape[1],z.shape[1]).to(device)
    prompt_optimizer = torch.optim.Adam(prompt.parameters(),lr=0.05,weight_decay=5e-3)
    prompt.train()
    best = 0
    best_reult = None
    for i in range(200):
        results = train_epoch(z,splits['train'], prompt, prompt_optimizer, batch_size=10000)
        if results['AUC'][0]>best:
            best_reult = results
            best = results['AUC'][0]
    print(best_reult)
    
    

    
    
    
