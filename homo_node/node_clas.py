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
from prompt import *
from model import Model, drop_feature, GNN_Encoder

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
    return val_acc,test_acc

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
    parser.add_argument('--dataset', type=str, default='Cora',help='Cora/CiteSeer/PubMed')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--aplha', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=1e-3)
    parser.add_argument('--delta', type=int, default=2,help='num of hops')
    parser.add_argument('--num_proj_hidden', type=int, default=256)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.4)
    parser.add_argument('--drop_fea_rate_1', type=float, default=0.4)
    parser.add_argument('--drop_fea_rate_2', type=float, default=0.)
    parser.add_argument('--activation', type=str, default='relu',help='prelu/relu/elu/tanh/sigmoid')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--tau', type=float, default=2)
    parser.add_argument("--layer", nargs="?", default="gcn", help="gcn/gat/sage")
    parser.add_argument('--encoder_layers', type=int, default=2, help='Number of layers of encoder. (default: 1)')
    parser.add_argument('--hidden_channels', type=int, default=256, help='Channels of GNN encoder. (default: 128)')
    parser.add_argument('--out_channels', type=int, default=128, help='Channels of GNN encoder. (default: 128)')
    parser.add_argument('--task_num', type=int, default=10, help='Channels of GNN encoder. (default: 128)')
    args = parser.parse_args()
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
    id_mat = make_identity_matrix(num_nodes).to(device)  
    save_name = args.dataset + 'classification.pth'
    model.load_state_dict(torch.load(save_name))

    model.eval()
    labels = data.y.to(device)
    id_mat = make_identity_matrix(num_nodes).to(device)  
    sem_emb = model.encoder(data.x, id_mat)
    sem_emb = F.normalize(sem_emb,p=2,dim=1)
    con_emb = model.encoder(data.x, data.edge_index)
    con_emb = F.normalize(con_emb,p=2,dim=1)
    accs = []
    np_accs = []
    trainset,valset,testset = fewshot_split(nodenum=num_nodes,tasknum=args.task_num,trainshot=1,valshot=5,labelnum=num_classes,labels=labels)
    for i in range(args.task_num):
        count = i
        _trainset = torch.tensor(trainset[count])
        _valset = torch.tensor(valset[count])
        _testset = torch.tensor(testset[count])
        trainmask = index2mask(_trainset, num_nodes).to(device)
        valmask = index2mask(_valset, num_nodes).to(device)
        testmask = index2mask(_testset, num_nodes).to(device)
        # -------------prompt model initialize------------
        prompt_model = node_prompt_fusion(con_emb.shape[1],lamb = args.aplha).to(device)
        prompt_model.reset_parameters()
        optimizer = torch.optim.AdamW(prompt_model.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
        optimizer.zero_grad()
        best_test_acc = -1
        best_val_acc = -1
        for epoch in range(100):
            prompt_model.eval()
            val_acc,test_acc = prompt_fusion(prompt_model,optimizer,con_emb, sem_emb, labels,trainmask,valmask,testmask,data.y.max()+1)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
        accs.append(best_test_acc)
    print(accs)
    print(np.mean(accs),np.std(accs))