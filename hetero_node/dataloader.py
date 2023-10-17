import argparse
import os.path as osp
import random
from torch.utils.data import DataLoader
import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Flickr, WebKB,TUDataset
from torch_geometric.utils import remove_self_loops

def load_data(dataset):
    path = osp.join('~/gch/public_data/pyg_data')
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(path, dataset)
    elif dataset in ['chameleon']:
        dataset = WikipediaNetwork(path, dataset)
    elif dataset in ['cornell', 'texas', 'wisconsin']:
        dataset = WebKB(path, dataset)
    elif dataset in ['dblp']:
        dataset = CitationFull(path, dataset, transform=T.NormalizeFeatures())
    elif dataset in ['ENZYMES']:
        dataset = TUDataset(root=path,name='ENZYMES',use_node_attr='True')
        data = dataset[0]
        x = dataset.data.x[:,:-3]
        dataset = TUDataset(root=path,name='ENZYMES')
        labels = dataset.data.x
        labels = torch.argmax(labels, dim=1)
        edges = remove_self_loops(dataset.data.edge_index)[0]
        nnodes = dataset.data.num_nodes
        nfeats = x.shape[1]
        nclasses = torch.max(labels).item() + 1
        train_mask,val_mask, test_mask = [],[],[]
        return x, edges, nclasses, train_mask, val_mask, test_mask, labels, nnodes, nfeats, data
    
    data = dataset[0]
    edges = remove_self_loops(data.edge_index)[0]
    features = data.x
    [nnodes, nfeats] = features.shape
    nclasses = torch.max(data.y).item() + 1
    labels = data.y
    train_mask, val_mask, test_mask = data.train_mask, data.val_mask, data.test_mask
    if len(train_mask.shape) < 2:
        train_mask = train_mask.unsqueeze(1)
        val_mask = val_mask.unsqueeze(1)
        test_mask = test_mask.unsqueeze(1)
    
    return features, edges, nclasses, train_mask, val_mask, test_mask, labels, nnodes, nfeats, data
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ENZYMES',help='Cora/CiteSeer/PubMed')
    args = parser.parse_args()
    features, edges, nclasses, train_mask, val_mask, test_mask, labels, nnodes, nfeats, data = load_data(args.dataset)