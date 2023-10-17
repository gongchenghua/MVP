import sys
import os
import torch
import random
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score
import copy

def get_centeremb(input,index,label_num=0):
    device=input.device
    mean = torch.ones(index.size(0), index.size(1)).to(device)
    index=torch.tensor(index,dtype=int).to(device)
    label_num = torch.max(index) + 1
    _mean = torch.zeros(label_num, 1,device=device).scatter_add_(dim=0, index=index, src=mean)
    preventnan=torch.ones(_mean.size(),device=device)*0.0000001
    _mean=_mean + preventnan
    index = index.expand(input.size())
    c = torch.zeros(label_num, input.size(1)).to(device)
    c = c.scatter_add_(dim=0, index=index, src=input)
    c = c / _mean
    return c

def mask_select_emb(emb,mask,device):
    index=[]
    count=0
    for i in mask:
        if i==1:
            index.append(count)
        count+=1
    index=torch.tensor(index,device=device)
    ret=torch.index_select(emb,0,index)
    return ret
    
def accuracy(y_pred, y_true):
    y_true = y_true.squeeze().long()
    preds = y_pred.max(1)[1].type_as(y_true)
    correct = preds.eq(y_true).double()
    correct = correct.sum().item()
    return correct / len(y_true)

def index2mask(index,nodenum):
    ret=torch.zeros(nodenum)
    ones=torch.ones_like(ret)
    ret=torch.scatter_add(input=ret,dim=0,index=index,src=ones)
    return ret.type(torch.bool)


def fewshot_split(nodenum,tasknum,trainshot,valshot,labelnum,labels):
    train=[]
    val=[]
    test=[]
    for count in range(tasknum):
        index = random.sample(range(0, nodenum), nodenum)
        trainindex=[]
        valindex=[]
        testindex=[]
        traincount = torch.zeros(labelnum)
        valcount = torch.zeros(labelnum)
        for i in index:
            label=labels[i]
            if traincount[label]<trainshot:
                trainindex.append(i)
                traincount[label]+=1
            elif valcount[label]<valshot:
                valcount[label]+=1
                valindex.append(i)
            else:
                testindex.append(i)
        train.append(trainindex)
        val.append(valindex)
        test.append(testindex)
    return train,val,test


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False

def make_identity_matrix(num_nodes):
    identity_matrix = torch.eye(num_nodes)
    identity_matrix  = sp.coo_matrix(identity_matrix)
    values = identity_matrix.data  
    indices = np.vstack((identity_matrix.row, identity_matrix.col)) 
    identity_matrix = torch.LongTensor(indices)
    return  identity_matrix

class LogisticRegression(nn.Module):
    def __init__(self, num_features, num_classes):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        torch.nn.init.xavier_uniform_(self.fc.weight.data)

    def forward(self, x):
        z = self.fc(x)
        return z


def LRE(x, y, idx_train, idx_val, idx_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = x.detach().to(device)
    input_dim = x.size()[1]
    y = y.detach().to(device)
    num_classes = y.max().item() + 1
    classifier = LogisticRegression(input_dim, num_classes).to(device)
    optimizer = Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    output_fn = nn.LogSoftmax(dim=-1)
    criterion = nn.NLLLoss()

    best_val_micro = -1
    best_test_micro = -1
    best_test_macro = -1
    best_epoch = -1
    num_epochs = 300


    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()
        output = classifier(x[idx_train])
        loss = criterion(output_fn(output), y[idx_train])
        loss.backward()
        optimizer.step()
        
        classifier.eval()
        y_test = y[idx_test].detach().cpu().numpy()
        y_pred_test = classifier(x[idx_test]).argmax(-1).detach().cpu().numpy()
        test_micro = f1_score(y_test, y_pred_test, average='micro')
        test_macro = f1_score(y_test, y_pred_test, average='macro')
        y_val = y[idx_val].detach().cpu().numpy()
        y_pred_val = classifier(x[idx_val]).argmax(-1).detach().cpu().numpy()
        val_micro = f1_score(y_val, y_pred_val, average='micro')
        if val_micro > best_val_micro:
            best_val_micro = val_micro
            best_test_micro = test_micro
            best_test_macro = test_macro
            best_epoch = epoch
    print('best_test_epoch:', best_epoch)
    return {'micro_f1': best_test_micro, 'macro_f1': best_test_macro}


def edge_drop(edge_index, p=0.4):
    # copy edge_index
    edge_index = copy.deepcopy(edge_index)
    num_edges = edge_index.size(1)
    num_droped = int(num_edges*p)
    perm = torch.randperm(num_edges)
    edge_index = edge_index[:, perm[:num_edges-num_droped]]
    return edge_index

def feat_drop(x, p=0.2):
    # copy x
    x = copy.deepcopy(x)
    mask = torch.empty((x.size(1),), dtype=torch.float32, device=x.device).uniform_(0, 1) < p
    x[:, mask] = 0
    return x