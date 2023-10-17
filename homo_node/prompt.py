import torch
import torch.nn as nn
import torch.nn.functional as F

class prompt_dot(nn.Module):
    def __init__(self,input_dim):
        super(prompt_dot, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))

    def forward(self,embedding):
        embedding = embedding * self.weight
        return embedding

class prompt_linear(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(prompt_linear, self).__init__()
        self.linear=torch.nn.Linear(input_dim,output_dim)

    def forward(self,embedding):
        embedding =  self.linear(embedding) 
        return embedding

class node_prompt_dot(nn.Module):
    def __init__(self, input_dim):
        super(node_prompt_dot, self).__init__()
        self.weight= torch.nn.Parameter(torch.Tensor(1,input_dim))

    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.ones_(self.weight)
    
    def forward(self,embedding):
        embedding = embedding * self.weight
        return embedding

class node_prompt_fusion(nn.Module):
    def __init__(self, input_dim, lamb = 1.):
        super(node_prompt_fusion, self).__init__()
        self.con_weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.sem_weight= torch.nn.Parameter(torch.Tensor(1,input_dim))
        self.lamb = lamb 

    def reset_parameters(self):
        torch.nn.init.ones_(self.con_weight)
        torch.nn.init.ones_(self.sem_weight)
    
    def forward(self,embedding):
        weight = torch.cat((self.con_weight,self.sem_weight * self.lamb),dim=1)
        embedding = embedding * weight
        return embedding

