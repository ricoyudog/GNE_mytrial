import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



#GCN  A*X*W as the ppt teaching, From Pygcn
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, use_bn=False):
        super(GraphConvolution, self).__init__()
        self.use_bn = use_bn
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))   # create W



    def forward(self, input, adj):
        support = torch.matmul(adj,input)  # A*X
        output = torch.matmul(support,self.weight)  # A*X*W

        if self.use_bn:
            self.bn = nn.BatchNorm1d(output.size(1)).to(device)
            output = self.bn(output)

        return output





class BatchedDiffPool(Module):
    def __init__(self, nfeat, nnext, nhid):
        super(BatchedDiffPool, self).__init__()
        self.embed = GraphConvolution(nfeat, nhid, use_bn=True)
        self.assign_mat = GraphConvolution(nfeat, nnext, use_bn=True)
        self.log = {}

    def forward(self, x, adj):
        z = self.embed(x, adj)   # GET Embedding_matrix, Last GCN(x,adj)
        s = F.softmax(self.assign_mat(x, adj), dim=-1)  # GET assign_matrix,softmax(GCN(x,adj))
        x_next = torch.matmul(s.transpose(-1, -2), z)  # Transpose(S) * Z
        adj_next = (s.transpose(-1, -2)).matmul(adj).matmul(s)  # Transpose(S) * Z * (S)
        return x_next, adj_next
