import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, use_bn=False):
        super(GraphConvolution, self).__init__()
        self.use_bn = use_bn
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = bias


    def forward(self, input, adj):
        support = torch.matmul(adj,input)
        output = torch.matmul(support,self.weight)

        if self.use_bn:
            self.bn = nn.BatchNorm1d(output.size(1)).to(device)
            output = self.bn(output)

        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class BatchedDiffPool(Module):
    def __init__(self, nfeat, nnext, nhid, is_final=False):
        super(BatchedDiffPool, self).__init__()
        self.is_final = is_final
        self.embed = GraphConvolution(nfeat, nhid, use_bn=True)
        self.assign_mat = GraphConvolution(nfeat, nnext, use_bn=True)
        self.log = {}
        self.link_pred_loss = 0
        self.entropy_loss = 0

    def forward(self, x, adj, mask=None, log=False):
        z_l = self.embed(x, adj)
        s_l = F.softmax(self.assign_mat(x, adj), dim=-1)
        if log:
            self.log['s'] = s_l.cpu().numpy()
        xnext = torch.matmul(s_l.transpose(-1, -2), z_l)
        adjnext = (s_l.transpose(-1, -2)).matmul(adj).matmul(s_l)
        return xnext, adjnext
