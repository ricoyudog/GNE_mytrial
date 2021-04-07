import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution, BatchedDiffPool


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.diff1 = BatchedDiffPool(256,256,128)
        self.gc2 = GraphConvolution(128, 128)
        self.diff2 = BatchedDiffPool(128,64,128)
        self.gc3 = GraphConvolution(128,128)
        self.diff3 = BatchedDiffPool(128,1,128)
        self.fc1 = nn.Linear(128,10)

        # self.dropout = dropout

    def forward(self, x, adj):
        x = self.gc1(x, adj)
        # x = F.dropout(x, self.dropout, training=self.training)
        x,adj = self.diff1(x,adj)
        x = self.gc2(x, adj)
        x,adj = self.diff2(x,adj)
        x = self.gc3(x,adj)
        x,adj = self.diff3(x,adj)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)

        return F.softmax(x, dim=1)


