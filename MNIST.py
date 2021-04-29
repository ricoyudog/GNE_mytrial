from __future__ import absolute_import
import numpy as np
import dgl
import os

import torch
from dgl.data.utils import download, extract_archive, get_download_dir, _get_dgl_url

from torch.utils.data import Dataset

import warnings
import torch.utils.data as data
from PIL import Image
import os
import os.path
import gzip
import numpy as np
import torch
import codecs
from torchvision import transforms

import networkx as nx

def adj_head(m):

    M = m**2

    adj_matrix = np.zeros((M,M),dtype=int)
    for i in range(m):
        for j in range(m):
            temp = np.zeros((m,m),dtype=int)
            for yy in [-1, 0, 1]:
                for xx in [-1, 0, 1]:
                    if 0 <= i + yy < m:
                        if 0<= j + xx < m:
                            temp[i+yy][j+xx]=1
            adj_matrix[i*m+j]=temp.reshape(M)

    return adj_matrix

def degree_(adj,m):
    lenghth = adj.shape[0]
    dia = 0
    M = m**2
    degree_matrix = np.zeros((M,M),dtype=int)
    for i in range(lenghth):
        degree = sum(adj[i])+1
        degree_matrix[i][i] = degree
        dia = dia + m + 1

    return degree_matrix


class GraphTransform:
    def __init__(self, device):
        self.adj = adj_head(28)
        self.degree = degree_(self.adj,28)
        self.degree = np.where(self.degree>0, np.float_power(self.degree,-0.5),0)
        self.adj = np.dot(self.degree,self.adj)
        self.adj = np.dot(self.adj,self.degree)


    def __call__(self, img):
        return self.adj, \
               np.array(img).reshape(-1, 1)
