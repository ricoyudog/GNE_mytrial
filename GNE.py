import torch
import torchvision
import numpy as np
import scipy as sio
import sklearn
import torch.nn as nn # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F # All functions that don't have any parameters
from torch.utils.data import DataLoader # Gives easier dataset managment and creates mini batches
import torchvision.datasets as datasets # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For nice progress bar!
import networkx as nx
from layers import GraphConvolution
import scipy.sparse as sp
from models import GCN
import time
from dataset import TUDataset, CollateFn
from MNIST import GraphTransform


# class GCN(nn.Module):
#     def __init__(self, ):
#         super(GCN, self).__init__()

torch.manual_seed(28)
torch.cuda.manual_seed(28)
# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
num_classes = 10
learning_rate = 1e-4
batch_size = 50
num_epochs = 100

#load dataset
train_dataset = datasets.MNIST(root='dataset/', train=True, transform=GraphTransform(device), download=False)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True , collate_fn=CollateFn(device))
test_dataset = datasets.MNIST(root='dataset/', train=False, transform=GraphTransform(device), download=False)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True,collate_fn=CollateFn(device))







def adj_head(m):
    '''
    To create adjacency matrix as per Defferrard et al. 2016
    '''
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
        degree = sum(adj[i])
        degree_matrix[i][i] = degree
        dia = dia + m + 1

    return degree_matrix



# degree_negative = np.where(degree > 0,degree**(-(1/2)),0)
# normalize = np.dot(degree_negative,adj)
# normalize = np.dot(normalize,degree_negative)
# normalize = np.dot(normalize,feature)
#
# normalize = sparse.csr_matrix(normalize)

def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.float_power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)



adj = adj_head(28)
degree = degree_(adj,28)
# G = nx.convert_matrix.from_numpy_matrix(adj,parallel_edges=True,create_using=nx.DiGraph)

# adj = normalize(adj)
# adj = sp.csr_matrix.todense(adj)
# d_inv = np.where(degree>0, np.float_power(degree,-1/2),0)
# processed_adj = np.dot(d_inv,adj)
# processed_adj = np.dot(adj,d_inv)
# adj = sp.csr_matrix(processed_adj)
# adj = sparse_mx_to_torch_sparse_tensor(adj)

# adj = torch.from_numpy(adj).float()






def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)







# feature = feature.reshape(feature.shape[0],-1)
# feature = feature.to(device=device)
# target = images
#
# model.train()
# optimizer.zero_grad()
# adj = adj.to(device)
# output = model(feature, adj)
# loss_train = F.nll_loss(output, target)
# print(loss_train)
# loss_train.backward()
# optimizer.step()

# feature = train_dataset.data.view(-1, 1, 28, 28).float()
# feature = feature.reshape(feature.shape[0],-1)
# # print(feature)
# labels = train_dataset.train_labels

# feature = feature.to(device)
# labels = labels.to(device)
# adj = adj.to(device)

model = GCN(nfeat=1,
            nhid=256,
            nclass=10).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)








start_time = time.time()
for epoch in range(num_epochs):

    for i, (adj, features, masks, batch_labels) in enumerate(train_loader):
        # input = feature[1].view(784,1)
        features = features.to(device)
        batch_labels = batch_labels.to(device)
        adj = adj.to(device)
        t = time.time()
        model.train()
        optimizer.zero_grad()
        output = model(features, adj)
        loss_train = F.nll_loss(output, batch_labels)
        acc_train = accuracy(output, batch_labels)
        loss_train.backward()
        optimizer.step()

        loss_val = F.nll_loss(output, batch_labels)
        acc_val = accuracy(output, batch_labels)

        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))


# Train model
t_total = time.time()

print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - start_time))




def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data")
    else:
        print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for adj, features, masks, batch_labels in loader:
            adj = adj.to(device=device)
            features = features.to(device=device)
            batch_labels = batch_labels.to(device=device)

            test_scores = model(features, adj)
            _, predictions = test_scores.max(1)
            num_correct += (predictions == batch_labels).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()

    return test_scores


check_accuracy(test_loader, model)