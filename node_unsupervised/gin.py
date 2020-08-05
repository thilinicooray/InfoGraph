import os.path as osp
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU, Tanh
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import sys


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.mm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers):
        super(Encoder, self).__init__()

        # num_features = dataset.num_features
        # dim = 32
        self.num_gc_layers = num_gc_layers

        # self.nns = []
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers+4):

            if i:
                conv = GCN(dim, dim)
            else:
                conv = GCN(num_features, dim)
            bn = torch.nn.BatchNorm1d(dim)

            self.convs.append(conv)
            self.bns.append(bn)

        '''
        # node
        self.node_mu = Linear(in_features=dim, out_features=dim, bias=True)
        self.node_logvar = Linear(in_features=dim, out_features=dim, bias=True)

        # class
        self.class_mu = Linear(in_features=dim, out_features=dim, bias=True)
        self.class_logvar = Linear(in_features=dim, out_features=dim, bias=True)'''


    '''def forward(self, x, edge_index):

        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            # if i == 2:
                # feature_map = x2
        j = self.num_gc_layers
        node_latent_space_mu = self.bns[j](torch.tanh(self.convs[j](x, edge_index)))
        node_latent_space_logvar = self.bns[j+1](torch.tanh(self.convs[j+1](x, edge_index)))

        class_latent_space_mu = self.bns[j+2](torch.tanh(self.convs[j+2](x, edge_index)))
        class_latent_space_logvar = self.bns[j+3](torch.tanh(self.convs[j+3](x, edge_index)))

        return node_latent_space_mu, node_latent_space_logvar, class_latent_space_mu, class_latent_space_logvar'''

    def forward(self, x, edge_index):

        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = x
            # if i == 2:
            # feature_map = x2
        j = self.num_gc_layers
        node_latent_space_mu = torch.relu(self.convs[j](x, edge_index))
        node_latent_space_logvar = torch.relu(self.convs[j+1](x, edge_index))

        class_latent_space_mu = torch.relu(self.convs[j+2](x, edge_index))
        class_latent_space_logvar = torch.relu(self.convs[j+3](x, edge_index))

        return node_latent_space_mu, node_latent_space_logvar, class_latent_space_mu, class_latent_space_logvar


class Decoder(torch.nn.Module):
    def __init__(self, node_dim, class_dim, feat_size):
        super(Decoder, self).__init__()

        self.linear_model = torch.nn.Sequential(OrderedDict([
            ('linear_1', torch.nn.Linear(in_features=node_dim + class_dim, out_features=node_dim, bias=True)),
            ('relu_1', ReLU()),

            ('linear_2', torch.nn.Linear(in_features=node_dim, out_features=feat_size, bias=True)),
            ('relu_final', ReLU()),
        ]))

    def forward(self, node_latent_space, class_latent_space):
        x = torch.cat((node_latent_space, class_latent_space), dim=1)

        x = self.linear_model(x)

        return x



class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        try:
            num_features = dataset.num_features
        except:
            num_features = 1
        dim = 32

        self.encoder = Encoder(num_features, dim)

        self.fc1 = Linear(dim*5, dim)
        self.fc2 = Linear(dim, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        x, _ = self.encoder(x, edge_index, batch)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=-1)

def train(epoch):
    model.train()

    if epoch == 51:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.5 * param_group['lr']

    loss_all = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        # print(data.x.shape)
        # [ num_nodes x num_node_labels ]
        # print(data.edge_index.shape)
        #  [2 x num_edges ]
        # print(data.batch.shape)
        # [ num_nodes ]
        output = model(data.x, data.edge_index, data.batch)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()

    return loss_all / len(train_dataset)

def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        output = model(data.x, data.edge_index, data.batch)
        pred = output.max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

