import os.path as osp
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
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

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim, num_gc_layers, node_dim, class_dim):
        super(Encoder, self).__init__()

        self.num_gc_layers = num_gc_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_gc_layers+4):

            if i == 0:
                nn = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)
            elif i >= num_gc_layers and i < num_gc_layers +2:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, node_dim))
                bn = torch.nn.BatchNorm1d(node_dim)
            elif i >= num_gc_layers and i >= num_gc_layers +2:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, class_dim))
                bn = torch.nn.BatchNorm1d(class_dim)
            else:
                nn = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
                bn = torch.nn.BatchNorm1d(dim)
            conv = GINConv(nn)

            self.convs.append(conv)
            self.bns.append(bn)



    def forward(self, x, edge_index, batch):

        xs = []
        for i in range(self.num_gc_layers):

            x = F.relu(self.convs[i](x, edge_index))
            x = self.bns[i](x)
            xs.append(x)

        j = self.num_gc_layers
        node_latent_space_mu = self.bns[j](F.relu(self.convs[j](x, edge_index)))
        node_latent_space_logvar = self.bns[j+1](F.relu(self.convs[j+1](x, edge_index)))

        class_latent_space_mu = self.bns[j+2](F.relu(self.convs[j+2](x, edge_index)))
        class_latent_space_logvar = self.bns[j+3](F.relu(self.convs[j+3](x, edge_index)))

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

        x = torch.softmax(self.linear_model(x), dim=-1)

        return x


