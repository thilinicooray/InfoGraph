import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch

from collections import OrderedDict

from infomax import *

from utils import *

class Encoder(torch.nn.Module):
    def __init__(self, num_features, dim):
        super(Encoder, self).__init__()
        self.lin0 = torch.nn.Linear(num_features, dim)

        nn = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.conv = NNConv(dim, dim, nn, aggr='mean', root_weight=False)
        self.gru = GRU(dim, dim)

        # self.lin1 = torch.nn.Linear(2 * dim, dim)
        # self.lin2 = torch.nn.Linear(dim, 1)

        #disentangling layers

        nn1 = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.node_mu_conv = NNConv(dim, dim, nn1, aggr='mean', root_weight=False)

        nn2 = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.node_lv_conv = NNConv(dim, dim, nn2, aggr='mean', root_weight=False)

        nn3 = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.graph_mu_conv = NNConv(dim, dim, nn3, aggr='mean', root_weight=False)

        nn4 = Sequential(Linear(5, 128), ReLU(), Linear(128, dim * dim))
        self.graph_lv_conv = NNConv(dim, dim, nn4, aggr='mean', root_weight=False)

        #graph distribution parameter accumulation

        self.set2set_mu = Set2Set(dim, processing_steps=3)
        self.set2set_lv = Set2Set(dim, processing_steps=3)
        self.set2set_nodes = Set2Set(dim, processing_steps=3)


    def forward(self, data):

        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)


        feat_map = []
        for i in range(3):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
            # print(out.shape) : [num_node x dim]
            feat_map.append(out)


        node_graph = self.set2set_nodes(out, data.batch)
        #node_graph = global_add_pool(out, data.batch)
        node_mu = F.relu(self.node_mu_conv(out, data.edge_index, data.edge_attr))
        node_lv = F.relu(self.node_lv_conv(out, data.edge_index, data.edge_attr))
        graph_mu_id = F.relu(self.graph_mu_conv(out, data.edge_index, data.edge_attr))
        graph_lv_id = F.relu(self.graph_lv_conv(out, data.edge_index, data.edge_attr))

        grouped_mu = self.set2set_mu(graph_mu_id, data.batch)
        grouped_lv = self.set2set_lv(graph_lv_id, data.batch)

        _, count = torch.unique(data.batch,  return_counts=True)

        grouped_mu_expanded = torch.repeat_interleave(grouped_mu, count, dim=0)
        grouped_lvar_expanded = torch.repeat_interleave(grouped_lv, count, dim=0)

        return node_mu, node_lv, grouped_mu_expanded, grouped_lvar_expanded, node_graph

class Decoder(torch.nn.Module):
    def __init__(self, node_dim, class_dim, feat_size):
        super(Decoder, self).__init__()

        self.linear_model = torch.nn.Sequential(OrderedDict([
            ('linear_1', torch.nn.Linear(in_features=node_dim + class_dim, out_features=node_dim, bias=True)),
            ('relu_1', ReLU()),

            ('linear_2', torch.nn.Linear(in_features=node_dim, out_features=feat_size, bias=True)),
            ('relu_final', ReLU()),
        ]))

    def forward(self, node_latent_space, class_latent_space, y):

        #x = torch.cat((node_latent_space, class_latent_space, y.unsqueeze(1)), dim=1)
        x = torch.cat((node_latent_space, node_latent_space, class_latent_space), dim=1)

        x = self.linear_model(x)

        return x



    # class PriorDiscriminator(nn.Module):
    # def __init__(self, input_dim):
    # super().__init__()
    # self.l0 = nn.Linear(input_dim, input_dim)
    # self.l1 = nn.Linear(input_dim, input_dim)
    # self.l2 = nn.Linear(input_dim, 1)

    # def forward(self, x):
    # h = F.relu(self.l0(x))
    # h = F.relu(self.l1(h))
    # return torch.sigmoid(self.l2(h))

class FF(nn.Module):
    def __init__(self, input_dim, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class Net(torch.nn.Module):
    def __init__(self, num_features, dim, std, use_unsup_loss=False, separate_encoder=False):
        super(Net, self).__init__()

        self.embedding_dim = dim
        self.std = std

        self.encoder = Encoder(num_features, dim)
        self.decoder = Decoder(dim*2, dim*2, num_features)

        self.fc1 = torch.nn.Linear(2 * dim, dim)
        self.fc2 = torch.nn.Linear(dim, 1)

        self.fc1_sup = torch.nn.Linear(dim, dim)
        self.fc2_sup = torch.nn.Linear(dim, 1)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def supervised_loss(self, data):

        node_mu, node_logvar, grouped_mu, grouped_logvar, node_graph = self.encoder(data)

        n_nodes = data.x.size(0)


        node_kl_divergence_loss = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * node_logvar - node_mu.pow(2) - node_logvar.exp().pow(2), 1))


        node_kl_divergence_loss = 10*node_kl_divergence_loss


        # kl-divergence error for class latent space
        '''class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )'''
        class_kl_divergence_loss = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp().pow(2), 1))

        #print('class kl unwei ', class_kl_divergence_loss)
        class_kl_divergence_loss = 10*class_kl_divergence_loss
        #print('class kl wei ', class_kl_divergence_loss)


        # reconstruct samples
        """
        sampling from group mu and logvar for each graph in mini-batch differently makes
        the decoder consider class latent embeddings as random noise and ignore them 
        """
        node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)
        class_latent_embeddings = group_wise_reparameterize(
            training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=data.batch, cuda=True
        )

        _, count = torch.unique(data.batch,  return_counts=True)

        graph_emb = global_mean_pool(class_latent_embeddings, data.batch)

        out = F.relu(self.fc1(graph_emb + node_graph))
        out = self.fc2(out)
        classification = out.view(-1)

        #y_expanded = torch.repeat_interleave(data.y, count, dim=0)
        y_expanded = torch.repeat_interleave(classification, count, dim=0)

        reconstructed_node = self.decoder(node_latent_embeddings, class_latent_embeddings, y_expanded)

        reconstruction_error =  0.001* mse_loss(reconstructed_node, data.x) # + self.recon_loss1(reconstructed_node, data.edge_index, data.batch)
        #reconstruction_error = 1e-5*self.recon_loss1(reconstructed_node, edge_index, batch)





        cls_loss = F.mse_loss(classification, data.y)

        '''out_node = F.relu(self.fc1_sup(node_graph))
        out_node = self.fc2_sup(out_node)
        classification_node = out_node.view(-1)

        cls_loss_node = F.mse_loss(classification_node, data.y)'''


        #cls_loss = torch.mean((classification * self.std - data.y * self.std).abs())

        total_loss = node_kl_divergence_loss + class_kl_divergence_loss + reconstruction_error + 100*cls_loss #+ cls_loss_node

        total_loss.backward()

        return node_kl_divergence_loss.item(), class_kl_divergence_loss.item(), reconstruction_error.item(), cls_loss.item()





    def unsupervised_loss(self, data):

        node_mu, node_logvar, grouped_mu, grouped_logvar, _ = self.encoder(data)

        n_nodes = data.x.size(0)


        node_kl_divergence_loss = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * node_logvar - node_mu.pow(2) - node_logvar.exp().pow(2), 1))


        node_kl_divergence_loss = 10*node_kl_divergence_loss


        # kl-divergence error for class latent space
        '''class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )'''
        class_kl_divergence_loss = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp().pow(2), 1))

        #print('class kl unwei ', class_kl_divergence_loss)
        class_kl_divergence_loss = 10*class_kl_divergence_loss
        #print('class kl wei ', class_kl_divergence_loss)


        # reconstruct samples
        """
        sampling from group mu and logvar for each graph in mini-batch differently makes
        the decoder consider class latent embeddings as random noise and ignore them 
        """
        node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)
        class_latent_embeddings = group_wise_reparameterize(
            training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=data.batch, cuda=True
        )

        graph_emb = global_mean_pool(class_latent_embeddings, data.batch)
        out = F.relu(self.fc1(graph_emb))
        out = self.fc2(out)
        classification = out.view(-1)

        _, count = torch.unique(data.batch,  return_counts=True)

        classification_expanded = torch.repeat_interleave(classification, count, dim=0)


        reconstructed_node = 0.001*self.decoder(node_latent_embeddings, class_latent_embeddings, classification_expanded)

        reconstruction_error =  mse_loss(reconstructed_node, data.x) #+ self.recon_loss1(reconstructed_node, data.edge_index, data.batch)
        #reconstruction_error = 1e-5*self.recon_loss1(reconstructed_node, edge_index, batch)


        total_loss = node_kl_divergence_loss + class_kl_divergence_loss + reconstruction_error

        total_loss.backward()

        return node_kl_divergence_loss.item(), class_kl_divergence_loss.item(), reconstruction_error.item(), None

    def forward(self, data):

        node_mu, node_logvar, grouped_mu, grouped_logvar, node_graph = self.encoder(data)


        class_latent_embeddings = group_wise_reparameterize(
            training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=data.batch, cuda=True
        )

        graph_emb = global_mean_pool(class_latent_embeddings, data.batch)

        out = F.relu(self.fc1(graph_emb + node_graph))
        out = self.fc2(out)
        classification = out.view(-1)

        return classification

    def edge_recon(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    def recon_loss1(self, z, edge_index, batch):

        EPS = 1e-15
        MAX_LOGSTD = 10
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
  
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        #org_adj = to_dense_adj(edge_index, batch)
        pos_weight = float(z.size(0) * z.size(0) - edge_index.size(0)) / edge_index.size(0)
        norm = z.size(0) * z.size(0) / float((z.size(0) * z.size(0) - edge_index.size(0)) * 2)



        recon_adj = self.edge_recon(z, edge_index)


        pos_loss = -torch.log(
            recon_adj + EPS).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0)) #random thingggg
        neg_loss = -torch.log(1 -
                              self.edge_recon(z, neg_edge_index) +
                              EPS).mean()

        return norm*(pos_loss*pos_weight + neg_loss)
