import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
# from core.encoders import *

#from torch_geometric.datasets import CitationFull
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder, Decoder
from evaluate_embedding import evaluate_embedding
from model import *
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence

from arguments import arg_parse

class GcnInfomax(nn.Module):
  def __init__(self, dataset_num_features, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    #self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
    self.decoder = Decoder(hidden_dim, hidden_dim, dataset_num_features)

    #self.local_d = FF(self.embedding_dim)
    #self.global_d = FF(self.embedding_dim)


    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index):

    # batch_size = data.num_graphs


    node_mu, node_logvar, class_mu, class_logvar = self.encoder(x, edge_index)

    '''print('node mu', node_mu)

    print('node node_logvar', node_logvar)

    print('class_mu', class_mu)

    print('class_logvar', class_logvar)'''

    '''node_mu = node_mu.view(-1, node_mu.size(-1))
    node_logvar = node_logvar.view(-1, node_logvar.size(-1))
    class_mu = class_mu.view(-1, class_mu.size(-1))
    class_logvar = class_logvar.view(-1, class_logvar.size(-1))'''

    n_nodes = node_mu.size(0)

    batch = torch.ones(node_mu.size(0)).cuda()

    '''grouped_mu, grouped_logvar = accumulate_group_evidence(
        class_mu.data, class_logvar.data, batch, True
    )'''

    #print('grouped_mu', grouped_mu)

    #print('grouped_logvar', grouped_logvar)

    # kl-divergence error for style latent space
    '''node_kl_divergence_loss = torch.mean(
        - 0.5 * torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
    )
    node_kl_divergence_loss = node_kl_divergence_loss
    node_kl_divergence_loss.backward(retain_graph=True)'''

    node_kl_divergence_loss = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * node_logvar - node_mu.pow(2) - node_logvar.exp().pow(2), 1))


    # kl-divergence error for class latent space
    '''class_kl_divergence_loss = torch.mean(
        - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
    )
    class_kl_divergence_loss = class_kl_divergence_loss
    class_kl_divergence_loss.backward(retain_graph=True)'''

    '''class_kl_divergence_loss = -0.5 / n_nodes * torch.mean(torch.sum(
        1 + 2 * grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp().pow(2), 1))'''

    # reconstruct samples
    """
    sampling from group mu and logvar for each graph in mini-batch differently makes
    the decoder consider class latent embeddings as random noise and ignore them 
    """
    node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)


    '''class_latent_embeddings = group_wise_reparameterize(
        training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
    )'''

    #print('class_latent_embeddings', class_latent_embeddings)

    #need to reduce ml between node and class latents
    '''measure='JSD'
    mi_loss = local_global_loss_disen(node_latent_embeddings, class_latent_embeddings, edge_index, batch, measure)
    mi_loss.backward(retain_graph=True)'''

    reconstructed_node = self.decoder(node_latent_embeddings)

    #check input feat first
    #print('recon ', x[0],reconstructed_node[0])
    reconstruction_error =  mse_loss(reconstructed_node, x)




    #reconstruction_error =  self.adj_recon(node_latent_embeddings, edge_index)



    #reconstruction_error.backward()

    print('recon, classkl, node kl', reconstruction_error.item(),  node_kl_divergence_loss.item())


    return reconstruction_error + node_kl_divergence_loss

  def get_embeddings(self, feat, adj):

      with torch.no_grad():
          node_mu, node_logvar, _, _ = self.encoder(feat, adj)
          node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)

      return node_latent_embeddings

  def adj_recon(self, node_latent, adj):

      pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
      norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


      recon_adj = torch.sigmoid(torch.mm(node_latent, node_latent.t()))

      #loss = norm* F.binary_cross_entropy_with_logits(recon_adj, adj, pos_weight=pos_weight)
      loss = norm* F.binary_cross_entropy_with_logits(recon_adj, adj, pos_weight=pos_weight)

      return loss

