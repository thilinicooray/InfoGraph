import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
# from core.encoders import *

from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops
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
  def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
    super(GcnInfomax, self).__init__()

    self.alpha = alpha
    self.beta = beta
    self.gamma = gamma
    self.prior = args.prior

    self.embedding_dim = mi_units = hidden_dim * num_gc_layers
    self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
    self.decoder = Decoder(hidden_dim, hidden_dim, dataset_num_features)

    self.local_d = FF(self.embedding_dim)
    self.global_d = FF(self.embedding_dim)


    self.init_emb()

  def init_emb(self):
    initrange = -1.5 / self.embedding_dim
    for m in self.modules():
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)


  def forward(self, x, edge_index, batch, num_graphs):

    # batch_size = data.num_graphs

    node_mu, node_logvar, class_mu, class_logvar = self.encoder(x, edge_index, batch)


    print('direct out ', node_mu[0,:5], node_logvar[0,:5], class_mu[0,:5], class_logvar[0,:5], batch)


    grouped_mu, grouped_logvar = accumulate_group_evidence(
        class_mu.data, class_logvar.data, batch, True
    )

    print('grouped ', grouped_mu[0,:5], grouped_logvar[0,:5])


    # kl-divergence error for style latent space
    node_kl_divergence_loss = torch.mean(
        - 0.5 * torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
    )
    print('node kl unwei ', node_kl_divergence_loss)
    node_kl_divergence_loss = 0.0000001*node_kl_divergence_loss *num_graphs
    print('node kl wei ', node_kl_divergence_loss)
    node_kl_divergence_loss.backward(retain_graph=True)

    # kl-divergence error for class latent space
    class_kl_divergence_loss = torch.mean(
        - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
    )
    print('class kl unwei ', class_kl_divergence_loss)
    class_kl_divergence_loss = 0.0000001* class_kl_divergence_loss * num_graphs
    print('class kl wei ', class_kl_divergence_loss)
    class_kl_divergence_loss.backward(retain_graph=True)

    # reconstruct samples
    """
    sampling from group mu and logvar for each graph in mini-batch differently makes
    the decoder consider class latent embeddings as random noise and ignore them 
    """
    node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)
    class_latent_embeddings = group_wise_reparameterize(
        training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
    )


    print('latent node ', node_latent_embeddings[0:5])
    print('latent class ', class_latent_embeddings[0:5])

    #need to reduce ml between node and class latents
    '''measure='JSD'
    mi_loss = local_global_loss_disen(node_latent_embeddings, class_latent_embeddings, edge_index, batch, measure)
    mi_loss.backward(retain_graph=True)'''

    reconstructed_node = self.decoder(node_latent_embeddings, class_latent_embeddings, edge_index)
    print('reconstructed_node ', reconstructed_node[0:5])


    #check input feat first
    #print('recon ', x[0],reconstructed_node[0])
    #reconstruction_error =  mse_loss(reconstructed_node, edge_index) * num_graphs
    reconstruction_error = self.recon_loss(reconstructed_node, edge_index)  #reeval adj loss
    print('reconstruction_error ', reconstruction_error)
    reconstruction_error.backward()

    #print(reconstruction_error.item(), class_kl_divergence_loss.item(), node_kl_divergence_loss.item())

    
    return reconstruction_error.item() , class_kl_divergence_loss.item() , node_kl_divergence_loss.item()


  def edge_recon(self, z, edge_index, sigmoid=False):
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

  def recon_loss(self, z, edge_index):

      EPS = 1e-15
      MAX_LOGSTD = 10
      r"""Given latent variables :obj:`z`, computes the binary cross
      entropy loss for positive edges :obj:`pos_edge_index` and negative
      sampled edges.

      Args:
          z (Tensor): The latent space :math:`\mathbf{Z}`.
          pos_edge_index (LongTensor): The positive edges to train against.
      """

      #reco = self.edge_recon(z, edge_index)

      #print('edge recon try ', reco.size(), edge_index.size(), reco [:10], edge_index[0][:10], edge_index[1][:10])

      pos_loss = -torch.log(
          self.edge_recon(z, edge_index) + EPS).mean()

      # Do not include self-loops in negative samples
      pos_edge_index, _ = remove_self_loops(edge_index)
      pos_edge_index, _ = add_self_loops(pos_edge_index)

      neg_edge_index = negative_sampling(pos_edge_index, z.size(0)) #random thingggg
      neg_loss = -torch.log(1 -
                            self.edge_recon(z, neg_edge_index) +
                            EPS).mean()

      return pos_loss + neg_loss

  def get_embeddings(self, loader):

      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      ret = []
      y = []
      with torch.no_grad():
          for data in loader:
              data.to(device)
              x, edge_index, batch = data.x, data.edge_index, data.batch
              x = torch.ones((batch.shape[0],1)).to(device)
              __, _, class_mu, class_logvar = self.encoder(x, edge_index, batch)

              grouped_mu, grouped_logvar = accumulate_group_evidence(
                  class_mu.data, class_logvar.data, batch, True
              )

              accumulated_class_latent_embeddings = group_wise_reparameterize(
                  training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
              )

              class_emb = global_mean_pool(accumulated_class_latent_embeddings, batch)
              ret.append(class_emb.cpu().numpy())
              y.append(data.y.cpu().numpy())
      ret = np.concatenate(ret, 0)
      y = np.concatenate(y, 0)
      return ret, y

if __name__ == '__main__':
    
    args = arg_parse()

    seed = 97
    #epochs = 30
    epochs = int(args.num_epochs)

    print('seed ', seed, 'epochs ', epochs)


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('init seed, seed ', torch.initial_seed(), seed)

    accuracies = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}

    log_interval = 1
    #batch_size = 128
    batch_size = args.batch_size
    lr = args.lr
    #lr = 0.000001
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS).shuffle()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        dataset_num_features = dataset.num_features
    except:
        dataset_num_features = 1

    if not dataset_num_features:
        dataset_num_features = 1
        #input_feat = torch.ones((batch_size, 1)).to(device)

    dataloader = DataLoader(dataset, batch_size=batch_size)


    model = GcnInfomax(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')


    '''model.eval()
    emb, y = model.get_embeddings(dataloader)
    res = evaluate_embedding(emb, y)
    accuracies['logreg'].append(res[0])
    accuracies['svc'].append(res[1])
    accuracies['linearsvc'].append(res[2])
    accuracies['randomforest'].append(res[3])'''

    #model.train()
    for epoch in range(1, epochs+1):
        recon_loss_all = 0
        kl_class_loss_all = 0
        kl_node_loss_all = 0
        mi_loss_all = 0
        model.train()
        for data in dataloader:
            data = data.to(device)
            #if data.x is None:
            data.x = torch.ones((data.batch.shape[0], 1)).to(device)


            optimizer.zero_grad()
            recon_loss, kl_class, kl_node = model(data.x, data.edge_index, data.batch, data.num_graphs)
            recon_loss_all += recon_loss
            kl_class_loss_all += kl_class
            kl_node_loss_all += kl_node
            optimizer.step()

        print('Epoch {}, Recon Loss {} KL class Loss {} KL node Loss {}'.format(epoch, recon_loss_all / len(dataloader),
                                                                                           kl_class_loss_all / len(dataloader), kl_node_loss_all / len(dataloader)))
        #used during finetune phase
        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.get_embeddings(dataloader)
            res = evaluate_embedding(emb, y)
            accuracies['logreg'].append(res[0])
            accuracies['svc'].append(res[1])
            accuracies['linearsvc'].append(res[2])
            accuracies['randomforest'].append(res[3])
            print(accuracies)


    '''model.eval()
    
    emb, y = model.get_embeddings(dataloader)
    res = evaluate_embedding(emb, y)
    accuracies['logreg'].append(res[0])
    accuracies['svc'].append(res[1])
    accuracies['linearsvc'].append(res[2])
    accuracies['randomforest'].append(res[3])
    print(accuracies)'''

    with open('unsupervised.log', 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s))
