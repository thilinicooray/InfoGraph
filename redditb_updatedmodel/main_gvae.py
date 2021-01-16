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
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch
import sys
import json
from torch import optim

from gin_gvae import *
from evaluate_embedding import evaluate_embedding
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence

from arguments import arg_parse

class GLDisen(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, node_dim):
        super(GLDisen, self).__init__()


        #self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, node_dim)
        self.decoder = Decoder(hidden_dim, 32)

        #self.proj1 = FF(self.embedding_dim)
        #self.proj2 = FF(self.embedding_dim)


        self.init_emb()

    def init_emb(self):
        #initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):

        n_nodes = x.size(0)

        # batch_size = data.num_graphs
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        node_mu, node_logvar = self.encoder(x, edge_index, batch)


        # kl-divergence error for style latent space
        '''node_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
        )'''
        node_kl_divergence_loss = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * node_logvar - node_mu.pow(2) - node_logvar.exp().pow(2), 1))

        #node_kl_divergence_loss = 0.0000001 * node_kl_divergence_loss *num_graphs
        node_kl_divergence_loss = node_kl_divergence_loss
        #node_kl_divergence_loss.backward(retain_graph=True)

        # kl-divergence error for class latent space
        '''class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + class_logvar - class_mu.pow(2) - class_logvar.exp())
        )'''


        # reconstruct samples
        """
        sampling from group mu and logvar for each graph in mini-batch differently makes
        the decoder consider class latent embeddings as random noise and ignore them 
        """
        node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)


        #need to reduce ml between node and class latents
        '''measure='JSD'
        mi_loss = local_global_loss_disen(node_latent_embeddings, class_latent_embeddings, edge_index, batch, measure)
        mi_loss.backward(retain_graph=True)'''

        reconstructed_node = self.decoder(node_latent_embeddings)
        #reconstructed_node = node_latent_embeddings
        #check input feat first
        #print('recon ', x[0],reconstructed_node[0])
        #reconstruction_error =  0.1*mse_loss(reconstructed_node, x) * num_graphs
        reconstruction_error =  self.recon_loss1(reconstructed_node, edge_index)#mse_loss(reconstructed_node, x) + self.recon_loss1(reconstructed_node, edge_index)
        #reconstruction_error.backward()

        loss =  node_kl_divergence_loss + reconstruction_error

        loss.backward()


        return reconstruction_error.item() , 0.0 , node_kl_divergence_loss.item()

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

    def recon_loss1(self, z, edge_index):

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

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if not dataset.num_features:
                    x = torch.ones((data.batch.shape[0], 5)).to(device)
                node_mu, node_logvar = self.encoder(x, edge_index, batch)
                mean_mu = global_mean_pool(node_mu, batch)
                mean_logvar = global_mean_pool(node_logvar, batch)
                #node_emb = reparameterize(training=False, mu=node_mu, logvar=node_logvar)
                #class_emb = global_mean_pool(node_emb, batch)
                class_emb = reparameterize(training=False, mu=mean_mu, logvar=mean_logvar)

                ret.append(class_emb.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

if __name__ == '__main__':

    args = arg_parse()

    #enable entire sets of hyperparameters for the full experiment

    best_acc = 0
    best_setup = {'seed':0, 'epoch':0, 'node_ratio':0, 'acc':0}

    seeds = [32,42,52,62,72]

    #seeds = [123,132,213,231,312,321] #this set also give similar results
    epochs_list =[1000, 50, 75, 100]
    node_ratio = [0.5]
    for seed in seeds:
        for epochs in epochs_list:
            for rat in node_ratio:

                node_dim = int(args.hidden_dim*2*rat)

                print('seed ', seed, 'epochs ', epochs, 'node dim ', node_dim)

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                print('init seed, seed ', torch.initial_seed(), seed)

                accuracies = {'svc':[]}
                losses = {'tot':[]}
                losses = {'recon':[], 'node_kl':[], 'class_kl': []}

                log_interval = 1
                batch_size = 128
                lr = args.lr
                DS = args.DS
                path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
                # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

                dataset = TUDataset(path, name=DS).shuffle()
                try:
                    dataset_num_features = dataset.num_features
                except:
                    dataset_num_features = 1

                if not dataset_num_features:
                    dataset_num_features = 5

                dataloader = DataLoader(dataset, batch_size=batch_size)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = GLDisen(args.hidden_dim, args.num_gc_layers, node_dim).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

                print('================')
                print('lr: {}'.format(lr))
                print('num_features: {}'.format(dataset_num_features))
                print('hidden_dim: {}'.format(args.hidden_dim))
                print('num_gc_layers: {}'.format(args.num_gc_layers))
                print('================')


                model.train()

                best_recon_loss = 10e40
                best_model = None

                for epoch in range(1, epochs+1):
                    loss_all = 0
                    recon_loss_all = 0
                    kl_class_loss_all = 0
                    kl_node_loss_all = 0

                    #model.train()
                    for data in dataloader:
                        data = data.to(device)
                        optimizer.zero_grad()
                        if not dataset.num_features:
                            data.x = torch.ones((data.batch.shape[0], 5)).to(device)
                        recon_loss, kl_class, kl_node = model(data.x, data.edge_index, data.batch, data.num_graphs)
                        recon_loss_all += recon_loss
                        kl_class_loss_all += kl_class
                        kl_node_loss_all += kl_node
                        optimizer.step()

                        #losses['tot'].append(loss_all/ len(dataloader))


                        #print('Epoch {}, Total Loss {} '.format(epoch, loss_all/ len(dataloader)))

                    losses['recon'].append(recon_loss_all/ len(dataloader))
                    losses['node_kl'].append(kl_node_loss_all/ len(dataloader))
                    losses['class_kl'].append(kl_class_loss_all/ len(dataloader))

                    recon_current = recon_loss_all/ len(dataloader)

                    '''if recon_current < best_recon_loss:
                        best_recon_loss = recon_current
                        torch.save(model.state_dict(), 'model.pkl')
                        print('current best model saved!')'''



                    print('Epoch {}, Recon Loss {} KL class Loss {} KL node Loss {}'.format(epoch, recon_loss_all / len(dataloader),
                                                                                            kl_class_loss_all / len(dataloader), kl_node_loss_all / len(dataloader)))

                #print('loading best model...')
                #model.load_state_dict(torch.load('model.pkl'))

                model.eval()

                emb, y = model.get_embeddings(dataloader)

                res = evaluate_embedding(emb, y)
                accuracies['svc'].append(res)
                print(accuracies)

                if res > best_acc:
                    best_acc = res
                    torch.save(model.state_dict(), 'gvae_best.pkl')
                    print('current best model saved!')
                    best_setup['seed']=seed
                    best_setup['epoch']=epochs
                    best_setup['node_ratio']=rat
                    best_setup['acc']=best_acc
                    with open('gvae_best_setup.json', 'w') as f:
                        json.dump(best_setup, f)

