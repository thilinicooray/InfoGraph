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
import sys
import json
from torch import optim

from gin import *
from evaluate_embedding import evaluate_embedding
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence

from arguments import arg_parse

class GLDisen(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, node_dim, class_dim):
        super(GLDisen, self).__init__()


        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, node_dim, class_dim)
        self.decoder = Decoder(hidden_dim, hidden_dim, dataset_num_features)

        self.proj1 = FF(self.embedding_dim)
        self.proj2 = FF(self.embedding_dim)


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
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        node_mu, node_logvar, class_mu, class_logvar = self.encoder(x, edge_index, batch)
        grouped_mu, grouped_logvar = accumulate_group_evidence(
            class_mu.data, class_logvar.data, batch, True
        )

        # kl-divergence error for style latent space
        node_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
        )
        #node_kl_divergence_loss = 0.0000001 * node_kl_divergence_loss *num_graphs
        node_kl_divergence_loss = 0.000001 * node_kl_divergence_loss
        node_kl_divergence_loss.backward(retain_graph=True)

        # kl-divergence error for class latent space
        class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )
        #class_kl_divergence_loss = 0.0000001 * class_kl_divergence_loss * num_graphs
        class_kl_divergence_loss = 0.000001 * class_kl_divergence_loss
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

        #need to reduce ml between node and class latents
        '''measure='JSD'
        mi_loss = local_global_loss_disen(node_latent_embeddings, class_latent_embeddings, edge_index, batch, measure)
        mi_loss.backward(retain_graph=True)'''

        reconstructed_node = self.decoder(node_latent_embeddings, class_latent_embeddings)
        #check input feat first
        #print('recon ', x[0],reconstructed_node[0])
        #reconstruction_error =  0.1*mse_loss(reconstructed_node, x) * num_graphs
        reconstruction_error =  mse_loss(reconstructed_node, x)
        reconstruction_error.backward()


        return reconstruction_error.item() + class_kl_divergence_loss.item() + node_kl_divergence_loss.item()

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
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

    #enable entire sets of hyperparameters for the full experiment

    seeds = [32]#[32,42,52,62,72]

    #seeds = [123,132,213,231,312,321]
    epochs_list = [20]#[20,30,40,50]
    node_ratio = [0.5]#[0.25,0.5,0.75]
    for seed in seeds:
        for epochs in epochs_list:
            for rat in node_ratio:

                node_dim = int(args.hidden_dim*2*rat)
                class_dim = int(args.hidden_dim*2 - node_dim)

                print('seed ', seed, 'epochs ', epochs, 'node dim ' , node_dim, 'class dim ', class_dim)

                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)

                print('init seed, seed ', torch.initial_seed(), seed)

                accuracies = {'svc':[]}
                losses = {'tot':[]}

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
                    dataset_num_features = 1

                dataloader = DataLoader(dataset, batch_size=batch_size)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = GLDisen(args.hidden_dim, args.num_gc_layers, node_dim, class_dim).to(device)
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

                model.train()
                for epoch in range(1, epochs+1):
                    loss_all = 0

                    #model.train()
                    for data in dataloader:
                        data = data.to(device)
                        optimizer.zero_grad()
                        loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
                        loss_all += loss
                        optimizer.step()

                    losses['tot'].append(loss_all/ len(dataloader))


                    print('Epoch {}, Total Loss {} '.format(epoch, loss_all/ len(dataloader)))


                model.eval()

                emb, y = model.get_embeddings(dataloader)

                res = evaluate_embedding(emb, y)
                accuracies['svc'].append(res)
                print(accuracies)

