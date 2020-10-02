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

from gin import Encoder, Decoder
from evaluate_embedding import evaluate_embedding

from utils import mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence

from arguments import arg_parse

class GL_Disen(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, node_dim, class_dim):
        super(GL_Disen, self).__init__()

        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, node_dim, class_dim)
        self.decoder = Decoder(hidden_dim, hidden_dim, dataset_num_features)


        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):



        node_mu, node_logvar, class_mu, class_logvar = self.encoder(x, edge_index, batch)
        grouped_mu, grouped_logvar = accumulate_group_evidence(
            class_mu.data, class_logvar.data, batch, True
        )

        # kl-divergence error for local latent space
        node_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
        )
        node_kl_divergence_loss = 0.0000001 * node_kl_divergence_loss
        node_kl_divergence_loss.backward(retain_graph=True)

        # kl-divergence error for global latent space
        class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )
        class_kl_divergence_loss = 0.0000001 * class_kl_divergence_loss
        class_kl_divergence_loss.backward(retain_graph=True)

        # sampling
        node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)
        class_latent_embeddings = group_wise_reparameterize(
            training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
        )

        #reconstruct node
        reconstructed_node = self.decoder(node_latent_embeddings, class_latent_embeddings)
        reconstruction_error =  mse_loss(reconstructed_node, x)
        reconstruction_error.backward()

        total_loss = reconstruction_error.item() + class_kl_divergence_loss.item() + node_kl_divergence_loss.item()


        return total_loss

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

    seeds = [32]

    #seeds = [123,132,213,231,312,321]
    epochs_list = [30]
    node_ratio = [0.5]
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

                accuracies = {'svc':[]}
                losses = {'total':[]}

                log_interval = 1
                batch_size = 128
                lr = args.lr
                DS = args.DS
                path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)

                dataset = TUDataset(path, name=DS).shuffle()
                try:
                    dataset_num_features = dataset.num_features
                except:
                    dataset_num_features = 1


                dataloader = DataLoader(dataset, batch_size=batch_size)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = GL_Disen(args.hidden_dim, args.num_gc_layers, node_dim, class_dim).to(device)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                print('================')
                print('lr: {}'.format(lr))
                print('num_features: {}'.format(dataset_num_features))
                print('hidden_dim: {}'.format(args.hidden_dim))
                print('num_gc_layers: {}'.format(args.num_gc_layers))
                print('================')

                model.train()
                for epoch in range(1, epochs+1):
                    tot_loss_all = 0
                    for data in dataloader:
                        data = data.to(device)
                        optimizer.zero_grad()
                        tot_loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
                        tot_loss_all += tot_loss
                        optimizer.step()

                    losses['total'].append(tot_loss_all/ len(dataloader))

                    print('Epoch {}, Total Loss {}'.format(epoch, tot_loss_all / len(dataloader)))

                model.eval()

                emb, y = model.get_embeddings(dataloader)

                res = evaluate_embedding(emb, y)
                accuracies['svc'].append(res)
                print(accuracies)




