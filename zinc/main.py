import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
# from core.encoders import *


from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
import sys
import json
from torch import optim

from torch_geometric.datasets import ZINC
from torch_geometric.data import DataLoader

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
        node_kl_divergence_loss = node_kl_divergence_loss
        node_kl_divergence_loss.backward(retain_graph=True)

        # kl-divergence error for class latent space
        class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )
        #class_kl_divergence_loss = 0.0000001 * class_kl_divergence_loss * num_graphs
        class_kl_divergence_loss = class_kl_divergence_loss
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


        return reconstruction_error.item() ,class_kl_divergence_loss.item() ,node_kl_divergence_loss.item()

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

    def train(self, epoch):
        model.train() need a supervised component

        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = (model(data).squeeze() - data.y).abs().mean()
            loss.backward()
            total_loss += loss.item() * data.num_graphs
            optimizer.step()
        return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def test(self, loader):
        model.eval()

        total_error = 0
        for data in loader:
            data = data.to(device)
            total_error += (model(data).squeeze() - data.y).abs().sum().item()
        return total_error / len(loader.dataset)

if __name__ == '__main__':

    args = arg_parse()



    node_dim = int(args.hidden_dim)
    class_dim = int(args.hidden_dim)



    accuracies = {'svc':[]}
    losses = {'tot':[]}

    log_interval = 1
    batch_size = 128
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    train_dataset = ZINC(path, subset=True, split='train', pre_transform=transform)
    val_dataset = ZINC(path, subset=True, split='val', pre_transform=transform)
    test_dataset = ZINC(path, subset=True, split='test', pre_transform=transform)

    train_loader = DataLoader(train_dataset, 128, shuffle=True, num_workers=12)
    val_loader = DataLoader(val_dataset, 1000, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_dataset, 1000, shuffle=False, num_workers=12)

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


    for run in range(1, 5):
        print()
        print(f'Run {run}:')
        print()

        model.reset_parameters()
        optimizer = Adam(model.parameters(), lr=0.001)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5,
                                      patience=10, min_lr=0.00001)

        best_val_mae = test_mae = float('inf')
        for epoch in range(1, args.epochs + 1):
            lr = scheduler.optimizer.param_groups[0]['lr']
            loss = train(epoch)
            val_mae = test(val_loader)
            scheduler.step(val_mae)

            if val_mae < best_val_mae:
                best_val_mae = val_mae
                test_mae = test(test_loader)

            print(f'Epoch: {epoch:03d}, LR: {lr:.5f}, Loss: {loss:.4f}, '
                  f'Val: {val_mae:.4f}, Test: {test_mae:.4f}')

        test_maes.append(test_mae)

    test_mae = torch.tensor(test_maes)
    print('===========================')
    print(f'Final Test: {test_mae.mean():.4f} ± {test_mae.std():.4f}')

