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

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin_eval import Encoder, Decoder
from evaluate_embedding import evaluate_embedding
from model import *
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence

from arguments import arg_parse

class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, node_dim, class_dim, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, node_dim, class_dim)
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
        node_kl_divergence_loss = 0.0000001 * node_kl_divergence_loss *num_graphs
        node_kl_divergence_loss.backward(retain_graph=True)

        # kl-divergence error for class latent space
        class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )
        class_kl_divergence_loss = 0.0000001 * class_kl_divergence_loss * num_graphs
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
        reconstruction_error =  0.1*mse_loss(reconstructed_node, x) * num_graphs
        reconstruction_error.backward()


        return reconstruction_error.item() , class_kl_divergence_loss.item() , node_kl_divergence_loss.item()

    def get_embeddings(self, loader):

        from scipy import stats
        from numpy import savetxt

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch

                node_mu, node_logvar, class_mu, class_logvar, entangled = self.encoder(x, edge_index, batch)

                node_latent_embeddings = reparameterize(training=False, mu=node_mu, logvar=node_logvar)

                grouped_mu, grouped_logvar = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, batch, True
                )

                accumulated_class_latent_embeddings = group_wise_reparameterize(
                    training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
                )

                sim_node = torch.tanh(torch.matmul(entangled[0].unsqueeze(0).t(), node_mu[0].unsqueeze(0))).cpu().numpy()
                graph_node = torch.tanh(torch.matmul(entangled[0].unsqueeze(0).t(), class_mu[0].unsqueeze(0))).cpu().numpy()

                '''np_entangled = entangled.cpu().numpy()
                np_node_emb = node_latent_embeddings.cpu().numpy()
                np_graph_emb = accumulated_class_latent_embeddings.cpu().numpy()

                n_rho, n_pval = stats.spearmanr(np_entangled, np_node_emb)
                g_rho, g_pval = stats.spearmanr(np_entangled, np_graph_emb)

                savetxt('node_rho.csv', n_rho, delimiter=',')
                savetxt('node_p.csv', n_pval, delimiter=',')
                savetxt('graph_rho.csv', g_rho, delimiter=',')
                savetxt('graph_pval.csv', g_pval, delimiter=',')'''

                savetxt('node.csv', sim_node, delimiter=',')
                savetxt('graph.csv', graph_node, delimiter=',')


                break

        return None

if __name__ == '__main__':

    args = arg_parse()

    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    batch_size = 1
    lr = args.lr
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    node_ratio = 0.5
    node_dim = int(args.hidden_dim*2*node_ratio)
    class_dim = int(args.hidden_dim*2 - node_dim)

    dataset = TUDataset(path, name=DS).shuffle()
    try:
        dataset_num_features = dataset.num_features
    except:
        dataset_num_features = 1

    if not dataset_num_features:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GcnInfomax(args.hidden_dim, args.num_gc_layers, node_dim, class_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.load_state_dict(torch.load(f'mutag_best_model.pkl'))

    model.eval()
    _ = model.get_embeddings(dataloader)
