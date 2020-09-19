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

from scipy import stats
from numpy import savetxt

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


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        k = 0
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch

                node_mu, node_logvar, class_mu, class_logvar, entangled = self.encoder(x, edge_index, batch)

                #print('class mu ', torch.softmax(class_mu, dim=-1)[:5,:5])

                node_latent_embeddings = reparameterize(training=False, mu=node_mu, logvar=node_logvar)
                indiclass_latent_embeddings = reparameterize(training=False, mu=class_mu, logvar=class_logvar)

                print('node size ', k, node_latent_embeddings.size(0))

                savetxt('node_emb.csv', node_latent_embeddings.cpu().numpy(), delimiter=',')
                savetxt('graph_emb.csv', indiclass_latent_embeddings.cpu().numpy(), delimiter=',')


                grouped_mu, grouped_logvar = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, batch, True
                )


                #print('div ', torch.dist(node_latent_embeddings[0], node_latent_embeddings[2],2), torch.dist(indiclass_latent_embeddings[0], indiclass_latent_embeddings[2],2))


                #print('div ', torch.dist(node_latent_embeddings[2], node_latent_embeddings[15],2), torch.dist(indiclass_latent_embeddings[2], indiclass_latent_embeddings[15],2))

                #print('div ', torch.dist(node_latent_embeddings[3], node_latent_embeddings[6],2), torch.dist(indiclass_latent_embeddings[3], indiclass_latent_embeddings[6],2))

                accumulated_class_latent_embeddings = group_wise_reparameterize(
                    training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
                )

                sim_node = torch.sigmoid(torch.matmul(class_mu.t(), node_mu)).cpu().numpy()
                graph_node = torch.sigmoid(torch.matmul(class_mu.t(), class_mu)).cpu().numpy()

                #cov = np.corrcoef(class_mu.cpu().numpy())

                #print('size ',  cov.shape)

                #savetxt('node.csv', sim_node, delimiter=',')
                #savetxt('graph.csv', cov, delimiter=',')

                n_rho, n_pval = stats.spearmanr(torch.cat([node_latent_embeddings,indiclass_latent_embeddings],0) .cpu().numpy(), axis=1)
                print('corr shape ', n_rho.shape)
                savetxt('graph_rho.csv', n_rho, delimiter=',')

                n_rho, n_pval = stats.spearmanr(torch.cat([class_mu,node_mu],0).cpu().numpy(), axis=1)
                print('corr shape ', n_rho.shape)
                savetxt('node_rho.csv', n_rho, delimiter=',')

                '''np_entangled = torch.sigmoid(entangled).cpu().numpy()
                np_node_emb = torch.sigmoid(node_mu).cpu().numpy()
                np_graph_emb = torch.sigmoid(grouped_mu).cpu().numpy()

                print('start corr ', np_entangled.shape, np_graph_emb.shape)

                n_rho, n_pval = stats.spearmanr(np_node_emb, np_entangled, axis=0 )
                print('start graph')
                g_rho, g_pval = stats.spearmanr(np_graph_emb,np_entangled, axis=0)

                print('sizes ', n_rho.shape, g_rho.shape)

                savetxt('node_rho.csv', n_rho, delimiter=',')
                savetxt('node_p.csv', n_pval, delimiter=',')
                savetxt('graph_rho.csv', g_rho, delimiter=',')
                savetxt('graph_pval.csv', g_pval, delimiter=',')'''

                k+=1

                '''k +=1

                if k == 56:
                    break'''

                #break

        return None

    def compute_two_gaussian_loss(self, mu1, logvar1, mu2, logvar2):
        """Computes the KL loss between the embedding attained from the answers
        and the categories.
        KL divergence between two gaussians:
            log(sigma_2/sigma_1) + (sigma_2^2 + (mu_1 - mu_2)^2)/(2sigma_1^2) - 0.5
        Args:
            mu1: Means from first space.
            logvar1: Log variances from first space.
            mu2: Means from second space.
            logvar2: Means from second space.
        """
        numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
        fraction = torch.div(numerator, (logvar2.exp() + 1e-8))
        kl = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1)
        return kl / (mu1.size(0) + 1e-8)

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

    #print('model ', model.encoder.convs[2].nn[0].weight.size())

    '''n_mu = model.encoder.convs[2].nn[0].weight
    n_logv = model.encoder.convs[3].nn[0].weight
    g_mu = model.encoder.convs[4].nn[0].weight
    g_logv = model.encoder.convs[5].nn[0].weight

    savetxt('n_mu.csv', n_mu.detach().cpu().numpy(), delimiter=',')
    savetxt('n_lv.csv', n_logv.detach().cpu().numpy(), delimiter=',')
    savetxt('g_mu.csv', g_mu.detach().cpu().numpy(), delimiter=',')
    savetxt('g_lv.csv', g_logv.detach().cpu().numpy(), delimiter=',')'''



    model.eval()
    _ = model.get_embeddings(dataloader)
