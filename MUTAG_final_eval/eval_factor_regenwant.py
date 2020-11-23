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
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch

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
        node_all_recon_error_all = 0
        node_node_only_recon_all = 0
        node_graph_only_recon_all = 0
        adj_all_recon_all = 0
        adj_node_only_recon_all = 0
        adj_graph_only_recon_all = 0
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


                reconstructed_all = self.decoder(node_latent_embeddings, accumulated_class_latent_embeddings)
                reconstructed_node_only = self.decoder(node_latent_embeddings,
                                                       torch.zeros_like(accumulated_class_latent_embeddings).to(device))
                reconstructed_graph_only = self.decoder(torch.zeros_like(node_latent_embeddings).to(device), accumulated_class_latent_embeddings)

                node_all_recon_error = mse_loss(reconstructed_all, x)
                node_node_only_recon_error = mse_loss(reconstructed_node_only, x)
                node_graph_only_recon_error = mse_loss(reconstructed_graph_only, x)

                adj_all_recon_error = self.recon_loss(reconstructed_all, edge_index, batch)
                adj_node_only_recon_error = self.recon_loss(reconstructed_node_only, edge_index, batch)
                adj_graph_only_recon_error = self.recon_loss(reconstructed_graph_only, edge_index, batch)

                node_all_recon_error_all += node_all_recon_error.item()
                node_node_only_recon_all += node_node_only_recon_error.item()
                node_graph_only_recon_all += node_graph_only_recon_error.item()
                adj_all_recon_all += adj_all_recon_error.item()
                adj_node_only_recon_all += adj_node_only_recon_error.item()
                adj_graph_only_recon_all += adj_graph_only_recon_error.item()


        return node_all_recon_error_all/ len(loader), node_node_only_recon_all/ len(loader), node_graph_only_recon_all/ len(loader), adj_all_recon_all/ len(loader), adj_node_only_recon_all/ len(loader), adj_graph_only_recon_all/ len(loader)

    def recon_loss(self, z, edge_index, batch):

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


        #recon_adj = self.edge_recon(z, edge_index)


        a, idx_tensor = to_dense_batch(z, batch)
        a_t = a.permute(0, 2, 1)

        #print('batch size', a.size(), a_t.size())

        rec = torch.bmm(a, a_t)

        #print('inner pro', rec.size())

        org_adj = to_dense_adj(edge_index, batch)


        pos_weight = float(z.size(0) * z.size(0) - org_adj.sum()) / org_adj.sum()
        norm = z.size(0) * z.size(0) / float((z.size(0) * z.size(0) - org_adj.sum()) * 2)


        #print('new' ,rec, 'org', org_adj)


        '''#adj = torch.matmul(z, z.t())
  
  
        #r_adj = to_dense_adj(recon_adj, batch)
        #org_adj = to_dense_adj(edge_index, batch)
  
        #print(r_adj.size(), org_adj.size())
  
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
  
        return pos_loss + neg_loss'''

        loss = norm * F.binary_cross_entropy_with_logits(rec, org_adj, pos_weight=pos_weight)

        return loss

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

    batch_size = 128
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

    model.load_state_dict(torch.load(f'mutag_best_model_128_new.pkl'))

    #print('model ', model.encoder.convs[2].nn[0].weight.size())

    '''n_mu = model.encoder.convs[2].nn[0].weight
    n_logv = model.encoder.convs[3].nn[0].weight
    g_mu = model.encoder.convs[4].nn[0].weight
    g_logv = model.encoder.convs[5].nn[0].weight

    savetxt('n_mu.csv', n_mu.detach().cpu().numpy(), delimiter=',')
    savetxt('n_lv.csv', n_logv.detach().cpu().numpy(), delimiter=',')
    savetxt('g_mu.csv', g_mu.detach().cpu().numpy(), delimiter=',')
    savetxt('g_lv.csv', g_logv.detach().cpu().numpy(), delimiter=',')'''

    '''t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [4, 2, 1], [5, 3, 6]], dtype=torch.float)
    index = torch.tensor([0, 0,0,1,1, 2,3,3,3,4])

    output, count = torch.unique(index,  return_counts=True)

    print('unique ', output, count)

    new = torch.repeat_interleave(t, count, dim=0)

    print('repeated ', new)'''



    model.eval()
    losses = model.get_embeddings(dataloader)

    print('losses ', losses)
