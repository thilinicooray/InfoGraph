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
from gin_gvae import Encoder, Decoder
from evaluate_embedding import evaluate_embedding
from model import *
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence

from arguments import arg_parse
from scipy import stats
from numpy import savetxt

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
        if x is None:
            x = torch.ones(batch.shape[0]).to(device)

        node_mu, node_logvar = self.encoder(x, edge_index, batch)


        # kl-divergence error for style latent space
        node_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
        )
        node_kl_divergence_loss = node_kl_divergence_loss *num_graphs
        node_kl_divergence_loss.backward(retain_graph=True)

        # kl-divergence error for class latent space


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
        #check input feat first
        #print('recon ', x[0],reconstructed_node[0])
        reconstruction_error =  0.1*mse_loss(reconstructed_node, x) * num_graphs
        reconstruction_error.backward()


        return reconstruction_error.item() , 0 , node_kl_divergence_loss.item()

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        k = 0
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch

                node_mu, node_logvar = self.encoder(x, edge_index, batch)

                latent_embeddings = reparameterize(training=False, mu=node_mu, logvar=node_logvar)
                print('latent size', latent_embeddings.size())

                node_latent_embeddings = latent_embeddings[:,:32]
                indiclass_latent_embeddings = latent_embeddings[:,32:]


                '''corre_matrix = np.ndarray(shape=(indiclass_latent_embeddings.shape[0],node_latent_embeddings.shape[0]))

                from scipy.stats.stats import pearsonr, spearmanr

                for i in range(indiclass_latent_embeddings.shape[0]):
                    for j in range(node_latent_embeddings.shape[0]):
                        #print('sizes ', word_rep[i].shape, global_rep[j].shape)
                        corre_matrix[i][j] = spearmanr(node_latent_embeddings[j],indiclass_latent_embeddings[i])[0]'''


                n_rho, n_pval = stats.spearmanr(torch.cat([node_latent_embeddings,indiclass_latent_embeddings],0) .cpu().numpy(), axis=1)
                savetxt('mutag_gvae_graph_rho_again1_{}.csv'.format(k), n_rho, delimiter=',')

                k+=1

                '''k +=1'''

                if k == 20:
                    break

                    #break

        return None

if __name__ == '__main__':

    args = arg_parse()


    seed = 42
    #epochs = 20

    epochs_list = [30]


    print('seed ', seed)


    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('init seed, seed ', torch.initial_seed(), seed)

    accuracies = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}

    log_interval = 1
    batch_size = 1
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
    model = GcnInfomax(args.hidden_dim, args.num_gc_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model.load_state_dict(torch.load(f'mutag_gvae_model_2.pkl'))

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
    _ = model.get_embeddings(dataloader)


