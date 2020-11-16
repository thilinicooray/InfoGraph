import os.path as osp
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import random
import os
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

from scipy import stats
from numpy import savetxt

from arguments import arg_parse
from graph_gen import SyntheticERDataset

class GLDisen(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, node_dim, class_dim):
        super(GLDisen, self).__init__()


        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, node_dim, class_dim)
        self.decoder = Decoder(node_dim, class_dim, dataset_num_features)

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


        return reconstruction_error.item() + class_kl_divergence_loss.item() + node_kl_divergence_loss.item()

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        i = 0
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)

                node_mu, node_logvar, class_mu, class_logvar = self.encoder(x[:,:2], edge_index, batch)


                node_latent_embeddings_org = reparameterize(training=False, mu=node_mu, logvar=node_logvar)

                indiclass_latent_embeddings_org = reparameterize(training=False, mu=class_mu, logvar=class_logvar)

                # global change, local fix
                input_g = None
                input_l = None
                z_g = None
                z_l= None
                fractions_list = torch.arange(0, 1.2, 0.2)
                print('fraction ', fractions_list)
                for frac in fractions_list:

                    x_new = torch.cat([(x[:,0]).unsqueeze(-1),(x[:,1]*frac).unsqueeze(-1)],-1)
                    print('x ', x.size(), x_new.size())

                    print('data', x, x_new)

                    if input_g is None:
                        input_g = x_new[:,1]
                    else:
                        input_g = torch.cat((input_g.clone(), x_new[:,1]), 0)

                    if input_l is None:
                        input_l = x_new[:,0]
                    else:
                        input_l = torch.cat((input_l.clone(), x_new[:,0]), 0)

                    node_mu, node_logvar, class_mu, class_logvar = self.encoder(x_new, edge_index, batch)


                    node_latent_embeddings = reparameterize(training=False, mu=node_mu, logvar=node_logvar)

                    indiclass_latent_embeddings = reparameterize(training=False, mu=class_mu, logvar=class_logvar)

                    if z_g is None:
                        z_g = indiclass_latent_embeddings
                    else:
                        z_g = torch.cat((z_g.clone(), (indiclass_latent_embeddings)), 0)

                    if z_l is None:
                        z_l = node_latent_embeddings[:,0]
                    else:
                        z_l = torch.cat((z_l.clone(), (node_latent_embeddings[:,0])), 0)


                savetxt('synth_inputg_{}.csv'.format('gfix'), input_g.cpu().numpy(), delimiter=',')
                savetxt('synth_inputl_{}.csv'.format('gfix'), input_l.cpu().numpy(), delimiter=',')
                savetxt('synth_zg_{}.csv'.format('gfix'), z_g.cpu().numpy(), delimiter=',')
                savetxt('synth_zl_{}.csv'.format('gfix'), z_l.cpu().numpy(), delimiter=',')

                i += 1

                '''if i == 6:
                    break'''
                break

        return None

if __name__ == '__main__':

    args = arg_parse()

    seed = 1234

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'SyntheticER')

    dataset = SyntheticERDataset(path)

    train_dataset = dataset[:3000]
    test_dataset = dataset[3000:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=512)
    test_loader = DataLoader(test_dataset, batch_size=1)

    lr = args.lr
    epochs = 50
    dataset_num_features = 2

    model = GLDisen(2, 2, 1, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)



    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    model.load_state_dict(torch.load(f'syner_model10.pkl'))


    '''model.train()
    for epoch in range(1, epochs+1):
        loss_all = 0

        #model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
            loss_all += loss
            optimizer.step()

            #losses['tot'].append(loss_all/ len(dataloader))


        print('Epoch {}, Total Loss {} '.format(epoch, loss_all/ len(train_loader)))

    torch.save(model.state_dict(), f'syner_model1.pkl')'''


    model.eval()

    _ = model.get_embeddings(test_loader)


