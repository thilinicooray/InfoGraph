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
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch

from scipy import stats
from numpy import savetxt

from arguments import arg_parse
from graph_gen import SyntheticERDataset
from graph_gen_random_nodeval import SyntheticER_N_Dataset
import networkx as nx
import matplotlib.pyplot as plt

import json

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


        y = []
        i = 0
        '''prob = [0.0000, 0.0500, 0.1000, 0.1500, 0.2000, 0.2500, 0.3000, 0.3500, 0.4000,
                0.4500, 0.5000, 0.5500, 0.6000, 0.6500, 0.7000, 0.7500, 0.8000, 0.8500,
                0.9000, 0.9500, 1.0000]
        labels = [0,1,2,3,4,5,6]'''

        prob = [0.0, 0.1,0.3, 0.5, 0.7, 0.9, 1.0]
        labels = [0,1,2,3,4,5,6]

        n = 10
        tot_edges = (n* (n-1))//2
        print('tot edges ', tot_edges)

        global_rep = []
        regen_p = []
        regen_adj = []
        org_p = []

        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)

                node_mu, node_logvar, class_mu, class_logvar = self.encoder(x, edge_index, batch)


                node_latent_embeddings = reparameterize(training=False, mu=node_mu, logvar=node_logvar)

                grouped_mu, grouped_logvar = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, batch, True
                )

                class_latent_embeddings = group_wise_reparameterize(
                    training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
                )

                reconstructed_node = self.decoder(node_latent_embeddings, class_latent_embeddings)

                a, idx_tensor = to_dense_batch(reconstructed_node, batch)
                a_t = a.permute(0, 2, 1)
                rec_adj = torch.bmm(a, a_t)

                np_rec_adj = np.squeeze(rec_adj.cpu().numpy())
                np_rec_adj_lin = np_rec_adj.reshape((-1,1))

                from sklearn.preprocessing import MinMaxScaler
                scaler = MinMaxScaler()
                scaler.fit(np_rec_adj_lin)
                long_adj = scaler.transform(np_rec_adj_lin)

                scaled_adj = long_adj.reshape((10,10))
                org_adj = to_dense_adj(edge_index, batch)

                best_cut_off ={}
                best_prob = 0

                label = data.y.item()
                #print('label', label)
                pr = prob[label]

                cut_best = 0
                dif_best = 1000
                p_best = 0
                our_adj = 0

                cut_values = []

                if pr != 1.0:

                    for k in range(10):
                        for m in range(10):
                            current_cut = np_rec_adj[k][m]
                            mask = (np_rec_adj > current_cut).astype(int)

                            p_gen = np.sum(mask)/tot_edges

                            err = np.power(pr - p_gen,2)

                            if err < dif_best:
                                dif_best = err
                                cut_best = current_cut
                                p_best = p_gen
                                our_adj = mask

                else:
                    cut_best = np_rec_adj[0][0]
                    p_best = 1.0
                    our_adj = np.ones((10,10), dtype=int)


                    #cut_values.append({'cut':str(current_cut), 'p':str(p_gen)})
                    #mse =

                #print('recon adj matrix' , rec_adj)

                #print('scaled recon adj matrix' , scaled_adj)

                #print('org matrix' , org_adj)


                print('ours ', pr, cut_best, p_best, np.sum(our_adj))


                exp = np.empty(1)
                exp.fill(pr)


                exp2 = np.empty(1)
                exp2.fill(p_best)



                global_rep.append(global_mean_pool(class_latent_embeddings, batch).cpu().numpy())
                regen_p.append(exp2)
                regen_adj.append(our_adj)
                org_p.append(exp)


                '''with open('sample_genp.json', 'w') as fout:
                    json.dump(cut_values, fout)'''

                '''G = nx.from_numpy_matrix(our_adj)
                nx.draw_circular(G)
                plt.axis('equal')
                #nx.draw(G)
                plt.savefig('graph_{}_{}{}.pdf'.format(i, str(pr*10), np.sum(our_adj)), bbox_inches='tight')'''

                i+=1



                if i == 10:
                    break


        global_rep = np.concatenate(global_rep, 0)
        regen = np.concatenate(regen_p, 0)
        org = np.concatenate(org_p, 0)
        regen_adj = np.concatenate(regen_adj, 0)

        savetxt('global_rep_tot_50_regen1_small.csv', global_rep, delimiter=',')
        savetxt('regen_p1_small.csv', regen, delimiter=',')
        savetxt('org_p1_small.csv', org, delimiter=',')
        savetxt('regen1_adj_small.csv', regen_adj, delimiter=',')






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

    dataset = SyntheticERDataset(path).shuffle()

    train_dataset = dataset[:3000]
    test_dataset = dataset[3000:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=512)
    test_loader = DataLoader(test_dataset, batch_size=1)

    lr = args.lr
    epochs = 50
    dataset_num_features = 1

    model = GLDisen(2, 2, 2, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)



    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')

    model.load_state_dict(torch.load(f'syner_model_correct2.pkl'))


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



