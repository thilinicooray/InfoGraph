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
        ret_node1 = []
        ret_node2 = []

        global_rep = []
        local1 = []
        local2 = []
        probability = []

        y = []
        i = 0
        prob = [0.0, 0.1,0.3, 0.5, 0.7, 0.9, 1.0]
        labels = [0,1,2,3,4,5,6]
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)

                node_mu, node_logvar, class_mu, class_logvar = self.encoder(x, edge_index, batch)


                #node_latent_embeddings_org = reparameterize(training=False, mu=node_mu, logvar=node_logvar)

                #indiclass_latent_embeddings_org = reparameterize(training=False, mu=class_mu, logvar=class_logvar)
                grouped_mu, grouped_logvar = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, batch, True
                )


                accumulated_class_latent_embeddings = group_wise_reparameterize(
                    training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
                )

                class_emb = global_mean_pool(accumulated_class_latent_embeddings, batch)

                grouped_mu_n, grouped_logvar_n = accumulate_group_evidence(
                    node_mu.data, node_logvar.data, batch, True
                )


                accumulated_class_latent_embeddings_g = group_wise_reparameterize(
                    training=False, mu=grouped_mu_n, logvar=grouped_logvar_n, labels_batch=batch, cuda=True
                )

                class_emb_n = global_mean_pool(accumulated_class_latent_embeddings_g, batch)


                label = data.y.item()
                #print('label', label)
                pr = prob[label]
                #print('pr', pr)

                exp = np.empty(1)
                exp.fill(pr)
                #print('exp', exp)

                global_rep.append(class_emb.cpu().numpy())
                local1.append(class_emb_n[:,0].cpu().numpy())
                local2.append(class_emb_n[:,1].cpu().numpy())
                probability.append(exp)


                '''grouped_mu, grouped_logvar = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, batch, True
                )


                accumulated_class_latent_embeddings = group_wise_reparameterize(
                    training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
                )

                class_emb = torch.tanh(global_mean_pool(accumulated_class_latent_embeddings, batch))
                ret.append(class_emb.cpu().numpy())

                grouped_mu_n, grouped_logvar_n = accumulate_group_evidence(
                    node_mu.data, node_logvar.data, batch, True
                )


                accumulated_class_latent_embeddings_g = group_wise_reparameterize(
                    training=False, mu=grouped_mu_n, logvar=grouped_logvar_n, labels_batch=batch, cuda=True
                )

                class_emb_n1 = torch.tanh(global_mean_pool(accumulated_class_latent_embeddings_g[:,0], batch))
                ret_node1.append(class_emb_n1.cpu().numpy())
                #class_emb_n2 = torch.tanh(global_mean_pool(accumulated_class_latent_embeddings_g[:,1], batch))
                #ret_node2.append(class_emb_n2.cpu().numpy())'''


        global_rep = np.concatenate(global_rep, 0)
        local1 = np.concatenate(local1, 0)
        local2 = np.concatenate(local2, 0)
        probability = np.concatenate(probability, 0)

        savetxt('global_rep_tot.csv', global_rep, delimiter=',')
        savetxt('local1_tot.csv', local1, delimiter=',')
        savetxt('local2_tot.csv', local2, delimiter=',')
        savetxt('probability_tot.csv', probability, delimiter=',')


        '''ret = np.concatenate(ret, 0)
        ret_node1 = np.concatenate(ret_node1, 0)
        #ret_node2 = np.concatenate(ret_node2, 0)
        y = np.concatenate(y, 0)

        savetxt('synthe_graph_emb_1.csv', ret, delimiter=',')
        savetxt('synthe_node_emb_1_1.csv', ret_node1, delimiter=',')
        #savetxt('synthe_node_emb_1_2.csv', ret_node2, delimiter=',')
        savetxt('synthe_graph_y_1.csv', y, delimiter=',')'''



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

    model.load_state_dict(torch.load(f'syner_n_model_correct2_ep30_all3000.pkl'))


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



