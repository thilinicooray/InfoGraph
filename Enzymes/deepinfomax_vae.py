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

from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn import preprocessing

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder, Decoder
from evaluate_embedding import evaluate_embedding, draw_plot
from model import *
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence, \
    accumulate_group_rep, expand_group_rep

from sklearn.linear_model import LogisticRegression

from arguments import arg_parse

class D_net_gauss(nn.Module):
    def __init__(self,N,z_dim):
        super(D_net_gauss, self).__init__()
        self.lin1 = nn.Linear(z_dim, N)
        self.lin2 = nn.Linear(N, N)
        self.lin3 = nn.Linear(N, 1)
    def forward(self, x):
        x = self.lin1(x)
        '''x = F.relu(x)
        x = F.dropout(self.lin2(x), p=0.0, training=self.training)'''
        x = F.relu(x)
        return torch.sigmoid(self.lin3(x))

class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)
        self.decoder = Decoder(hidden_dim, hidden_dim, dataset_num_features)
        self.node_discriminator = D_net_gauss(hidden_dim, hidden_dim)
        self.class_discriminator = D_net_gauss(hidden_dim, hidden_dim)



        self.init_emb()

    def init_emb(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)


    def forward(self, x, edge_index, batch, num_graphs):



        n_nodes = x.size(0)

        # batch_size = data.num_graphs

        node_mu, node_logvar, class_mu, class_logvar = self.encoder(x, edge_index, batch)





        grouped_mu, grouped_logvar = accumulate_group_evidence(
            class_mu.data, class_logvar.data, batch, True
        )




        # kl-divergence error for style latent space
        '''node_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
        )'''

        node_kl_divergence_loss = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * node_logvar - node_mu.pow(2) - node_logvar.exp().pow(2), 1))


        node_kl_divergence_loss = node_kl_divergence_loss


        # kl-divergence error for class latent space
        '''class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )'''
        class_kl_divergence_loss = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp().pow(2), 1))

        #print('class kl unwei ', class_kl_divergence_loss)
        class_kl_divergence_loss = class_kl_divergence_loss
        #print('class kl wei ', class_kl_divergence_loss)


        # reconstruct samples
        """
        sampling from group mu and logvar for each graph in mini-batch differently makes
        the decoder consider class latent embeddings as random noise and ignore them 
        """
        node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)
        class_latent_embeddings = group_wise_reparameterize(
            training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
        )


        reconstructed_node = self.decoder(node_latent_embeddings, class_latent_embeddings)

        #reconstruction_error =  mse_loss(reconstructed_node, x) * num_graphs
        reconstruction_error = self.recon_loss1(reconstructed_node, edge_index, batch)


        #class_kl_divergence_loss.backward(retain_graph=True)
        #node_kl_divergence_loss.backward(retain_graph=True)
        #reconstruction_error.backward()

        loss =  class_kl_divergence_loss + node_kl_divergence_loss + reconstruction_error

        loss.backward()


        #return  reconstruction_error.item(), class_kl_divergence_loss.item() , node_kl_divergence_loss.item()
        return loss.item()


    def edge_recon(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value



    def recon_loss1(self, z, edge_index, batch):

        EPS = 1e-15
        MAX_LOGSTD = 10
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
  
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        #org_adj = to_dense_adj(edge_index, batch)
        pos_weight = float(z.size(0) * z.size(0) - edge_index.size(0)) / edge_index.size(0)
        norm = z.size(0) * z.size(0) / float((z.size(0) * z.size(0) - edge_index.size(0)) * 2)



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

        return norm*(pos_loss*pos_weight + neg_loss)

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret_node = []
        y_node = []
        ret_class = []
        y_class = []
        with torch.no_grad():
            for data in loader:

                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch


                node_mu, node_logvar, class_mu, class_logvar, entangledrep = self.encoder(x[:,:18], edge_index, batch)


                node_latent_embeddings = reparameterize(training=False, mu=node_mu, logvar=node_logvar)

                grouped_mu, grouped_logvar = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, batch, True
                )

                accumulated_class_latent_embeddings = group_wise_reparameterize(
                    training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
                )

                class_emb = global_mean_pool(accumulated_class_latent_embeddings, batch)


                ret_node.append(node_latent_embeddings.cpu().numpy())
                node_label_idx = (data.x[:,18:] != 0).nonzero()[:,1]
                y_node.append(node_label_idx.cpu().numpy())
                ret_class.append(class_emb.cpu().numpy())
                y_class.append(data.y.cpu().numpy())

        ret_node = np.concatenate(ret_node, 0)
        y_node = np.concatenate(y_node, 0)
        ret_class = np.concatenate(ret_class, 0)
        y_class = np.concatenate(y_class, 0)
        return ret_node, y_node, ret_class, y_class

def test(train_z, train_y, val_z, val_y,test_z, test_y,  solver='lbfgs',
         multi_class='ovr', *args, **kwargs):
    r"""Evaluates latent space quality via a logistic regression downstream
    task."""

    log_reg = LogisticRegression(solver=solver, multi_class=multi_class, max_iter=15000)
    clf = MultiOutputClassifier(log_reg)

    scaler = preprocessing.StandardScaler().fit(train_z)

    updated = scaler.transform(train_z)


    clf.fit(updated,train_y)

    predict_val = clf.predict(scaler.transform(val_z))

    micro_f1_val = f1_score(val_y, predict_val, average='micro')

    predict_test = clf.predict(scaler.transform(test_z))

    micro_f1_test = f1_score(test_y, predict_test, average='micro')

    return micro_f1_val, micro_f1_test


class SimpleClassifier(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, dropout):
        super(SimpleClassifier, self).__init__()
        layers = [
            #nn.Linear(in_dim, hid_dim),
            #nn.ReLU(),
            #nn.Dropout(dropout, inplace=True),
            nn.Linear(in_dim, out_dim)
        ]
        self.main = nn.Sequential(*layers)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        logits = self.main(x)
        return logits

if __name__ == '__main__':

    args = arg_parse()

    #for seed in [32,42,52,62,72]:
    for seed in [52]:

        #seed = 42
        #epochs = 37
        epochs = int(args.num_epochs)

        #for epochs in range(20,41):

        print('seed ', seed, 'epochs ', epochs)


        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(seed)

        print('init seed, seed ', torch.initial_seed(), seed)

        accuracies_node = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}
        accuracies_class = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}

        #losses = {'recon':[], 'node_kl':[], 'class_kl': []}
        losses = []

        warmup_steps = 10
        #batch_size = 128
        batch_size = args.batch_size
        lr = args.lr
        gen_lr = 1 * lr
        reg_lr = 0.5 * lr

        EPS = 1e-15

        #lr = 0.000001
        DS = args.DS
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        #dataset = TUDataset(path, name=DS, pre_transform = torch_geometric.transforms.OneHotDegree(max_degree=88)).shuffle()
        dataset = TUDataset(path, use_node_attr=True, name=DS).shuffle()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            dataset_num_features = 18
        except:
            dataset_num_features = 1

        if not dataset_num_features:

            dataset_num_features = 5

            #dataset_num_features = 5
            #input_feat = torch.ones((batch_size, 1)).to(device)

        dataloader = DataLoader(dataset, batch_size=batch_size)


        model = GcnInfomax(args.hidden_dim, args.num_gc_layers).double().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)



        print('================')
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

        #model.train()
        for epoch in range(1, epochs+1):
            tot_loss = 0
            recon_loss_all = 0
            kl_class_loss_all = 0
            kl_node_loss_all = 0
            mi_loss_all = 0
            model.train()
            for data in dataloader:
                data = data.to(device)

                optimizer.zero_grad()
                #recon_loss, kl_class, kl_node = model(data.x[:,:18], data.edge_index, data.batch, data.num_graphs)
                current_loss = model(data.x[:,:18], data.edge_index, data.batch, data.num_graphs)
                '''recon_loss_all += recon_loss
                kl_class_loss_all += kl_class
                kl_node_loss_all += kl_node'''
                tot_loss += current_loss



                '''for name, param in model.named_parameters():
                    print(name, param.grad)'''




                #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()

            '''losses['recon'].append(recon_loss_all/ len(dataloader))
            losses['node_kl'].append(kl_node_loss_all/ len(dataloader))
            losses['class_kl'].append(kl_class_loss_all/ len(dataloader))'''
            losses.append(tot_loss)



            '''print('Epoch {}, Recon Loss {} KL class Loss {} KL node Loss {}'.format(epoch, recon_loss_all / len(dataloader),
                                                                                    kl_class_loss_all / len(dataloader), kl_node_loss_all / len(dataloader)))'''
            print('Epoch {}, Loss {}'.format(epoch, tot_loss / len(dataloader)))

            #used during finetune phase
            if epoch > warmup_steps :
                model.eval()

                emb_node, y_node, emb_class, y_class = model.get_embeddings(dataloader)
                print('node classificaion')
                res = evaluate_embedding(emb_node, y_node)
                accuracies_node['logreg'].append(res[0])
                accuracies_node['svc'].append(res[1])
                accuracies_node['linearsvc'].append(res[2])
                accuracies_node['randomforest'].append(res[3])
                print('node ', accuracies_node)
                print('train_loss', losses)
                '''print('graph classificaion')
                res = evaluate_embedding(emb_class, y_class)
                accuracies_class['logreg'].append(res[0])
                accuracies_class['svc'].append(res[1])
                accuracies_class['linearsvc'].append(res[2])
                accuracies_class['randomforest'].append(res[3])
                print('class ', accuracies_node)'''









        #draw_plot(y, emb, 'imdb_b_normal.png')

        '''with open('unsupervised.log', 'a+') as f:
            s = json.dumps(accuracies)
            f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s))'''
