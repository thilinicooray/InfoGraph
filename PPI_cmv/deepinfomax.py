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
from torch_geometric.datasets import PPI
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder
from evaluate_embedding import evaluate_embedding, draw_plot
from model import *
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence, \
    accumulate_group_rep, expand_group_rep

from sklearn.linear_model import LogisticRegression

from arguments import arg_parse

class GCNLayer(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCNLayer, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, feat, adj):
        feat = self.fc(feat)
        out = torch.bmm(adj, feat)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, num_layers):
        super(GCN, self).__init__()
        n_h = out_ft
        self.layers = []
        self.num_layers = num_layers
        self.layers.append(GCNLayer(in_ft, n_h).cuda())
        for __ in range(num_layers - 1):
            self.layers.append(GCNLayer(n_h, n_h).cuda())

    def forward(self, feat, adj, mask):
        h_1 = self.layers[0](feat, adj)
        h_1g = torch.sum(h_1, 1)
        for idx in range(self.num_layers - 1):
            h_1 = self.layers[idx + 1](h_1, adj)
            h_1g = torch.cat((h_1g, torch.sum(h_1, 1)), -1)
        return h_1, h_1g


class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)


class Model(nn.Module):
    def __init__(self, n_in, n_h, num_layers):
        super(Model, self).__init__()
        self.mlp1 = MLP(1 * n_h, n_h)
        self.mlp2 = MLP(num_layers * n_h, n_h)
        self.gnn1 = GCN(n_in, n_h, num_layers)
        self.gnn2 = GCN(n_in, n_h, num_layers)

    def forward(self, adj, diff, feat, mask):
        lv1, gv1 = self.gnn1(feat, adj, mask)
        lv2, gv2 = self.gnn2(feat, diff, mask)

        lv1 = self.mlp1(lv1)
        lv2 = self.mlp1(lv2)

        gv1 = self.mlp2(gv1)
        gv2 = self.mlp2(gv2)

        return lv1, gv1, lv2, gv2

    def embed(self, feat, adj, diff, mask):
        __, gv1, __, gv2 = self.forward(adj, diff, feat, mask)
        return (gv1 + gv2).detach()


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_positive_expectation(p_samples, measure, average=True):
    """Computes the positive part of a divergence / difference.
    Args:
        p_samples: Positive samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples

    if average:
        return Ep.mean()
    else:
        return Ep


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def get_negative_expectation(q_samples, measure, average=True):
    """Computes the negative part of a divergence / difference.
    Args:
        q_samples: Negative samples.
        measure: Measure to compute for.
        average: Average the result over samples.
    Returns:
        torch.Tensor
    """
    log_2 = np.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-q_samples) + q_samples
    elif measure == 'JSD':
        Eq = F.softplus(-q_samples) + q_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(q_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(q_samples)
    elif measure == 'RKL':
        Eq = q_samples - 1.
    elif measure == 'H2':
        Eq = torch.exp(q_samples) - 1.
    elif measure == 'W1':
        Eq = q_samples

    if average:
        return Eq.mean()
    else:
        return Eq


# Borrowed from https://github.com/fanyun-sun/InfoGraph
def local_global_loss_(l_enc, g_enc, batch, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g_enc.shape[0]
    num_nodes = l_enc.shape[0]
    max_nodes = num_nodes // num_graphs

    pos_mask = torch.zeros((num_nodes, num_graphs)).cuda()
    neg_mask = torch.ones((num_nodes, num_graphs)).cuda()
    msk = torch.ones((num_nodes, num_graphs)).cuda()
    for nodeidx, graphidx in enumerate(batch):
        pos_mask[nodeidx][graphidx] = 1.
        neg_mask[nodeidx][graphidx] = 0.

    res = torch.mm(l_enc, g_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))
    return E_neg - E_pos


def global_global_loss_(g1_enc, g2_enc, measure):
    '''
    Args:
        l: Local feature map.
        g: Global features.
        measure: Type of f-divergence. For use with mode `fd`
        mode: Loss mode. Fenchel-dual `fd`, NCE `nce`, or Donsker-Vadadhan `dv`.
    Returns:
        torch.Tensor: Loss.
    '''
    num_graphs = g1_enc.shape[0]

    pos_mask = torch.zeros((num_graphs, num_graphs)).cuda()
    neg_mask = torch.ones((num_graphs, num_graphs)).cuda()
    for graphidx in range(num_graphs):
        pos_mask[graphidx][graphidx] = 1.
        neg_mask[graphidx][graphidx] = 0.

    res = torch.mm(g1_enc, g2_enc.t())

    E_pos = get_positive_expectation(res * pos_mask, measure, average=False).sum()
    E_pos = E_pos / num_graphs
    E_neg = get_negative_expectation(res * neg_mask, measure, average=False).sum()
    E_neg = E_neg / (num_graphs * (num_graphs - 1))
    return E_neg - E_pos



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

        accuracies = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}

        losses = {'recon':[], 'node_kl':[], 'class_kl': []}

        log_interval = 10
        #batch_size = 128
        batch_size = args.batch_size
        lr = args.lr
        gen_lr = 1 * lr
        reg_lr = 0.5 * lr

        EPS = 1e-15

        #lr = 0.000001
        DS = 'PPI'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        #dataset = TUDataset(path, name=DS, pre_transform = torch_geometric.transforms.OneHotDegree(max_degree=88)).shuffle()

        train_dataset = PPI(path, split='train').shuffle()
        val_dataset = PPI(path, split='val')
        test_dataset = PPI(path, split='train')





        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            dataset_num_features = train_dataset.num_features
        except:
            dataset_num_features = 1

        if not dataset_num_features:

            dataset_num_features = 5

            #dataset_num_features = 5
            #input_feat = torch.ones((batch_size, 1)).to(device)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size)


        model = Model(dataset_num_features, args.hidden_dim, args.num_gc_layers).double().to(device)
        #encode/decode optimizers
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
            loss_all = 0
            model.train()
            for data in train_dataloader:
                data = data.to(device)


                model.zero_grad()

                data.x = data.x.double()

                optimizer.zero_grad()
                lv1, gv1, lv2, gv2 = model(data.edge_index, diff[batch], data.x)


                loss1 = local_global_loss_(lv1, gv2, data.batch, 'JSD')
                loss2 = local_global_loss_(lv2, gv1, data.batch, 'JSD')
                # loss3 = global_global_loss_(gv1, gv2, 'JSD')
                loss = loss1 + loss2 #+ loss3
                loss_all += loss.item()
                loss.backward()
                optimizer.step()
            print('Epoch {}, Loss {}'.format(epoch, loss_all / len(train_dataloader)))

            #used during finetune phase
            '''if epoch % log_interval == 0:
                model.eval()
                #first train logistic regressor
                #put it eval mode
                #get eval F1
                #get test F1




                train_emb, train_y = model.get_embeddings(train_dataloader)
                val_emb, val_y = model.get_embeddings(val_dataloader)
                test_emb, test_y = model.get_embeddings(test_dataloader)
                val_f1, test_f1 = test(train_emb, train_y, val_emb, val_y,test_emb, test_y)

                print('val and test micro F1', val_f1, test_f1)'''


            model.eval()

            accs = []
            best_f1 = 0
            best_round = 0


            print('Logistic regression started!')

            log = SimpleClassifier(args.hidden_dim*2, args.hidden_dim, 121, 0.5)
            opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=0.0)
            log.double().cuda()


            for round in range(1500):

                log.train()
                for data_new in train_dataloader:

                    opt.zero_grad()
                    data_new = data_new.to(device)

                    _, rep = model.encoder(data_new.x.double(), data_new.edge_index, data_new.batch)

                    logits = log(rep)

                    '''tot = torch.sum(data_new.y, 0)

                    val = 1.0 / tot

                    pos_weight = val'''

                    criterion = nn.BCEWithLogitsLoss()
                    loss = criterion(logits, data_new.y )

                    loss.backward()
                    opt.step()

                log.eval()

                pred_list = []
                y_list = []
                with torch.no_grad():
                    for data_val in val_dataloader:
                        data_val = data_val.to(device)
                        _, rep = model.encoder(data_val.x.double(), data_val.edge_index, data_val.batch)

                        pred = torch.sigmoid(log(rep)) >= 0.5

                        pred_list.append(pred.cpu().numpy())
                        y_list.append(data_val.y.cpu().numpy())
                ret = np.concatenate(pred_list, 0)
                y = np.concatenate(y_list, 0)
                mi_f1 = f1_score(y, ret, average='micro')

                if mi_f1 > best_f1:
                    best_f1 = mi_f1
                    best_round = round

            print('best f1 obtained in round:', best_f1, best_round)

        #accs = torch.stack(accs)
        #print(accs.mean().item(), accs.std().item())'''


        '''model.eval()

        accs = []
        best_f1 = 0
        best_round = 0

        criterion = nn.BCEWithLogitsLoss()
        print('Logistic regression started!')

        log = SimpleClassifier(args.hidden_dim, args.hidden_dim, 121, 0.5)
        opt = torch.optim.Adam(log.parameters(), lr=5e-2, weight_decay=0.0)
        log.double().cuda()

        for round in range(300):

            log.train()
            for data_new in train_dataloader:

                opt.zero_grad()
                data_new = data_new.to(device)

                z_sample, z_class, entangled_rep = model.encoder(data_new.x, data_new.edge_index, data_new.batch)

                logits = log(z_sample)
                loss = criterion(logits, data_new.y)

                loss.backward()
                opt.step()

            log.eval()

            pred_list = []
            y_list = []
            with torch.no_grad():
                for data_val in val_dataloader:
                    data_val = data_val.to(device)
                    z_sample, z_class, entangled_rep = model.encoder(data_val.x, data_val.edge_index, data_val.batch)

                    pred = torch.sigmoid(log(z_sample)) >= 0.5

                    pred_list.append(pred.cpu().numpy())
                    y_list.append(data_val.y.cpu().numpy())
            ret = np.concatenate(pred_list, 0)
            y = np.concatenate(y_list, 0)
            mi_f1  = f1_score(y, ret, average='micro')

            print('current f1 ', round, mi_f1)

            if mi_f1 > best_f1:
                best_f1 = mi_f1
                best_round = round

        print('best f1 obtained in round:', best_f1, best_round)'''



    


        #draw_plot(y, emb, 'imdb_b_normal.png')

        '''with open('unsupervised.log', 'a+') as f:
            s = json.dumps(accuracies)
            f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s))'''
