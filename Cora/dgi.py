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
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch
from torch.nn import Sequential, Linear, ReLU, Tanh, Sigmoid, PReLU
import sys
import json
from torch import optim

from torch_geometric.nn import GINConv, global_add_pool, GCNConv

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gcn_mlaae import Encoder, Decoder
from evaluate_embedding import evaluate_embedding, draw_plot
from model import *
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence, \
    accumulate_group_rep, expand_group_rep

from sklearn.linear_model import LogisticRegression

from arguments import arg_parse

import torch
import torch.nn as nn
from layers import GCN, AvgReadout, Discriminator

class DGI(nn.Module):
    def __init__(self, n_in, n_h):
        super(DGI, self).__init__()
        self.gcn = GCNConv(n_in, n_h)
        self.read = AvgReadout()
        self.act = PReLU()
        self.bn = torch.nn.BatchNorm1d(n_h)

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, x, x_permute, edge_index):
        h_1 = self.bn(self.act(self.gcn(x, edge_index)))

        c = self.read(h_1, None)
        c = self.sigm(c)

        h_2 = self.act(self.gcn(x_permute, edge_index))

        ret = self.disc(c, h_1, h_2)

        return ret

    # Detach the return variables
    def embed(self, x, edge_index):
        h_1 = self.act(self.gcn(x, edge_index))
        c = self.read(h_1, None)

        return h_1.detach(), c.detach()

    def get_embeddings(self, data):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():

            data.to(device)
            x, edge_index = data.x, data.edge_index


            node_latent, _ = self.embed(x.double(), edge_index)

        #node_latent = node_latent.cpu().numpy()
        #y = da

        train_emb = node_latent[data.train_mask].cpu().numpy()
        train_y = data.y[data.train_mask].cpu().numpy()
        val_emb = node_latent[data.val_mask].cpu().numpy()
        val_y = data.y[data.val_mask].cpu().numpy()
        test_emb = node_latent[data.test_mask].cpu().numpy()
        test_y = data.y[data.test_mask].cpu().numpy()

        '''train_msk_exp = data.train_mask.unsqueeze(-1).expand_as(node_latent)
        val_msk_exp = data.val_mask.unsqueeze(-1).expand_as(node_latent)
        test_msk_exp = data.test_mask.unsqueeze(-1).expand_as(node_latent)


        train_emb = torch.masked_select(node_latent, train_msk_exp).cpu().numpy()
        train_y = torch.masked_select(data.y, data.train_mask).cpu().numpy()
        val_emb = torch.masked_select(node_latent, val_msk_exp).cpu().numpy()
        val_y = torch.masked_select(data.y, data.val_mask).cpu().numpy()
        test_emb = torch.masked_select(node_latent, test_msk_exp).cpu().numpy()
        test_y = torch.masked_select(data.y, data.test_mask).cpu().numpy()'''



        return train_emb, train_y, val_emb, val_y,test_emb, test_y



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

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


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

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        losses = {'recon':[], 'gen':[], 'disc': []}

        log_interval = 10
        #batch_size = 128
        batch_size = args.batch_size
        lr = args.lr
        gen_lr = 1 * lr
        reg_lr = 0.5 * lr

        EPS = 1e-15

        #lr = 0.000001
        DS = 'Cora'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        dataset = Planetoid(path, name=DS)

        print('cora dataset summary', dataset[0])
        data = dataset[0].to(device)



        try:
            dataset_num_features = data.x.size(-1)
        except:
            dataset_num_features = 1

        if not dataset_num_features:

            dataset_num_features = 5

            #dataset_num_features = 5
            #input_feat = torch.ones((batch_size, 1)).to(device)

        train_x = data.x[data.train_mask]
        train_y = data.y[data.train_mask]
        val_x = data.x[data.val_mask]
        val_y = data.y[data.val_mask]
        test_x = data.x[data.test_mask]
        test_y = data.y[data.test_mask]

        nb_classes = np.unique(data.y.cpu().numpy()).shape[0]


        model = model = DGI(dataset_num_features, args.hidden_dim).double().to(device)
        #encode/decode optimizers
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)



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

        logreg_val = []
        logreg_valbased_test = []

        best_val_round = -1
        best_val = 0

        xent = nn.CrossEntropyLoss()
        b_xent = nn.BCEWithLogitsLoss()

        #model.train()
        for epoch in range(1, epochs+1):
            recon_loss_all = 0
            gen_loss_all = 0
            disc_loss_all = 0
            mi_loss_all = 0
            model.train()


            idx = np.random.permutation(data.x.size(0))
            shuf_fts = torch.from_numpy(data.x.cpu().numpy()[idx, :]).double()

            lbl_1 = torch.ones(data.x.size(0))
            lbl_2 = torch.zeros(data.x.size(0))
            lbl = torch.cat((lbl_1, lbl_2), 0)

            if torch.cuda.is_available():
                shuf_fts = shuf_fts.cuda()
                lbl = lbl.cuda()

            logits = model(data.x.double(), shuf_fts, data.edge_index)

            loss = b_xent(logits, lbl)

            print('Current Loss:', loss.item())

            losses['recon'].append(loss.item())

            print('all losses ', losses)


            model.eval()

            #if epoch == epochs:




            accs = []
            best_f1 = 0
            best_round = 0

            print('Logistic regression started!')


            train_emb, train_y_labels, val_emb, val_y_labels,test_emb, test_y_labels  = model.get_embeddings(data)

            train_emb, train_lbls = torch.from_numpy(train_emb).cuda(), torch.from_numpy(train_y_labels).cuda()
            val_emb, val_lbls= torch.from_numpy(val_emb).cuda(), torch.from_numpy(val_y_labels).cuda()
            test_emb, test_lbls= torch.from_numpy(test_emb).cuda(), torch.from_numpy(test_y_labels).cuda()

            #print('emb', train_emb.size(), val_emb.size(), test_emb.size())
            #print('y',  train_lbls.size(), val_lbls.size(), test_lbls.size())

            '''from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            scaler.fit(train_emb)
            train_emb = scaler.transform(train_emb)
            val_emb = scaler.transform(val_emb)
            test_emb = scaler.transform(test_emb)'''



            '''from sklearn.linear_model import SGDClassifier
            from sklearn.metrics import accuracy_score
            log = SGDClassifier(loss="log")
            log.fit(train_emb, train_y)


            val_pred = log.predict(val_emb)
            test_pred = log.predict(test_emb)

            tot_acc_val = accuracy_score(val_y.flatten(), val_pred.flatten())

            tot_acc_test = accuracy_score(test_y.flatten(), test_pred.flatten())

            if tot_acc_val > best_val:
                best_val_round = epoch - 1'''

            accs_val = []
            accs_test = []
            for _ in range(50):
                log = LogReg(args.hidden_dim, nb_classes).double().cuda()
                opt = torch.optim.Adam(log.parameters(), lr=1e-2, weight_decay=0)
                log.cuda()
                current_val_best = 0
                current_best_iter = 0
                current_val_list = []
                current_test_list = []

                for iter in range(300):
                    log.train()
                    opt.zero_grad()

                    logits = log(train_emb)
                    loss = xent(logits, train_lbls)

                    loss.backward()
                    opt.step()

                logits_test = log(test_emb)
                preds_test = torch.argmax(logits_test, dim=1)
                acc_test = torch.sum(preds_test == test_lbls).float() / test_lbls.shape[0]
                #current_test_list.append(acc_test)


                logits_val = log(val_emb)
                preds_val = torch.argmax(logits_val, dim=1)
                acc_val = torch.sum(preds_val == val_lbls).float() / val_lbls.shape[0]
                #current_val_list.append(acc_val)


                '''if acc_val.item() > current_val_best:
                        current_best_iter = iter'''


                #accs_test.append(current_val_list[current_best_iter] * 100)
                #accs_val.append(current_test_list[current_best_iter] * 100)

                accs_test.append(acc_test * 100)
                accs_val.append(acc_val * 100)

            accs_test = torch.stack(accs_test)
            print('test ', accs_test.mean().item(), accs_test.std().item())

            accs_val = torch.stack(accs_val)
            print('val ', accs_val.mean().item(), accs_val.std().item())

            if accs_val.mean().item() > best_val:
                best_val_round = epoch - 1


            logreg_val.append(accs_val.mean().item())
            logreg_valbased_test.append(accs_test.mean().item())

            print('logreg val', logreg_val)
            print('logreg test', logreg_valbased_test)

        print('best perf based on validation score val, test, in epoch', logreg_val[best_val_round], logreg_valbased_test[best_val_round], best_val_round)


