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

class GcnInfomax(nn.Module):
    def __init__(self, hidden_dim, num_gc_layers, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.embedding_dim = mi_units = hidden_dim * num_gc_layers
        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)
        # self.local_d = MI1x1ConvNet(self.embedding_dim, mi_units)
        # self.global_d = MIFCNet(self.embedding_dim, mi_units)

        if self.prior:
            self.prior_d = PriorDiscriminator(self.embedding_dim)

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

        y, M = self.encoder(x, edge_index, batch)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        mode='fd'
        measure='JSD'
        local_global_loss = local_global_loss_(l_enc, g_enc, edge_index, batch, measure)

        if self.prior:
            prior = torch.rand_like(y)
            term_a = torch.log(self.prior_d(prior)).mean()
            term_b = torch.log(1.0 - self.prior_d(y)).mean()
            PRIOR = - (term_a + term_b) * self.gamma
        else:
            PRIOR = 0

        return local_global_loss + PRIOR



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


        model = GcnInfomax(args.hidden_dim, args.num_gc_layers).double().to(device)
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
                loss = model(data.x, data.edge_index, data.batch, data.num_graphs)
                loss_all += loss.item() * data.num_graphs
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