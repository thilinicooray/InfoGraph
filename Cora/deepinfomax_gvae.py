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
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin_vae import Encoder, Decoder
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


    def forward(self, x, edge_index):



        n_nodes = x.size(0)

        # batch_size = data.num_graphs

        node_mu, node_logvar = self.encoder(x, edge_index)









        # kl-divergence error for style latent space
        '''node_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
        )'''

        node_kl_divergence_loss = -0.5 * torch.mean(torch.sum(
            1 + 2 * node_logvar - node_mu.pow(2) - node_logvar.exp().pow(2), 1))


        node_kl_divergence_loss = node_kl_divergence_loss


        # kl-divergence error for class latent space
        '''class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )'''


        #print('class kl unwei ', class_kl_divergence_loss)
        #print('class kl wei ', class_kl_divergence_loss)


        # reconstruct samples
        """
        sampling from group mu and logvar for each graph in mini-batch differently makes
        the decoder consider class latent embeddings as random noise and ignore them 
        """
        node_latent_embeddings = reparameterize(training=True, mu=node_mu, logvar=node_logvar)


        reconstructed_node = self.decoder(node_latent_embeddings)

        #reconstruction_error =  mse_loss(reconstructed_node, x) * num_graphs
        reconstruction_error = self.recon_loss1(reconstructed_node, edge_index)


        #class_kl_divergence_loss.backward(retain_graph=True)
        #node_kl_divergence_loss.backward(retain_graph=True)
        #reconstruction_error.backward()

        loss =   node_kl_divergence_loss + reconstruction_error

        loss.backward()


        return  reconstruction_error.item(), 0 , node_kl_divergence_loss.item()


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



    def recon_loss1(self, z, edge_index):

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



    def get_embeddings(self, data):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():

            data.to(device)
            x, edge_index = data.x, data.edge_index


            node_mu, node_logvar = self.encoder(x.double(), edge_index)

            '''grouped_mu, grouped_logvar = accumulate_group_evidence(
                class_mu.data, class_logvar.data, batch, True
            )
            class_latent_embeddings = group_wise_reparameterize(
                training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
            )'''

            node_latent = reparameterize(training=False, mu=node_mu, logvar=node_logvar)

        train_emb = node_latent[data.train_mask].cpu().numpy()
        train_y = data.y[data.train_mask].cpu().numpy()
        val_emb = node_latent[data.val_mask].cpu().numpy()
        val_y = data.y[data.val_mask].cpu().numpy()
        test_emb = node_latent[data.test_mask].cpu().numpy()
        test_y = data.y[data.test_mask].cpu().numpy()

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
    for seed in [123]:

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
        DS = 'PubMed'
        path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
        # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

        dataset = Planetoid(path, name=DS)

        print('dataset summary', dataset[0])
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


        model = GcnInfomax(args.hidden_dim, args.num_gc_layers).double().to(device)
        #encode/decode optimizers
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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

        logreg_val = []
        logreg_valbased_test = []

        best_val_round = -1
        best_val = 0

        xent = nn.CrossEntropyLoss()

        #model.train()
        for epoch in range(1, 50+1):
            recon_loss_all = 0
            kl_class_loss_all = 0
            kl_node_loss_all = 0
            mi_loss_all = 0
            model.train()


            optimizer.zero_grad()
            recon_loss, kl_class, kl_node = model(data.x.double(), data.edge_index)
            recon_loss_all += recon_loss
            kl_class_loss_all += kl_class
            kl_node_loss_all += kl_node



            '''for name, param in model.named_parameters():
                print(name, param.grad)'''




            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()

            losses['recon'].append(recon_loss_all)
            losses['node_kl'].append(kl_node_loss_all)
            losses['class_kl'].append(kl_class_loss_all)



            print('Epoch {}, Recon Loss {} KL class Loss {} KL node Loss {}'.format(epoch, recon_loss_all ,
                                                                                    kl_class_loss_all , kl_node_loss_all ))



            #print('all losses ', losses)


        model.eval()


        #if epoch == epochs:





        '''accs = []
        
        best_f1 = 0
        
        best_round = 0'''


        print('Logistic regression started!')






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





        train_emb, train_y_labels, val_emb, val_y_labels,test_emb, test_y_labels  = model.get_embeddings(data)

        train_emb, train_lbls = torch.from_numpy(train_emb).cuda(), torch.from_numpy(train_y_labels).cuda()

        val_emb, val_lbls= torch.from_numpy(val_emb).cuda(), torch.from_numpy(val_y_labels).cuda()

        test_emb, test_lbls= torch.from_numpy(test_emb).cuda(), torch.from_numpy(test_y_labels).cuda()

        best_valacc = 0
        best_test_score = 0


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


            '''if acc_val > best_valacc:

                best_val_round = epoch - 1
                best_test_score = acc_test'''

            accs = torch.stack(accs_test)
            mean = accs.mean().item()
            std = accs.std().item()


        #input = lamda *  local_emb + (1 - lamda)*global_emb
        print('lambda and test acc ', mean, std)

        #print('best perf based on validation score val, test, in epoch', logreg_val[best_val_round], logreg_valbased_test[best_val_round], best_val_round)

