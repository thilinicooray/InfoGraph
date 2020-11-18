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
from gin import Encoder, Decoder
from evaluate_embedding import evaluate_embedding, draw_plot
from model import *
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence, \
    accumulate_group_rep, expand_group_rep

from sklearn.linear_model import LogisticRegression
from scipy import stats

from arguments import arg_parse
from numpy import savetxt

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
    def __init__(self, hidden_dim, num_gc_layers, node_dim, class_dim, lamda, alpha=0.5, beta=1., gamma=.1):
        super(GcnInfomax, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.prior = args.prior

        self.encoder = Encoder(dataset_num_features, hidden_dim, num_gc_layers, node_dim, class_dim)
        self.decoder = Decoder(node_dim, class_dim, dataset_num_features)
        self.node_discriminator = D_net_gauss(hidden_dim, hidden_dim)
        self.class_discriminator = D_net_gauss(hidden_dim, hidden_dim)
        self.lamda = lamda



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

        node_mu, node_logvar, class_mu, class_logvar = self.encoder(x, edge_index)





        grouped_mu, grouped_logvar = accumulate_group_evidence(
            class_mu.data, class_logvar.data, None, True
        )




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
        class_kl_divergence_loss = - 0.5  * torch.mean(torch.sum(
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
            training=True, mu=grouped_mu, logvar=grouped_logvar, labels_batch=None, cuda=True
        )


        #reconstructed_node = self.decoder(node_latent_embeddings, class_latent_embeddings)
        reconstructed_node = self.decoder(self.lamda* node_latent_embeddings + (1- self.lamda)*class_latent_embeddings)

        #reconstruction_error =  mse_loss(reconstructed_node, x) * num_graphs
        reconstruction_error = self.recon_loss1(reconstructed_node, edge_index)


        #class_kl_divergence_loss.backward(retain_graph=True)
        #node_kl_divergence_loss.backward(retain_graph=True)
        #reconstruction_error.backward()

        loss =  class_kl_divergence_loss + node_kl_divergence_loss + reconstruction_error

        loss.backward()


        return  reconstruction_error.item(), class_kl_divergence_loss.item() , node_kl_divergence_loss.item()


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



    def get_embeddings(self, data, lamda):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():

            data.to(device)
            x, edge_index = data.x, data.edge_index


            node_mu, node_logvar, class_mu, class_logvar = self.encoder(x.double(), edge_index)


            global_latent_all = reparameterize(training=False, mu=class_mu, logvar=class_logvar)

        val_targets = global_latent_all[data.val_mask].cpu().numpy()
        test_targets = global_latent_all[data.test_mask].cpu().numpy()
        val_x = x[data.val_mask].cpu().numpy()
        test_x = x[data.test_mask].cpu().numpy()
        val_y = data.y[data.val_mask].cpu().numpy()
        test_y = data.y[data.test_mask].cpu().numpy()





        return val_x, val_targets, val_y, test_x, test_targets,test_y

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


        node_dim = 512
        class_dim = 512





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

        gate_val = 0.05
        runs = 20

        lambdas = []
        overall_acc = []

        #for coef in range(runs+1):
        #lamda = round(int(coef) * gate_val,2)
        lamda = 0.9


        model = GcnInfomax(args.hidden_dim, args.num_gc_layers, node_dim, class_dim, lamda).double().to(device)
        #encode/decode optimizers
        #optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)



        print('================')
        print('num_features: {}'.format(dataset_num_features))
        print('hidden_dim: {}'.format(args.hidden_dim))
        print('num_gc_layers: {}'.format(args.num_gc_layers))
        print('================')

        model.load_state_dict(torch.load(f'pubmed_512.pkl'))

        model.eval()
        import csv

        train_feat, train_targets, train_y, test_feat, test_targets, test_y  = model.get_embeddings(data, lamda)

        '''print('first train feat ', train_feat.shape, train_y)

        #K = 500

        #index_array = np.argpartition(train_feat, kth=-K, axis=-1)[:,-K:]
        #index_array = (-coef).argsort(axis=-1)[:, :K]
        #print(index_array.shape)
        #savetxt('pubmed_topwordidx_1.csv', index_array, delimiter=',')
        word_freq_dict_1 = {}
        word_freq_dict_2 = {}
        word_freq_dict_3 = {}

        for i in range(60):
            if train_y[i] == 0:
                for j in range(500):
                    val = train_feat[i][j]

                    if val > 0 :
                        if j not in word_freq_dict_1:
                            word_freq_dict_1[j] = 1
                        else:
                            word_freq_dict_1[j] += 1
            elif train_y[i] == 1:
                for j in range(500):
                    val = train_feat[i][j]

                    if val > 0 :
                        if j not in word_freq_dict_2:
                            word_freq_dict_2[j] = 1
                        else:
                            word_freq_dict_2[j] += 1

            elif train_y[i] == 2:
                for j in range(500):
                    val = train_feat[i][j]

                    if val > 0 :
                        if j not in word_freq_dict_3:
                            word_freq_dict_3[j] = 1
                        else:
                            word_freq_dict_3[j] += 1

        sorted_words_1 = sorted(word_freq_dict_1.items(), reverse=True, key=lambda kv: kv[1])

        with open('word_freq_input_train_new_1.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['word_idx', 'freq'])
            for i in range(len(sorted_words_1)):
                item = sorted_words_1[i]
                writer.writerow([item[0], item[1]])

        sorted_words_2 = sorted(word_freq_dict_2.items(), reverse=True, key=lambda kv: kv[1])

        with open('word_freq_input_train_new_2.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['word_idx', 'freq'])
            for i in range(len(sorted_words_2)):
                item = sorted_words_2[i]
                writer.writerow([item[0], item[1]])

        sorted_words_3 = sorted(word_freq_dict_3.items(), reverse=True, key=lambda kv: kv[1])

        with open('word_freq_input_train_new_3.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['word_idx', 'freq'])
            for i in range(len(sorted_words_3)):
                item = sorted_words_3[i]
                writer.writerow([item[0], item[1]])'''



        '''from sklearn import linear_model

        regr = linear_model.LinearRegression()
        regr.fit(train_feat, train_targets)

        coef = regr.coef_

        print('Coefficients:', coef.shape)

        #savetxt('pubmed_coef1.csv', coef, delimiter=',')
        K = 20

        index_array = np.argpartition(coef, kth=-K, axis=-1)[:,-K:]
        #index_array = (-coef).argsort(axis=-1)[:, :K]
        print(index_array.shape)
        #savetxt('pubmed_topwordidx_1.csv', index_array, delimiter=',')
        word_freq_dict = {}

        for i in range(512):
            for j in range(20):
                idx = index_array[i][j]
                if idx not in word_freq_dict:
                    word_freq_dict[idx] = 1
                else:
                    word_freq_dict[idx] += 1

        sorted_words = sorted(word_freq_dict.items(), reverse=True, key=lambda kv: kv[1])

        import csv

        with open('word_freq.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['word_idx', 'freq'])
            for i in range(len(sorted_words)):
                item = sorted_words[i]
                writer.writerow([item[0], item[1]])'''


        '''from sklearn.ensemble import RandomForestRegressor
        from sklearn.inspection import permutation_importance

        #regr = RandomForestRegressor(random_state=0)
        regr = RandomForestRegressor(n_estimators=500,
                              criterion='mse', random_state=0)
        regr.fit(train_feat, train_targets)


        #predic = regr.predict(test_feat)
        #score = regr.score(test_feat, test_targets)

        #print('ored ', score, predic.shape, test_targets.shape)

        #savetxt('regression_pred_local.csv', predic, delimiter=',')
        #savetxt('regression_true_local.csv', test_targets, delimiter=',')

        coef = regr.feature_importances_

        #result = permutation_importance(regr, train_feat, train_targets, n_repeats=10,random_state=0)

        #coef = result.importances_mean
        print('Coefficients:', coef.shape)


        #savetxt('pubmed_coef1.csv', coef, delimiter=',')'''
        '''K = 500

        index_array = np.argpartition(coef, kth=-K, axis=-1)[-K:]
        #index_array = (-coef).argsort(axis=-1)[:, :K]
        print(index_array.shape)
        #savetxt('pubmed_topwordidx_1.csv', index_array, delimiter=',')
        word_freq_dict = {}

        for j in range(K):
            idx = index_array[j]
            if idx not in word_freq_dict:
                word_freq_dict[idx] = 1
            else:
                word_freq_dict[idx] += 1

        sorted_words = sorted(word_freq_dict.items(), reverse=True, key=lambda kv: kv[1])

        import csv

        with open('word_freq_randomforestreg_permimpor_global2.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['word_idx', 'freq'])
            for i in range(len(sorted_words)):
                item = sorted_words[i]
                writer.writerow([item[0], item[1]])'''

        print('feat size ', train_feat.shape, train_targets.shape)

        print('values ', train_feat[0,:10], train_targets[0,:10])

        #n_rho, n_pval = stats.spearmanr(train_feat, train_targets, axis=0)
        #n_rho, n_pval = stats.spearmanr(torch.cat([node_latent_embeddings,indiclass_latent_embeddings],0) .cpu().numpy(), axis=1)

        #n_rho = np.matmul(train_feat.transpose(), train_targets)

        #print('corre', n_rho.shape)
        #a = np.sum(np.power(n_rho,2), axis=1)
        #print('corre', a.shape)

        word_rep = train_feat.transpose()
        global_rep = train_targets.transpose()
        print('feat size new ', word_rep.shape, global_rep.shape)

        #normalizing
        word_rep = preprocessing.scale(word_rep)
        global_rep = preprocessing.scale(global_rep)

        corre_matrix = np.ndarray(shape=(word_rep.shape[0],global_rep.shape[0]))

        from scipy.stats.stats import pearsonr, spearmanr

        for i in range(word_rep.shape[0]):
            for j in range(global_rep.shape[0]):
                #print('sizes ', word_rep[i].shape, global_rep[j].shape)
                corre_matrix[i][j] = pearsonr(word_rep[i],global_rep[j])[0]#np.cov(word_rep[i], global_rep[j])[0][1]# spearmanr(word_rep[i],global_rep[j])[0]

        print('cor matrix ',corre_matrix )

        mask = (corre_matrix > 0).astype(int)
        pos_corr = corre_matrix * mask


        #savetxt('corre.csv', corre_matrix, delimiter=',')

        #abs_corre = np.absolute(corre_matrix)
        abs_corre = np.power(corre_matrix, 2)
        #savetxt('abs_corre.csv', abs_corre, delimiter=',')

        #print('corre size ', corre_matrix.shape)

        word_wise_corr = np.sum(abs_corre, axis=1)
        print('word_wise_corr size ', word_wise_corr.shape)


        global_importance_dict = {}

        for i in range(500):
            global_importance_dict[i] = word_wise_corr[i]

        sorted_words = sorted(global_importance_dict.items(), reverse=True, key=lambda kv: kv[1])

        import csv

        with open('word_freq_corre_input_global_pearsonr_val_sum_again.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['word_idx', 'importance'])
            for i in range(len(sorted_words)):
                item = sorted_words[i]
                writer.writerow([item[0], item[1]])




        '''global_importance_dict = {}

        for i in range(500):
            global_importance_dict[i] = a[i]

        sorted_words = sorted(global_importance_dict.items(), reverse=True, key=lambda kv: kv[1])

        import csv

        with open('word_freq_randomforestreg_permimpor_global4.csv','w') as f:
            writer = csv.writer(f)
            writer.writerow(['word_idx', 'importance'])
            for i in range(len(sorted_words)):
                item = sorted_words[i]
                writer.writerow([item[0], item[1]])'''








        #savetxt('gate_acc_citeseer_testsep1_ep50_4.csv', overall_acc, delimiter=',')
        #savetxt('gate_val_citeseer_testsep1_ep50_4.csv', lambdas, delimiter=',')







