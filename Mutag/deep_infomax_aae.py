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
from gin_aae import Encoder, Decoder
from evaluate_embedding import evaluate_embedding, draw_plot
from model import *
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence, \
    accumulate_group_rep, expand_group_rep

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

        node_z, class_z = self.encoder(x, edge_index, batch)





        grouped_mu, grouped_logvar = accumulate_group_evidence(
            class_mu.data, class_logvar.data, batch, True
        )







        reconstructed_node = self.decoder(node_latent_embeddings, class_latent_embeddings, edge_index)

        #reconstruction_error =  mse_loss(reconstructed_node, x) * num_graphs
        reconstruction_error = self.recon_loss1(reconstructed_node, edge_index, batch)


        #class_kl_divergence_loss.backward(retain_graph=True)
        #node_kl_divergence_loss.backward(retain_graph=True)
        #reconstruction_error.backward()

        loss =  + 1e-7*reconstruction_error

        loss.backward()


        return  1e-7*reconstruction_error.item(), class_kl_divergence_loss.item() , node_kl_divergence_loss.item()


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

    def recon_loss(self, z, edge_index, batch):

        EPS = 1e-15
        MAX_LOGSTD = 10
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
  
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
        """

        #reco = self.edge_recon(z, edge_index)

        #print('edge recon try ', reco.size(), edge_index.size(), reco [:10], edge_index[0][:10], edge_index[1][:10])


        #recon_adj = self.edge_recon(z, edge_index)


        a, idx_tensor = to_dense_batch(z, batch)
        a_t = a.permute(0, 2, 1)

        #print('batch size', a.size(), a_t.size())

        rec = torch.bmm(a, a_t)

        #print('inner pro', rec.size())

        org_adj = to_dense_adj(edge_index, batch)


        pos_weight = float(z.size(0) * z.size(0) - org_adj.sum()) / org_adj.sum()
        norm = z.size(0) * z.size(0) / float((z.size(0) * z.size(0) - org_adj.sum()) * 2)


        #print('new' ,rec, 'org', org_adj)


        '''#adj = torch.matmul(z, z.t())
  
  
        #r_adj = to_dense_adj(recon_adj, batch)
        #org_adj = to_dense_adj(edge_index, batch)
  
        #print(r_adj.size(), org_adj.size())
  
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
  
        return pos_loss + neg_loss'''

        loss = norm * F.binary_cross_entropy_with_logits(rec, org_adj, pos_weight=pos_weight)

        return loss


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

        #loss = F.binary_cross_entropy_with_logits(rec, org_adj)

        #return loss


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


                node_mu, node_logvar, class_mu, class_logvar = self.encoder(x.double(), edge_index, batch)


                node_latent_embeddings = reparameterize(training=False, mu=node_mu, logvar=node_logvar)

                grouped_mu, grouped_logvar = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, batch, True
                )

                accumulated_class_latent_embeddings = group_wise_reparameterize(
                    training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
                )

                class_emb = global_mean_pool(accumulated_class_latent_embeddings, batch)
                node_emb = global_mean_pool(node_latent_embeddings, batch)


                ret_node.append(node_emb.cpu().numpy())
                y_node.append(data.y.cpu().numpy())
                ret_class.append(class_emb.cpu().numpy())
                y_class.append(data.y.cpu().numpy())

        ret_node = np.concatenate(ret_node, 0)
        y_node = np.concatenate(y_node, 0)
        ret_class = np.concatenate(ret_class, 0)
        y_class = np.concatenate(y_class, 0)
        return ret_node, y_node, ret_class, y_class

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

        losses = {'recon':[], 'node_kl':[], 'class_kl': []}

        warmup_steps = 0
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
        dataset = TUDataset(path, name=DS).shuffle()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            dataset_num_features = dataset.num_features
        except:
            dataset_num_features = 1

        if not dataset_num_features:

            dataset_num_features = 5

            #dataset_num_features = 5
            #input_feat = torch.ones((batch_size, 1)).to(device)

        dataloader = DataLoader(dataset, batch_size=batch_size)


        model = GcnInfomax(args.hidden_dim, args.num_gc_layers).double().to(device)
        #encode/decode optimizers
        optim_P = torch.optim.Adam(model.decoder.parameters(), lr=gen_lr)
        optim_Q_enc = torch.optim.Adam(model.encoder.parameters(), lr=gen_lr)
        #regularizing optimizers
        optim_Q_gen = torch.optim.Adam(model.encoder.parameters(), lr=reg_lr)
        optim_D = torch.optim.Adam([
            {'params': model.node_discriminator.parameters()},
            {'params': model.class_discriminator.parameters()}
        ], lr=reg_lr)



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
            recon_loss_all = 0
            kl_class_loss_all = 0
            kl_node_loss_all = 0
            mi_loss_all = 0
            model.train()
            for data in dataloader:
                data = data.to(device)


                #if data.x is None:
                if not dataset.num_features:

                    data.x = torch.ones((data.batch.shape[0], 5)).double().to(device)


                #model.encoder.zero_grad()
                #model.decoder.zero_grad()

                model.zero_grad()

                z_sample, z_class = model.encoder(data.x.double(), data.edge_index, data.batch)
                grouped_class = accumulate_group_rep(
                    z_class, data.batch
                )


                #encode to z
                X_sample = model.decoder(z_sample, grouped_class) #decode to X reconstruction
                recon_loss = 1e-5* model.recon_loss1(X_sample, data.edge_index, data.batch)
                recon_loss_all += recon_loss.item()

                recon_loss.backward()
                optim_P.step()
                optim_Q_enc.step()

                # Discriminator
                ## true prior is random normal (randn)
                ## this is constraining the Z-projection to be normal!
                model.encoder.eval()
                #model.class_discriminator.zero_grad()
                #model.class_discriminator.zero_grad()


                z_real_gauss_node = Variable(torch.randn(data.batch.shape[0], args.hidden_dim) ).double().cuda()
                D_real_gauss_node = model.node_discriminator(z_real_gauss_node)

                z_real_gauss_class = Variable(torch.randn(data.num_graphs, args.hidden_dim)).double().cuda()

                z_real_gauss_class_exp = expand_group_rep(z_real_gauss_class, data.batch, data.batch.shape[0], args.hidden_dim)


                D_real_gauss_class = model.class_discriminator(z_real_gauss_class_exp)

                z_fake_gauss_node, z_fake_gauss_class = model.encoder(data.x.double(), data.edge_index, data.batch)

                grouped_z_fake_gauss_class = accumulate_group_rep(
                    z_fake_gauss_class, data.batch
                )

                D_fake_gauss_node = model.node_discriminator(z_fake_gauss_node)
                D_fake_gauss_class = model.class_discriminator(grouped_z_fake_gauss_class)



                D_loss_node = -torch.mean(torch.log(D_real_gauss_node + EPS) + torch.log(1 - D_fake_gauss_node + EPS))
                D_loss_class = -torch.mean(torch.log(D_real_gauss_class + EPS) + torch.log(1 - D_fake_gauss_class + EPS))




                D_loss = D_loss_node + D_loss_class

                kl_class_loss_all += D_loss.item()

                D_loss.backward()
                optim_D.step()

                # Generator
                model.encoder.train()

                z_fake_gauss_node, z_fake_gauss_class = model.encoder(data.x.double(), data.edge_index, data.batch)

                grouped_z_fake_gauss_class = accumulate_group_rep(
                    z_fake_gauss_class, data.batch
                )

                D_fake_gauss_node = model.node_discriminator(z_fake_gauss_node)
                D_fake_gauss_class = model.class_discriminator(grouped_z_fake_gauss_class)



                G_loss_node = -torch.mean(torch.log(D_fake_gauss_node + EPS))
                G_loss_class = -torch.mean(torch.log(D_fake_gauss_class + EPS))

                G_loss = G_loss_node + G_loss_class
                #G_loss = G_loss_node

                kl_node_loss_all += G_loss.item()

                optim_Q_gen.zero_grad()
                G_loss.backward()
                optim_Q_gen.step()








                '''for name, param in model.named_parameters():
                    print(name, param.grad)'''




                #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)

            losses['recon'].append(recon_loss_all/ len(dataloader))
            losses['node_kl'].append(kl_node_loss_all/ len(dataloader))
            losses['class_kl'].append(kl_class_loss_all/ len(dataloader))



            print('Epoch {}, Recon Loss {} KL class Loss {} KL node Loss {}'.format(epoch, recon_loss_all / len(dataloader),
                                                                                    kl_class_loss_all / len(dataloader), kl_node_loss_all / len(dataloader)))
            #print('\n\n', losses, '\n')

            #used during finetune phase
            if epoch > warmup_steps :
                model.eval()

                emb_node, y_node, emb_class, y_class = model.get_embeddings(dataloader)
                print('node mean graph classificaion')
                res = evaluate_embedding(emb_node, y_node)
                accuracies_node['logreg'].append(res[0])
                accuracies_node['svc'].append(res[1])
                accuracies_node['linearsvc'].append(res[2])
                accuracies_node['randomforest'].append(res[3])
                print('node ', accuracies_node)
                print('train_loss', losses)
                print('graph classificaion')
                res = evaluate_embedding(emb_class, y_class)
                accuracies_class['logreg'].append(res[0])
                accuracies_class['svc'].append(res[1])
                accuracies_class['linearsvc'].append(res[2])
                accuracies_class['randomforest'].append(res[3])
                print('class ', accuracies_class)


        '''model.eval()
    
        #for i in range(5):
        emb, y = model.get_embeddings(dataloader)
        res = evaluate_embedding(emb, y)
        accuracies['logreg'].append(res[0])
        accuracies['svc'].append(res[1])
        accuracies['linearsvc'].append(res[2])
        accuracies['randomforest'].append(res[3])
        print(accuracies)'''

        #draw_plot(y, emb, 'imdb_b_normal.png')

        with open('unsupervised.log', 'a+') as f:
            s = json.dumps(accuracies)
            f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s))
