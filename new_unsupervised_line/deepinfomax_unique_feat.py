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
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch, add_remaining_self_loops
import sys
import json
from torch import optim

from cortex_DIM.nn_modules.mi_networks import MIFCNet, MI1x1ConvNet
from losses import *
from gin import Encoder, Decoder
from evaluate_embedding import evaluate_embedding
from model import *
from utils import imshow_grid, mse_loss, reparameterize, group_wise_reparameterize, accumulate_group_evidence

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
        self.decoder = Decoder(hidden_dim, hidden_dim, dataset_num_features)

        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)


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

        node_mu, node_logvar, class_mu, class_logvar = self.encoder(x, edge_index, batch)


        '''n_digits = 4
        #print('before ', node_mu)
        node_mu = (node_mu * 10**n_digits).round() / (10**n_digits)
        node_logvar = (node_logvar * 10**n_digits).round() / (10**n_digits)
        class_mu = (class_mu * 10**n_digits).round() / (10**n_digits)
        class_logvar = (class_logvar * 10**n_digits).round() / (10**n_digits)'''


        #print('direct out ', node_mu[0,:5], node_logvar[0,:5], class_mu[0,:5], class_logvar[0,:5], batch)


        grouped_mu, grouped_logvar = accumulate_group_evidence(
            class_mu.data, class_logvar.data, batch, True
        )

        #print('grouped ', grouped_mu[0,:5], grouped_logvar[0,:5])


        #print('after ', node_mu)


        # kl-divergence error for style latent space
        node_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + node_logvar - node_mu.pow(2) - node_logvar.exp())
        )
        #print('node kl unwei ', node_kl_divergence_loss, node_logvar, node_mu)


        node_kl_divergence_loss = node_kl_divergence_loss *num_graphs
        #print('node kl wei ', node_kl_divergence_loss)
        node_kl_divergence_loss.backward(retain_graph=True)

        # kl-divergence error for class latent space
        class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )
        #print('class kl unwei ', class_kl_divergence_loss)
        class_kl_divergence_loss = class_kl_divergence_loss * num_graphs
        #print('class kl wei ', class_kl_divergence_loss)
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


        reconstructed_node = self.decoder(node_latent_embeddings, class_latent_embeddings, edge_index)

        #reconstruction_error =  mse_loss(reconstructed_node, x) * num_graphs
        reconstruction_error = self.recon_loss(reconstructed_node, edge_index, batch) * num_graphs
        reconstruction_error.backward()

        #print(reconstruction_error.item(), class_kl_divergence_loss.item(), node_kl_divergence_loss.item())


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

        loss = F.binary_cross_entropy_with_logits(rec, org_adj)

        return loss

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:

                data.to(device)

                adj_looped, _ = add_remaining_self_loops(data.edge_index)

                new_adj = to_dense_adj(data.edge_index, data.batch)


                x_unique = data.batch.unique(sorted=True)
                x_unique_count = torch.stack([(data.batch==x_u).sum() for x_u in x_unique])


                nodes = None

                batch_count = x_unique_count.size(0)

                for gid in range(batch_count):
                    count = x_unique_count[gid]

                    current_nodes = new_adj[gid][:count]


                    if nodes is None:
                        nodes = current_nodes
                    else :
                        nodes = torch.cat([nodes.clone(), current_nodes], 0)

                pad_count = dataset_num_features - new_adj.size(-1)

                if pad_count > 0:
                    nodes = torch.cat([nodes.clone(), torch.zeros(data.batch.shape[0], pad_count).to(device)],  1)

                data.x = nodes.double().to(device)




                x, edge_index, batch = data.x, data.edge_index, data.batch

                #print(x, edge_index, data.x)
                #x = torch.rand(data.batch.shape[0], 5).to(device)
                #x = torch.ones((batch.shape[0],5)).to(device)
                #print('eval train', x.type())
                __, _, class_mu, class_logvar = self.encoder(x, edge_index, batch)

                grouped_mu, grouped_logvar = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, batch, True
                )

                accumulated_class_latent_embeddings = group_wise_reparameterize(
                    training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
                )

                class_emb = global_mean_pool(accumulated_class_latent_embeddings, batch)

                #print('clz emb ', class_emb[:5,:3])


                ret.append(class_emb.cpu().numpy())
                y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

if __name__ == '__main__':

    args = arg_parse()

    #for seed in range(80,101):

    seed = 52
    #epochs = 30
    epochs = int(args.num_epochs)

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

    log_interval = 1
    #batch_size = 128
    batch_size = args.batch_size
    lr = args.lr
    #lr = 0.000001
    DS = args.DS
    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', DS)
    # kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

    dataset = TUDataset(path, name=DS).shuffle()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        dataset_num_features = dataset.num_features
    except:
        dataset_num_features = 1

    dataloader = DataLoader(dataset, batch_size=batch_size)

    if not dataset_num_features:

        dataset_num_features = 0

        for data in dataloader:
            data = data.to(device)

            new_adj = to_dense_adj(data.edge_index, data.batch)
            current_n_count = new_adj.size(-1)

            if current_n_count > dataset_num_features:
                dataset_num_features = current_n_count


    model = GcnInfomax(args.hidden_dim, args.num_gc_layers).double().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    print('================')
    print('lr: {}'.format(lr))
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

            adj_looped, _ = add_remaining_self_loops(data.edge_index)

            new_adj = to_dense_adj(data.edge_index, data.batch)


            x_unique = data.batch.unique(sorted=True)
            x_unique_count = torch.stack([(data.batch==x_u).sum() for x_u in x_unique])



            nodes = None
            batch_count = x_unique_count.size(0)

            for gid in range(batch_count):
                count = x_unique_count[gid]

                current_nodes = new_adj[gid][:count]


                if nodes is None:
                    nodes = current_nodes
                else :
                    nodes = torch.cat([nodes.clone(), current_nodes], 0)


            pad_count = dataset_num_features - new_adj.size(-1)

            if pad_count > 0:
                nodes = torch.cat([nodes.clone(), torch.zeros(data.batch.shape[0], pad_count).to(device)],  1)


            data.x = nodes.double().to(device)



            #data.x = torch.ones((data.batch.shape[0], 5)).double().to(device)

            optimizer.zero_grad()
            recon_loss, kl_class, kl_node = model(data.x, data.edge_index, data.batch, data.num_graphs)
            recon_loss_all += recon_loss
            kl_class_loss_all += kl_class
            kl_node_loss_all += kl_node

            '''for name, param in model.named_parameters():
                print(name, param.grad)'''




            #torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            optimizer.step()



        print('Epoch {}, Recon Loss {} KL class Loss {} KL node Loss {}'.format(epoch, recon_loss_all / len(dataloader),
                                                                                kl_class_loss_all / len(dataloader), kl_node_loss_all / len(dataloader)))
        #used during finetune phase
        if epoch % log_interval == 0:
            model.eval()
            emb, y = model.get_embeddings(dataloader)
            res = evaluate_embedding(emb, y)
            accuracies['logreg'].append(res[0])
            accuracies['svc'].append(res[1])
            accuracies['linearsvc'].append(res[2])
            accuracies['randomforest'].append(res[3])
            print(accuracies)


    '''model.eval()
    
    #for i in range(5):

    emb, y = model.get_embeddings(dataloader)
    res = evaluate_embedding(emb, y)
    accuracies['logreg'].append(res[0])
    accuracies['svc'].append(res[1])
    accuracies['linearsvc'].append(res[2])
    accuracies['randomforest'].append(res[3])
    print(accuracies)'''

    with open('unsupervised.log', 'a+') as f:
        s = json.dumps(accuracies)
        f.write('{},{},{},{},{},{}\n'.format(args.DS, args.num_gc_layers, epochs, log_interval, lr, s))