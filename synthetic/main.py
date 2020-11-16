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
        node_kl_divergence_loss = 0.0001 *node_kl_divergence_loss
        node_kl_divergence_loss.backward(retain_graph=True)

        # kl-divergence error for class latent space
        class_kl_divergence_loss = torch.mean(
            - 0.5 * torch.sum(1 + grouped_logvar - grouped_mu.pow(2) - grouped_logvar.exp())
        )
        #class_kl_divergence_loss = 0.0000001 * class_kl_divergence_loss * num_graphs
        class_kl_divergence_loss =  0.0001 *class_kl_divergence_loss
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
        #reconstruction_error =  0.1*mse_loss(reconstructed_node, x)
        #reconstruction_error =  mse_loss(reconstructed_node, x)
        reconstruction_error = 0.001 *self.recon_loss1(reconstructed_node, edge_index, batch)
        reconstruction_error.backward()


        return reconstruction_error.item() , class_kl_divergence_loss.item() , node_kl_divergence_loss.item()

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

    def get_embeddings(self, loader):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ret = []
        y = []
        with torch.no_grad():
            for data in loader:
                data.to(device)
                x, edge_index, batch = data.x, data.edge_index, data.batch
                if x is None:
                    x = torch.ones((batch.shape[0],1)).to(device)
                __, _, class_mu, class_logvar = self.encoder(x, edge_index, batch)

                grouped_mu, grouped_logvar = accumulate_group_evidence(
                    class_mu.data, class_logvar.data, batch, True
                )

                accumulated_class_latent_embeddings = group_wise_reparameterize(
                    training=False, mu=grouped_mu, logvar=grouped_logvar, labels_batch=batch, cuda=True
                )

                class_emb = global_mean_pool(accumulated_class_latent_embeddings, batch)
                ret.append(class_emb.cpu().numpy())
                #y.append(data.y.cpu().numpy())
        ret = np.concatenate(ret, 0)
        y = np.concatenate(y, 0)
        return ret, y

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

    dataset = SyntheticERDataset(path)#.shuffle()

    train_dataset = dataset[:3000]
    test_dataset = dataset[3000:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=512)
    test_loader = DataLoader(test_dataset, batch_size=512)

    lr = args.lr
    epochs = 50
    dataset_num_features = 2

    model = GLDisen(2, 2, 3, 1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    #scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)



    print('================')
    print('lr: {}'.format(lr))
    print('num_features: {}'.format(dataset_num_features))
    print('hidden_dim: {}'.format(args.hidden_dim))
    print('num_gc_layers: {}'.format(args.num_gc_layers))
    print('================')


    model.train()
    for epoch in range(1, epochs+1):
        loss_all = 0
        recon_loss_all = 0
        kl_class_loss_all = 0
        kl_node_loss_all = 0

        #model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            recon_loss, kl_class, kl_node = model(data.x[:,:2], data.edge_index, data.batch, data.num_graphs)
            recon_loss_all += recon_loss
            kl_class_loss_all += kl_class
            kl_node_loss_all += kl_node
            optimizer.step()

            #losses['tot'].append(loss_all/ len(dataloader))


        print('Epoch {}, Recon Loss {} KL class Loss {} KL node Loss {}'.format(epoch, recon_loss_all / len(train_loader),
                                                                                kl_class_loss_all / len(train_loader), kl_node_loss_all / len(train_loader)))


    torch.save(model.state_dict(), f'syner_model7.pkl')


    #model.eval()

    #emb, y = model.get_embeddings(test_loader)

