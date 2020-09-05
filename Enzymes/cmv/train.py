import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from dataset import load

from evaluate_embedding import evaluate_embedding


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
        l1, gv1, l2, gv2 = self.forward(adj, diff, feat, mask)
        return (l1 + l2).detach(),(gv1 + gv2).detach()


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
def local_global_loss_(l_enc, g_enc, batch, measure, mask):
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

    for idx, m in enumerate(mask):
        msk[idx * max_nodes + m: idx * max_nodes + max_nodes, idx] = 0.

    res = torch.mm(l_enc, g_enc.t()) * msk

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


def train(dataset, gpu, num_layer=4, epoch=40, batch=64):
    nb_epochs = epoch
    batch_size = batch
    patience = 20
    lr = 0.001
    l2_coef = 0.0
    hid_units = 32

    adj, diff, feat, labels, num_nodes = load(dataset)

    #print('feat', labels[0])

    feat = torch.FloatTensor(feat).cuda()
    accuracies_node = {'logreg':[], 'svc':[], 'linearsvc':[], 'randomforest':[]}

    diff = torch.FloatTensor(diff).cuda()
    adj = torch.FloatTensor(adj).cuda()
    labels = torch.LongTensor(labels).cuda()



    ft_size = feat[0].shape[1]
    max_nodes = feat[0].shape[0]

    model = Model(ft_size, hid_units, num_layer)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    model.cuda()

    cnt_wait = 0
    best = 1e9

    itr = (adj.shape[0] // batch_size) + 1
    for epoch in range(nb_epochs):
        epoch_loss = 0.0
        train_idx = np.arange(adj.shape[0])
        np.random.shuffle(train_idx)

        for idx in range(0, len(train_idx), batch_size):
            model.train()
            optimiser.zero_grad()

            batch = train_idx[idx: idx + batch_size]
            mask = num_nodes[idx: idx + batch_size]

            lv1, gv1, lv2, gv2 = model(adj[batch], diff[batch], feat[batch], mask)

            lv1 = lv1.view(batch.shape[0] * max_nodes, -1)
            lv2 = lv2.view(batch.shape[0] * max_nodes, -1)

            batch = torch.LongTensor(np.repeat(np.arange(batch.shape[0]), max_nodes)).cuda()

            loss1 = local_global_loss_(lv1, gv2, batch, 'JSD', mask)
            loss2 = local_global_loss_(lv2, gv1, batch, 'JSD', mask)
            # loss3 = global_global_loss_(gv1, gv2, 'JSD')
            loss = loss1 + loss2 #+ loss3
            epoch_loss += loss
            loss.backward()
            optimiser.step()

        epoch_loss /= itr

        print('Epoch: {0}, Loss: {1:0.4f}'.format(epoch, epoch_loss))




        features = feat.cuda()
        adj = adj.cuda()
        diff = diff.cuda()
        labels = labels.cuda()

        node_embeds, graph_embeds = model.embed(features, adj, diff, num_nodes)


        real_nodes = None
        real_labels = None

        print('eval ', node_embeds.size(), num_nodes, labels.size())

        '''x = node_embeds.cpu().numpy()
        y = labels.cpu().numpy()

        res = evaluate_embedding(x, y)
        accuracies_node['logreg'].append(res[0])
        accuracies_node['svc'].append(res[1])
        accuracies_node['linearsvc'].append(res[2])
        accuracies_node['randomforest'].append(res[3])
        print('node ', accuracies_node)'''




if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    gpu = 1
    #torch.cuda.set_device(gpu)
    layers = [2]
    batch = [128]
    epoch = [500]
    #ds = ['MUTAG', 'PTC_MR', 'IMDB-BINARY', 'IMDB-MULTI', 'REDDIT-BINARY', 'REDDIT-MULTI-5K']
    ds = ['ENZYMES']
    seeds = [52]
    for d in ds:
        print(f'####################{d}####################')
        for l in layers:
            for b in batch:
                for e in epoch:
                    for i in range(5):
                        seed = seeds[i]
                        torch.manual_seed(seed)
                        torch.backends.cudnn.deterministic = True
                        torch.backends.cudnn.benchmark = False
                        np.random.seed(seed)
                        print(f'Dataset: {d}, Layer:{l}, Batch: {b}, Epoch: {e}, Seed: {seed}')
                        train(d, gpu, l, e, b)
        print('################################################')