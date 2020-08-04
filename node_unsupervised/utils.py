import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid

import numpy as np
import networkx as nx
import torch
from scipy.linalg import fractional_matrix_power, inv
import scipy.sparse as sp


def accumulate_group_evidence(class_mu, class_logvar, is_cuda):
    """
    :param class_mu: mu values for class latent embeddings of each sample in the mini-batch
    :param class_logvar: logvar values for class latent embeddings for each sample in the mini-batch
    :param labels_batch: all samples from the same graph
    :param is_cuda:
    :return:
    """
    var_dict = {}
    mu_dict = {}

    # convert logvar to variance for calculations
    class_var = class_logvar.exp_()

    # calculate var inverse for each group using group vars
    #for nodeidx, graphidx in enumerate(labels_batch):
    for i in range(1):
        group_label = 1

        # remove 0 values from variances
        class_var[i][class_var[i] == float(0)] = 1e-6

        if group_label in var_dict.keys():
            var_dict[group_label] += 1 / class_var[i]
        else:
            var_dict[group_label] = 1 / class_var[i]

    # invert var inverses to calculate mu and return value
    for group_label in var_dict.keys():
        var_dict[group_label] = 1 / var_dict[group_label]

    # calculate mu for each group
    for i in range(1):
        group_label = 1

        if group_label in mu_dict.keys():
            mu_dict[group_label] += class_mu[i] * (1 / class_var[i])
        else:
            mu_dict[group_label] = class_mu[i] * (1 / class_var[i])

    # multiply group var with sums calculated above to get mu for the group
    for group_label in mu_dict.keys():
        mu_dict[group_label] *= var_dict[group_label]

    # replace individual mu and logvar values for each sample with group mu and logvar
    group_mu = torch.FloatTensor(class_mu.size(0), class_mu.size(1))
    group_var = torch.FloatTensor(class_var.size(0), class_var.size(1))

    if is_cuda:
        group_mu = group_mu.cuda()
        group_var = group_var.cuda()

    for i in range(1):
        group_label = 1

        group_mu[i] = mu_dict[group_label]
        group_var[i] = var_dict[group_label]

        # remove 0 from var before taking log
        group_var[i][group_var[i] == float(0)] = 1e-6

    # convert group vars into logvars before returning
    return Variable(group_mu, requires_grad=True), Variable(torch.log(group_var), requires_grad=True)


def mse_loss(input, target):
    return torch.sum((input - target).pow(2)) / input.data.nelement()


def l1_loss(input, target):
    return torch.sum(torch.abs(input - target)) / input.data.nelement()


def reparameterize(training, mu, logvar):
    if training:
        std = logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        return eps.mul(std).add_(mu)
    else:
        return mu


def group_wise_reparameterize(training, mu, logvar, labels_batch, cuda):
    eps_dict = {}

    # generate only 1 eps value per group label
    for label in torch.unique(labels_batch):
        if cuda:
            eps_dict[label.item()] = torch.cuda.FloatTensor(1, logvar.size(1)).normal_(0., 0.1)
        else:
            eps_dict[label.item()] = torch.FloatTensor(1, logvar.size(1)).normal_(0., 0.1)

    if training:
        std = logvar.mul(0.5).exp_()
        reparameterized_var = Variable(std.data.new(std.size()))

        # multiply std by correct eps and add mu
        for i in range(logvar.size(0)):
            reparameterized_var[i] = std[i].mul(Variable(eps_dict[labels_batch[i].item()]))
            reparameterized_var[i].add_(mu[i])

        return reparameterized_var
    else:
        return mu


def weights_init(layer):
    if isinstance(layer, nn.Conv2d):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.BatchNorm2d):
        layer.weight.data.normal_(1.0, 0.02)
        layer.bias.data.zero_()
    elif isinstance(layer, nn.Linear):
        layer.weight.data.normal_(0.0, 0.05)
        layer.bias.data.zero_()


def imshow_grid(images, shape=[2, 8], name='default', save=False):
    """Plot images in a grid of a given shape."""
    fig = plt.figure(1)
    grid = ImageGrid(fig, 111, nrows_ncols=shape, axes_pad=0.05)

    size = shape[0] * shape[1]
    for i in range(size):
        grid[i].axis('off')
        grid[i].imshow(images[i])  # The AxesGrid object work as a list of axes.

    if save:
        plt.savefig('reconstructed_images/' + str(name) + '.png')
        plt.clf()
    else:
        plt.show()


def compute_ppr(graph: nx.Graph, alpha=0.2, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])                                # A^ = A + I_n
    d = np.diag(np.sum(a, 1))                                     # D^ = Sigma A^_ii
    dinv = fractional_matrix_power(d, -0.5)                       # D^(-1/2)
    at = np.matmul(np.matmul(dinv, a), dinv)                      # A~ = D^(-1/2) x A^ x D^(-1/2)
    return alpha * inv((np.eye(a.shape[0]) - (1 - alpha) * at))   # a(I_n-(1-a)A~)^-1


def compute_heat(graph: nx.Graph, t=5, self_loop=True):
    a = nx.convert_matrix.to_numpy_array(graph)
    if self_loop:
        a = a + np.eye(a.shape[0])
    d = np.diag(np.sum(a, 1))
    return np.exp(t * (np.matmul(a, inv(d)) - 1))


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    if isinstance(features, np.ndarray):
        return features
    else:
        return features.todense(), sparse_to_tuple(features)


def normalize_adj(adj, self_loop=True):
    """Symmetrically normalize adjacency matrix."""
    if self_loop:
        adj = adj + sp.eye(adj.shape[0])
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
