import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
from mpl_toolkits.axes_grid1 import ImageGrid

from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool


def accumulate_group_evidence_old(class_mu, class_logvar, batch, is_cuda):
    """
    :param class_mu: mu values for class latent embeddings of each sample in the mini-batch
    :param class_logvar: logvar values for class latent embeddings for each sample in the mini-batch
    :param labels_batch: class labels of each sample (the operation of accumulating class evidence can also
        be performed using group labels instead of actual class labels)
    :param is_cuda:
    :return:
    """
    var_dict = {}
    mu_dict = {}

    # convert logvar to variance for calculations
    class_var = class_logvar.exp_()

    # calculate var inverse for each group using group vars
    #for nodeidx, graphidx in enumerate(labels_batch):
    for i in range(len(batch)):
        group_label = batch[i].item()

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
    for i in range(len(batch)):
        group_label = batch[i].item()

        if group_label in mu_dict.keys():
            mu_dict[group_label] += class_mu[i] * (1 / class_var[i])
        else:
            mu_dict[group_label] = class_mu[i] * (1 / class_var[i])

    # multiply group var with sums calculated above to get mu for the group
    for group_label in mu_dict.keys():
        mu_dict[group_label] *= var_dict[group_label]

    # replace individual mu and logvar values for each sample with group mu and logvar
    group_mu = torch.DoubleTensor(class_mu.size(0), class_mu.size(1))
    group_var = torch.DoubleTensor(class_var.size(0), class_var.size(1))

    if is_cuda:
        group_mu = group_mu.cuda()
        group_var = group_var.cuda()

    for i in range(len(batch)):
        group_label = batch[i].item()

        group_mu[i] = mu_dict[group_label]
        group_var[i] = var_dict[group_label]

        # remove 0 from var before taking log
        group_var[i][group_var[i] == float(0)] = 1e-6

    # convert group vars into logvars before returning
    return Variable(group_mu, requires_grad=True), Variable(torch.log(group_var), requires_grad=True)

def accumulate_group_evidence_mul(class_mu, class_logvar, batch, is_cuda):
    """
    :param class_mu: mu values for class latent embeddings of each sample in the mini-batch
    :param class_logvar: logvar values for class latent embeddings for each sample in the mini-batch
    :param labels_batch: class labels of each sample (the operation of accumulating class evidence can also
        be performed using group labels instead of actual class labels)
    :param is_cuda:
    :return:
    """

    class_var = class_logvar.exp_()

    class_var[class_var == float(0)] = 1e-6

    var_opp = class_var.pow(-1)

    grouped_var = global_add_pool(var_opp, batch)

    grouped_var = grouped_var.pow(-1)

    mu_new = class_mu * class_var.pow(-1)

    grouped_mu = global_add_pool(mu_new, batch)


    grouped_mu = grouped_mu * grouped_var

    grouped_var[grouped_var == float(0)] = 1e-6


    grouped_lvar = torch.log(grouped_var)


    _, count = torch.unique(batch,  return_counts=True)

    grouped_mu_expanded = torch.repeat_interleave(grouped_mu, count, dim=0)
    grouped_lvar_expanded = torch.repeat_interleave(grouped_lvar, count, dim=0)


    return grouped_mu_expanded, grouped_lvar_expanded



def accumulate_group_evidence(class_mu, class_logvar, batch, is_cuda):

    grouped_mu = global_mean_pool(class_mu, batch)
    grouped_lvar = global_mean_pool(class_logvar, batch)

    _, count = torch.unique(batch,  return_counts=True)

    grouped_mu_expanded = torch.repeat_interleave(grouped_mu, count, dim=0)
    grouped_lvar_expanded = torch.repeat_interleave(grouped_lvar, count, dim=0)


    return grouped_mu_expanded, grouped_lvar_expanded




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


def group_wise_reparameterize_old(training, mu, logvar, labels_batch, cuda):
    eps_dict = {}

    # generate only 1 eps value per group label
    for label in torch.unique(labels_batch):
        if cuda:
            eps_dict[label.item()] = torch.cuda.DoubleTensor(1, logvar.size(1)).normal_(0., 0.1)
        else:
            eps_dict[label.item()] = torch.DoubleTensor(1, logvar.size(1)).normal_(0., 0.1)

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


def group_wise_reparameterize(training, mu, logvar, labels_batch, cuda):

    if training:

        g_mu = global_mean_pool(mu, labels_batch)
        g_logvar = global_mean_pool(logvar, labels_batch)

        std = g_logvar.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())
        graph_wise_sample = eps.mul(std).add_(g_mu)

        _, count = torch.unique(labels_batch,  return_counts=True)

        grouped_mu_expanded = torch.repeat_interleave(graph_wise_sample, count, dim=0)

        return grouped_mu_expanded


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