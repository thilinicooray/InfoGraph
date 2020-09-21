import os
import sys
import os.path as osp
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, GRU

import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

class MyTransform(object):
    def __call__(self, data):
        # Specify target.
        data.y = data.y[:, target]
        return data


class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data

def train(epoch, use_unsup_loss):
    model.train()
    loss_all = 0
    recon_loss_all = 0
    kl_class_loss_all = 0
    kl_node_loss_all = 0
    cls_loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        node_kl_divergence_loss, class_kl_divergence_loss, reconstruction_error, cls_loss = model.supervised_loss(data)
        recon_loss_all += reconstruction_error
        kl_class_loss_all += class_kl_divergence_loss
        kl_node_loss_all += node_kl_divergence_loss
        cls_loss_all += cls_loss

        optimizer.step()

    return recon_loss_all / len(train_loader.dataset), kl_class_loss_all / len(train_loader.dataset), kl_node_loss_all / len(train_loader.dataset), \
           cls_loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += (model(data) * std - data.y * std).abs().sum().item()  # MAE
    return error / len(loader.dataset)


def seed_everything(seed=1234):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    seed_everything()
    from mlvae_model import Net
    from arguments import arg_parse
    args = arg_parse()

    target = args.target
    dim = 64
    epochs = 500
    batch_size = 20
    lamda = args.lamda
    use_unsup_loss = args.use_unsup_loss
    separate_encoder = args.separate_encoder

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'QM9')
    transform = T.Compose([MyTransform(), Complete(), T.Distance(norm=False)])
    dataset = QM9(path, transform=transform).shuffle()
    print('num_features : {}\n'.format(dataset.num_features))

    print('dataset ', dataset.data)

    # Normalize targets to mean = 0 and std = 1.
    mean = dataset.data.y[:, target].mean().item()
    std = dataset.data.y[:, target].std().item()
    dataset.data.y[:, target] = (dataset.data.y[:, target] - mean) / std

    # print(type(dataset[0]))
    # print(type(dataset.data.x)) #tensor
    # print(type(dataset.data.y)) #tensor
    # input()

    # Split datasets.
    test_dataset = dataset[:10000]
    val_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:20000+args.train_num]

    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if use_unsup_loss:
        unsup_train_dataset = dataset[20000:]
        unsup_train_loader = DataLoader(unsup_train_dataset, batch_size=batch_size, shuffle=True)

        print(len(train_dataset), len(val_dataset), len(test_dataset), len(unsup_train_dataset))
    else:
        print(len(train_dataset), len(val_dataset), len(test_dataset))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(dataset.num_features, dim, use_unsup_loss, separate_encoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=args.weight_decay)

    #val_error = test(val_loader)
    #test_error = test(test_loader)
    #print('Epoch: {:03d}, Validation MAE: {:.7f}, Test MAE: {:.7f},'.format(0, val_error, test_error))

    losses = {'recon':[], 'node_kl':[], 'class_kl': [], 'cls_loss' : []}

    best_val_error = None
    for epoch in range(1, epochs+1):
        #lr = scheduler.optimizer.param_groups[0]['lr']
        recon, class_kl, node_kl, cls = train(epoch, use_unsup_loss)


        losses['recon'].append(recon)
        losses['class_kl'].append(class_kl)
        losses['node_kl'].append(node_kl)
        losses['cls_loss'].append(cls)


        val_error = test(val_loader)
        #scheduler.step(val_error)

        if best_val_error is None or val_error <= best_val_error:
            print('Update')
            test_error = test(test_loader)
            best_val_error = val_error


        print('Epoch: {:03d}, LR: {:7f} Validation MAE: {:.7f}, '
              'Test MAE: {:.7f},'.format(epoch, 0.001, val_error, test_error))

        print('all losses ', losses)


