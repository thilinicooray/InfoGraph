import argparse

from loader import Dataset_
from torch_geometric.data import DataLoader
from torch_sparse import spspmm, coalesce

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import os, sys
import numpy as np
import random
import copy

from model import Embed_SAD_model, Node_Classifier
from sklearn.metrics import roc_auc_score
from scipy.stats import beta

from splitters import scaffold_split
from util import degree_attr

import os
import shutil
import pdb
import pandas as pd

from tensorboardX import SummaryWriter

def modify_node_attr(x, modify_rate=0.15, modify_num=0, in_mol=True):
    x_ = copy.deepcopy(x)
    num_node = x.shape[0]

    if modify_num > 0:
        num_modify = min(modify_num, num_node)
    else:
        num_modify = max(int(num_node * modify_rate), 1)
    modify_node_idx = random.sample(range(num_node), num_modify)

    if in_mol:
        random_atom_type = torch.randint(0, 120, (num_modify,)).long().to(x_.device)
        random_chirality_tag = torch.randint(0, 3, (num_modify,)).long().to(x_.device)
        x_[modify_node_idx, 0] = random_atom_type
        x_[modify_node_idx, 1] = random_chirality_tag
    else:
        random_token = np.int64(np.random.random(num_modify) * x_.shape[1])
        x_[modify_node_idx, :] *= 0
        x_[modify_node_idx, random_token] = 1

    return x_

def create_pair(args, data, label, device):
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    batch = data.batch
    y = data.y.view(batch[-1] + 1, -1)

    x_list = list()
    edge_index_list = list()
    edge_attr_list = list()
    batch_list = list()
    node_cnt = 0

    num_graph = batch[-1] + 1
    for batch_idx in range(num_graph):
        # only modify the attribute of a graph
        if label[batch_idx] == 0:
            # get the node indices in original mini-batch
            node_idx = torch.nonzero((batch == batch_idx).float()).squeeze(-1)

            # modify the node attribute
            orig_x = x[node_idx, :]
            modified_x = modify_node_attr(orig_x, args.modify_rate, args.modify_num, args.in_mol)
            x_list.append(modified_x)

            # define batch indices
            node_idx_ = node_idx - node_idx[0] + node_cnt
            tmp_batch = torch.ones(node_idx_.shape[0], device=device) * batch_idx
            tmp_batch = tmp_batch.to(batch.dtype)
            batch_list.append(tmp_batch)

            # retain edges and their attributes
            edge_idx = torch.nonzero(
                (edge_index[0, :] >= node_idx[0]).float() * (edge_index[0, :] <= node_idx[-1]).float()).squeeze(-1)
            tmp_edge_index = edge_index[:, edge_idx] - node_idx[0] + node_cnt
            edge_index_list.append(tmp_edge_index)
            if edge_attr is not None:
                tmp_edge_attr = edge_attr[edge_idx, :]
                edge_attr_list.append(tmp_edge_attr)

            node_cnt += tmp_batch.shape[0]

        # only modify the structure of a graph
        else:
            # get another sample with most similar label
            node_idx = torch.nonzero((batch == batch_idx).float()).squeeze(-1)
            edge_idx = torch.nonzero(
                (edge_index[0, :] >= node_idx[0]).float() * (edge_index[0, :] <= node_idx[-1]).float()).squeeze(-1)
            edge_idx_last = torch.nonzero((edge_index[0, :] == node_idx[-1]).float()).squeeze(-1)

            # select the remaining edges after dropout
            num_node = node_idx.shape[0]
            num_edge = edge_idx.shape[0]

            if args.drop_num > 0:
                sample_size = max(1, int(num_edge - args.drop_num))
            else:
                sample_size = max(1, int(num_edge * (1 - args.drop_rate)))
            sample_size = min(sample_size, num_edge)
            remain_edge_idx = random.sample(edge_idx.tolist(), sample_size)
            remain_edge_idx = remain_edge_idx + edge_idx_last.tolist()

            tmp_edge_index = edge_index[:, remain_edge_idx]
            tmp_edge_index_ = copy.deepcopy(tmp_edge_index)
            if not args.in_mol:
                tmp_edge_index, _ = coalesce(tmp_edge_index, None, num_node, num_node)
            tmp_edge_index = tmp_edge_index - node_idx[0] + node_cnt
            edge_index_list.append(tmp_edge_index)

            if edge_attr is not None:
                tmp_edge_attr = edge_attr[remain_edge_idx, :]
                if not args.in_mol:
                    _, tmp_edge_attr = coalesce(tmp_edge_index_, tmp_edge_attr, num_node, num_node)
                edge_attr_list.append(tmp_edge_attr)

            # update the node attribute
            if not args.in_mol:
                tmp_x = degree_attr(tmp_edge_index, args.max_degree)
                tmp_x = tmp_x[-node_idx.shape[0]:, :].to(device)
                x_list.append(tmp_x)
            else:
                tmp_x = x[node_idx, :]
                x_list.append(tmp_x)

            # define batch indices
            node_idx_ = node_idx - node_idx[0] + node_cnt
            tmp_batch = torch.ones(node_idx_.shape[0], device=device) * batch_idx
            tmp_batch = tmp_batch.to(batch.dtype)
            batch_list.append(tmp_batch)

            node_cnt += tmp_batch.shape[0]

    # constitute a new mini-batch
    x_ = torch.cat(x_list, dim = 0)
    data.x = x_

    edge_index_ = torch.cat(edge_index_list, dim = 1)
    data.edge_index = edge_index_

    if edge_attr is not None:
        edge_attr_ = torch.cat(edge_attr_list, dim = 0)
        data.edge_attr = edge_attr_

    batch_ = torch.cat(batch_list, dim = 0)
    data.batch = batch_

    return data

def train(args, epoch, model, classifier, device, loader, optimizer, scheduler):
    model.eval()
    classifier.train()
    scheduler.step()
    criterion = nn.CrossEntropyLoss()

    loss_cls_cnt = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        batch_ = copy.deepcopy(batch)
        y = (torch.rand(int(batch.batch[-1] + 1), device = device) > 0.5).long()
        batch_ = create_pair(args, batch_, y, device)

        graph_emb, _, _, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        graph_emb_, _, _, _, _ = model(batch_.x, batch_.edge_index, batch_.edge_attr, batch_.batch)

        diff_emb = torch.abs(graph_emb - graph_emb_)
        pred = classifier(diff_emb)
        loss_cls = criterion(pred, y)
        loss_cls_cnt += loss_cls.item()

        optimizer.zero_grad()
        loss_cls.backward()
        optimizer.step()

    print('Epoch: {:04d}\tCls Loss: {:.5f}'.format(epoch, loss_cls_cnt / len(loader)))

    return

def eval(args, model, classifier, device, loader):
    model.eval()
    classifier.eval()
    total_cnt = 0
    correct_cnt = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        batch_ = copy.deepcopy(batch)
        y = (torch.rand(int(batch.batch[-1] + 1), device=device) > 0.5).long()
        batch_ = create_pair(args, batch_, y, device)

        graph_emb, _, _, _, _ = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
        graph_emb_, _, _, _, _ = model(batch_.x, batch_.edge_index, batch_.edge_attr, batch_.batch)

        diff_emb = torch.abs(graph_emb - graph_emb_)
        output = classifier(diff_emb)
        pred = output.max(1)[1]
        correct_cnt += pred.eq(y).sum().item()
        total_cnt += y.shape[0]

    return float(correct_cnt) / float(total_cnt)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Evaluation procedure for SAD-Metric')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=3,
                        help='number of GNN message passing layers (default: 3).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gcn",
                        help="the Graph Neural Network to use")
    parser.add_argument('--num_heads', type=int, default=5,
                        help='the number of attention heads for GAT (default: 5)')
    parser.add_argument('--dataset', type=str, default = 'bbbp',
                        help='the name of dataset. For now, only classification.')
    parser.add_argument('--data_path', type=str, default = './data',
                        help='the path to store data')
    parser.add_argument('--input_model_file', type=str, default = '',
                        help='the model to load from (if there is any)')
    parser.add_argument('--seed', type=int, default=100,
                        help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=None,
                        help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', action='store_true', default=False,
                        help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 1,
                        help='number of workers for dataset loading')
    # parameters for evaluating representation disentanglement
    parser.add_argument('--modify_rate', type=float, default = 0.05,
                        help='the rate of modified atoms')
    parser.add_argument('--modify_num', type=int, default=1,
                        help='the number of modified atoms')
    parser.add_argument('--drop_rate', type=float, default=0.05,
                        help='the rate of dropped edges')
    parser.add_argument('--drop_num', type=int, default=1,
                        help='the number of dropped edge')
    args = parser.parse_args()

    # Set the seeds for a run
    if args.runseed:
        runseed = args.runseed
        print ('Manual runseed: ', runseed)
    else:
        runseed = random.randint(0, 10000)
        print ('Random runseed: ', runseed)

    if args.seed:
        seed = args.seed
        print ('Manual seed: ', seed)
    else:
        seed = random.randint(0, 10000)
        print ('Random seed: ', seed)

    torch.manual_seed(runseed)
    np.random.seed(runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(runseed)

    # Load and split the dataset
    dataset_ = Dataset_(args, args.dataset, args.data_path)
    dataset = dataset_.dataset
    args.max_degree = dataset_.max_degree
    in_mol = args.dataset in dataset_.moleculenet_list
    args.in_mol = in_mol

    train_dataset, val_dataset, test_dataset = \
        dataset_.get_scaffold_split(train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1)

    print (dataset)
    print (train_dataset[0])

    # Set up dataloader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    # Set up model
    # in_dim = dataset.num_features
    in_dim = args.max_degree + 1
    out_dim = dataset_.num_task if in_mol else dataset.num_classes
    model = Embed_SAD_model(args.num_layer, in_dim, args.emb_dim, out_dim, JK = args.JK,
                            drop_ratio = args.dropout_ratio, graph_pooling = args.graph_pooling,
                            gnn_type = args.gnn_type, in_mol = in_mol, num_heads = args.num_heads,
                            return_graph_emb = True)
    if not args.input_model_file == "":
        model.from_pretrained(args.input_model_file)

    classifier = Node_Classifier(args.num_layer, args.emb_dim, 2, JK=args.JK, graph_pooling=args.graph_pooling)
    
    model.to(device)
    classifier.to(device)
    print(model)
    print(classifier)

    # Set up optimizer
    optimizer = optim.Adam(classifier.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    print(optimizer)

    # Training
    best_test_acc = 0
    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch), " lr: ", optimizer.param_groups[-1]['lr'])
        
        train(args, epoch, model, classifier, device, train_loader, optimizer, scheduler)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval(args, model, classifier, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval(args, model, classifier, device, val_loader)
        test_acc = eval(args, model, classifier, device, test_loader)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print("train: %f val: %f test: %f best test: %f" %(train_acc, val_acc, test_acc, best_test_acc))
        print("")

    os.system('watch nvidia-smi')

if __name__ == "__main__":
    main()
