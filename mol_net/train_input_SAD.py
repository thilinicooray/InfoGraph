import argparse

from loader import Dataset_
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree, add_self_loops

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import os, sys
import numpy as np
import random

from model import GNN, FCEncoder, Graph_Classifier
from sklearn.metrics import roc_auc_score
from scipy.stats import beta

from splitters import scaffold_split
from util import degree_attr

import os
import shutil
import pdb
import pandas as pd

from tensorboardX import SummaryWriter

def train(args, epoch, struct_encoder, attr_encoder, classifier, device, loader, optimizer, scheduler):
    struct_encoder.train()
    attr_encoder.train()
    classifier.train()
    scheduler.step()
    criterion = nn.CrossEntropyLoss()

    loss_cls_cnt = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        # edge_index_ = add_self_loops(batch.edge_index, num_nodes=batch.x.size(0))[0]
        edge_index_ = batch.edge_index
        x_ = degree_attr(edge_index_, args.max_degree)
        x_ = x_.to(device)

        struct_node_emb = struct_encoder(x_, batch.edge_index, batch.edge_attr)
        attr_node_emb = attr_encoder(batch.x, batch.edge_index, None)
        node_emb = torch.cat([struct_node_emb, attr_node_emb], dim = 1)
        pred = classifier(node_emb, batch.batch)

        loss_cls = criterion(pred, batch.y)
        loss_cls_cnt += loss_cls.item()

        optimizer.zero_grad()
        loss_cls.backward()
        optimizer.step()

    print('Epoch: {:04d}\tCls Loss: {:.5f}'.format(epoch, loss_cls_cnt / len(loader)))

    return

def train_mol(args, epoch, struct_encoder, attr_encoder, classifier, device, loader, optimizer, scheduler):
    struct_encoder.train()
    attr_encoder.train()
    classifier.train()
    scheduler.step()
    criterion = nn.BCEWithLogitsLoss(reduce = False)

    loss_cls_cnt = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        edge_index_ = add_self_loops(batch.edge_index, num_nodes=batch.x.size(0))[0]
        x_ = degree_attr(edge_index_, args.max_degree_)
        x_ = x_.to(device)

        struct_node_emb = struct_encoder(x_, batch.edge_index, batch.edge_attr)
        attr_node_emb = attr_encoder(batch.x, batch.edge_index, None)
        node_emb = torch.cat([struct_node_emb, attr_node_emb], dim=1)
        pred = classifier(node_emb, batch.batch)

        y = batch.y.view(pred.shape).to(torch.float64)
        is_valid = (y ** 2) > 0
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        loss_mat = torch.where(is_valid, loss_mat,
                               torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))
        loss_cls = torch.sum(loss_mat) / torch.sum(is_valid)
        loss_cls_cnt += loss_cls.item()

        optimizer.zero_grad()
        loss_cls.backward()
        optimizer.step()

    print('Epoch: {:04d}\tCls Loss: {:.5f}'.format(epoch, loss_cls_cnt / len(loader)))

    return

def eval(args, struct_encoder, attr_encoder, classifier, device, loader):
    struct_encoder.eval()
    attr_encoder.eval()
    classifier.eval()
    total_cnt = 0
    correct_cnt = 0

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if batch.x is None:
            batch.x = degree_attr(batch.edge_index, args.max_degree)
        batch = batch.to(device)
        # edge_index_ = add_self_loops(batch.edge_index, num_nodes=batch.x.size(0))[0]
        edge_index_ = batch.edge_index
        x_ = degree_attr(edge_index_, args.max_degree)
        x_ = x_.to(device)

        with torch.no_grad():
            struct_node_emb = struct_encoder(x_, batch.edge_index, batch.edge_attr)
            attr_node_emb = attr_encoder(batch.x, batch.edge_index, None)
            node_emb = torch.cat([struct_node_emb, attr_node_emb], dim=1)
            output = classifier(node_emb, batch.batch)
        pred = output.max(1)[1]
        correct_cnt += pred.eq(batch.y).sum().item()
        total_cnt += batch.y.shape[0]

    return float(correct_cnt) / float(total_cnt)

def eval_mol(args, struct_encoder, attr_encoder, classifier, device, loader):
    struct_encoder.eval()
    attr_encoder.eval()
    classifier.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        if batch.x is None:
            batch.x = degree_attr(batch.edge_index, args.max_degree)
        batch = batch.to(device)
        edge_index_ = add_self_loops(batch.edge_index, num_nodes=batch.x.size(0))[0]
        x_ = degree_attr(edge_index_, args.max_degree_)
        x_ = x_.to(device)

        with torch.no_grad():
            struct_node_emb = struct_encoder(x_, batch.edge_index, batch.edge_attr)
            attr_node_emb = attr_encoder(batch.x, batch.edge_index, None)
            node_emb = torch.cat([struct_node_emb, attr_node_emb], dim=1)
            pred = classifier(node_emb, batch.batch)
        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim = 0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim = 0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        #AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:,i] == 1) > 0 and np.sum(y_true[:,i] == -1) > 0:
            is_valid = y_true[:,i]**2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid,i] + 1)/2, y_scores[is_valid,i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" %(1 - float(len(roc_list))/y_true.shape[1]))

    return sum(roc_list)/len(roc_list)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Input-SAD for graph classification')
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
    parser.add_argument('--emb_dim', type=int, default=150,
                        help='embedding dimensions (default: 150)')
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
    parser.add_argument('--output_model_file', type=str, default = '',
                        help='the path to save model')
    parser.add_argument('--filename', type=str, default = '',
                        help='the name of output model')
    parser.add_argument('--seed', type=int, default=100,
                        help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=None,
                        help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', action='store_true', default=False,
                        help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 1,
                        help='number of workers for dataset loading')
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

    max_degree_ = 0
    degs = list()
    for data in dataset:
        degs += [degree(data.edge_index[0], data.x.shape[0], dtype=torch.long)]
        max_degree_ = max(max_degree_, degs[-1].max().item())
    args.max_degree_ = max_degree_ + 1

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
    if in_mol:
        in_dim = args.max_degree_ + 1
    else:
        in_dim = args.max_degree + 1
    out_dim = dataset_.num_task if in_mol else dataset.num_classes

    struct_encoder = GNN(args.num_layer, in_dim, args.emb_dim, args.JK, args.dropout_ratio,
                         gnn_type = args.gnn_type, in_mol = in_mol, num_heads = args.num_heads)
    attr_encoder = FCEncoder(args.num_layer, in_dim, args.emb_dim, args.dropout_ratio, in_mol = args.in_mol)
    classifier = Graph_Classifier(args.num_layer, 2 * args.emb_dim, out_dim, args.JK,
                                  args.graph_pooling, nonlinear = True)
    
    struct_encoder.to(device)
    attr_encoder.to(device)
    classifier.to(device)
    print(struct_encoder)
    print(attr_encoder)
    print(classifier)

    # Set up optimizer
    param_list = list(struct_encoder.parameters())
    param_list += list(attr_encoder.parameters())
    param_list += list(classifier.parameters())
    optimizer = optim.Adam(param_list, lr=args.lr, weight_decay=args.decay)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.5)
    print(optimizer)

    # Training
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    best_test_acc = 0

    if not args.filename == "":
        fname = os.path.join('runs', args.filename, 'seed_' + str(seed))
        #delete the directory if there exists one
        if os.path.exists(fname):
            shutil.rmtree(fname)
            print("removed the existing file.")
        writer = SummaryWriter(fname)

    if args.dataset in dataset_.moleculenet_list:
        train_func = train_mol
        eval_func = eval_mol
    else:
        train_func = train
        eval_func = eval
    
    # record the training parameters.
    results = open(os.path.join(fname, 'results.txt'), 'a')
    results.write(str(args) + '\n')
    results.close()

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch), " lr: ", optimizer.param_groups[-1]['lr'])
        
        train_func(args, epoch, struct_encoder, attr_encoder, classifier, device, train_loader, optimizer, scheduler)

        print("====Evaluation")
        if args.eval_train:
            train_acc = eval_func(args, struct_encoder, attr_encoder, classifier, device, train_loader)
        else:
            print("omit the training accuracy computation")
            train_acc = 0
        val_acc = eval_func(args, struct_encoder, attr_encoder, classifier, device, val_loader)
        test_acc = eval_func(args, struct_encoder, attr_encoder, classifier, device, test_loader)

        if test_acc > best_test_acc:
            best_test_acc = test_acc

        print("train: %f val: %f test: %f best test: %f" %(train_acc, val_acc, test_acc, best_test_acc))
        
        results = open(os.path.join(fname, 'results.txt'), 'a')
        results.write('Epoch {:>3}, train_acc {:.5f} val_acc {:.5f} test_acc {:.5f}  best_test_acc {:.5f} :\n'.format(
                epoch, train_acc, val_acc, test_acc, best_test_acc))
        results.close()

        val_acc_list.append(val_acc)
        test_acc_list.append(test_acc)
        train_acc_list.append(train_acc)

        if not args.filename == "":
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()

    if not args.output_model_file == "":
        torch.save({
            'struct_encoder': struct_encoder.state_dict(),
            'attr_encoder': attr_encoder.state_dict(),
            'classifier': classifier.state_dict()
        }, args.output_model_file)

    os.system('watch nvidia-smi')

if __name__ == "__main__":
    main()
