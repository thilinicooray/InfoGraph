import argparse

from loader import Dataset_
from torch_geometric.data import DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm
import os, sys
import numpy as np
import random

from model import Embed_SAD_model, MI_estimator, Node_Classifier
from sklearn.metrics import roc_auc_score
from scipy.stats import beta

from splitters import scaffold_split
from util import degree_attr

import os
import shutil
import pdb
import pandas as pd

from tensorboardX import SummaryWriter

def random_split(dataset, num_classes):
    train_mask = torch.zeros(dataset.x.shape[0])
    val_mask = torch.zeros(dataset.x.shape[0])
    test_mask = torch.zeros(dataset.x.shape[0])

    for cls_id in range(num_classes):
        avail_indices = torch.nonzero((dataset.y == cls_id).float()).squeeze(1).tolist()
        train_size = 20 if len(avail_indices) >= 100 else int(len(avail_indices) * 0.2)
        train_indices = random.sample(avail_indices, train_size)
        train_mask[train_indices] = 1

        avail_indices_ = torch.nonzero((dataset.y == cls_id).float() * (train_mask == 0).float()).squeeze(1).tolist()
        val_size = 30 if len(avail_indices) >= 100 else int(len(avail_indices) * 0.3)
        val_indices = random.sample(avail_indices_, val_size)
        val_mask[val_indices] = 1

        test_indices = torch.nonzero(
            (dataset.y == cls_id).float() * (train_mask == 0).float() * (val_mask == 0).float()).squeeze(1).tolist()
        test_mask[test_indices] = 1

    return (train_mask == 1), (val_mask == 1), (test_mask == 1)

def train(args, epoch, model, estimator, classifier, device, dataset, optimizer, optimizer_MI, scheduler, scheduler_MI):
    model.train()
    estimator.train()
    classifier.train()
    scheduler.step()
    scheduler_MI.step()
    criterion = nn.CrossEntropyLoss()

    dataset = dataset.to(device)
    dataset.batch = torch.zeros(dataset.x.shape[0], device = device).long()
    node_emb, _, loss_recon, attribute_emb, structure_emb = model(dataset.x, dataset.edge_index, dataset.edge_attr,
                                                                 dataset.batch)
    pred = classifier(node_emb)

    if args.dataset in ['pubmed', 'cs', 'physics']:
        sample_idx = random.sample(range(dataset.x.shape[0]), args.sample_size)
        attribute_emb_ = attribute_emb[sample_idx, :]
        structure_emb_ = structure_emb[sample_idx, :]
        batch_ = torch.zeros(args.sample_size, device = device).long()
        loss_MI = estimator(attribute_emb_, structure_emb_, batch_)
    else:
        loss_MI = estimator(attribute_emb, structure_emb, dataset.batch)

    loss_cls = criterion(pred[dataset.train_mask], dataset.y[dataset.train_mask])
    loss = loss_cls + args.lambda_recon * loss_recon - args.lambda_MI_min * loss_MI

    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()

    if args.dataset in ['pubmed', 'cs', 'physics']:
        loss_estimator = estimator(attribute_emb_.detach(), structure_emb_.detach(), batch_)
    else:
        loss_estimator = estimator(attribute_emb.detach(), structure_emb.detach(), dataset.batch)
    optimizer_MI.zero_grad()
    loss_estimator.backward()
    optimizer_MI.step()

    print('Epoch: {:04d}\tCls Loss: {:.5f}\tRecon Loss: {:.5f}\tMI Loss: {:.5f}'.format(
        epoch, loss_cls.item(), loss_recon.item(), loss_MI.item()))

    return

def eval(args, model, classifier, device, dataset):
    model.eval()
    classifier.eval()

    dataset = dataset.to(device)
    dataset.batch = torch.zeros(dataset.x.shape[0], device=device).long()
    node_emb, _, loss_recon, attribute_emb, structure_emb = model(dataset.x, dataset.edge_index, dataset.edge_attr,
                                                                 dataset.batch)
    _, pred = classifier(node_emb).max(dim=1)
    target = dataset.y

    if args.eval_train:
        train_correct = pred[dataset.train_mask].eq(target[dataset.train_mask]).sum().item()
        train_number = dataset.train_mask.sum().item()
        train_acc = float(train_correct) / float(train_number)
    else:
        train_acc = 0

    val_correct = pred[dataset.val_mask].eq(target[dataset.val_mask]).sum().item()
    val_number = dataset.val_mask.sum().item()
    val_acc = float(val_correct) / float(val_number)

    test_correct = pred[dataset.test_mask].eq(target[dataset.test_mask]).sum().item()
    test_number = dataset.test_mask.sum().item()
    test_acc = float(test_correct) / float(test_number)

    return train_acc, val_acc, test_acc

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Embed-SAD for node classification')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=2,
                        help='number of GNN message passing layers (default: 2).')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="gcn",
                        help="the Graph Neural Network to use")
    parser.add_argument('--num_heads', type=int, default=5,
                        help='the number of attention heads for GAT (default: 5)')
    parser.add_argument('--dataset', type=str, default = 'cora',
                        help='the name of dataset. For now, only classification.')
    parser.add_argument('--data_path', type=str, default = './data',
                        help='the path to store data')
    parser.add_argument('--input_model_file', type=str, default = '',
                        help='the model to load from (if there is any)')
    parser.add_argument('--output_model_file', type=str, default = '',
                        help='the path to save model')
    parser.add_argument('--filename', type=str, default = '',
                        help='the name of output model')
    parser.add_argument('--result_file', type=str, default='',
                        help='the file to store results')
    parser.add_argument('--seed', type=int, default=100,
                        help = "Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=None,
                        help = "Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', action='store_true', default=False,
                        help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default = 1,
                        help='number of workers for dataset loading')
    # parameters for representation disentanglement
    parser.add_argument('--lambda_recon', type=float, default=1.0,
                        help='the weight of embedding reconstruction loss')
    parser.add_argument('--lambda_MI_min', type=float, default=0.1,
                        help='the weight of mutual information minimization')
    parser.add_argument('--tau', type=float, default=0.05,
                        help='the temperature parameter')
    parser.add_argument('--sample_size', type=int, default=3000,
                        help="the size of sampled nodes")
    args = parser.parse_args()

    if args.dataset == 'cora':
        args.decay = 0
        args.lr = 0.001
    elif args.dataset == 'citeseer':
        args.decay = 0.005
        args.lr = 0.001
    elif args.dataset == 'pubmed':
        args.decay = 0.0005
        args.lr = 0.01
    elif args.dataset == 'cs':
        args.decay = 0.001
        args.lr = 0.0005
        args.lambda_recon = 0.01
    elif args.dataset == 'physics':
        args.decay = 0
        args.lr = 0.001
        args.lambda_recon = 0.01

    print (args.lr, args.decay)

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
    network = dataset[0]
    args.max_degree = dataset_.max_degree
    in_mol = args.dataset in dataset_.moleculenet_list
    args.in_mol = in_mol

    if args.dataset in ['cs', 'physics']:
        train_mask, val_mask, test_mask = random_split(network, dataset.num_classes)
        network.train_mask = train_mask
        network.val_mask = val_mask
        network.test_mask = test_mask

    print(network)

    # Set up model
    # in_dim = dataset.num_features
    in_dim = args.max_degree + 1
    out_dim = dataset_.num_task if in_mol else dataset.num_classes

    model = Embed_SAD_model(args.num_layer, in_dim, args.emb_dim, out_dim, JK = args.JK,
                            drop_ratio = args.dropout_ratio, gnn_type = args.gnn_type, in_mol = in_mol,
                            num_heads = args.num_heads, return_node_emb=True, use_bn=False, use_proj=False)
    if args.JK == 'concat':
        if args.dataset in ['pubmed', 'cs', 'physics']:
            estimator = MI_estimator(int(args.emb_dim / 2) * args.num_layer, tau=args.tau, save_space=True)
        else:
            estimator = MI_estimator(int(args.emb_dim / 2) * args.num_layer, tau = args.tau)
    else:
        if args.dataset in ['pubmed', 'cs', 'physics']:
            estimator = MI_estimator(int(args.emb_dim / 2), tau=args.tau, save_space=True)
        else:
            estimator = MI_estimator(int(args.emb_dim / 2), tau = args.tau)
    classifier = Node_Classifier(args.num_layer, args.emb_dim, out_dim, JK=args.JK)
    
    model.to(device)
    estimator.to(device)
    classifier.to(device)
    print(model)
    print(estimator)
    print(classifier)

    # Set up optimizer
    param_list = list(model.parameters())
    param_list += list(classifier.parameters())
    optimizer = optim.Adam(param_list, lr=args.lr, weight_decay=args.decay)
    optimizer_MI = optim.Adam(estimator.parameters(), lr=args.lr, weight_decay=args.decay)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.5)
    scheduler_MI = StepLR(optimizer_MI, step_size=200, gamma=0.5)
    print(optimizer)
    print(optimizer_MI)

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
    
    # record the training parameters.
    results = open(os.path.join(fname, 'results.txt'), 'a')
    results.write(str(args) + '\n')
    results.close()

    for epoch in range(1, args.epochs+1):
        print("====epoch " + str(epoch), " lr: ", optimizer.param_groups[-1]['lr'])
        
        train(args, epoch, model, estimator, classifier, device, network, optimizer, optimizer_MI,
              scheduler, scheduler_MI)

        print("====Evaluation")
        train_acc, val_acc, test_acc = eval(args, model, classifier, device, network)

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
            writer.add_scalar('data/train acc', train_acc, epoch)
            writer.add_scalar('data/val acc', val_acc, epoch)
            writer.add_scalar('data/test acc', test_acc, epoch)

        print("")

    if not args.filename == "":
        writer.close()

    if not args.result_file == "":
        outfile = open(args.result_file, 'a')
        outfile.write(str(best_test_acc) + '\n')
        outfile.close()

    if not args.output_model_file == "":
        torch.save({
            'gnn': model.gnn.state_dict(),
            'proj': model.proj.state_dict(),
            'classifier': classifier.state_dict()
        }, args.output_model_file)

    # os.system('watch nvidia-smi')

if __name__ == "__main__":
    main()
