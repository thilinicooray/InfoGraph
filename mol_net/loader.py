import os, sys
import pandas as pd

import torch
from torch_geometric.utils import degree
import torch_geometric.datasets as datasets
import torch_geometric.transforms as T

from molecule_loader import MoleculeDataset
from splitters import cv_random_split, scaffold_split

class Dataset_(object):
    def __init__(self, args, name, data_path):
        self.args = args
        self.name = name
        self.data_path = data_path
        self.moleculenet_list = {'bbbp': 1, 'tox21': 12, 'toxcast': 617, 'sider': 27, 'clintox': 2, 'muv': 17,
                                 'hiv': 1, 'bace': 1}

        self.load_dataset()

    # Load the dataset
    def load_dataset(self):
        # Molecule dataset
        if self.name in self.moleculenet_list:
            path = os.path.join(self.data_path, 'MoleculeNet', self.name)
            self.dataset = MoleculeDataset(path, dataset = self.name)
            self.num_task = self.moleculenet_list[self.name]
        # Citation network
        elif self.name == 'cora':
            path = os.path.join(self.data_path, 'Planetoid', self.name)
            self.dataset = datasets.Planetoid(path, name = 'Cora')
        elif self.name == 'citeseer':
            path = os.path.join(self.data_path, 'Planetoid', self.name)
            self.dataset = datasets.Planetoid(path, name = 'CiteSeer')
        elif self.name == 'pubmed':
            path = os.path.join(self.data_path, 'Planetoid', self.name)
            self.dataset = datasets.Planetoid(path, name = 'PubMed')
        # Co-authorship graphs
        elif self.name == 'cs':
            path = os.path.join(self.data_path, 'Coauthor', self.name)
            self.dataset = datasets.Coauthor(path, name='CS')
        elif self.name == 'physics':
            path = os.path.join(self.data_path, 'Coauthor', self.name)
            self.dataset = datasets.Coauthor(path, name='Physics')
        # Invalid dataset
        else:
            raise ValueError("Invalid dataset name.")

        # Get the transform function for specific dataset
        self.get_transform()

        return

    # Define the transform function
    def get_transform(self):
        if self.dataset.data.x is None:
            max_degree = 0
            degs = list()
            for data in self.dataset:
                degs += [degree(data.edge_index[0], dtype = torch.long)]
                max_degree = max(max_degree, degs[-1].max().item())

            self.max_degree = max_degree
            self.dataset.transform = T.OneHotDegree(self.max_degree)
        else:
            self.max_degree = self.dataset.data.x.shape[1] - 1

        return

    # Return a fold for cross validation
    def get_cv_fold(self, seed, fold_idx = 0):
        self.train_dataset, self.val_dataset = cv_random_split(self.dataset, fold_idx = fold_idx, seed = seed)

        return self.train_dataset, self.val_dataset

    # Split dataset by Bemis-Murcko scaffolds (for MoleculeNet)
    def get_scaffold_split(self, train_ratio = 0.8, val_ratio = 0.1, test_ratio = 0.1):
        assert self.name in self.moleculenet_list, 'The dataset is not in MoleculeNet'
        smiles_file = os.path.join(self.data_path, 'MoleculeNet', self.name, 'processed', 'smiles.csv')
        smiles_list = pd.read_csv(smiles_file, header=None)[0].tolist()
        self.train_dataset, self.val_dataset, self.test_dataset = \
            scaffold_split(self.dataset, smiles_list, frac_train = train_ratio, frac_valid = val_ratio,
                           frac_test = test_ratio)

        return self.train_dataset, self.val_dataset, self.test_dataset