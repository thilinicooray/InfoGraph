import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_geometric.io import read_tu_data

from torch_geometric.transforms import LineGraph


class PairData(Data):
    def __init__(self, edge_index_s, x_s, edge_index_t, x_t):
        super(PairData, self).__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.x_t = x_t


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_map):
        self.dataset_map = dataset_map

    def __getitem__(self, i):
        #items =  tuple(d[i] for d in self.datasets)

        items_map = {}

        for key, dataset in self.dataset_map.items():
            items_map[key] = dataset[i]

        print('items ', items_map)

        return items_map

    def __len__(self):
        return len(self.dataset_map['node'])