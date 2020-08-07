import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data

from torch_geometric.transforms import LineGraph


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
        return min(len(d) for d in self.datasets)