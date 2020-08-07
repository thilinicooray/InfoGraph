import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip
from torch_geometric.io import read_tu_data

from torch_geometric.transforms import LineGraph


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        items =  tuple(d[i] for d in self.datasets)

        print('items ', items)

        return items

    def __len__(self):
        return min(len(d) for d in self.datasets)