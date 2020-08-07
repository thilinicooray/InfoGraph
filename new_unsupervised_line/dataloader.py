import os
import os.path as osp
import shutil

import torch
from torch_geometric.data import InMemoryDataset, download_url, extract_zip, Data
from torch_geometric.io import read_tu_data

from torch.utils.data.dataloader import default_collate

from torch_geometric.data import Data, Batch

from torch_geometric.transforms import LineGraph

from torch._six import container_abcs, string_classes, int_classes

class Collater(object):
    def __init__(self, follow_batch):
        self.follow_batch = follow_batch

    def collate(self, batch):

        #return (Batch.from_data_list(batch['node'], self.follow_batch), Batch.from_data_list(batch['line'], self.follow_batch))

        elem = batch['node']
        if isinstance(elem, Data):
            print('came to data')
            return Batch.from_data_list(batch, self.follow_batch)
        elif isinstance(elem, torch.Tensor):
            print('came to tensor')
            return default_collate(batch)
        elif isinstance(elem, float):
            print('came to float')
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            print('came to int')
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            print('came to string')
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            print('came to map')
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            print('came to field')
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            print('came to seq')
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))


    def __call__(self, batch):
        return self.collate(batch)


class DataLoader(torch.utils.data.DataLoader):
    r"""Data loader which merges data objects from a
    :class:`torch_geometric.data.dataset` to a mini-batch.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (list or tuple, optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`[]`)
    """

    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[],
                 **kwargs):
        super(DataLoader,
              self).__init__(dataset, batch_size, shuffle,
                             collate_fn=Collater(follow_batch), **kwargs)





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