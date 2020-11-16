import torch
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm
from torch_geometric.utils import erdos_renyi_graph
import numpy as np
import random
import os
import os.path as osp
from torch_geometric.data import DataLoader

class SyntheticERDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(SyntheticERDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []
    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):

        data_list = []
        n = 10

        for i in range(4000):
            p = np.random.uniform(0,1)
            g_val = np.array([np.random.randint(1,10)])
            mean_1 = np.array([np.random.randint(2,9)])
            mean_2 = np.array([np.random.randint(2,9)])
            g_id = torch.from_numpy(g_val).unsqueeze(0).float()
            g = g_id.expand(n,1)
            node_feat1 = torch.normal(mean_1, 1, (n,1))
            node_feat2 = torch.normal(mean_2, 1, (n,1))
            edge_index = erdos_renyi_graph(n,p)

            x = torch.cat([node_feat1,node_feat2,g], -1)

            print('data x', x)

            data = Data(x=x, edge_index=edge_index)
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':

    seed = 1234

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

    path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'SyntheticER')

    dataset = SyntheticERDataset(path).shuffle()

    train_dataset = dataset[:3000]
    test_dataset = dataset[3000:]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader = DataLoader(train_dataset, batch_size=512)

    for data in train_loader:
        data = data.to(device)
        #print('data ', data.batch, data.x)
        #break



