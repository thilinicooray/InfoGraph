import torch
import numpy as np
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch
import random
import os


seed = 1234

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)



num_nodes = 50
edge_prob_list = [0.1, 0.3, 0.5, 0.7, 0.9]

for i in range(5):

    for edge_prob in edge_prob_list:

        idx = torch.combinations(torch.arange(num_nodes))

        size = idx.size(0)


        a = torch.rand(idx.size(0))


        mask = a  < edge_prob

        #int_mask = mask.astype(int)

        #print('mask', mask.size(),torch.sum(mask).item(), (torch.sum(mask)/1125).item())

        idx = idx[mask]


        edge_index = to_undirected(idx.t(), num_nodes)

        org_adj = to_dense_adj(edge_index)




        #p = torch.sum(org_adj).item()/2


        adj = np.tril((org_adj.cpu().numpy()),k=-1)


        print('adj ',i, edge_prob, np.sum(adj)/size)

        #break