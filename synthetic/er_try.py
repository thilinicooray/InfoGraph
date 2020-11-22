import torch
import numpy as np
from torch_geometric.utils import to_undirected, remove_self_loops
from torch_geometric.utils import negative_sampling, remove_self_loops, add_self_loops, to_dense_adj, to_dense_batch



num_nodes = 50
edge_prob = 0.1

idx = torch.combinations(torch.arange(num_nodes))

print('idx ', idx.size(), idx)

a = torch.rand(idx.size(0))

print('a', a)

mask = a  < edge_prob

print('mask', mask.size(), torch.sum(mask), torch.sum(mask)/1125, mask)

idx = idx[mask]

print('after ', idx)


edge_index = to_undirected(idx.t(), num_nodes)

org_adj = to_dense_adj(edge_index)

print('adj ', org_adj, torch.sum(org_adj))