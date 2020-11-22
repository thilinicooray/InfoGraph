import torch
import numpy as np
from torch_geometric.utils import to_undirected, remove_self_loops



num_nodes = 50
edge_prob = 0.1

idx = torch.combinations(torch.arange(num_nodes))

print('idx ', idx.size())

mask = torch.rand(idx.size(0)) < edge_prob

print('mask', mask.size())

idx = idx[mask]