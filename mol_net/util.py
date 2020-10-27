import copy
import random
import pdb

import torch
from torch.nn.functional import one_hot
from torch_sparse import spspmm, coalesce
from torch_geometric.utils import add_self_loops, remove_self_loops, to_dense_adj, degree

# get the node attributes based on degree
def degree_attr(edge_index, max_degree, in_degree = False):
    idx = edge_index[1 if in_degree else 0]
    deg = degree(idx, dtype = torch.long)
    deg = one_hot(deg, num_classes = max_degree + 1).to(torch.float)

    return deg

# randomly dropout some edges in graphs
def batch_drop(args, data):
    x = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr
    batch = data.batch
    num_graph = batch[-1] + 1

    x_ = list()
    edge_index_ = list()
    edge_attr_ = list()

    for batch_idx in range(num_graph):
        # get node and edge indices of original graph
        node_idx = torch.nonzero((batch == batch_idx).float()).squeeze(-1)
        edge_idx = torch.nonzero(
            (edge_index[0, :] >= node_idx[0]).float() * (edge_index[0, :] <= node_idx[-1]).float()).squeeze(-1)
        edge_idx_last = torch.nonzero((edge_index[0, :] == node_idx[-1]).float()).squeeze(-1)

        # if a graph contains no edge
        if edge_idx.shape[0] == 0:
            if x is None:
                tmp_x = torch.zeros((node_idx.shape[0], args.max_degree + 1)).to(torch.float)
                tmp_x[:, 0] = 1.
                x_.append(tmp_x)

            continue

        # select the remaining edges after dropout
        num_node = node_idx.shape[0]
        num_edge = edge_idx.shape[0]
        sample_size = max(1, int(num_edge * (1 - args.drop_rate)))
        remain_edge_idx = random.sample(edge_idx.tolist(), sample_size)
        remain_edge_idx = remain_edge_idx + edge_idx_last.tolist()

        tmp_edge_index = edge_index[:, remain_edge_idx]
        tmp_edge_index_ = copy.deepcopy(tmp_edge_index)
        if not args.in_mol:
            tmp_edge_index, _ = coalesce(tmp_edge_index, None, num_node, num_node)
        edge_index_.append(tmp_edge_index)
        if edge_attr is not None:
            tmp_edge_attr = edge_attr[remain_edge_idx, :]
            if not args.in_mol:
                _, tmp_edge_attr = coalesce(tmp_edge_index_, tmp_edge_attr, num_node, num_node)
            edge_attr_.append(tmp_edge_attr)

        if x is None:
            tmp_x = degree_attr(tmp_edge_index, args.max_degree)
            tmp_x = tmp_x[-node_idx.shape[0]:, :]
            x_.append(tmp_x)

        # if args.in_mol:
        #     edge_idx_even = torch.nonzero(((edge_idx % 2) == 0).float()).squeeze(-1)
        #     edge_idx_even = torch.index_select(edge_idx, 0, edge_idx_even)
        #     num_edge = edge_idx_even.shape[0]
        #
        #     sample_size = max(1, int(num_edge * (1 - args.drop_rate)))
        #     remain_edge_idx = random.sample(edge_idx_even.tolist(), sample_size)
        #     remain_edge_idx = remain_edge_idx + [j + 1 for j in remain_edge_idx]
        #
        #     tmp_edge_index = edge_index[:, remain_edge_idx]
        #     edge_index_.append(tmp_edge_index)
        #     if edge_attr is not None:
        #         tmp_edge_attr = edge_attr[remain_edge_idx, :]
        #         edge_attr_.append(tmp_edge_attr)
        #
        #     if x is None:
        #         tmp_x = degree_attr(tmp_edge_index, args.max_degree)
        #         tmp_x = tmp_x[-node_idx.shape[0]:, :]
        #         x_.append(tmp_x)
        # else:
        #     edge_idx_half = edge_idx[:edge_idx.shape[0] // 2]
        #     num_edge = edge_idx_half.shape[0]
        #
        #     sample_size = max(1, int(num_edge * (1 - args.drop_rate)))
        #     remain_edge_idx = random.sample(edge_idx_half.tolist(), sample_size)
        #     remain_edge_idx = remain_edge_idx + [j + num_edge for j in remain_edge_idx]
        #
        #     tmp_edge_index = edge_index[:, remain_edge_idx]
        #     edge_index_.append(tmp_edge_index)
        #     if edge_attr is not None:
        #         tmp_edge_attr = edge_attr[remain_edge_idx, :]
        #         edge_attr_.append(tmp_edge_attr)
        #
        #     if x is None:
        #         tmp_x = degree_attr(tmp_edge_index, args.max_degree)
        #         tmp_x = tmp_x[-node_idx.shape[0]:, :]
        #         x_.append(tmp_x)

    if x is None:
        x_ = torch.cat(x_, dim = 0)
        data.x = x_

    edge_index_ = torch.cat(edge_index_, dim = 1)
    data.edge_index = edge_index_

    if edge_attr is not None:
        edge_attr_ = torch.cat(edge_attr_, dim = 0)
        data.edge_attr = edge_attr_

    return data