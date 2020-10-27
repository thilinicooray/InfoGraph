import pdb
import torch
from torch.nn import Parameter, Linear
from torch.distributions import Normal
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import negative_sampling, add_self_loops, remove_self_loops, degree, softmax
from torch_geometric.nn import GINConv, GCNConv, SAGEConv, GATConv
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add
from torch_geometric.nn.inits import glorot, zeros, reset

num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

num_bond_type = 6 #including aromatic and self-loop edge, and extra masked tokens
num_bond_direction = 3


# GIN layer for molecular graphs
class GINConv_mol(MessagePassing):
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv_mol, self).__init__(aggr = aggr)
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim),
                                       torch.nn.ReLU(),
                                       torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))[0]

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        return self.propagate(edge_index, x = x, edge_attr = edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return self.mlp(aggr_out)


# GCN layer for molecular graphs
class GCNConv_mol(MessagePassing):
    def __init__(self, emb_dim, aggr = "add"):
        super(GCNConv_mol, self).__init__(aggr = aggr)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def norm(self, edge_index, num_nodes, dtype):
        ### assuming that self-loops have been already added in edge_index
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))[0]

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = self.linear(x)

        return self.propagate(edge_index, x = x, edge_attr = edge_embeddings, norm = norm)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j + edge_attr)


# GAT layer for molecular graphs
class GATConv_mol(MessagePassing):
    def __init__(self, emb_dim, heads=1, concat = True, negative_slope = 0.2, dropout = 0,
                 bias = True, aggr = "add"):
        super(GATConv_mol, self).__init__(aggr = aggr)

        self.emb_dim = emb_dim
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.__alpha__ = None

        self.lin = Linear(emb_dim, heads * emb_dim, bias=False)

        self.att_i = Parameter(torch.Tensor(1, heads, emb_dim))
        self.att_j = Parameter(torch.Tensor(1, heads, emb_dim))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * emb_dim))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(emb_dim))
        else:
            self.register_parameter('bias', None)

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, heads * emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, heads * emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin.weight)
        glorot(self.att_i)
        glorot(self.att_j)
        zeros(self.bias)

    def forward(self, x, edge_index, edge_attr, return_attention_weights=False):
        # add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:, 0] = 4  # bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_embeddings = self.edge_embedding1(edge_attr[:, 0]) + self.edge_embedding2(edge_attr[:, 1])

        if torch.is_tensor(x):
            x = self.lin(x)
            x = (x, x)
        else:
            x = (self.lin(x[0]), self.lin(x[1]))

        edge_index = remove_self_loops(edge_index)[0]
        edge_index = add_self_loops(edge_index, num_nodes=x[1].size(self.node_dim))[0]

        out = self.propagate(edge_index, x=x, edge_attr = edge_embeddings,
                             return_attention_weights=return_attention_weights)

        if self.concat:
            out = out.view(-1, self.heads * self.emb_dim)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out = out + self.bias

        if return_attention_weights:
            alpha, self.__alpha__ = self.__alpha__, None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, x_i, x_j, edge_index_i, size_i,
                return_attention_weights):
        # Compute attention coefficients.
        x_i = x_i.view(-1, self.heads, self.emb_dim)
        x_j = x_j.view(-1, self.heads, self.emb_dim)

        alpha = (x_i * self.att_i).sum(-1) + (x_j * self.att_j).sum(-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        if return_attention_weights:
            self.__alpha__ = alpha

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def __repr__(self):
        return '{}({}, heads={})'.format(self.__class__.__name__, self.emb_dim, self.heads)


# GraphSAGE layer for molecular graphs
class GraphSAGEConv_mol(MessagePassing):
    def __init__(self, emb_dim, aggr = "mean"):
        super(GraphSAGEConv_mol, self).__init__(aggr = aggr)

        self.emb_dim = emb_dim
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)

        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        #add self loops in the edge space
        edge_index = add_self_loops(edge_index, num_nodes = x.size(0))[0]

        #add features corresponding to self-loop edges.
        self_loop_attr = torch.zeros(x.size(0), 2)
        self_loop_attr[:,0] = 4 #bond type for self-loop edge
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim = 0)

        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])

        x = self.linear(x)

        return self.propagate(edge_index, x = x, edge_attr = edge_embeddings)

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out):
        return F.normalize(aggr_out, p = 2, dim = -1)


# GNN encoder
class GNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, in_dim, emb_dim, JK = "last", drop_ratio = 0, gnn_type = "gin",
                 in_mol = False, num_heads = 2, use_embedding = True, use_bn = True):
        super(GNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.in_mol = in_mol
        self.num_heads = num_heads
        self.use_embedding = use_embedding
        self.use_bn = use_bn

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        self.x_proj = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim),
                                            torch.nn.ReLU())
        self.gin_mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(2*emb_dim, emb_dim))

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        ###List of MLPs
        self.gnns = torch.nn.ModuleList()
        for layer in range(num_layer):
            if gnn_type == "gin":
                if self.in_mol:
                    self.gnns.append(GINConv_mol(emb_dim, aggr = "add"))
                else:
                    self.gnns.append(GINConv(self.gin_mlp))
            elif gnn_type == "gcn":
                if self.in_mol:
                    self.gnns.append(GCNConv_mol(emb_dim))
                else:
                    self.gnns.append(GCNConv(emb_dim, emb_dim))
            elif gnn_type == "gat":
                if self.in_mol:
                    self.gnns.append(GATConv_mol(emb_dim, heads = self.num_heads, concat = False))
                else:
                    self.gnns.append(GATConv(emb_dim, emb_dim, heads = self.num_heads, concat = False))
            elif gnn_type == "graphsage":
                if self.in_mol:
                    self.gnns.append(GraphSAGEConv_mol(emb_dim))
                else:
                    self.gnns.append(SAGEConv(emb_dim, emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    #def forward(self, x, edge_index, edge_attr):
    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        if self.use_embedding:
            if self.in_mol and x.dtype == torch.long:
                x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
            else:
                x = self.x_proj(x)

        h_list = [x]
        for layer in range(self.num_layer):
            if self.in_mol:
                h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            else:
                h = self.gnns[layer](h_list[layer], edge_index)

            if self.use_bn:
                h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                # remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            h_list.append(h)

        ### Different implementations of JK-concat
        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim = 1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim = 0), dim = 0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim = 0), dim = 0)[0]

        return node_representation


# GNN model for baseline
class GNN_graphpred(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        in_dim (int): dimensionality of input
        emb_dim (int): dimensionality of embeddings
        out_dim (int): dimensionality of output
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat
        in_mol (bool): whether the dataset is in MoleculeNet
        num_heads (int): the number of attention heads
    """
    def __init__(self, num_layer, in_dim, emb_dim, out_dim, JK = "last", drop_ratio = 0, graph_pooling = "mean",
                 gnn_type = "gin", in_mol = False, num_heads = 5):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.in_mol = in_mol

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, in_dim, emb_dim, JK, drop_ratio, gnn_type = gnn_type, in_mol = self.in_mol,
                       num_heads = num_heads)

        #Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        #For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1
        
        if self.JK == "concat":
            rep_dim = self.mult * (self.num_layer + 1) * self.emb_dim
            self.graph_classifier = torch.nn.Sequential(
                torch.nn.Linear(rep_dim, rep_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(rep_dim, self.out_dim)
            )
        else:
            rep_dim = self.mult * self.emb_dim
            self.graph_classifier = torch.nn.Sequential(
                torch.nn.Linear(rep_dim, rep_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(rep_dim, self.out_dim)
            )

    def from_pretrained(self, model_file):
        print('Load model from: ', model_file)
        self.gnn.load_state_dict(torch.load(model_file))

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_representation = self.gnn(x, edge_index, edge_attr)
        graph_representation = self.pool(node_representation, batch)
        pred = self.graph_classifier(graph_representation)

        return graph_representation, pred


# Attribute component encoder for Input-SAD
class FCEncoder(torch.nn.Module):
    def __init__(self, num_layer, in_dim, emb_dim, drop_ratio = 0, in_mol = False, use_bn = True):
        super(FCEncoder, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.in_mol = in_mol
        self.use_bn = use_bn

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, emb_dim)
        self.x_proj = torch.nn.Sequential(torch.nn.Linear(in_dim, emb_dim),
                                          torch.nn.ReLU())

        self.edge_embedding1 = torch.nn.Embedding(num_bond_type, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(num_bond_direction, emb_dim)

        self.fcs = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.fcs.append(torch.nn.Linear(emb_dim, emb_dim))

        ###List of batchnorms
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))

    def forward(self, *argv):
        if len(argv) == 3:
            x, edge_index, edge_attr = argv[0], argv[1], argv[2]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        else:
            raise ValueError("unmatched number of arguments.")

        if self.in_mol:
            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
        else:
            x = self.x_proj(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.fcs[layer](h_list[layer])
            if self.use_bn:
                h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            h_list.append(h)

        node_emb = h_list[-1]

        return node_emb


# Decoder for topology reconstruction
class InnerProductDecoder(torch.nn.Module):
    def forward(self, z, edge_index, sigmoid=True):
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        output = torch.sigmoid(value) if sigmoid else value

        return output

    def forward_all(self, z, sigmoid=True):
        adj = torch.matmul(z, z.t())
        output = torch.sigmoid(adj) if sigmoid else adj

        return output


# Classifier for edge attribute
class EdgeAttrClassifier(torch.nn.Module):
    def __init__(self, emb_dim, num_attr):
        super(EdgeAttrClassifier, self).__init__()
        self.emb_dim = emb_dim
        self.num_attr = num_attr

        self.classifier = torch.nn.Linear(emb_dim, num_attr)

    def forward(self, node_emb):
        pred = self.classifier(node_emb)

        return pred


# Model for Embed-SAD
class Embed_SAD_model(torch.nn.Module):

    def __init__(self, num_layer, in_dim, emb_dim, out_dim, JK="last", drop_ratio=0, graph_pooling="mean",
                 gnn_type="gin", in_mol=False, num_heads=5, return_graph_emb = False, return_node_emb = False,
                 use_bn = True, use_proj = True, bond_cls = False):
        super(Embed_SAD_model, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.half_dim = int(emb_dim / 2)
        self.out_dim = out_dim
        self.in_mol = in_mol
        self.return_graph_emb = return_graph_emb
        self.return_node_emb = return_node_emb
        self.use_proj = use_proj
        self.bond_cls = bond_cls
        self.num_bond_type = 4

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnn = GNN(num_layer, in_dim, emb_dim, JK, drop_ratio, gnn_type=gnn_type, in_mol=self.in_mol,
                       num_heads=num_heads, use_bn=use_bn)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        # For graph-level binary classification
        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            rep_dim = self.mult * (self.num_layer + 1) * self.emb_dim
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(rep_dim, rep_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(rep_dim, rep_dim)
            )
            self.classifier = torch.nn.Linear(rep_dim, self.out_dim)
        else:
            rep_dim = self.mult * self.emb_dim
            self.proj = torch.nn.Sequential(
                torch.nn.Linear(rep_dim, rep_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(rep_dim, rep_dim)
            )
            self.classifier = torch.nn.Linear(rep_dim, self.out_dim)

        self.decoder = InnerProductDecoder()
        if self.bond_cls:
            self.type_classifier = EdgeAttrClassifier(self.emb_dim, self.num_bond_type)

    def from_pretrained(self, model_file):
        print('Load model from: ', model_file)
        self.gnn.load_state_dict(torch.load(model_file)['gnn'])
        self.proj.load_state_dict(torch.load(model_file)['proj'])
        self.classifier.load_state_dict(torch.load(model_file)['classifier'])

    def recon_loss(self, z, edge_index, edge_attr, epsilon = 1e-6):
        pos_edge_index = edge_index
        pos_loss = -torch.log(
            self.decoder(z, pos_edge_index, sigmoid=True) + epsilon).mean()

        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)

        neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_loss = -torch.log(1 -
                              self.decoder(z, neg_edge_index, sigmoid=True) +
                              epsilon).mean()

        # Classify edge types
        if self.bond_cls:
            node_emb_cat = torch.cat([z[edge_index[0]], z[edge_index[1]]], dim=1)
            pred_type = self.type_classifier(node_emb_cat)

            criterion = torch.nn.CrossEntropyLoss()
            type_loss = criterion(pred_type, edge_attr[:, 0])
            loss = pos_loss + neg_loss + type_loss
        else:
            loss = pos_loss + neg_loss

        return loss

    def forward(self, *argv):
        if len(argv) == 4:
            x, edge_index, edge_attr, batch = argv[0], argv[1], argv[2], argv[3]
        elif len(argv) == 1:
            data = argv[0]
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        else:
            raise ValueError("unmatched number of arguments.")

        node_emb = self.gnn(x, edge_index, edge_attr)
        if self.use_proj:
            node_emb = self.proj(node_emb)

        if self.JK == 'concat':
            attribute_emb_list = [node_emb[:, self.emb_dim * i:(self.emb_dim * i + self.half_dim)] for i in
                                 range(1, self.num_layer + 1)]
            attribute_emb = torch.cat(attribute_emb_list, dim = 1)
            structure_emb_list = [node_emb[:, (self.emb_dim * i + self.half_dim):self.emb_dim * (i+1)] for i in
                                 range(1, self.num_layer + 1)]
            structure_emb = torch.cat(structure_emb_list, dim = 1)
        else:
            attribute_emb = node_emb[:, :self.half_dim]
            structure_emb = node_emb[:, self.half_dim:]

        graph_emb = self.pool(node_emb, batch)
        pred = self.classifier(graph_emb)

        # loss definition
        loss_recon = self.recon_loss(structure_emb, edge_index, edge_attr)

        if self.return_graph_emb:
            return graph_emb, pred, loss_recon, attribute_emb, structure_emb
        elif self.return_node_emb:
            return node_emb, pred, loss_recon, attribute_emb, structure_emb
        else:
            return pred, loss_recon, attribute_emb, structure_emb


# NCE estimator of mutual information
class MI_estimator(torch.nn.Module):
    def __init__(self, emb_dim, tau = 1.0, save_space = False):
        super(MI_estimator, self).__init__()
        self.emb_dim = emb_dim
        self.tau = tau
        self.save_space = save_space

        self.x_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )
        self.y_mlp = torch.nn.Sequential(
            torch.nn.Linear(emb_dim, emb_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim, emb_dim)
        )

    def InfoNCE(self, x, y, batch, epsilon = 1e-6):
        x_norm = torch.norm(x, dim=1).unsqueeze(-1)
        y_norm = torch.norm(y, dim=1).unsqueeze(-1)
        sim = torch.mm(x, y.t()) / (torch.mm(x_norm, y_norm.t()) + epsilon)
        exp_sim = torch.exp(sim / self.tau)

        graph_mask = torch.stack([(batch == i).float() for i in batch.tolist()], dim=1)
        exp_sim_mask = exp_sim * graph_mask

        if self.save_space:
            diag_idx = [range(x.shape[0]), range(x.shape[0])]
            positive = exp_sim_mask[diag_idx]
            positive_ratio = positive / (exp_sim_mask.sum(0) + epsilon)
        else:
            node_mask = torch.eye(x.shape[0], device=x.device)
            positive = (exp_sim_mask * node_mask).sum(0)
            negative = (exp_sim_mask * (1 - node_mask)).sum(0)
            positive_ratio = positive / (positive + negative + epsilon)

        loss_NCE = -torch.log(positive_ratio).sum() / x.shape[0]

        return loss_NCE

    def forward(self, x, y, batch):
        x_proj = self.x_mlp(x)
        y_proj = self.y_mlp(y)
        loss_NCE = self.InfoNCE(x_proj, y_proj, batch)

        return loss_NCE


# Classifier for node classification
class Node_Classifier(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, out_dim, JK="last", graph_pooling="mean", nonlinear = False):
        super(Node_Classifier, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.JK = JK
        self.graph_pooling = graph_pooling
        self.nonlinear = nonlinear

        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.rep_dim = self.mult * (self.num_layer + 1) * self.emb_dim
        else:
            self.rep_dim = self.mult * self.emb_dim

        if nonlinear:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.rep_dim, self.rep_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.rep_dim, self.out_dim)
            )
        else:
            self.classifier = torch.nn.Linear(self.rep_dim, self.out_dim)

    def forward(self, node_emb):
        pred = self.classifier(node_emb)

        return pred


# Classifier for graph classification
class Graph_Classifier(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, out_dim, JK="last", graph_pooling="mean", nonlinear = False,
                 return_graph_emb = False):
        super(Graph_Classifier, self).__init__()
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.JK = JK
        self.graph_pooling = graph_pooling
        self.nonlinear = nonlinear
        self.return_graph_emb = return_graph_emb

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn = torch.nn.Linear(emb_dim, 1))
        elif graph_pooling[:-1] == "set2set":
            set2set_iter = int(graph_pooling[-1])
            if self.JK == "concat":
                self.pool = Set2Set((self.num_layer + 1) * emb_dim, set2set_iter)
            else:
                self.pool = Set2Set(emb_dim, set2set_iter)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling[:-1] == "set2set":
            self.mult = 2
        else:
            self.mult = 1

        if self.JK == "concat":
            self.rep_dim = self.mult * (self.num_layer + 1) * self.emb_dim
        else:
            self.rep_dim = self.mult * self.emb_dim

        if nonlinear:
            self.classifier = torch.nn.Sequential(
                torch.nn.Linear(self.rep_dim, self.rep_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(self.rep_dim, self.out_dim)
            )
        else:
            self.classifier = torch.nn.Linear(self.rep_dim, self.out_dim)

    def forward(self, node_emb, batch):
        graph_emb = self.pool(node_emb, batch)
        pred = self.classifier(graph_emb)

        if self.return_graph_emb:
            return graph_emb, pred
        else:
            return pred


if __name__ == "__main__":
    pass

