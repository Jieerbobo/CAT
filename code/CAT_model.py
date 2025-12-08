import dgl
import torch
import torch.nn as nn
import dgl.function as Fn
import torch.nn.functional as F
from dgl.ops import edge_softmax
from dgl.nn.pytorch import HeteroLinear
import numpy as np
from torch_scatter import scatter_mean
from utils import setup_seed


def to_hetero_feat(h, type, name):
    """Feature convert API.

    It uses information about the type of the specified node
    to convert features ``h`` in homogeneous graph into a heteorgeneous
    feature dictionay ``h_dict``.

    Parameters
    ----------
    h: Tensor
        Input features of homogeneous graph
    type: Tensor
        Represent the type of each node or edge with a number.
        It should correspond to the parameter ``name``.
    name: list
        The node or edge types list.

    Return
    ------
    h_dict: dict
        output feature dictionary of heterogeneous graph

    Example
    -------

    >>> h = torch.tensor([[1, 2, 3],
                          [1, 1, 1],
                          [0, 2, 1],
                          [1, 3, 3],
                          [2, 1, 1]])
    >>> print(h.shape)
    torch.Size([5, 3])
    >>> type = torch.tensor([0, 1, 0, 0, 1])
    >>> name = ['author', 'paper']
    >>> h_dict = to_hetero_feat(h, type, name)
    >>> print(h_dict)
    {'author': tensor([[1, 2, 3],
    [0, 2, 1],
    [1, 3, 3]]), 'paper': tensor([[1, 1, 1],
    [2, 1, 1]])}

    """
    h_dict = {}
    for index, ntype in enumerate(name):
        h_dict[ntype] = h[torch.where(type == index)]

    return h_dict


class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """

    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()

    def reset_parameters(self, ):
        self.w.weight = nn.Parameter(
            (torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = True
        self.w.bias.requires_grad = True

    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


class cat_model(nn.Module):
    def __init__(self, num_layers, in_dim, hidden_dim,
                 ntypes, negative_slope, seed):
        super(cat_model, self).__init__()
        self.num_layers = num_layers
        self.activation = F.elu
        self.cat_layers = nn.ModuleList()
        self.node_trans = nn.ModuleList()
        self.seed = seed
        setup_seed(self.seed)

        # the first layer
        self.cat_layers.append(
            TypeAttention(hidden_dim,
                          ntypes,
                          negative_slope))
        self.cat_layers.append(
            NodeAttention(hidden_dim,
                          negative_slope))

        # the second layer
        if num_layers > 1:
            for i in range(num_layers - 1):
                self.cat_layers.append(
                    TypeAttention(hidden_dim,
                                  ntypes,
                                  negative_slope))
                self.cat_layers.append(
                    NodeAttention(hidden_dim,
                                  negative_slope))

        self.mlp = nn.Sequential(
                        nn.Linear(hidden_dim * 3, hidden_dim),
                        nn.ReLU(),
                        nn.Linear(hidden_dim, 2)
                        )

        self.time_emb = TimeEncode(hidden_dim)
        self.gate = nn.Parameter(torch.ones(25)/25, requires_grad=True)  # 25 for the Epinions dataset

    def forward(self, hg, h_dict, edge_label_index, item_type, item_id):
        """
        The forward part of CAT.

        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types

        Returns
        -------
        dict
            The embeddings after the output projection.
        """

        with hg.local_scope():
            hg.ndata['h'] = h_dict
            hg.edata['edge_time'] = {key: self.time_emb(val) for key, val in hg.edata['edge_time'].items()}

            for l in range(self.num_layers):
                attention = self.cat_layers[2 * l](hg, hg.ndata['h'])
                hg.edata['alpha'] = attention

                g = dgl.to_homogeneous(hg, ndata='h', edata=['alpha', 'edge_time'])
                h = self.cat_layers[2 * l + 1](g, g.ndata['h'])

                # Convert features ``h`` in homogeneous graph into a heterogeneous feature dictionary ``h_dict``.
                h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
                hg.ndata['h'] = h_dict

        src = h_dict['user'][edge_label_index[0]]
        dst = h_dict['user'][edge_label_index[1]]

        # item category
        item_type = item_type.to(src.device)
        item_id = item_id.to(src.device)

        category_index = item_type[1][item_id - 1]
        # item_type[1] contains categories of items in order
        if len(item_id) == h_dict['item'].shape[0]:
            input_emb = h_dict['item']
        else:  # val/testing phase
            input_emb = h_dict['item'][1:,:]

        context = scatter_mean(input_emb, category_index, dim=0)  # avg operation
        context = context[1:, :]

        src_dst = torch.cat([src, dst], dim=-1)
        src_dst_expand = src_dst.unsqueeze(1).expand(-1, context.shape[0], -1)
        context_expand = context.repeat(src.shape[0], 1, 1)

        src_dst_context = torch.cat([src_dst_expand, context_expand], dim=-1)
        output = self.mlp(src_dst_context)

        gate_normalized = torch.softmax(self.gate, dim=0).view(1, -1, 1).expand_as(output)
        out = (output * gate_normalized).sum(dim=1)

        return out


class TypeAttention(nn.Module):
    """
    Type Attention

    Parameters
    ----------
    in_dim: int
        the input dimension of the feature
    ntypes: list
        the list of the node type in the graph
    slope: float
        the negative slope used in the LeakyReLU
    """

    def __init__(self, in_dim, ntypes, slope):
        super(TypeAttention, self).__init__()
        attn_vector = {}
        for ntype in ntypes:
            attn_vector[ntype] = in_dim
        self.mu_l = HeteroLinear(attn_vector, in_dim, bias=False)
        self.mu_r = HeteroLinear(attn_vector, in_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU(slope)

    def forward(self, hg, h_dict):
        """
        The forward part of the Type Attention.

        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph
        h_dict: dict
            the feature dict of different node types

        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        h_t = {}
        with hg.local_scope():  # Any modifications to nodes or edges below will not affect the original feature values in the graph
            hg.ndata['h'] = h_dict
            for srctype, etype, dsttype in hg.canonical_etypes:

                rel_graph = hg[srctype, etype, dsttype]  # Graph for each type of interaction
                if srctype not in h_dict:
                    continue
                with rel_graph.local_scope():
                    degs = rel_graph.out_degrees().float().clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    feat_src = h_dict[srctype]
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)

                    feat_src = feat_src * norm
                    rel_graph.srcdata['h'] = feat_src
                    src_nodes = rel_graph.edges()[0]
                    edge_src_nodes = norm[src_nodes]
                    edge_src_norm = edge_src_nodes.view(-1, 1)
                    rel_graph.edata['e'] = rel_graph.edata['edge_time'] * edge_src_norm
                    rel_graph.update_all(Fn.u_add_e('h', 'e', 'm'), Fn.sum(msg='m', out='h'))
                    rst = rel_graph.dstdata['h']

                    degs = rel_graph.in_degrees().float().clamp(min=1)
                    norm = torch.pow(degs, -0.5)
                    shp = norm.shape + (1,) * (feat_src.dim() - 1)
                    norm = torch.reshape(norm, shp)
                    rst = rst * norm
                    h_t[srctype] = rst

                    h_l = self.mu_l(h_dict)[dsttype]
                    h_r = self.mu_r(h_t)[srctype]
                    edge_attention = F.elu(h_l + h_r)

                    if srctype == dsttype:
                        rel_graph.ndata['m'] = edge_attention
                    else:
                        rel_graph.ndata['m'] = {dsttype: edge_attention,
                                            srctype: torch.zeros((rel_graph.num_nodes(ntype=srctype),)).to(
                                                edge_attention.device)}
                    reverse_graph = dgl.reverse(rel_graph)  # Since dgl doesn't have copy_v function, we need to reverse
                    reverse_graph.apply_edges(Fn.copy_u('m', 'alpha'))

                hg.edata['alpha'] = {
                    (srctype, etype, dsttype): reverse_graph.edata['alpha']}

            attention = edge_softmax(hg, hg.edata['alpha'])

        return attention


class NodeAttention(nn.Module):
    """
    Node attention

    Parameters
    ----------
    in_dim: int
        the input/output dimension of the feature
    slope: float
        the negative slope used in the LeakyReLU
    """

    def __init__(self, in_dim, slope):
        super(NodeAttention, self).__init__()
        self.Mu_l = nn.Linear(in_dim, in_dim, bias=False)  # attention vector
        self.Mu_r = nn.Linear(in_dim, in_dim, bias=False)
        self.leakyrelu = nn.LeakyReLU(slope)
        self.fc = nn.Linear(in_dim, in_dim)
        self.activation = nn.ReLU()

    def forward(self, g, x):
        """
        The forward part of the Node Attention.

        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        x: tensor
            the original features of the graph

        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        with g.local_scope():
            src = g.edges()[0]
            dst = g.edges()[1]

            h_l = self.Mu_l(x)[src] + self.Mu_l(g.edata['edge_time'])
            h_r = self.Mu_r(x)[dst]
            edge_attention = self.leakyrelu((h_l + h_r) * g.edata['alpha'])

            edge_attention = edge_softmax(g, edge_attention)
            g.edata['alpha'] = edge_attention
            g.srcdata['x'] = x

            g.apply_edges(Fn.u_mul_e('x', 'alpha', 'm1'))
            g.apply_edges(lambda edges: {'m2': edges.data['edge_time'] * edges.data['alpha']})
            g.apply_edges(lambda edges: {'m': edges.data['m1'] + edges.data['m2']})
            g.update_all(Fn.copy_e('m', 'm'), Fn.sum('m', 'x'))
            h = g.ndata['x']

            h = h + self.fc(h)
            h = self.activation(h)

        return h