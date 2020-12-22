from collections import OrderedDict

from torch_geometric.nn import global_mean_pool
from torch_geometric.nn.inits import uniform, glorot, zeros
from torch_geometric.utils import add_self_loops, remove_self_loops, softmax
import torch.nn.functional as F
import torch
import math
from torch.nn import Parameter, BatchNorm1d
from torch_scatter import scatter_add
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops
import torch.nn.functional as F
from embinitialization import EmbInitial, EmbInitial_DBLP, EmbInitial_CHEM


class MetaGINConv(MessagePassing):
    r"""The graph isomorphism operator from the `"How Powerful are
        Graph Neural Networks?" <https://arxiv.org/abs/1810.00826>`_ paper

        .. math::
            \mathbf{x}^{\prime}_i = h_{\mathbf{\Theta}} \left( (1 + \epsilon) \cdot
            \mathbf{x}_i + \sum_{j \in \mathcal{N}(i)} \mathbf{x}_j \right)

        or

        .. math::
            \mathbf{X}^{\prime} = h_{\mathbf{\Theta}} \left( \left( \mathbf{A} +
            (1 + \epsilon) \cdot \mathbf{I} \right) \cdot \mathbf{X} \right),

        here :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* an MLP.

        Args:
            nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
                maps node features :obj:`x` of shape :obj:`[-1, in_channels]` to
                shape :obj:`[-1, out_channels]`, *e.g.*, defined by
                :class:`torch.nn.Sequential`.
            eps (float, optional): (Initial) :math:`\epsilon` value.
                (default: :obj:`0`)
            train_eps (bool, optional): If set to :obj:`True`, :math:`\epsilon`
                will be a trainable parameter. (default: :obj:`False`)
            **kwargs (optional): Additional arguments of
                :class:`torch_geometric.nn.conv.MessagePassing`.
        """

    def __init__(self, in_channels, out_channels, edge_in_channels, **kwargs):
        super(MetaGINConv, self).__init__(aggr='add', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels  # 9 for bio
        # self.w1 = Parameter(torch.Tensor(in_channels, 2 * out_channels))
        self.w1 = Parameter(torch.Tensor(2 * in_channels, 2 * out_channels))
        self.b1 = Parameter(torch.Tensor(2 * out_channels))
        # batch norm.
        self.bn = BatchNorm1d(2 * out_channels)
        # linear 2
        self.w2 = Parameter(torch.Tensor(out_channels, 2 * out_channels))
        self.b2 = Parameter(torch.Tensor(out_channels))

        self.edge_w = Parameter(torch.Tensor(out_channels, edge_in_channels))
        self.edge_b = Parameter(torch.Tensor(out_channels))
        glorot(self.edge_w)
        zeros(self.edge_b)
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.w1)
        zeros(self.b1)
        glorot(self.w2)
        zeros(self.b2)

    def forward(self, x, edge_index, edge_attr, w1, b1, bn_w, bn_b, w2, b2, edge_encoder_w=None, edge_encoder_b=None):
        """"""
        # add self loops in the edge space
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        self_loop_attr = torch.zeros(x.size(0), self.edge_in_channels)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_emb = F.linear(edge_attr, edge_encoder_w, edge_encoder_b)

        out = self.propagate(edge_index, x=x, edge_attr=edge_emb)
        out = F.relu(F.batch_norm(F.linear(out, w1, b1),
                     self.bn.running_mean, self.bn.running_var, bn_w, bn_b, training=True))  # always true
        return F.linear(out, w2, b2)

    def message(self, x_j, edge_attr):
        return torch.cat((x_j, edge_attr), dim=1)
        # return x_j + edge_attr

    def __repr__(self):
        return '{}(nn={})'.format(self.__class__.__name__, self.nn)


class MetaGIN(torch.nn.Module):
    def __init__(self, emb_dim, edge_fea_dim, drop_ratio=0):
        super(MetaGIN, self).__init__()
        self.drop_ratio = drop_ratio
        self.edge_in_channels = edge_fea_dim
        # List of message-passing GNN convs
        self.conv1 = MetaGINConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv2 = MetaGINConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv3 = MetaGINConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv4 = MetaGINConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv5 = MetaGINConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)

    def forward(self, x, edge_index, edge_attr, weights=None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())

        x = F.relu(self.conv1(x, edge_index, edge_attr,
                              weights['conv1.w1'], weights['conv1.b1'],
                              weights['conv1.bn.weight'], weights['conv1.bn.bias'],
                              weights['conv1.w2'], weights['conv1.b2'],
                              weights['conv1.edge_w'], weights['conv1.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr,
                              weights['conv2.w1'], weights['conv2.b1'],
                              weights['conv2.bn.weight'], weights['conv2.bn.bias'],
                              weights['conv2.w2'], weights['conv2.b2'],
                              weights['conv2.edge_w'], weights['conv2.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_attr,
                              weights['conv3.w1'], weights['conv3.b1'],
                              weights['conv3.bn.weight'], weights['conv3.bn.bias'],
                              weights['conv3.w2'], weights['conv3.b2'],
                              weights['conv3.edge_w'], weights['conv3.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv4(x, edge_index, edge_attr,
                              weights['conv4.w1'], weights['conv4.b1'],
                              weights['conv4.bn.weight'], weights['conv4.bn.bias'],
                              weights['conv4.w2'], weights['conv4.b2'],
                              weights['conv4.edge_w'], weights['conv4.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = self.conv5(x, edge_index, edge_attr,
                       weights['conv5.w1'], weights['conv5.b1'],
                       weights['conv5.bn.weight'], weights['conv5.bn.bias'],
                       weights['conv5.w2'], weights['conv5.b2'],
                       weights['conv5.edge_w'], weights['conv5.edge_b'])
        x = F.dropout(x, self.drop_ratio, training=self.training)
        return x


class MetaGCNConv(MessagePassing):
    r"""The graph convolutional operator from the `"Semi-supervised
    Classfication with Graph Convolutional Networks"
    <https://arxiv.org/abs/1609.02907>`_ paper
    .. math::
        \mathbf{X}^{\prime} = \mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
        \mathbf{\hat{D}}^{-1/2} \mathbf{X} \mathbf{\Theta},
    where :math:`\mathbf{\hat{A}} = \mathbf{A} + \mathbf{I}` denotes the
    adjacency matrix with inserted self-loops and
    :math:`\hat{D}_{ii} = \sum_{j=0} \hat{A}_{ij}` its diagonal degree matrix.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`{\left(\mathbf{\hat{D}}^{-1/2}
            \mathbf{\hat{A}} \mathbf{\hat{D}}^{-1/2} \right)}`.
            (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 edge_in_channels,
                 improved=False,
                 cached=False,
                 bias=True,
                 **kwargs):
        super(MetaGCNConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels  # 9 for bio
        self.improved = improved
        self.cached = cached
        self.cached_result = None

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.edge_w = Parameter(torch.Tensor(out_channels, edge_in_channels))
        self.edge_b = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        glorot(self.edge_w)
        zeros(self.edge_b)
        self.cached_result = None

    @staticmethod
    def norm(edge_index, num_nodes, dtype=None):
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)
        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_attr, weight, bias, edge_encoder_w, edge_encoder_b, edge_weight=None):
        """"""
        # add features to self-loop edges
        edge_index, edge_weight = add_remaining_self_loops(edge_index)
        self_loop_attr = torch.zeros(x.size(0), self.edge_in_channels)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)

        edge_emb = F.linear(edge_attr, edge_encoder_w, edge_encoder_b)

        norm = self.norm(edge_index, x.size(0), x.dtype)

        x = torch.matmul(x, weight)

        return self.propagate(edge_index=edge_index, x=x, edge_attr=edge_emb, norm=norm, bias=bias)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * (x_j+edge_attr)

    def update(self, aggr_out, bias):
        if self.bias is not None:
            aggr_out = aggr_out + bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MetaGCN(torch.nn.Module):
    def __init__(self, emb_dim, edge_fea_dim, drop_ratio=0):
        super(MetaGCN, self).__init__()
        self.drop_ratio = drop_ratio
        # List of message-passing GNN convs
        self.conv1 = MetaGCNConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv2 = MetaGCNConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv3 = MetaGCNConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv4 = MetaGCNConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv5 = MetaGCNConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)

    def forward(self, x, edge_index, edge_attr, weights=None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
        # 5 layers
        x = F.relu(self.conv1(x, edge_index, edge_attr, weights['conv1.weight'], weights['conv1.bias'],
                              weights['conv1.edge_w'], weights['conv1.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr, weights['conv2.weight'], weights['conv2.bias'],
                              weights['conv2.edge_w'], weights['conv2.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_attr, weights['conv3.weight'], weights['conv3.bias'],
                              weights['conv3.edge_w'], weights['conv3.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv4(x, edge_index, edge_attr, weights['conv4.weight'], weights['conv4.bias'],
                              weights['conv4.edge_w'], weights['conv4.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = self.conv5(x, edge_index, edge_attr, weights['conv5.weight'], weights['conv5.bias'],
                       weights['conv5.edge_w'], weights['conv5.edge_b'])
        x = F.dropout(x, self.drop_ratio, training=self.training)

        return x


class MetaGATConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self, in_channels, out_channels,edge_in_channels, heads=2,
                 negative_slope=0.2, dropout=0, bias=True, **kwargs):
        super(MetaGATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels, heads * out_channels))
        self.att = Parameter(torch.Tensor(1, heads, 2 * out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.edge_w = Parameter(torch.Tensor(heads * out_channels, edge_in_channels))
        self.edge_b = Parameter(torch.Tensor(heads * out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)
        glorot(self.edge_w)
        zeros(self.edge_b)

    def forward(self, x, edge_index, edge_attr, weight, bias, edge_encoder_w, edge_encoder_b, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # add features to self-loop edges
        self_loop_attr = torch.zeros(x.size(0), self.edge_in_channels)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_emb = F.linear(edge_attr, edge_encoder_w, edge_encoder_b)
        x = torch.matmul(x, weight)

        return self.propagate(edge_index, size=size, x=x, bias=bias, edge_attr=edge_emb)

    def message(self, edge_index_i, x_i, x_j, size_i, edge_attr):
        # Compute attention coefficients.
        x_j = x_j.view(-1, self.heads, self.out_channels)  # (#node, head, dim)
        edge_attr = edge_attr.view(-1, self.heads, self.out_channels)
        x_j += edge_attr

        if x_i is None:
            alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(-1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)  # (#node, 1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        aggr_out = aggr_out.mean(dim=1)
        aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


class MetaGAT(torch.nn.Module):
    def __init__(self, emb_dim, edge_fea_dim, drop_ratio=0):
        super(MetaGAT, self).__init__()
        self.drop_ratio = drop_ratio
        # List of message-passing GNN convs
        self.conv1 = MetaGATConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv2 = MetaGATConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv3 = MetaGATConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv4 = MetaGATConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv5 = MetaGATConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)

    def forward(self, x, edge_index, edge_attr, weights=None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())

        # 5 layers
        x = F.relu(self.conv1(x, edge_index, edge_attr,
                              weights['conv1.weight'], weights['conv1.bias'],
                              weights['conv1.edge_w'], weights['conv1.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr,
                              weights['conv2.weight'], weights['conv2.bias'],
                              weights['conv2.edge_w'], weights['conv2.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_attr,
                              weights['conv3.weight'], weights['conv3.bias'],
                              weights['conv3.edge_w'], weights['conv3.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv4(x, edge_index, edge_attr,
                              weights['conv4.weight'], weights['conv4.bias'],
                              weights['conv4.edge_w'], weights['conv4.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = self.conv5(x, edge_index, edge_attr,
                       weights['conv5.weight'], weights['conv5.bias'],
                       weights['conv5.edge_w'], weights['conv5.edge_b'])
        x = F.dropout(x, self.drop_ratio, training=self.training)

        return x


class MetaGraphSAGEConv(MessagePassing):
    r"""The GraphSAGE operator from the `"Inductive Representation Learning on
    Large Graphs" <https://arxiv.org/abs/1706.02216>`_ paper

    .. math::
        \mathbf{\hat{x}}_i &= \mathbf{\Theta} \cdot
        \mathrm{mean}_{j \in \mathcal{N(i) \cup \{ i \}}}(\mathbf{x}_j)

        \mathbf{x}^{\prime}_i &= \frac{\mathbf{\hat{x}}_i}
        {\| \mathbf{\hat{x}}_i \|_2}.

    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        normalize (bool, optional): If set to :obj:`True`, output features
            will be :math:`\ell_2`-normalized. (default: :obj:`False`)
        concat (bool, optional): If set to :obj:`True`, will concatenate
            current node features with aggregated ones. (default: :obj:`False`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,in_channels,out_channels,edge_in_channels,
                 normalize=False, concat=False, bias=True, **kwargs):
        super(MetaGraphSAGEConv, self).__init__(aggr='mean', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_in_channels = edge_in_channels  # 9 for bio
        self.normalize = normalize
        self.concat = concat

        in_channels = 2 * in_channels if concat else in_channels
        self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.edge_w = Parameter(torch.Tensor(out_channels, edge_in_channels))
        self.edge_b = Parameter(torch.Tensor(out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        zeros(self.bias)
        glorot(self.edge_w)
        zeros(self.edge_b)

    def forward(self, x, edge_index, edge_attr, weight, bias, edge_encoder_w, edge_encoder_b,
                edge_weight=None, size=None, res_n_id=None):
        """"""
        if not self.concat and torch.is_tensor(x):
            edge_index, edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, 1, x.size(self.node_dim))
        # add features to self-loop edges
        self_loop_attr = torch.zeros(x.size(0), self.edge_in_channels)
        self_loop_attr = self_loop_attr.to(edge_attr.device).to(edge_attr.dtype)
        edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
        edge_emb = F.linear(edge_attr, edge_encoder_w, edge_encoder_b)

        return self.propagate(edge_index, size=size, x=x, edge_attr=edge_emb,
                              edge_weight=edge_weight, res_n_id=res_n_id)

    def message(self, x_j, edge_attr, edge_weight):
        return x_j+edge_attr if edge_weight is None else edge_weight.view(-1, 1) * (x_j+edge_attr)

    def update(self, aggr_out, x, res_n_id):
        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)
        elif self.concat and (isinstance(x, tuple) or isinstance(x, list)):
            assert res_n_id is not None
            aggr_out = torch.cat([x[0][res_n_id], aggr_out], dim=-1)

        aggr_out = torch.matmul(aggr_out, self.weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class MetaGraphSAGE(torch.nn.Module):
    def __init__(self, emb_dim, edge_fea_dim, drop_ratio=0):
        super(MetaGraphSAGE, self).__init__()
        self.drop_ratio = drop_ratio
        # List of message-passing GNN convs
        self.conv1 = MetaGraphSAGEConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv2 = MetaGraphSAGEConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv3 = MetaGraphSAGEConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv4 = MetaGraphSAGEConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)
        self.conv5 = MetaGraphSAGEConv(in_channels=emb_dim, out_channels=emb_dim, edge_in_channels=edge_fea_dim)

    def forward(self, x, edge_index, edge_attr, weights=None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())

        # 5 layers
        x = F.relu(self.conv1(x, edge_index, edge_attr, weights['conv1.weight'], weights['conv1.bias'],
                              weights['conv1.edge_w'], weights['conv1.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv2(x, edge_index, edge_attr, weights['conv2.weight'], weights['conv2.bias'],
                              weights['conv2.edge_w'], weights['conv2.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv3(x, edge_index, edge_attr, weights['conv3.weight'], weights['conv3.bias'],
                              weights['conv3.edge_w'], weights['conv3.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = F.relu(self.conv4(x, edge_index, edge_attr, weights['conv4.weight'], weights['conv4.bias'],
                              weights['conv4.edge_w'], weights['conv4.edge_b']))
        x = F.dropout(x, self.drop_ratio, training=self.training)
        x = self.conv5(x, edge_index, edge_attr, weights['conv5.weight'], weights['conv5.bias'],
                       weights['conv5.edge_w'], weights['conv5.edge_b'])
        x = F.dropout(x, self.drop_ratio, training=self.training)

        return x


class MetaPool(torch.nn.Module):
    def __init__(self, emb_dim):
        super(MetaPool, self).__init__()
        self.weight = Parameter(torch.Tensor(emb_dim, emb_dim))
        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)

    def forward(self, x, weights=None, batch=None):
        if weights is None:
            weights = OrderedDict(self.named_parameters())
        if batch is None:
            pooled_emb = x.mean(0).unsqueeze(0)  # (1, dim)
        else:
            pooled_emb = global_mean_pool(x, batch)  # (num_graph, dim)
        graph_emb = torch.matmul(pooled_emb, weights['weight'])
        return graph_emb


class GraphPred(torch.nn.Module):
    def __init__(self, args, emb_dim, edge_fea_dim, num_tasks, drop_ratio=0, gnn_type="gin"):
        super(GraphPred, self).__init__()
        self.emb_dim = emb_dim
        self.edge_fea_dim = edge_fea_dim
        self.num_tasks = num_tasks
        self.drop_ratio = drop_ratio
        self.gnn_type = gnn_type
        self.dataset = args.dataset

        if args.gnn_type == 'gcn':
            self.gnn = MetaGCN(args.emb_dim, args.edge_fea_dim, args.dropout_ratio)
        elif args.gnn_type == 'gat':
            self.gnn = MetaGAT(args.emb_dim, args.edge_fea_dim, args.dropout_ratio)
        elif args.gnn_type == 'graphsage':
            self.gnn = MetaGraphSAGE(args.emb_dim, args.edge_fea_dim, args.dropout_ratio)
        elif args.gnn_type == 'gin':
            self.gnn = MetaGIN(args.emb_dim, args.edge_fea_dim, args.dropout_ratio)
        else:
            raise ValueError("Not implement GNN type!")

        self.pool = MetaPool(emb_dim)

        self.emb_initial = EmbInitial_DBLP(args.emb_dim, args.node_fea_dim)
        self.graph_pred_linear = torch.nn.Linear(self.emb_dim, self.num_tasks)

    def from_pretrained(self, model_file, pool_file, emb_file):
        print('loading pre-trained gnn...')
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        if '.pth' in pool_file:  # node-level adaptation
            print('loading pre-trained pool...')
            self.pool.load_state_dict(torch.load(pool_file, map_location=lambda storage, loc: storage))
        print('loading pre-trained emb...')
        self.emb_initial.load_state_dict(torch.load(emb_file, map_location=lambda storage, loc: storage))

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.emb_initial(x)
        node_representation = self.gnn(x, edge_index, edge_attr)
        pooled = self.pool(node_representation,batch=batch)  # (batch, dim)

        graph_rep = pooled
        return self.graph_pred_linear(graph_rep)


if __name__ == "__main__":
    pass
