import torch
import random
from torch_geometric.utils import negative_sampling
import numpy as np


class TaskConstruction:
    def __init__(self, args):
        """
        construct tasks
        """
        self.args = args

    def __call__(self, data):
        num_nodes = data.num_nodes
        num_edges = data.num_edges

        # sample support set and query set for each data/task/graph
        num_sampled_edges = self.args.node_batch_size * (self.args.support_set_size + self.args.query_set_size)
        perm = np.random.randint(num_edges, size=num_sampled_edges)
        pos_edges = data.edge_index[:, perm]

        x = 1 - 1.1 * (data.edge_index.size(1) / (num_nodes * num_nodes) )
        if x != 0:
            alpha = 1 / (1 - 1.1 * (data.edge_index.size(1) / (num_nodes * num_nodes) ))
        else:
            alpha = 0
        if alpha > 0:
            neg_edges = negative_sampling(data.edge_index, num_nodes, num_sampled_edges)
        else:
            i, _, k = structured_negative_sampling(data.edge_index)
            neg_edges = torch.stack((i,k), 0)
        cur_num_neg = neg_edges.shape[1]
        if cur_num_neg != num_sampled_edges:
            perm = np.random.randint(cur_num_neg, size=num_sampled_edges)
            neg_edges = neg_edges[:, perm]

        data.pos_sup_edge_index = pos_edges[:, :self.args.node_batch_size * self.args.support_set_size]
        data.neg_sup_edge_index = neg_edges[:, :self.args.node_batch_size * self.args.support_set_size]
        data.pos_que_edge_index = pos_edges[:, self.args.node_batch_size * self.args.support_set_size:]
        data.neg_que_edge_index = neg_edges[:, self.args.node_batch_size * self.args.support_set_size:]

        return data

