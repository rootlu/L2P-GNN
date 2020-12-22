import random
import time
from collections import OrderedDict
import torch
import torch.nn as nn
from embinitialization import EmbInitial, EmbInitial_DBLP, EmbInitial_CHEM


class MetaPre(torch.nn.Module):
    def __init__(self, args):
        super(MetaPre, self).__init__()
        self.args = args
        self.sup_size = args.support_set_size
        self.que_size = args.query_set_size
        if args.dataset == 'bio':
            self.emb_initial = EmbInitial(args.emb_dim, args.node_fea_dim)
            from model_bio import MetaGCN, MetaGIN, MetaPool, MetaGAT, MetaGraphSAGE
        elif args.dataset == 'dblp':
            self.emb_initial = EmbInitial_DBLP(args.emb_dim, args.node_fea_dim)
            from model_dblp import MetaGCN, MetaGIN, MetaPool, MetaGAT, MetaGraphSAGE
        elif args.dataset == 'chem':
            self.emb_initial = EmbInitial_CHEM(args.emb_dim, args.node_fea_dim)
            from model_chem import MetaGCN, MetaGIN, MetaPool, MetaGAT, MetaGraphSAGE

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
        self.pool = MetaPool(args.emb_dim)
        self.loss = nn.BCEWithLogitsLoss()

    def from_pretrained(self, model_file, pool_file, emb_file):
        self.gnn.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
        self.pool.load_state_dict(torch.load(pool_file, map_location=lambda storage, loc: storage))
        self.emb_initial.load_state_dict(torch.load(emb_file, map_location=lambda storage, loc: storage))

    def cycle_index(self, num, shift):
        arr = torch.arange(num) + shift
        arr[-shift:] = torch.arange(shift)
        return arr

    def meta_gradient_step(self, batch_data, optimizer, train=True):
        task_losses = []
        task_acc = []
        torch.autograd.set_detect_anomaly(True)
        create_graph = (True if self.args.order == 2 else False) and train

        batch_data = batch_data.to(self.args.device)
        x = self.emb_initial(batch_data.x)
        sup_task_nodes_emb, que_task_nodes_emb = [], []

        # node-level
        node_loss = []
        node_acc = []
        for idx in range(self.args.node_batch_size):
            # cur data
            cur_pos_sup_e_idx = batch_data.pos_sup_edge_index[:,idx * self.sup_size * self.args.graph_batch_size:
                                                                (idx + 1) * self.sup_size * self.args.graph_batch_size]
            cur_neg_sup_e_idx = batch_data.neg_sup_edge_index[:,idx * self.sup_size * self.args.graph_batch_size:
                                                                (idx + 1) * self.sup_size * self.args.graph_batch_size]
            cur_pos_que_e_idx = batch_data.pos_que_edge_index[:,idx * self.que_size * self.args.graph_batch_size:
                                                                (idx + 1) * self.que_size * self.args.graph_batch_size]
            cur_neg_que_e_idx = batch_data.neg_que_edge_index[:,idx * self.que_size * self.args.graph_batch_size:
                                                                (idx + 1) * self.que_size * self.args.graph_batch_size]

            fast_weights = OrderedDict(self.gnn.named_parameters())
            for step in range(self.args.node_update):
                node_emb = self.gnn(x, batch_data.edge_index, batch_data.edge_attr, fast_weights)
                pos_score = torch.sum(node_emb[cur_pos_sup_e_idx[0]] *
                                      node_emb[cur_pos_sup_e_idx[1]], dim=1)  # ([n_batch*#sup_set])
                neg_score = torch.sum(node_emb[cur_neg_sup_e_idx[0]] *
                                      node_emb[cur_neg_sup_e_idx[1]], dim=1)
                loss = self.loss(pos_score, torch.ones_like(pos_score)) + \
                       self.loss(neg_score, torch.zeros_like(neg_score))
                gradients = torch.autograd.grad(loss, fast_weights.values(), create_graph=create_graph)
                # update weights manually
                fast_weights = OrderedDict(
                    (name, param - self.args.node_lr * grad)
                    for ((name, param), grad) in zip(fast_weights.items(), gradients)
                )
            # for graph-level adaptation
            sup_task_nodes_emb.append(
                node_emb[cur_pos_sup_e_idx].reshape(-1, self.args.emb_dim))  # ([(#sup_set, dim), ...])
            que_task_nodes_emb.append(
                node_emb[cur_pos_que_e_idx].reshape(-1, self.args.emb_dim))  # ([(#que_set, dim), ...])
            # node-level loss on query set
            node_emb = self.gnn(x, batch_data.edge_index, batch_data.edge_attr, fast_weights)
            pos_score = torch.sum(node_emb[cur_pos_que_e_idx[0]] *
                                  node_emb[cur_pos_que_e_idx[1]], dim=1)  # ([n_batch*#sup_set])
            neg_score = torch.sum(node_emb[cur_neg_que_e_idx[0]] *
                                  node_emb[cur_neg_que_e_idx[1]], dim=1)
            loss = self.loss(pos_score, torch.ones_like(pos_score)) + \
                   self.loss(neg_score, torch.zeros_like(neg_score))
            acc = (torch.sum(pos_score > 0) + torch.sum(neg_score < 0)).to(torch.float32) / float(2 * len(pos_score))
            node_loss.append(loss)
            node_acc.append(acc)

        node_loss = torch.stack(node_loss).mean()
        node_acc = torch.stack(node_acc).mean()

        # graph level
        g_fast_weights = OrderedDict(self.pool.named_parameters())
        graph_emb = self.pool(node_emb, weights=g_fast_weights).squeeze()  # (#num_graph, dim)
        neg_graph_emb = graph_emb[self.cycle_index(len(graph_emb), 1)]

        task_emb = torch.cat(([self.pool(ns_e, weights=g_fast_weights)
                               for ns_e in sup_task_nodes_emb]), dim=0)  # (node_batch_size, dim)
        graph_pos_score = torch.sum(task_emb * graph_emb.repeat(self.args.node_batch_size, 1), dim=1)
        graph_neg_score = torch.sum(task_emb * neg_graph_emb.repeat(self.args.node_batch_size, 1), dim=1)
        loss = self.loss(graph_pos_score, torch.ones_like(graph_pos_score)) + \
               self.loss(graph_neg_score, torch.zeros_like(graph_neg_score))
        gradients = torch.autograd.grad(loss, g_fast_weights.values(), create_graph=create_graph)
        g_fast_weights = OrderedDict(
            (name, param - self.args.graph_lr * grad)
            for ((name, param), grad) in zip(g_fast_weights.items(), gradients)
        )
        task_emb = torch.cat(([self.pool(ns_e, weights=g_fast_weights)
                               for ns_e in que_task_nodes_emb]), dim=0)  # (#num_graph, dim)
        graph_pos_score = torch.sum(task_emb * graph_emb.repeat(self.args.node_batch_size, 1), dim=1)
        graph_neg_score = torch.sum(task_emb * neg_graph_emb.repeat(self.args.node_batch_size, 1), dim=1)
        loss = self.loss(graph_pos_score, torch.ones_like(graph_pos_score)) + \
               self.loss(graph_neg_score, torch.zeros_like(graph_neg_score))
        node_loss += loss

        task_losses.append(node_loss)
        task_acc.append(node_acc)

        # optimization
        optimizer.zero_grad()
        meta_batch_loss = torch.stack(task_losses).mean()
        meta_batch_loss.backward()
        optimizer.step()
        return meta_batch_loss, torch.stack(task_acc).mean()
