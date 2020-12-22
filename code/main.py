import argparse
import random
from loader import BioDataset, DblpDataset
from loader_chem import MoleculeDataset
from dataloader import DataLoaderAE
from util import TaskConstruction
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from metapre import MetaPre
import time
from progressbar import *


def train(args, model, loader, optimizer):
    # train
    model.train()
    for epoch in range(1, args.epochs + 1):
        train_loss = []
        train_acc = []
        print("====epoch " + str(epoch))
        for step, batch in enumerate(tqdm(loader, desc="Iteration",ncols=80)):
            loss, acc = model.meta_gradient_step(batch, optimizer)
            train_loss.append(loss)
            train_acc.append(acc)

        print('loss:', torch.stack(train_loss).mean().detach().cpu().item())
        print('acc:', torch.stack(train_acc).mean().detach().cpu().item())

        if epoch % 10 == 0 or epoch == 1:
            if not args.model_file == '':
                print('saving model...')
                torch.save(model.gnn.state_dict(),
                           '../res/' + args.dataset + '/' + args.model_file + '_' + args.gnn_type + '_' + str(epoch) + "_gnn.pth")
                torch.save(model.emb_initial.state_dict(),
                           '../res/' + args.dataset + '/' +args.model_file + '_' + args.gnn_type + '_' + str(epoch) + "_emb.pth")
                torch.save(model.pool.state_dict(),
                           '../res/' + args.dataset + '/' + args.model_file + '_' + args.gnn_type + '_' + str(
                               epoch) + "_pool.pth")


def main(args):
    # set up dataset
    if args.dataset == 'bio':
        root_unsupervised = '../data/bio/unsupervised'
        dataset = BioDataset(root_unsupervised, data_type='unsupervised', transform=TaskConstruction(args))
    elif args.dataset == 'dblp':
        root_unsupervised = '../data/dblp/unsupervised'
        dataset = DblpDataset(root_unsupervised, data_type='unsupervised',transform=TaskConstruction(args))
    elif args.dataset == 'chem':
        root_unsupervised = '../data/chem/zinc_standard_agent'
        dataset = MoleculeDataset(root_unsupervised,  dataset='zinc_standard_agent',transform=TaskConstruction(args))
    print(dataset)
    args.node_fea_dim = dataset[0].x.shape[1]
    args.edge_fea_dim = dataset[0].edge_attr.shape[1]
    print(args)

    loader = DataLoaderAE(dataset, batch_size=args.graph_batch_size, shuffle=True, num_workers=args.num_workers)

    # set up model
    metapre = MetaPre(args).to(args.device)
    # set up optimizer
    optimizer = optim.Adam(metapre.parameters(), lr=args.lr, weight_decay=args.decay)
    train(args, metapre, loader, optimizer)


if __name__ == "__main__":
    print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    parser = argparse.ArgumentParser(
        description='PyTorch implementation of meta-learning-like pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--graph_batch_size', type=int, default=64,
                        help='input batch size for parent tasks (default: 64)')
    parser.add_argument('--node_batch_size', type=int, default=1,
                        help='input batch size for parent tasks (default: 3)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')

    # gnn setting
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--dropout_ratio', type=float, default=0,
                        help='dropout ratio (default: 0)')
    parser.add_argument('--graph_pooling', type=str, default="mean")
    parser.add_argument('--model_file', type=str, default='test', help='filename to output the pre-trained model')

    # meta-learning settings
    parser.add_argument('--order', type=int, default=2, help='gradient order')
    parser.add_argument('--node_level', type=int, default=1, help='node-level adaptation')
    parser.add_argument('--graph_level', type=int, default=1, help='graph-level adaptation')
    parser.add_argument('--node_lr', type=float, default=0.001, help='learning rate for node-level adaptation')
    parser.add_argument('--node_update', type=int, default=1, help='update step for node-level adaptation')
    parser.add_argument('--graph_lr', type=float, default=0.001, help='learning rate for graph-level adaptation')
    parser.add_argument('--graph_update', type=int, default=1, help='update step for graph-level adaptation')
    parser.add_argument('--support_set_size', type=int, default=10, help='size of support set')
    parser.add_argument('--query_set_size', type=int, default=5, help='size of query set')

    # dataset settings
    parser.add_argument('--dataset', type=str, default='chem',
                        help='dataset name (bio; dblp; chem)')
    parser.add_argument('--node_fea_dim', type=int, default=2,
                        help='node feature dimensions (BIO: 2; DBLP: 10; CHEM: ))')
    parser.add_argument('--edge_fea_dim', type=int, default=9,
                        help='edge feature dimensions (BIO: 9; DBLP: 1; CHEM: ))')
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    # device = torch.device("cpu")
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    args.device = device

    main(args)

