import argparse
from loader_chem import MoleculeDataset
from splitters import random_split, species_split, scaffold_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import pandas as pd
from tensorboardX import SummaryWriter
from model_chem import GraphPred

criterion = nn.BCEWithLogitsLoss(reduction="none")


def train(args, model, device, loader, optimizer):
    model.train()
    train_loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration", ncols=80)):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.y.view(pred.shape).to(torch.float64)
        is_valid = y ** 2 > 0
        loss_mat = criterion(pred.double(), (y + 1) / 2)
        loss_mat = torch.where(is_valid, loss_mat,
                               torch.zeros(loss_mat.shape).to(loss_mat.device).to(loss_mat.dtype))

        optimizer.zero_grad()
        loss = torch.sum(loss_mat) / torch.sum(is_valid)
        loss.backward()
        optimizer.step()
        train_loss_accum += float(loss.detach().cpu().item())

    return train_loss_accum / (step + 1)


def eval(args, model, device, loader):
    model.eval()
    y_true = []
    y_scores = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration", ncols=80)):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)

        y_true.append(batch.y.view(pred.shape))
        y_scores.append(pred)

    y_true = torch.cat(y_true, dim=0).cpu().numpy()
    y_scores = torch.cat(y_scores, dim=0).cpu().numpy()

    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i] ** 2 > 0
            roc_list.append(roc_auc_score((y_true[is_valid, i] + 1) / 2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list)) / y_true.shape[1]))

    return sum(roc_list) / len(roc_list)  # y_true.shape[1]


def main(args):
    torch.manual_seed(args.run_seed)
    np.random.seed(args.run_seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.run_seed)

    # set up dataset
    dataset = MoleculeDataset("../data/chem/" + args.down_dataset, dataset=args.down_dataset)
    args.split = 'scaffold'
    # Bunch of classification tasks
    if args.down_dataset == "tox21":
        num_tasks = 12
    elif args.down_dataset == "hiv":
        num_tasks = 1
    elif args.down_dataset == "pcba":
        num_tasks = 128
    elif args.down_dataset == "muv":
        num_tasks = 17
    elif args.down_dataset == "bace":
        num_tasks = 1
    elif args.down_dataset == "bbbp":
        num_tasks = 1
    elif args.down_dataset == "toxcast":
        num_tasks = 617
    elif args.down_dataset == "sider":
        num_tasks = 27
    elif args.down_dataset == "clintox":
        num_tasks = 2
    elif args.down_dataset == "mutag":
        num_tasks = 1
    elif args.down_dataset == "ptc_mr":
        num_tasks = 1
    else:
        raise ValueError("Invalid dataset name.")

    print(dataset)
    args.node_fea_dim = dataset[0].x.shape[1]
    args.edge_fea_dim = dataset[0].edge_attr.shape[1]
    print(args)

    smiles_list = pd.read_csv('../data/chem/' + args.down_dataset + '/processed/smiles.csv', header=None)[0].tolist()
    train_dataset, valid_dataset, test_dataset = scaffold_split(dataset, smiles_list, null_value=0, frac_train=0.8,
                                                                    frac_valid=0.1, frac_test=0.1)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(train_dataset[0])

    # set up model
    model = GraphPred(args, args.emb_dim, args.edge_fea_dim, num_tasks,
                      drop_ratio=args.dropout_ratio, gnn_type=args.gnn_type)

    if not args.pre_trained_model_file == "":
        model.from_pretrained('../res/' + args.dataset + '/' + args.pre_trained_model_file,
                              '../res/' + args.dataset + '/' + args.pool_trained_model_file,
                              '../res/' + args.dataset + '/' + args.emb_trained_model_file)
    model.to(device)
    # set up optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    os.makedirs("../res/"+ args.dataset + '/' +args.down_dataset + '/' +"finetune_seed" + str(args.run_seed), exist_ok=True)
    fname = "../res/" + args.dataset + '/' +args.down_dataset + '/' + "finetune_seed" + str(args.run_seed) + "/" + args.result_file
    writer = SummaryWriter(fname)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
        train_loss = train(args, model, device, train_loader, optimizer)
        print('train loss:', train_loss)
        train_acc = eval(args, model, device, train_loader)
        train_acc_list.append(train_acc)
        print('train auc:', train_acc)
        val_acc = eval(args, model, device, val_loader)
        val_acc_list.append(val_acc)
        print('val auc:', val_acc)
        test_acc = eval(args, model, device, test_loader)
        test_acc_list.append(test_acc)
        print(test_acc)
        if not args.result_file == "":  # chem dataset
            writer.add_scalar('data/train auc', train_acc, epoch)
            writer.add_scalar('data/val auc', val_acc, epoch)
            writer.add_scalar('data/test auc', test_acc, epoch)

        print("")

    if not args.result_file == "":
        writer.close()
        print('saving model...')
        torch.save(model.gnn.state_dict(),
                   "../res/" + args.dataset + '/' + args.down_dataset + '/' + "finetune_seed" + str(args.run_seed) + "/"
                   + args.result_file + '_' + str(epoch) + "_finetuned_gnn.pth")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=3,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--run_seed', type=int, default=0, help="Seed for running experiments.")
    parser.add_argument('--num_workers', type=int, default=8, help='number of workers for dataset loading')

    # gnn settings
    parser.add_argument('--gnn_type', type=str, default="gin")
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5).')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="mean",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="last",
                        help='how the node features across layers are combined. last, sum, max or concat')
    # pre-trained file
    parser.add_argument('--pre_trained_model_file', type=str, default='node_adaptation_5_300_gin_10_gnn.pth',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--pool_trained_model_file', type=str, default='node_adaptation_5_300_gin_10_pool.pth',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--emb_trained_model_file', type=str, default='node_adaptation_5_300_gin_10_emb.pth',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--result_file', type=str, default='', help='output filename')

    # dataset settings
    parser.add_argument('--dataset', type=str, default='chem',
                        help='dataset name (bio; dblp)')
    parser.add_argument('--down_dataset', type=str, default='bbbp',
                        help='downstream dataset name')
    parser.add_argument('--split', type=str, default="species", help='Random or species split')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting dataset.")
    parser.add_argument('--node_fea_dim', type=int, default=2,
                        help='node feature dimensions (BIO: 2; DBLP: 10))')
    parser.add_argument('--edge_fea_dim', type=int, default=9,
                        help='edge feature dimensions (BIO: 9; DBLP: 1))')
    args = parser.parse_args()

    for i in range(10):
        args.run_seed = i
        main(args)
