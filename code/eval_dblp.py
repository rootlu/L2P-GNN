import argparse
from loader import BioDataset, DblpDataset
from loader_chem import MoleculeDataset
from splitters import random_split, species_split, scaffold_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import os
import pickle
from model_dblp import GraphPred

criterion = nn.CrossEntropyLoss()


def train(args, model, device, loader, optimizer):
    model.train()
    train_loss_accum = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration", ncols=80)):
        batch = batch.to(device)
        pred = model(batch)
        y = batch.go_target_downstream.view(pred.shape).to(torch.float64)
        y = torch.topk(y, 1)[1].squeeze(1)  # one-hot to int
        optimizer.zero_grad()
        loss = criterion(pred.double(), y)
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

        tmp_y_true = batch.go_target_downstream.view(pred.shape).detach().cpu()
        y_true += torch.topk(tmp_y_true, 1)[1].squeeze(1).tolist()
        tmp_y_scores = pred.cpu().data.numpy().argmax(axis=1)
        y_scores += tmp_y_scores.tolist()

    f1 = f1_score(y_true, y_scores, average='micro')
    acc = accuracy_score(y_true, y_scores)
    return f1, acc


def main(args):
    torch.manual_seed(args.run_seed)
    np.random.seed(args.run_seed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.run_seed)

    # set up dataset
    root_supervised = '../data/dblp/supervised'
    dataset = DblpDataset(root_supervised, data_type='supervised')
    args.split = 'random'
    num_tasks = len(dataset[0].go_target_downstream)

    print(dataset)
    args.node_fea_dim = dataset[0].x.shape[1]
    args.edge_fea_dim = dataset[0].edge_attr.shape[1]
    print(args)

    train_dataset, valid_dataset, test_dataset = random_split(dataset, seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=10 * args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=10 * args.batch_size, shuffle=False, num_workers=args.num_workers)

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

    train_f1_list = []
    val_f1_list = []
    test_f1_list = []
    train_acc_list = []
    val_acc_list = []
    test_acc_list = []
    os.makedirs("../res/"+args.dataset + '/' +"finetune_seed" + str(args.run_seed), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print("====epoch " + str(epoch))
        train_loss = train(args, model, device, train_loader, optimizer)
        print('train loss:', train_loss)
        # train_f1, train_acc = eval(args, model, device, train_loader)
        # train_f1_list.append(train_f1)
        # train_acc_list.append(train_acc)
        # print('train f1: {}, acc: {}'.format(train_f1, train_acc))

        val_f1, val_acc = eval(args, model, device, val_loader)
        val_f1_list.append(val_f1)
        val_acc_list.append(val_acc)
        print('val f1: {}, acc: {}'.format(val_f1, val_acc))

        test_f1, test_acc = eval(args, model, device, test_loader)
        test_f1_list.append(test_f1)
        test_acc_list.append(test_acc)
        print('test f1: {}, acc: {}'.format(test_f1, test_acc))

        print("")

    if not args.result_file == "":
        with open("../res/" + args.dataset + '/' + "finetune_seed" + str(args.run_seed) + "/" + 'f1_'+ args.result_file,
                  'wb') as f:
            pickle.dump(
                {"train": np.array(train_f1_list),
                 "val": np.array(val_f1_list),
                 "test": np.array(test_f1_list)}, f)
        with open("../res/" + args.dataset + '/' + "finetune_seed" + str(args.run_seed) + "/" + 'acc_'+ args.result_file,
                  'wb') as f:
            pickle.dump(
                {"train": np.array(train_acc_list),
                 "val": np.array(val_acc_list),
                 "test": np.array(test_acc_list)}, f)
        print('saving model...')
        torch.save(model.gnn.state_dict(), "../res/" + args.dataset + '/' + "finetune_seed" + str(args.run_seed) + "/"
                   + args.result_file + '_' + str(epoch) + "_finetuned_gnn.pth")


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of epochs to train (default: 50)')
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
    parser.add_argument('--pre_trained_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--pool_trained_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--emb_trained_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--result_file', type=str, default='', help='output filename')

    # dataset settings
    parser.add_argument('--dataset', type=str, default='dblp',
                        help='dataset name (bio; dblp)')
    parser.add_argument('--down_dataset', type=str, default='',
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
