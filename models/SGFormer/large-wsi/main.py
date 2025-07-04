

import sys
sys.path.append('../../')  # Only for Remote use on Clusters

import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_scatter import scatter

from logger import Logger, save_result
from dataset import load_dataset
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, to_sparse_tensor, \
    load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from eval import evaluate
from parse import parse_method 

import time
import pickle

import warnings
warnings.filterwarnings('ignore')

import hydra
from omegaconf import DictConfig, OmegaConf



@hydra.main(config_path="../../../configs/SGFormer", config_name="config_largewsi", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # print config nicely


    # NOTE: for consistent data splits, see data_utils.rand_train_test_idx
    def fix_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    fix_seed(cfg.seed)

    if cfg.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(cfg.device)) if torch.cuda.is_available() else torch.device("cpu")


    ### Load and preprocess data ###
    dataset = load_dataset(cfg.data_dir, cfg.dataset, cfg.sub_dataset)

    if len(dataset.label.shape) == 1:
        dataset.label = dataset.label.unsqueeze(1)
    dataset.label = dataset.label.to(device)


    #### get the splits for all runs
    if cfg.rand_split:
        split_idx_lst = [dataset.get_idx_split(train_prop=cfg.train_prop, valid_prop=cfg.valid_prop)
                         for _ in range(cfg.runs)]

    elif cfg.rand_split_class:
        split_idx_lst = [dataset.get_idx_split(split_type='class', label_num_per_class=cfg.label_num_per_class)
                         for _ in range(cfg.runs)]

    elif cfg.dataset in ['ogbn-proteins', 'ogbn-arxiv', 'ogbn-products']:
        split_idx_lst = [dataset.load_fixed_splits()
                         for _ in range(cfg.runs)]

    elif cfg.dataset == 'skinwsi':
        split_idx_lst = dataset.load_fixed_splits()

    else:
        split_idx_lst = load_fixed_splits(cfg.data_dir, dataset, name=cfg.dataset, protocol=cfg.protocol)



    ### Basic information of datasets ###
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    # infer the number of classes for non one-hot and one-hot labels
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    print(f"dataset {cfg.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")



    ### whether or not to symmetrize
    if not cfg.directed and cfg.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)



    ### Load method ###
    model = parse_method(cfg, c, d, device)



    ### Loss function (Single-class, Multi-class) ###
    if cfg.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()




    ### Performance metric (Acc, AUC, F1) ###
    if cfg.metric == 'rocauc':
        eval_func = eval_rocauc
    elif cfg.metric == 'f1':
        eval_func = eval_f1
    else:
        eval_func = eval_acc

    logger = Logger(cfg.runs, cfg)

    model.train()
    print('MODEL:', model)




    ### Training loop ###
    for run in range(cfg.runs):
        if cfg.dataset in ['cora', 'citeseer', 'pubmed'] and cfg.protocol == 'semi':
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[run]
        train_idx = split_idx['train'].to(device)
        model.reset_parameters()
        if cfg.method == 'sgformer':
            optimizer = torch.optim.Adam([
                {'params': model.params1, 'weight_decay': cfg.trans_weight_decay},
                {'params': model.params2, 'weight_decay': cfg.gnn_weight_decay}
            ],
                lr=cfg.lr)
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), weight_decay=cfg.weight_decay, lr=cfg.lr)
        best_val = float('-inf')

        for epoch in range(cfg.epochs):
            model.train()
            optimizer.zero_grad()

            train_start = time.time()
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
            if cfg.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
                if dataset.label.shape[1] == 1:
                    true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
                else:
                    true_label = dataset.label
                loss = criterion(out[train_idx], true_label.squeeze(1)[
                    train_idx].to(torch.float))
            else:
                out = F.log_softmax(out, dim=1)
                loss = criterion(
                    out[train_idx], dataset.label.squeeze(1)[train_idx])
            loss.backward()
            optimizer.step()

            if epoch % cfg.eval_step == 0:
                result = evaluate(model, dataset, split_idx, eval_func, criterion, cfg)
                logger.add_result(run, result[:-1])

                if epoch % cfg.display_step == 0:
                    print_str = f'Epoch: {epoch:02d}, ' + \
                                f'Loss: {loss:.4f}, ' + \
                                f'Train: {100 * result[0]:.2f}%, ' + \
                                f'Valid: {100 * result[1]:.2f}%, ' + \
                                f'Test: {100 * result[2]:.2f}%'
                    print(print_str)
        logger.print_statistics(run)

    logger.print_statistics()


if __name__ == "__main__":
    main()
