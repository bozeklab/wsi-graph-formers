
import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, k_hop_subgraph
from torch_scatter import scatter
from torch_geometric.loader import DataLoader

from logger import Logger
from dataset import load_dataset, load_dataset_extra
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, \
    eval_binary_acc, eval_binary_rocauc, eval_binary_f1, eval_binary_bacc, eval_bacc, \
    to_sparse_tensor, load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from eval import evaluate_large, evaluate_batch
from parse import parse_method, parser_add_main_cfg
from collections import Counter

import time
import pickle
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import hydra
from omegaconf import DictConfig, OmegaConf




@hydra.main(config_path="../../../configs/SGFormer", config_name="config_largewsi", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # print config nicely

    if cfg.dataset == 'skinwsi':
            graph_list = load_dataset_extra(
                cfg.data_dir, 
                cfg.dataset, 
                cfg.nodestype, 
                cfg.train_prop, 
                cfg.valid_prop,
                cfg.sub_dataset
                )

    elif cfg.dataset == 'onegraphskinwsi': 
        raise ValueError(
            "onegraphskinwsi dataset fit only for training without batches."
            "For the training with batches, use skinwsi")

    else:
        raise ValueError(
            "Only skinwsi dataset can be used to run this code")



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
    graph_list = load_dataset_extra(
        cfg.data_dir, 
        cfg.dataset, 
        cfg.nodestype, 
        cfg.train_prop, 
        cfg.valid_prop,
        cfg.sub_dataset
        )

    
    ### splitting and batching ###
    torch.manual_seed(cfg.seed)
    graph_list = graph_list[:]
    n = len(graph_list)
    train_data = graph_list[:int(cfg.train_prop * n)]
    val_data = graph_list[int(cfg.train_prop * n):int(cfg.valid_prop * n)]
    test_data = graph_list[int(cfg.valid_prop * n):]

    train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1)
    test_loader = DataLoader(test_data, batch_size=1)



    ### Display information of dataset (nbr graphs and so on..) ###
    num_graphs = len(graph_list)
    num_nodes_list = [data.num_nodes for data in graph_list]
    num_edges_list = [data.num_edges for data in graph_list]

    # Collect all node labels
    all_labels = []
    for data in graph_list:
        y = data.y
        if y.ndim == 1:
            all_labels.extend(y.tolist())
        elif y.ndim == 2 and y.size(1) == 1:
            all_labels.extend(y.squeeze(1).tolist())
        else:
            raise ValueError("Unexpected label shape: expected 1D or (N,1), got " + str(y.shape))

    # Determine number of classes
    label_tensor = torch.tensor(all_labels)
    num_classes = label_tensor.max().item() + 1 if label_tensor.numel() > 0 else "unknown"
    c = num_classes

    # Node feature dimension (assume consistent shape)
    d = graph_list[0].x.shape[1]

    print(f"\ndataset {cfg.dataset} | num graphs: {num_graphs}")
    print(f"avg #nodes/graph: {sum(num_nodes_list)/num_graphs:.2f}, min: {min(num_nodes_list)}, max: {max(num_nodes_list)}")
    print(f"avg #edges/graph: {sum(num_edges_list)/num_graphs:.2f}, min: {min(num_edges_list)}, max: {max(num_edges_list)}")
    print(f"node feature dim: {d}")
    print(f"num node classes: {num_classes}")

    # Count node labels per class
    class_counts = Counter(all_labels)
    print("\nNode count per class:")
    for cls, count in sorted(class_counts.items()):
        print(f"Class {cls}: {count} nodes")



    ### Load method ### 
    model = parse_method(cfg, c, d, device) #(args, num_classes, num_feats, device)



    ### Loss function (Single-class, Multi-class) ###
    if cfg.trainingtask == "binnodeclass_mask":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.NLLLoss()



    ### Performance metric (Acc, AUC, F1) ###
    if cfg.metric == 'rocauc':
        if cfg.trainingtask == "binnodeclass_mask":
            eval_func = eval_binary_rocauc
        else:
            eval_func = eval_rocauc
    elif cfg.metric == 'f1':
        if cfg.trainingtask == "binnodeclass_mask":
            eval_func = eval_binary_f1
        else:
            eval_func = eval_f1
    elif cfg.metric == 'bacc':
        if cfg.trainingtask == "binnodeclass_mask":
            eval_func = eval_binary_bacc
        else:
            eval_func = eval_bacc
    else:
        if cfg.trainingtask == "binnodeclass_mask":
            eval_func = eval_binary_acc
        else:
            eval_func = eval_acc


    logger = Logger(cfg.runs, cfg)

    model.train()
    print('\n MODEL:', model)




    ### We can test until this point
    anchor = True

    #true_label = dataset.label





    ### Training loop ###
    for run in range(cfg.runs):
        if cfg.dataset in ['cora', 'citeseer', 'pubmed'] and cfg.protocol == 'semi':
            split_idx = split_idx_lst[0]
        else:
            split_idx = split_idx_lst[run]
        train_mask = torch.zeros(n, dtype=torch.bool)
        train_mask[split_idx['train']] = True

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
        num_batch = n // cfg.batch_size + (n%cfg.batch_size>0)
        for epoch in range(cfg.epochs):
            model.to(device)
            model.train()

            idx = torch.randperm(n)
            for i in range(num_batch):
                idx_i = idx[i*cfg.batch_size:(i+1)*cfg.batch_size]
                train_mask_i = train_mask[idx_i]
                x_i = x[idx_i].to(device)
                edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
                edge_index_i = edge_index_i.to(device)
                y_i = true_label[idx_i].to(device)
                optimizer.zero_grad()
                out_i = model(x_i, edge_index_i)
                if cfg.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
                    loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i].to(torch.float))

                else:
                    out_i = F.log_softmax(out_i, dim=1)
                    loss = criterion(out_i[train_mask_i], y_i.squeeze(1)[train_mask_i])
                loss.backward()
                optimizer.step()

            if epoch % cfg.eval_step == 0:
                if cfg.dataset=='ogbn-papers100M':
                    result = evaluate_batch(model, dataset, split_idx, cfg, device, n, true_label)
                else:
                    result = evaluate_large(model, dataset, split_idx, eval_func, criterion, cfg, device="cpu")
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