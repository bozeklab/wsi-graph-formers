

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
from dataset import load_dataset, load_dataset_extra
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, \
    eval_binary_acc, eval_binary_rocauc, eval_binary_f1, to_sparse_tensor, load_fixed_splits, adj_mul, \
    get_gpu_memory_map, count_parameters
from eval import evaluate, evaluate_binary_masked
from parse import parse_method 
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
    if cfg.customload:
        if cfg.dataset == 'skinwsi':
            dataset = load_dataset_extra(
                cfg.data_dir, 
                cfg.dataset, 
                cfg.nodestype, 
                cfg.train_prop, 
                cfg.valid_prop,
                cfg.sub_dataset
                )
        else:
            pass 

    else:
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
        split_idx_dict = dataset.load_fixed_splits()

    else:
        split_idx_lst = load_fixed_splits(cfg.data_dir, dataset, name=cfg.dataset, protocol=cfg.protocol)



    ### Basic information of datasets ###
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    # infer the number of classes for non one-hot and one-hot labels
    c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
    d = dataset.graph['node_feat'].shape[1]

    print(f"\n dataset {cfg.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

    #number of nodes per class
    listlabels = dataset.label.view(-1).tolist()
    class_counts = Counter(listlabels)

    print("\n Node count per class:")
    for cls, count in sorted(class_counts.items()):
        print(f"Class {cls}: {count} nodes")



    ### whether or not to symmetrize
    if not cfg.directed and cfg.dataset != 'ogbn-proteins':
        dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])

    dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
    dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)

    dataset.graph['edge_index'], dataset.graph['node_feat'] = \
        dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)



    ### Load method ### 
    model = parse_method(cfg, c, d, device) #(args, num_classes, num_feats, device)



    ### Loss function (Single-class, Multi-class) ###
    if cfg.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
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
    else:
        if cfg.trainingtask == "binnodeclass_mask":
            eval_func = eval_binary_acc
        else:
            eval_func = eval_acc


    logger = Logger(cfg.runs, cfg)

    model.train()
    print('\n MODEL:', model)




    ### Training loop ###
    for run in range(cfg.runs):
        if cfg.dataset in ['cora', 'citeseer', 'pubmed'] and cfg.protocol == 'semi':
            split_idx = split_idx_lst[0]
        elif cfg.dataset == "skinwsi":
            # there is only one run of train/val/test (for now)
            split_idx = split_idx_dict
        else:
            split_idx = split_idx_lst[run]


        train_idx = split_idx['train'].to(device)

        if cfg.trainingtask == "binnodeclass_mask":
            # Mask for target classification nodes (4 or 5
            values = torch.tensor([4, 5], device=dataset.label.device)
            train_mask = torch.stack([dataset.label == v for v in values]).any(dim=0)
            train_mask = train_mask.view(-1)

            binary_labels = (dataset.label == 5).long().view(-1)


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

                if cfg.trainingtask == "binnodeclass_mask":
                    # Compute loss only on nodes of class 4 and 5 (tumor and nontumor epithelial)
                    train_idx_filtered = train_idx[train_mask[train_idx]]
                    loss = criterion(
                        out[train_idx_filtered], 
                        binary_labels[train_idx_filtered]
                        )

                else:
                    loss = criterion(
                        out[train_idx], 
                        dataset.label.squeeze(1)[train_idx]
                        )

            loss.backward()
            optimizer.step()


            if epoch % cfg.eval_step == 0:

                if cfg.trainingtask == "binnodeclass_mask":
                    result = evaluate_binary_masked(model, dataset, split_idx, eval_func, criterion, cfg)
                else:
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




    train_ids = set(split_idx['train'].tolist())
    valid_ids = set(split_idx['valid'].tolist())

    intersection = train_ids & valid_ids
    print(f"Overlap between train and valid: {len(intersection)} nodes")

    labels = dataset.label.view(-1)

    for split_name in ['train', 'valid', 'test']:
        idx = split_idx[split_name]
        binary_mask = (labels[idx] == 4) | (labels[idx] == 5)
        binary_labels = (labels[idx][binary_mask] == 5).long()
        print(f"{split_name} → size: {idx.shape[0]}, class 4/5 only: {binary_labels.shape[0]}")
        print(f"    label counts: {binary_labels.bincount().tolist()}")
    
    logger.print_statistics()


    if cfg.save_model:
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        #create the model directory if does not exists:
        if not os.path.exists(cfg.model_dir):
            os.mkdir(cfg.model_dir)

        # add in the name of tjhe weights if the task was 
        # Binary Node Classification with Known Context Nodes
        if cfg.trainingtask == "binarynodeclass_withkcn":
            save_path = os.path.join(
                cfg.model_dir, 
                f"bnckcn_{cfg.method}_{cfg.dataset}_run{timestamp}.pth"
            )
        else:
            save_path = os.path.join(
                cfg.model_dir, 
                f"{cfg.method}_{cfg.dataset}_run{timestamp}.pth"
            )


        torch.save(model.state_dict(), save_path)
        print(f"[INFO] Model weights saved to: {save_path}")        



if __name__ == "__main__":
    main()
