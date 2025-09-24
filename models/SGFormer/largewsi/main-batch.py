
import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops, subgraph, k_hop_subgraph
from torch_scatter import scatter
from torch_geometric.data import Batch , Data # for PyG v1.7
from torch.utils.data import DataLoader 


from logger import Logger
from dataset import load_dataset, load_dataset_extra
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, \
    eval_binary_acc, eval_binary_rocauc, eval_binary_f1, eval_binary_bacc, eval_bacc, \
    to_sparse_tensor, load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
from eval import evaluate_large, evaluate_batch, evaluate_wloader, evaluate_binmasked_wloader
from parse import parse_method
from collections import Counter

import time
import pickle
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.graph_utils import fit_zscore_stats_pyg, normalize_zscore_pyg, \
    append_celltype_onehot_pyg, normalize_encode_celltype_pyg, mask_on_graph_list, \
    sanity_check_graph_list 



@hydra.main(config_path="../../../configs/SGFormer", config_name="config_largewsi", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # print config nicely

    # we don't want to extract the nodestype from the cofig all along but only once as it can
    # change depending on the dataset choosen for instance
    nodestype = cfg.nodestype 

    print("Here is the dataset selected: {}".format(cfg.dataset))

    ### Load and preprocess data ###
    if cfg.dataset == 'skinwsi':
            graph_list = load_dataset_extra(
                cfg.data_dir, 
                cfg.dataset, 
                nodestype, 
                cfg.train_prop, 
                cfg.valid_prop,
                cfg.sub_datasetname
                )

    elif cfg.dataset == 'subgraphs-skinwsi':
            # for subgraphs we only keep one node type 
            nodestype = "allclasses"
            graph_list = load_dataset_extra(
                cfg.data_dir, 
                cfg.dataset, 
                nodestype, 
                cfg.train_prop, 
                cfg.valid_prop,
                cfg.sub_datasetname
                )

    elif cfg.dataset == 'subgraphs-onegraphskinwsi':
            # for subgraphs we only keep one node type 
            nodestype = "allclasses"
            graph_list = load_dataset_extra(
                cfg.data_dir, 
                cfg.dataset, 
                nodestype, 
                cfg.train_prop, 
                cfg.valid_prop,
                cfg.sub_datasetname
                )


    elif cfg.dataset == 'onegraphskinwsi': 
        raise ValueError(
            "onegraphskinwsi dataset fit only for training without batches."
            "For the training with batches, use skinwsi")

    else:
        raise ValueError(
            "Only skinwsi dataset and subgraphs skinwsi datasets can be used to run this code")



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


    torch.manual_seed(cfg.seed)



    ### Convert graph list into a list of torch_geometric.data.Data objects:
    converted = []
    for g in graph_list:
        if isinstance(g, Data):
            data = g
        else:
            # if g is a dict with those keys
            data = Data(
                x= g.graph['node_feat'],
                edge_index= g.graph['edge_index'],
                edge_feat= None,
                num_nodes= g.graph['num_nodes'],
                label= g.label
            )
        converted.append(data)


    # replacement 
    graph_list = converted


    # sanity-check
    assert all(isinstance(g, Data) for g in graph_list), "Make sure graph_list[i] is a torch_geometric.data.Data"



    ### create folds ###
    n = len(graph_list)

    if cfg.dataset == 'skinwsi':
        # Calculate split indices
        train_end = int(cfg.train_prop * n)
        val_end = int(cfg.train_prop * n + cfg.valid_prop * n)

        train_data = graph_list[:train_end]
        val_data = graph_list[train_end:val_end]
        test_data = graph_list[val_end:]

    if cfg.dataset == 'subgraphs-skinwsi' or cfg.dataset == 'subgraphs-onegraphskinwsi':
        # we want the subgraph to be randomly spread in the training set 
        random.shuffle(graph_list)
        # the shuffle follow the defined seeds above

        # Calculate split indices
        train_end = int(cfg.train_prop * n)
        val_end = int(cfg.train_prop * n + cfg.valid_prop * n)

        train_data = graph_list[:train_end]
        val_data = graph_list[train_end:val_end]
        test_data = graph_list[val_end:]


    ### Normalization of features ###
    if cfg.zscore_normalization:
        centroid_idx = [1,2]
        cont_idx = [0]
        cont_idx = cont_idx + list(range(3, data.x.size(1)))  # all but centroid

        # Fit on TRAIN graphs only
        stats = fit_zscore_stats_pyg(train_data, cont_idx=cont_idx, centroid_idx=centroid_idx, device='cpu')

        if cfg.celltype_asfeature:

            ## Add cell type as a feature after normalization###

            # Apply the *same* stats to every split
            train_data = [normalize_encode_celltype_pyg(
                                          g, 
                                          stats, 
                                          cont_idx=cont_idx,
                                          gamma=cfg.gamma,
                                          centroid_idx=centroid_idx,
                                          normalize_centroid=None
                                          ) for g in train_data]
            val_data = [normalize_encode_celltype_pyg(
                                          g, 
                                          stats, 
                                          cont_idx=cont_idx,
                                          gamma=cfg.gamma,
                                          centroid_idx=centroid_idx,
                                          normalize_centroid=None
                                          ) for g in val_data]
            test_data = [normalize_encode_celltype_pyg(
                                          g, 
                                          stats, 
                                          cont_idx=cont_idx,
                                          gamma=cfg.gamma,
                                          centroid_idx=centroid_idx,
                                          normalize_centroid=None
                                          ) for g in test_data]
        else: 
            # Apply the *same* stats to every split
            train_data = [normalize_zscore_pyg(
                                        g, 
                                        stats, 
                                        cont_idx=cont_idx,
                                        centroid_idx=centroid_idx,
                                        normalize_centroid=None
                                        ) for g in train_data]
            val_data   = [normalize_zscore_pyg(
                                        g, 
                                        stats, 
                                        cont_idx=cont_idx,
                                        centroid_idx=centroid_idx,
                                        normalize_centroid=None
                                        ) for g in val_data]
            test_data  = [normalize_zscore_pyg(
                                        g, 
                                        stats, 
                                        cont_idx=cont_idx,
                                        centroid_idx=centroid_idx,
                                        normalize_centroid=None
                                        ) for g in test_data]


    ## Add cell type as a feature and skip normalization###
    # gamma is useful only if there is a z-scoring normalization so put to 1 here 
    if not cfg.zscore_normalization and cfg.celltype_asfeature:
        train_data = [append_celltype_onehot_pyg(g,gamma=1) for g in train_data]
        val_data = [append_celltype_onehot_pyg(g,gamma=1) for g in val_data]
        test_data = [append_celltype_onehot_pyg(g,gamma=1) for g in test_data]


    ### Mask the hot-encoded cell type feature if needed ###
    # do it for supervised classes to avoid data leakage
    if cfg.celltype_asfeature:
        mask_on_graph_list(train_data, classes=[4, 5], label_base=0)
        mask_on_graph_list(val_data, classes=[4, 5], label_base=0)
        mask_on_graph_list(test_data, classes=[4, 5], label_base=0)

        # Sanity check 
        print("Sanity check on train split masking... ")
        sanity_check_graph_list(train_data, (4,5), label_base=0)
        print("Sanity check on val split masking... ")
        sanity_check_graph_list(val_data,   (4,5), label_base=0)
        print("Sanity check on test split masking... ")
        sanity_check_graph_list(test_data,  (4,5), label_base=0)


    ### dataloader and batching ###
    if cfg.dataset == 'skinwsi':
       # Torch Dataloader, We use collate_graphs that the dataloader can take NCDataset instance as input
        train_loader = DataLoader(train_data, batch_size=1, shuffle=True,  collate_fn=Batch.from_data_list)
        val_loader = DataLoader(val_data, batch_size=1,  collate_fn=Batch.from_data_list)
        test_loader = DataLoader(test_data, batch_size=1,  collate_fn=Batch.from_data_list)

    if cfg.dataset == 'subgraphs-skinwsi' or cfg.dataset == 'subgraphs-onegraphskinwsi':
       # Torch Dataloader, We use collate_graphs that the dataloader can take NCDataset instance as input
        train_loader = DataLoader(train_data, batch_size=cfg.trainsubgraphs_batch_size, shuffle=True,  collate_fn=Batch.from_data_list)
        val_loader = DataLoader(val_data, batch_size=cfg.trainsubgraphs_batch_size,  collate_fn=Batch.from_data_list)
        test_loader = DataLoader(test_data, batch_size=cfg.testsubgraphs_batch_size,  collate_fn=Batch.from_data_list)


    ### Display information of dataset (nbr graphs and so on..) ###
    num_graphs = len(graph_list)
    num_nodes_list = [data.num_nodes for data in graph_list]
    num_edges_list = [len(data.edge_index[0]) for data in graph_list]

    # Collect all node labels
    all_labels = []
    for data in graph_list:
        y = data.label
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

    # display information 
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



    ### Training loop ###
    for run in range(cfg.runs):

        model.reset_parameters()
        model.to(device)

        if cfg.method == 'sgformer':
            optimizer = torch.optim.Adam([
                {'params': model.params1, 'weight_decay': cfg.trans_weight_decay},
                {'params': model.params2, 'weight_decay': cfg.gnn_weight_decay}
            ], lr=cfg.lr)
        else:
            optimizer = torch.optim.Adam(
                model.parameters(), weight_decay=cfg.weight_decay, lr=cfg.lr)
            

        train_start_training = time.time()

        for epoch in range(cfg.epochs):
            model.train()
            # total_loss = 0.0

            for data in train_loader:           # each `data` is one graph

                data = data.to(device)          # moves x, edge_index, y, etc.
                optimizer.zero_grad()     

                out = model(data.x, data.edge_index)

                if cfg.trainingtask == "binnodeclass_mask":

                    if not nodestype == 'notumor': 

                        # Binary masked loss: supervise only nodes with labels {4,5}
                        logits = out
                        if logits.dim() == 2 and logits.size(1) == 1:
                            logits = logits.squeeze(1)
                        else:
                            assert logits.dim() == 1, "Binary head must output [N] or [N,1]."

                        y = data.label.view(-1)

                        # PyTorch 1.9: no torch.isin, so use logical OR
                        mask_45 = (y == 4) | (y == 5)

                        if not mask_45.any():
                            continue  # no eligible nodes in this batch

                        y_bin = (y == 5).float()          # 5 -> 1, 4 -> 0
                        loss = criterion(logits[mask_45], y_bin[mask_45])  # BCEWithLogitsLoss

                        # The logic is quite different than for main.py, both because now we are working with batches
                        # and because we are working with the train loader instances instead of graph dictionnaries 


                    else:

                        raise ValueError("No notumor mode for several graph dataset implemented yet.") 

                else:
                    out = F.log_softmax(out, dim=1)
                    target = data.label.squeeze()

                    loss = criterion(out, target)


                loss.backward()
                optimizer.step()


            # total_loss += loss.item()

            # avg_train_loss = total_loss / len(train_loader)

            ### Periodic evaluatio and logging
            if epoch % cfg.eval_step == 0:

                if cfg.trainingtask == "binnodeclass_mask":
                    train_metric, train_loss = evaluate_binmasked_wloader(
                        model, 
                        train_loader, 
                        eval_func, 
                        criterion, 
                        cfg, 
                        device, 
                        celltype_asfeature=cfg.celltype_asfeature
                    )
                    val_metric,   val_loss   = evaluate_binmasked_wloader(
                        model, 
                        val_loader, 
                        eval_func, 
                        criterion, 
                        cfg, 
                        device, 
                        celltype_asfeature=cfg.celltype_asfeature
                    )
                    test_metric,  _          = evaluate_binmasked_wloader(
                        model, 
                        test_loader, 
                        eval_func, 
                        criterion, 
                        cfg, 
                        device, 
                        celltype_asfeature=cfg.celltype_asfeature
                    )
                else:
                    train_metric, train_loss = evaluate_wloader(
                        model, 
                        train_loader, 
                        eval_func, 
                        criterion, 
                        cfg, 
                        device, 
                        celltype_asfeature=cfg.celltype_asfeature
                    )
                    val_metric,   val_loss   = evaluate_wloader(
                        model, 
                        val_loader, 
                        eval_func, 
                        criterion, 
                        cfg, 
                        device, 
                        celltype_asfeature=cfg.celltype_asfeature
                    )
                    test_metric,  _          = evaluate_wloader(
                        model, 
                        test_loader, 
                        eval_func, 
                        criterion, 
                        cfg, 
                        device, 
                        celltype_asfeature=cfg.celltype_asfeature
                    )
                    
                logger.add_result(run, [train_metric, val_metric, test_metric, val_loss])

                if epoch % cfg.display_step == 0:

                    print_str = f'Epoch: {epoch:02d}, ' + \
                                f'Train Loss: {train_loss:.4f}, ' + \
                                f'Train: {100*train_metric:.2f}%, ' + \
                                f'Val Loss: {val_loss:.4f}, ' + \
                                f'Val:   {100*val_metric:.2f}%, '  + \
                                f'Test: {100*test_metric:.2f}%'
                    print(print_str)
    
    logger.print_statistics(run)

    ### Print global training stats
    ### NOT necessary here?

    ### Save model ###
    if cfg.save_model:
        # Get current timestamp
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')

        #create the model directory if does not exists:
        if not os.path.exists(cfg.model_dir):
            os.mkdir(cfg.model_dir)

        # Add prefix to the name of the weights if the task was 
        # Binary Node Classification with Known Context Nodes
        if cfg.trainingtask == "binnodeclass_mask":
            save_path = os.path.join(
                cfg.model_dir, 
                f"binnodeclass_{cfg.method}_{cfg.dataset}_run{timestamp}.pth"
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