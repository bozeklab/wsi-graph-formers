
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
from eval import evaluate_large, evaluate_batch, evaluate_wloader
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


    ### Load and preprocess data ###
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



    ### splitting and batching ###
    n = len(graph_list)

    # Calculate split indices
    train_end = int(cfg.train_prop * n)
    val_end = int(cfg.train_prop * n + cfg.valid_prop * n)

    train_data = graph_list[:train_end]
    val_data = graph_list[train_end:val_end]
    test_data = graph_list[val_end:]

    # Torch Dataloader, We use collate_graphs that the dataloader can take NCDataset instance as input
    train_loader = DataLoader(train_data, batch_size=1, shuffle=True,  collate_fn=Batch.from_data_list)
    val_loader = DataLoader(val_data, batch_size=1,  collate_fn=Batch.from_data_list)
    test_loader = DataLoader(test_data, batch_size=1,  collate_fn=Batch.from_data_list)




    ### Display information of dataset (nbr graphs and so on..) ###
    num_graphs = len(graph_list)
    num_nodes_list = [data.num_nodes for data in graph_list]
    num_edges_list = [data.num_nodes for data in graph_list]

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

                startbatchload = time.perf_counter()
                print('Data name:', data)

                data = data.to(device)          # moves x, edge_index, y, etc.
                optimizer.zero_grad()

                out = model(data.x, data.edge_index)

                if cfg.trainingtask == "binnodeclass_mask":
                    # TO ADD LATER
                    raise ValueError("binnodeclass_mask mode not implemented yet")
                    

                else:
                    out = F.log_softmax(out, dim=1)
                    target = data.label.squeeze()

                    loss = criterion(out, target)

                # endbatchload = time.perf_counter()
                # print(f"Learning with this batch took {endbatchload - startbatchload:.6f} seconds")


            loss.backward()
            optimizer.step()


            # total_loss += loss.item()

            # avg_train_loss = total_loss / len(train_loader)

            ### Periodic evaluatio and logging
            if epoch % cfg.eval_step == 0:
                
                train_metric, train_loss = evaluate_wloader(model, train_loader, eval_func, criterion, cfg, device)
                val_metric,   val_loss   = evaluate_wloader(model, val_loader,   eval_func, criterion, cfg, device)
                test_metric,  _          = evaluate_wloader(model, test_loader,  eval_func, criterion, cfg, device)

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