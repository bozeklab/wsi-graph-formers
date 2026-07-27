
import sys
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
# sys.path.append('../../../')  # Only for Remote use on Clusters


import argparse
import sys
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, remove_self_loops, add_self_loops
from torch_scatter import scatter
from torch_geometric.data import Batch , Data # for PyG v1.7


from logger import Logger, save_result
from dataset import load_dataset, load_dataset_extra, NCDataset, custom_fixed_split, \
    custom_cv_split_train_test
from data_utils import normalize, gen_normalized_adjs, eval_acc, eval_rocauc, eval_f1, \
    eval_binary_acc, eval_binary_rocauc, eval_binary_f1, eval_binary_bacc, eval_bacc, \
    to_sparse_tensor, load_fixed_splits, adj_mul, get_gpu_memory_map, count_parameters
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

from utils.graph_utils import fit_zscore_stats_pyg, normalize_zscore_pyg, \
    append_celltype_onehot_pyg, normalize_encode_celltype_pyg, mask_on_graph_list, \
    sanity_check_graph_list, induce_split_subgraph
from utils.train_utils import fix_seed
 


@hydra.main(config_path="../configs/Graph_Transformers", config_name="config_largewsi", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # print config nicely

    # we don't want to extract the nodestype from the cofig all along but only once as it can
    # change depending on the dataset choosen for instance
    nodestype = cfg.nodestype 

    print("Here is the dataset selected: {}".format(cfg.dataset))

    ### Load and preprocess data ###
    if cfg.dataset == 'onegraphskinwsi':
            preproc_dataset = load_dataset_extra(
                                    cfg.data_dir, 
                                    cfg.dataset, 
                                    cfg.nodestype, 
                                    cfg.train_prop, 
                                    cfg.valid_prop,
                                    cfg.sub_datasetname
                                    )

    elif cfg.dataset == 'skinwsi' or cfg.dataset == 'subgraphs-skinwsi': 
        raise ValueError(
            "skinwsi and subgraphs-skinwsi datasets fit only for training with batches."
            "For the training without batches, use onegraphskinwsi")


    else:
        raise ValueError(
            "Only onegraphskinwsi dataset can be used to run this code")



    # if len(dataset.label.shape) == 1:
    #     dataset.label = dataset.label.unsqueeze(1)
    # dataset.label = dataset.label.to(cfg.device)


    # NOTE: for consistent data splits, see data_utils.rand_train_test_idx
    def fix_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    fix_seed(cfg.seed)
    torch.manual_seed(cfg.seed)


    # Define cuda device 
    if cfg.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:" + str(cfg.device)) if torch.cuda.is_available() else torch.device("cpu")




    ### Convert split list (graphs) into a list of torch_geometric.data.Data objects:
    # converted = []
    # for g in split_list:
    if isinstance(preproc_dataset, Data):
        pass 
    else:
        # if g is a dict with those keys
        data = Data(
            x= preproc_dataset.graph['node_feat'],
            edge_index= preproc_dataset.graph['edge_index'],
            edge_feat= None,
            num_nodes= preproc_dataset.graph['num_nodes'],
            label= preproc_dataset.label
        )

        preproc_dataset = data


    # create limit index for the loop
    if cfg.cv:
        l = cfg.k_folds
    else:
        l= 1


    # we plan a loop for cv, but if not cv there will be only one run
    for testfold_idx in range(0,l):

        #### get the splits for all runs
        if cfg.cv:
            train_idx, test_idx = custom_cv_split_train_test(
                preproc_dataset.label,
                k_folds=cfg.k_folds,
                testfold=testfold_idx,
                seed=cfg.seed
            )
            valid_idx = None # to test
            print("\n***Cross-validation with test fold {}***\n".format(testfold_idx))

        else:
            train_idx, valid_idx, test_idx = custom_fixed_split(
                preproc_dataset.label,
                train_prop=cfg.train_prop,
                valid_prop=cfg.valid_prop,
                seed=cfg.seed
            )


        # construction used for the evaluation functions
        split_idx_dict =  {
                'train': train_idx,
                'valid': valid_idx,
                'test': test_idx,
            }


        # generate a graph from train_idx to be able to run fit_zscore_stats_pyg with no further changes    
        train_splitgraph = induce_split_subgraph(preproc_dataset, train_idx)



        ### Normalization of features ###
        if cfg.zscore_normalization:
            centroid_idx = [1,2]
            cont_idx = [0]
            cont_idx = cont_idx + list(range(3, preproc_dataset.x.shape[1]))  # all but centroid

            # Fit on TRAIN graphs only
            stats = fit_zscore_stats_pyg([train_splitgraph], cont_idx=cont_idx, centroid_idx=centroid_idx, device=device)

            if cfg.celltype_asfeature:

                ## Add cell type as a feature after normalization###
                graph = normalize_encode_celltype_pyg(
                                              preproc_dataset, 
                                              stats, 
                                              cont_idx=cont_idx,
                                              gamma=cfg.gamma,
                                              centroid_idx=centroid_idx,
                                              normalize_centroid=None
                                              )

            else: 
                # Apply the *same* stats to every split
                graph = normalize_zscore_pyg(
                                            preproc_dataset, 
                                            stats, 
                                            cont_idx=cont_idx,
                                            centroid_idx=centroid_idx,
                                            normalize_centroid=None
                                            ) 


        ## Add cell type as a feature and skip normalization###
        # gamma is useful only if there is a z-scoring normalization so put to 1 here 
        if not cfg.zscore_normalization and cfg.celltype_asfeature:
            graph = append_celltype_onehot_pyg(preproc_dataset, gamma=1) 

        if not cfg.zscore_normalization and not cfg.celltype_asfeature:
            graph = preproc_dataset


        ### Mask the hot-encoded cell type feature if needed ###
        # do it for supervised classes to avoid data leakage
        if cfg.celltype_asfeature:
            mask_on_graph_list([graph], classes=[4, 5], label_base=0)

            # Sanity check 
            print("Sanity check for masking... ")
            sanity_check_graph_list([graph], (4,5), label_base=0)


        ## We transformed the dataset object to  an torch_geometric.data.Data object for the feature transformations
        ## Now we do the transformation backward to fit with the rest of the script
        ## That take into account NCdataset object 



        # Create NCDataset
        dataset = NCDataset('onegraphskinwsi')

        dataset.graph = {
            'edge_index': graph.edge_index,
            'node_feat': graph.x,
            'edge_feat': None,
            'num_nodes': graph.num_nodes
        }
        dataset.label = graph.label

        # In realty graph as more attribute than newly dataset created object




        ### Basic information of datasets ###
        n = dataset.graph['num_nodes']
        e = dataset.graph['edge_index'].shape[1]
        # infer the number of classes for non one-hot and one-hot labels
        # c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
        c = dataset.label.max().item() + 1
        d = dataset.graph['node_feat'].shape[1]

        print(f"\ndataset {cfg.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

        #number of nodes per class
        listlabels = dataset.label.view(-1).tolist()
        class_counts = Counter(listlabels)

        print("\nNode count per class:")
        for cls, count in sorted(class_counts.items()):
            print(f"Class {cls}: {count} nodes")



        ### whether or not to symmetrize
        if not cfg.directed and cfg.dataset != 'ogbn-proteins':
            dataset.graph['edge_index'] = to_undirected(dataset.graph['edge_index'])


        ### clean graph and load to device 
        dataset.graph['edge_index'], _ = remove_self_loops(dataset.graph['edge_index'])
        dataset.graph['edge_index'], _ = add_self_loops(dataset.graph['edge_index'], num_nodes=n)
        
        dataset.graph['edge_index'], dataset.graph['node_feat'] = \
            dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)



        ### Load method ### 
        model = parse_method(cfg, c, d, device) #(args, num_classes, num_feats, device)



        ### Loss function (Single-class, Multi-class) ###
        singleclass_datasets = ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins')
        if cfg.dataset in singleclass_datasets or cfg.trainingtask == "binnodeclass_mask":
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




        for run in range(cfg.runs):


            train_idx = train_idx.to(device)
            dataset.label = dataset.label.to(device)

            if cfg.trainingtask == "binnodeclass_mask":

                # Mask for target classification nodes (4 or 5)
                values = torch.tensor([4, 5], device=device)
                train_mask = torch.stack([dataset.label == v for v in values]).any(dim=0)
                train_mask = train_mask.view(-1)

                binary_labels = (dataset.label == 5).float().view(-1)


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
                if cfg.method ==  'nodeformerbin':
                    out, link_loss_ = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
                    # link_loss_ is typically a list/tuple of per-layer link log-likelihood terms
                    link_reg = sum(link_loss_) / len(link_loss_)
                else:
                    out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])


                if cfg.trainingtask == "binnodeclass_mask":
                    # Make sure model output is of shape [N]
                    out = out.squeeze(1)  # Because binary model outputs 
                    # Compute loss only on nodes of class 4 and 5 (tumor and nontumor epithelial)
                    train_idx_filtered = train_idx[train_mask[train_idx]]
                    target = binary_labels[train_idx_filtered]
                    loss = criterion(out[train_idx_filtered], target)

                    if cfg.method ==  'nodeformerbin':
                        # we update the loss with he regularization term
                        loss = loss - cfg.lamda * link_reg
                
                else:
                    out = F.log_softmax(out, dim=1)
                    target = dataset.label.squeeze(1)[train_idx]

                    loss = criterion(out[train_idx], target)


                loss.backward()
                optimizer.step()


                ### Periodic evaluatio and logging
                if epoch % cfg.eval_step == 0 or epoch==(cfg.epochs-1):

                    if cfg.trainingtask == "binnodeclass_mask":
                        result = evaluate_binary_masked(model, dataset, split_idx_dict, eval_func, criterion, cfg, only2splits=cfg.cv)
                    else:
                        if cfg.cv:
                            raise ValueError("Cross-validation not implemented for other task than binnodeclass_mask")
                        else:
                            result = evaluate(model, dataset, split_idx_dict, eval_func, criterion, cfg)

                    logger.add_result(run, result[:-1])

                    if epoch % cfg.display_step == 0 or epoch==(cfg.epochs-1):

                            print_str = f'Epoch: {epoch:02d}, ' + \
                                        f'Loss: {loss:.4f}, ' + \
                                        f'Train: {100 * result[0]:.2f}%, ' + \
                                        f'Valid: {100 * result[1]:.2f}%, ' + \
                                        f'Test: {100 * result[2]:.2f}%'
                            print(print_str)

            logger.print_statistics(run, cv=cfg.cv)



        ### Print global training stats 
        train_ids = set(train_idx.tolist())
        test_ids = set(test_idx.tolist())
        intersection3 = train_ids & test_ids
        if not cfg.cv: 
            intersection1 = train_ids & valid_ids
            intersection2 = valid_ids & test_ids
            valid_ids = set(valid_idx.tolist())

        print(f"Sanity check: are splits fully separated:")
        if not cfg.cv: 
            print(f"\nOverlap between train and valid: {len(intersection1)} nodes")
            print(f"\nOverlap between valid and test: {len(intersection2)} nodes")
        print(f"\nOverlap between train and test: {len(intersection3)} nodes")

        
        if cfg.trainingtask == "binnodeclass_mask":
            labels = dataset.label.view(-1)
            if cfg.cv: 
                splitnames = ['train', 'test']
            else:
                splitnames = ['train', 'valid', 'test']
                
            for split_name in splitnames:
                idx = split_idx_dict[split_name]
                binary_mask = (labels[idx] == 4) | (labels[idx] == 5)
                binary_labels = (labels[idx][binary_mask] == 5).long()
                print(f"{split_name} → size: {idx.shape[0]}, class 4/5 only: {binary_labels.shape[0]}")
                print(f"    label counts: {binary_labels.bincount().tolist()}")
        
        logger.print_statistics()


        ### Save model ###
        if (not cfg.cv) and (cfg.save_model):
            # for now we don't allow saving models for cross-val
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

        ### some possible gathering before running the next cross val instance

    ### possible gathered information from cross validation  



if __name__ == "__main__":
    main()
