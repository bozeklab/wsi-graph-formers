import os
import sys
import glob

import torch
import torch.nn.functional as F
from torch_geometric.utils import subgraph

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.graph_utils import mask_celltype_onehot_cols
from data_utils import eval_acc, eval_rocauc, eval_f1, \
    eval_binary_acc, eval_binary_rocauc, eval_binary_f1, eval_binary_bacc, eval_bacc
from dataset import load_dataset
from tqdm import tqdm


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, cfg, result=None):
    if result is not None:
        out = result
    else:
        model.eval()
        try:
            # Some models (like NodeFormer) return (out, link_loss_)
            out,_ = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        except ValueError:
            # for most of the models:
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

        
    train_metric = eval_func(
        dataset.label[split_idx['train']], 
        out[split_idx['train']]
    )
    valid_metric = eval_func(
        dataset.label[split_idx['valid']], 
        out[split_idx['valid']]
    )
    test_metric = eval_func(
        dataset.label[split_idx['test']], 
        out[split_idx['test']]
    )

    # NOT ONEGRAPHSKINWSI HERE --------------------------------------------------------
    if cfg.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(
            out[split_idx['valid']], 
            true_label.squeeze(1)[split_idx['valid']].to(torch.float)
        )
    # END OF NOT ONEGRAPHSKINWSI -----------------------------------------------------------
            

    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], 
            dataset.label.squeeze(1)[split_idx['valid']]
        )

    return train_metric, valid_metric, test_metric, valid_loss, out





@torch.no_grad()
def evaluate_wloader(model, 
                     loader, 
                     eval_func, 
                     criterion, 
                     cfg, 
                     device,
                     celltype_asfeature: bool = False):
    """
    Evaluate a node‑classification model on a PyG DataLoader of graphs.
    Returns:
      metric (float): whatever eval_func(all_true, all_pred) produces
      avg_loss (float): mean loss per node
    """
    model.eval()
    all_preds = []
    all_trues = []
    loss_sum = 0.0
    total_nodes = 0

    for batch in loader:

        batch = batch.to(device)

        # if cfg.celltype_asfeature:
        #     mask_celltype_onehot_cols(batch, classes=[4, 5], label_base=0)

        try:
            # Some models (like NodeFormer) return (out, link_loss_)
            logits, _ = model(batch.x, batch.edge_index)
        except ValueError:
            # for most of the models:
            logits = model(batch.x, batch.edge_index)

        out = F.log_softmax(out, dim=1)
       
        #target = batch.label.to(torch.float)
        #target = batch.label.squeeze.to(torch.long)
        target = batch.label.to(torch.long)

        loss = criterion(out, target)
        loss_sum += loss.item() * target.size(0)
        total_nodes += target.size(0)

        all_preds.append(out.cpu())
        all_trues.append(batch.label.cpu())

    # Concatenate over all graphs
    # get shape (total_nodes, C) for y_pred  and (total_nodes,) for y_true
    y_pred = torch.cat(all_preds, dim=0)
    y_true = torch.cat(all_trues, dim=0)

    # resize to have y_true with a shape fitting SGFormer utils
    num_classes = y_pred.size(1)
    y_true = F.one_hot(y_true, num_classes=num_classes)  # (total_nodes, C)
    y_true = y_true.argmax(dim=-1, keepdim=True) # (total_nodes, 1)

    # Compute your metric (accuracy / rocauc / f1, etc.)
    metric = eval_func(y_true, y_pred)
    avg_loss = loss_sum / total_nodes

    return metric, avg_loss






@torch.no_grad()
def evaluate_binary_masked(model, dataset, split_idx, eval_func, criterion, cfg, result=None, only2splits = False):
    if result is not None:
        out = result
    else:
        model.eval()
        try:
            # Some models (like NodeFormer) return (out, link_loss_)
            out,_ = model(dataset.graph['node_feat'], dataset.graph['edge_index'])
        except ValueError:
            # for most of the models:
            out = model(dataset.graph['node_feat'], dataset.graph['edge_index'])

    out = out.squeeze(1)

    labels = dataset.label.view(-1)
    binary_labels = (labels == 5).float()   # 4 → 0, 5 → 1
    class_mask = (labels == 4) | (labels == 5)
    #  STILL TO TEST FOR  not (cfg.nodestype == 'notumor')  

    def filtered_eval(split_name):
        idx = split_idx[split_name]
        filtered_idx = idx[class_mask[idx]]
        y_pred = (out[filtered_idx] > 0).long()  # predicted class index
        y_true = binary_labels[filtered_idx]
        return eval_func(y_true, y_pred)

    def filtered_loss(split_name):
        idx = split_idx[split_name]
        filtered_idx = idx[class_mask[idx]]
        return criterion(out[filtered_idx], binary_labels[filtered_idx])


    train_metric = filtered_eval('train')
    if only2splits:
        valid_metric = 0
    else:
        valid_metric = filtered_eval('valid')
    test_metric  = filtered_eval('test')    
    if only2splits:
        valid_loss = 0
    else:
        valid_loss = filtered_loss('valid') 

    return train_metric, valid_metric, test_metric, valid_loss, out

    # target = batch.label.squeeze().to(torch.long)
    # loss also needs to change # target = batch.label.to(torch.float)




@torch.no_grad()
def evaluate_binmasked_wloader(model, 
                               loader, 
                               eval_func, 
                               criterion, 
                               cfg, 
                               device, 
                               threshold_logit: float = 0.0, 
                               celltype_asfeature: bool = False):
    """
    Binary masked evaluation on a PyG DataLoader:
      - Uses only nodes with labels in {4,5}
      - Maps 5 -> 1 (positive), 4 -> 0 (negative)
      - Computes loss with BCEWithLogitsLoss on masked nodes
      - For the metric, thresholds logits at `threshold_logit` (default 0) to get class predictions
    Returns:
      metric (float), avg_loss (float)
    """
    model.eval()
    all_preds = []
    all_trues = []
    loss_sum = 0.0
    total_masked = 0

    for batch in loader:
        batch = batch.to(device)

        # if cfg.celltype_asfeature:
        #     mask_celltype_onehot_cols(batch, classes=[4, 5], label_base=0)
        try:
            # Some models (like NodeFormer) return (out, link_loss_)
            logits, _ = model(batch.x, batch.edge_index)
        except ValueError:
            # for most of the models:
            logits = model(batch.x, batch.edge_index)

        
        # ensure shape [N]
        if logits.dim() == 2 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        else:
            assert logits.dim() == 1, "Binary head must output [N] or [N,1]."

        y = batch.label.view(-1)

        if cfg.nodestype == 'notumor':
            raise ValueError("No notumor mode for several graph dataset implemented yet.")

        else:
            # create Binary nodes 
            # 5 -> 1, 4 -> 0
            mask_45 = (y == 4) | (y == 5)
            if not mask_45.any():
                continue

            y_bin = (y == 5).float()  # 5 -> 1, 4 -> 0

            # loss on masked nodes only
            masked_logits = logits[mask_45]
            masked_targets = y_bin[mask_45]
            loss = criterion(masked_logits, masked_targets)
            m = mask_45.sum().item()
            loss_sum += loss.item() * m
            total_masked += m
            # normally there is no need to calculate the loss inside the loop, could create a variable "mask"
            # and run it outisde, but readibility is better this way 

        # predictions for metric: sign(logit) -> class
        y_pred = (masked_logits > threshold_logit).long()

        # keep the same convention as evaluate_binary_masked (true as 0/1 float)
        all_preds.append(y_pred.cpu())
        all_trues.append(masked_targets.cpu())

    if total_masked == 0:
        # no eligible nodes across the loader
        return 0.0, 0.0

    y_pred_all = torch.cat(all_preds, dim=0)            # (M,)
    y_true_all = torch.cat(all_trues, dim=0)            # (M,) floats in {0.,1.}

    metric = eval_func(y_true_all, y_pred_all)
    avg_loss = loss_sum / total_masked
    return metric, avg_loss



# WORK IN PROGRESS !!!!!!!! ********************
@torch.no_grad()
def evaluate_binmasked_infer(pred_input_dir,
                             pred_output_dir,
                             model_name,  
                             eval_func, 
                             threshold_logit: float = 0.0
                             ):
    """
    Binary masked evaluation on a PyG DataLoader:
      - Uses only nodes with labels in {4,5}
      - Maps 5 -> 1 (positive), 4 -> 0 (negative)
      - Computes loss with BCEWithLogitsLoss on masked nodes
      - For the metric, thresholds logits at `threshold_logit` (default 0) to get class predictions
    Returns:
      metric (float), avg_loss (float)
    """
    all_preds = []
    all_trues = []
    total_masked = 0
    noext_modelname = os.path.splitext(model_name)[0] 

    gt_files = os.path.join(pred_input_dir, '*.pt')
    gt_files = glob.glob(gt_files)

    print("Load binary node labels...")
    for path_gtgraph in tqdm(gt_files):
        gtgraphname = os.path.split(path_gtgraph)[1]
        gtgraphname_noext = os.path.splitext(gtgraphname)[0]

        path_predgraph = str(
            pred_output_dir + 
            "predictions_" +
            gtgraphname_noext +
            "_" +            
            noext_modelname +
            ".pt"
        )

        if os.path.exists(path_gtgraph) and os.path.exists(path_predgraph):

            gtname = os.path.split(path_gtgraph)[1]
            predname = os.path.split(path_predgraph)[1]
            gtdataset_type = "inferin" # as gt it is infer input
            preddataset_type = "predout"
            gtgraph = load_dataset(pred_input_dir, gtdataset_type, sub_dataname=gtname)
            predgraph = load_dataset(pred_output_dir,preddataset_type, sub_dataname=predname)

            gtlabels = gtgraph.label.view(-1)
            predlabels = predgraph.label.view(-1)


            gtmask_45 = (gtlabels == 4) | (gtlabels == 5)
            if not gtmask_45.any():
                continue

            # for pred graph, the nodes already have 0 and 1, but 2 for nodes not to classify.
            # we want not to have this nodes anymore for eval, only the calssified ones
            # predmask = (predlabels == 0) | (predlabels == 1)
            # if not predmask.any():
            #     continue

            # make labels of the GT binary 
            gtlabels_bin = (gtlabels == 5).int()  # 5 -> 1, 4 -> 0
            # eval on masked nodes only            
            masked_gt = gtlabels_bin[gtmask_45]
            masked_pred = predlabels[gtmask_45]

            u = masked_pred.unique()
            if not ((u == 0) | (u == 1)).all():
                raise ValueError(f"Found non-binary preds on masked nodes: {u}")

            # loss = criterion(masked_logits, masked_targets)
            # m = mask_45.sum().item()
            # loss_sum += loss.item() * m
            # total_masked += m
            # # normally there is no need to calculate the loss inside the loop, could create a variable "mask"
            # and run it outisde, but readibility is better this way 
            # predictions for metric: sign(logit) -> class
            # y_pred = (masked_logits > threshold_logit).long()
            # keep the same convention as evaluate_binary_masked (true as 0/1 float)
            all_preds.append(masked_pred.cpu())
            all_trues.append(masked_gt.cpu())

    # if total_masked == 0:
    #     # no eligible nodes across the loader
    #     return 0.0, 0.0

    y_pred_all = torch.cat(all_preds, dim=0)            # (M,)
    y_true_all = torch.cat(all_trues, dim=0)            # (M,) floats in {0.,1.}

    print("Calculate metric....")
    metric = eval_func(y_true_all, y_pred_all)
    return metric

# ************************************************





@torch.no_grad()
def evaluate_large(model, dataset, split_idx, eval_func, criterion, args, device="cpu", result=None):
    if result is not None:
        out = result
    else:
        model.eval()

    model.to(torch.device(device))
    dataset.label = dataset.label.to(torch.device(device))
    edge_index, x = dataset.graph['edge_index'].to(torch.device(device)), dataset.graph['node_feat'].to(torch.device(device))
    out = model(x, edge_index)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    test_acc = eval_func(
        dataset.label[split_idx['test']], out[split_idx['test']])
    if args.dataset in ('yelp-chi', 'deezer-europe', 'twitch-e', 'fb100', 'ogbn-proteins'):
        if dataset.label.shape[1] == 1:
            true_label = F.one_hot(dataset.label, dataset.label.max() + 1).squeeze(1)
        else:
            true_label = dataset.label
        valid_loss = criterion(out[split_idx['valid']], true_label.squeeze(1)[
            split_idx['valid']].to(torch.float))
    else:
        out = F.log_softmax(out, dim=1)
        valid_loss = criterion(
            out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, test_acc, valid_loss, out




def evaluate_batch(model, dataset, split_idx, args, device, n, true_label):
    num_batch = n // args.batch_size + 1
    edge_index, x = dataset.graph['edge_index'], dataset.graph['node_feat']
    train_mask = torch.zeros(n, dtype=torch.bool)
    train_mask[split_idx['train']] = True
    valid_mask = torch.zeros(n, dtype=torch.bool)
    valid_mask[split_idx['valid']] = True
    test_mask = torch.zeros(n, dtype=torch.bool)
    test_mask[split_idx['test']] = True

    model.to(device)
    model.eval()

    idx = torch.randperm(n)
    train_total, train_correct=0, 0
    valid_total, valid_correct=0, 0
    test_total, test_correct=0, 0

    with torch.no_grad():
        for i in range(num_batch):
            idx_i = idx[i*args.batch_size:(i+1)*args.batch_size]
            x_i = x[idx_i].to(device)
            edge_index_i, _ = subgraph(idx_i, edge_index, num_nodes=n, relabel_nodes=True)
            edge_index_i = edge_index_i.to(device)
            y_i = true_label[idx_i].to(device)
            train_mask_i = train_mask[idx_i]
            valid_mask_i = valid_mask[idx_i]
            test_mask_i = test_mask[idx_i]

            out_i = model(x_i, edge_index_i)

            cur_train_total, cur_train_correct=eval_acc(y_i[train_mask_i], out_i[train_mask_i])
            train_total+=cur_train_total
            train_correct+=cur_train_correct
            cur_valid_total, cur_valid_correct=eval_acc(y_i[valid_mask_i], out_i[valid_mask_i])
            valid_total+=cur_valid_total
            valid_correct+=cur_valid_correct
            cur_test_total, cur_test_correct=eval_acc(y_i[test_mask_i], out_i[test_mask_i])
            test_total+=cur_test_total
            test_correct+=cur_test_correct

            # train_acc = eval_func(
            #     dataset.label[split_idx['train']], out[split_idx['train']])
            # valid_acc = eval_func(
            #     dataset.label[split_idx['valid']], out[split_idx['valid']])
            # test_acc = eval_func(
            #     dataset.label[split_idx['test']], out[split_idx['test']])
        train_acc=train_correct/train_total
        valid_acc=valid_correct/valid_total
        test_acc=test_correct/test_total

    return train_acc, valid_acc, test_acc, 0, None

def eval_acc(true, pred):
    '''
    true: (n, 1)
    pred: (n, c)
    '''
    pred=torch.max(pred,dim=1,keepdim=True)[1]
    # cmp=torch.eq(true, pred)
    # print(f'pred:{pred}')
    # print(cmp)
    true_cnt=(true==pred).sum()

    return true.shape[0], true_cnt.item()






@hydra.main(config_path="../../../configs/SGFormer", config_name="config_largewsi", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))  # print config nicely

    if cfg.evalinfer:

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


        evalmetric = evaluate_binmasked_infer(
            cfg.pred_input_dir,
            cfg.pred_output_dir, 
            cfg.model_name, 
            eval_func,
            threshold_logit=0.5
        )

        print("{} of the test sample is {}".format(cfg.metric, evalmetric))





if __name__=='__main__':
    main()

    # x=torch.arange(4).unsqueeze(1)
    # y=torch.Tensor([[3,0,0,0],
    #                 [3,2,1.5,2.8],
    #                 [0,0,2,1],
    #                 [0,0,1,3]
    #                 ])
    # a, b=eval_acc(x, y)
    # print(x)
    # print(a,b)










