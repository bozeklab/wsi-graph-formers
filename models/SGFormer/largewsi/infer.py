import os
import sys
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
# sys.path.append('../../../') 

import torch
from collections import Counter

from models.SGFormer.largewsi.parse import parse_method
from models.SGFormer.largewsi.dataset import load_dataset

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig
from copy import deepcopy




@hydra.main(config_path="../../../configs/SGFormer", config_name="config_largewsi", version_base=None)
def infer(cfg: DictConfig):

    # Print hydra overrides
    # overrides = HydraConfig.get().overrides.task
    # print("Hydra overrides:")
    # for override in overrides:
    #     print("  -", override)

    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    infergraph = load_dataset(cfg.pred_input_dir, cfg.infer_graphtype, sub_dataname=cfg.pred_input_name)

    ### Basic information of inputgraphs 
    n = infergraph.graph['num_nodes']
    e = infergraph.graph['edge_index'].shape[1]
    # infer the number of classes for non one-hot and one-hot labels
    c = cfg.nbrclass_toinfer
    d = infergraph.graph['node_feat'].shape[1]

    print(f"\ninputgraph: {cfg.pred_input_name} | graph type: {cfg.infer_graphtype}\
    | num nodes: {n} | num edge: {e} | num node feats: {d} | num classes: {c}")

    # # adapt infer if we use the binary classification model 
    if cfg.nbrclass_toinfer == 2:

        if cfg.nodestype == 'notumor': 
            values = torch.tensor([4], device=infergraph.label.device)
            infer_mask = torch.stack([infergraph.label == v for v in values]).any(dim=0)
            infer_mask = infer_mask.view(-1)

        else:
            # the most useful 
            # Mask for target classification nodes (4 or 5)
            values = torch.tensor([4, 5], device=infergraph.label.device)
            train_mask = torch.stack([infergraph.label == v for v in values]).any(dim=0)
            train_mask = train_mask.view(-1)
            # values = torch.tensor([4], device=infergraph.label.device)
            # infer_mask = torch.stack([infergraph.label == v for v in values]).any(dim=0)
            # infer_mask = infer_mask.view(-1)


    ### Load method  
    model = parse_method(cfg, c, d, device) #(args, num_classes, num_feats, device)

    ### Load weigths
    model.load_state_dict(torch.load(cfg.model_dir + cfg.model_name, map_location=device))
    model.to(device).eval()

    ### create prediction dir if not already done
    if not os.path.exists(cfg.pred_output_dir):
        os.mkdir(cfg.pred_output_dir)



    with torch.no_grad():

        if cfg.nbrclass_toinfer == 2:

            # # Identify nodes with label == 4 (the only ones to classify)
            # infer_mask = (infergraph.label == 4).to(device)

            out = model(
                infergraph.graph['node_feat'].to(device),
                infergraph.graph['edge_index'].to(device)
            ).squeeze(1)  

            if cfg.nodestype == 'notumor': 
                # Identify nodes with label == 4 (the only ones to classify)
                infer_mask = (infergraph.label == 4).to(device)

            else:
                # the most useful 
                y = infergraph.label.view(-1)

                # PyTorch 1.9: no torch.isin, so use logical OR
                infer_mask = (y == 4) | (y == 5)

                # infer_mask = (infergraph.label == 5).float().view(-1) 


            # Initialize prediction tensor: fill everything with class 2 (for nodes not to classify)
            pred = torch.full_like(infergraph.label, fill_value=2, dtype=torch.long).to(device)

            # Apply sigmoid and threshold for binary prediction on class-4 nodes
            probs = torch.sigmoid(out[infer_mask])
            binary_pred = (probs > 0.5).long()  # 0 or 1

            # Assign predicted 0/1 to class-4 nodes
            pred[infer_mask] = binary_pred.to(device)


        else:
            out = model(
                infergraph.graph['node_feat'].to(device), 
                infergraph.graph['edge_index'].to(device)
            )
            pred = out.argmax(dim=1)

        # construct the inference graph output as a dict to fit inference script
        # if saved as a NCDataset object then the loading will require all dependancies 
        # so we create a simple dict in case visualization env is differen than train/inf env
        outputgraph = dict()

        outputgraph['x'] = infergraph.graph['node_feat']  
        outputgraph['y'] = pred
        outputgraph['edge_index'] = infergraph.graph['edge_index']
        outputgraph['centroid'] = infergraph.graph['centroid']   


        # save the graph
        noext_modelname = os.path.splitext(cfg.model_name)[0] 
        saving_path = cfg.pred_output_dir + \
                      f'predictions_{cfg.infer_graphtype}_{cfg.pred_output_name}_{noext_modelname}.pt'
        torch.save(outputgraph, saving_path)
        print("Inference saved here: {}".format(saving_path))




if __name__ == "__main__":
    infer()