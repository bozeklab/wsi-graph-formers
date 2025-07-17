
import os
import torch
from parse import parse_method
from dataset import load_dataset

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="../../../configs/SGFormer", config_name="config_largewsi", version_base=None)
def infer(cfg: DictConfig):

    # Print hydra overrides
    # overrides = HydraConfig.get().overrides.task
    # print("Hydra overrides:")
    # for override in overrides:
    #     print("  -", override)

    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    dataset = load_dataset(cfg.data_dir, cfg.dataset, cfg.sub_dataset)

    ### Basic information of datasets 
    n = dataset.graph['num_nodes']
    e = dataset.graph['edge_index'].shape[1]
    # infer the number of classes for non one-hot and one-hot labels
    c = cfg.nbrclass_toinfer
    d = dataset.graph['node_feat'].shape[1]

    print(f"dataset {cfg.dataset} | num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")


    ### Load method  
    model = parse_method(cfg, c, d, device) #(args, num_classes, num_feats, device)

    ### Load weigths
    model.load_state_dict(torch.load(cfg.model_dir + cfg.model_name, map_location=device))
    model.to(device).eval()

    ### create prediction dir if not already doen
    if not os.path.exists(cfg.pred_dir):
        os.mkdir(cfg.pred_dir)


    with torch.no_grad():
        out = model(dataset.graph['node_feat'].to(device), dataset.graph['edge_index'].to(device))
        pred = out.argmax(dim=1)
        
        noext_modelname = os.path.splitext(cfg.model_name)[0] 
        saving_path = cfg.pred_dir + f'predictions_{cfg.dataset}_{noext_modelname}.pt'
        torch.save(pred, saving_path)
        print("Inference saved here: {}".format(saving_path))




if __name__ == "__main__":
    infer()