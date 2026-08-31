import glob
import os

import hydra

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))
# sys.path.append('../../../') 
import torch
from dataset import (
    NCDataset,
    load_dataset,
)
from omegaconf import DictConfig
from torch_geometric.data import Data  # for PyG v1.7

from models.large_wsigraphs.dataset import load_dataset
from models.large_wsigraphs.parse import parse_method
from utils.graph_utils import (
    append_celltype_onehot_pyg,
    mask_on_graph_list,
    normalize_encode_celltype_pyg,
    normalize_zscore_pyg,
    sanity_check_graph_list,
)


@hydra.main(config_path="../../configs/SGFormer", config_name="config_largewsi", version_base=None)
def infer(cfg: DictConfig):

    # Print hydra overrides
    # overrides = HydraConfig.get().overrides.task
    # print("Hydra overrides:")
    # for override in overrides:
    #     print("  -", override)

    device = torch.device(f"cuda:{cfg.device}" if torch.cuda.is_available() else "cpu")
    infcount = 1

    if cfg.multi_graph_infer :
        files = os.path.join(cfg.pred_input_dir, '*.pt')
        files = glob.glob(files)
    else:
        files = [cfg.pred_input_dir + cfg.pred_input_name]


    # for inference we load the graphs one by one so the dataset type is onegraphskinwsi
    dataset_type = "inferin"
    for fname in files:
        if os.path.exists(fname):
            if cfg.multi_graph_infer:
                print(f"Runnning inference {infcount} on {len(files)}")
                infcount += 1
                filename = os.path.split(fname)[1]
                infergraph = load_dataset(cfg.pred_input_dir, dataset_type, sub_dataname=filename)
            else:
                infergraph = load_dataset(cfg.pred_input_dir, dataset_type, sub_dataname=cfg.pred_input_name)


            ### Convert split list (graphs) into a list of torch_geometric.data.Data objects:
            # converted = []
            # for g in split_list:
            if isinstance(infergraph, Data):
                pass 
            else:
                # if g is a dict with those keys
                data = Data(
                    x= infergraph.graph['node_feat'],
                    edge_index= infergraph.graph['edge_index'],
                    edge_feat= None,
                    num_nodes= infergraph.graph['num_nodes'],
                    centroid = infergraph.graph['centroid'],
                    label= infergraph.label
                )

                infergraph = data



            # ## We remove unclassified nodes because the models are not trained with it so we cannot infer with 7 classes
            # keep_mask = (infergraph.label != 0) & (infergraph.label != -1)
            # classified_nodes = keep_mask.nonzero(as_tuple=True)[0]
            # infergraph = subgraph_filtering(infergraph, classified_nodes, filtering_step=1)

            # ## rename labels to fit with training
            # for cellclass in range(1,7):  
            #     infergraph.label[infergraph.label == cellclass] = cellclass - 1
           


            ### Normalization of features before test to correspond to training graph, having the same features ###
            if cfg.zscore_normalization:
                centroid_idx = [1,2]
                cont_idx = [0]
                cont_idx = cont_idx + list(range(3, infergraph.x.shape[1]))  # all but centroid

                # stats = fit_zscore_stats_pyg([infergraph], cont_idx=cont_idx, centroid_idx=centroid_idx, device=device)
                raise NotImplementedError(
                    "Inference with zscore_normalization=True is not implemented "
                    "yet: the z-score statistics fitted on the training graphs are "
                    "not persisted with the checkpoint, so they cannot be restored "
                    "here. Run inference with zscore_normalization=False, or save "
                    "the statistics at training time and load them at this point."
                )

                if cfg.celltype_asfeature:

                    ## Add cell type as a feature after normalization###
                    graph = normalize_encode_celltype_pyg(
                                                  infergraph, 
                                                  stats, 
                                                  cont_idx=cont_idx,
                                                  gamma=cfg.gamma,
                                                  centroid_idx=centroid_idx,
                                                  normalize_centroid=None
                                                  )

                else: 
                    # Apply the *same* stats to every split
                    graph = normalize_zscore_pyg(
                                                infergraph, 
                                                stats, 
                                                cont_idx=cont_idx,
                                                centroid_idx=centroid_idx,
                                                normalize_centroid=None
                                                ) 


            ## Add cell type as a feature and skip normalization###
            # gamma is useful only if there is a z-scoring normalization so put to 1 here 
            if not cfg.zscore_normalization and cfg.celltype_asfeature:
                graph = append_celltype_onehot_pyg(infergraph, gamma=1) 

            if not cfg.zscore_normalization and not cfg.celltype_asfeature:
                graph = infergraph


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
            infergraph = NCDataset('onegraphskinwsi')

            infergraph.graph = {
                'edge_index': graph.edge_index,
                'node_feat': graph.x,
                'edge_feat': None,
                'num_nodes': graph.num_nodes,
                'centroid': graph.centroid
            }
            infergraph.label = graph.label





            ### Basic information of inputgraphs 
            n = infergraph.graph['num_nodes']
            e = infergraph.graph['edge_index'].shape[1]
            # infer the number of classes for non one-hot and one-hot labels
            c = cfg.nbrclass_toinfer
            d = infergraph.graph['node_feat'].shape[1]

            print(f"inputgraph: {cfg.pred_input_name} \
            | num nodes: {n} | num edge: {e} | num node feats: {d} | num classes: {c}")

            # # adapt infer if we use the binary classification model 
            # if cfg.nbrclass_toinfer == 2:

            #     if cfg.nodestype == 'notumor': 
            #         values = torch.tensor([4], device=infergraph.label.device)
            #         infer_mask = torch.stack([infergraph.label == v for v in values]).any(dim=0)
            #         infer_mask = infer_mask.view(-1)

            #     else:
            #         # the most useful 
            #         # Mask for target classification nodes (4 or 5)
            #         values = torch.tensor([4, 5], device=infergraph.label.device)
            #         train_mask = torch.stack([infergraph.label == v for v in values]).any(dim=0)
            #         train_mask = train_mask.view(-1)
            #         # values = torch.tensor([4], device=infergraph.label.device)
            #         # infer_mask = torch.stack([infergraph.label == v for v in values]).any(dim=0)
            #         # infer_mask = infer_mask.view(-1)


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


                    # Initialize prediction tensor: fill everything with class -1 (for nodes not to classify)
                    pred = torch.full_like(infergraph.label, fill_value=-1, dtype=torch.long).to(device)

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
                if cfg.multi_graph_infer:
                    filename_noext = os.path.splitext(filename)[0]
                    saving_path = cfg.pred_output_dir + \
                              f'predictions_{filename_noext}_{noext_modelname}.pt'
                else:
                    saving_path = cfg.pred_output_dir + \
                              f'predictions_{cfg.pred_output_name}_{noext_modelname}.pt'
                torch.save(outputgraph, saving_path)
                print(f"Inference saved here: {saving_path} \n\n")




if __name__ == "__main__":
    infer()