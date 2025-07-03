from dataclasses import dataclass

@dataclass
class LargeWSISchema:
    dataset: str = 'largewsi'
    sub_dataset: str = ''
    data_dir: str = '../../../Data_General/Predictions/Former_Models_InferData/data/'
    device: int = 0                    # which gpu to use if any (default: 0)
    cpu: bool = False 
    seed: int = 123
    epochs: int = 500
    runs: int = 1                      # number of distinct runs
    directed: bool = False             # set to not symmetrize adjacency
    train_prop: float = 0.5            # training label proportion
    valid_prop: float = 0.25           # validation label proportion
    protocol: str = 'semi'             # protocol for cora datasets, semi or supervised
    rand_split: bool = False           # use random splits
    rand_split_class: bool = False     # use random splits with a fixed number of labeled nodes for each class
    label_num_per_class: int = 20      # labeled nodes randomly selected
    metric: str = 'acc'                # evaluation metric

    # Model
    method: str = 'sgformer'           
    hidden_channels: int = 32
    use_graph: bool = False            # use input graph
    aggregate: str = 'add'             # aggregate type, add or cat.
    graph_weight: float = 0.8          # graph weight
 
    # GNN
    gnn_use_bn: bool = False           # use batchnorm for each GNN layer
    gnn_use_residual: bool = False     # use residual link for each GNN layer
    gnn_use_weight: bool = False       # use weight for GNN convolution
    gnn_use_init: bool = False         # use initial feat for each GNN layer
    gnn_use_act: bool = False          # use activation for each GNN layer
    gnn_num_layers: int = 2            # number of layers for GNN
    gnn_dropout: float = 0.0
    gnn_weight_decay: float = 1e-3 

    # Transformer
    trans_num_heads: int = 1           # number of heads for attention
    trans_use_weight: bool = False     # use weight for trans convolution
    trans_use_bn: bool = False         # use layernorm for trans
    trans_use_residual: bool = False   # use residual link for each trans layer
    trans_use_act: bool = False        # use activation for each trans layer
    trans_num_layers: int = 2          # number of layers for all-pair attention
    trans_dropout: float = 0.0        
    trans_weight_decay: float = 1e-3

    # Training
    lr: float = 0.01
    batch_size: int = 10000            # mini batch training for large graphs
    patience: int = 200                # early stopping patience

    # Utility
    display_step: int = 1              # how often to print
    eval_step: int = 1                 # how often to evaluate
    cached: bool = False               # set to use faster sgc
    print_prop: bool = False           # print proportions of predicted class
    save_result: bool = False          
    save_model: bool = False
    use_pretrained: bool = False
    save_att: bool = False             # whether to save attention (for visualization)
    model_dir: str = '../../model/'

    # Misc
    hops: int = 2                      # number of hops for SGC
    gat_heads: int = 4                 # attention heads for gat
    out_heads: int = 1                 # out heads for gat
