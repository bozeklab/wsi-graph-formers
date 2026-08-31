from models.difformer import *
from models.gnns import *
from models.nodeformer import *
from models.sgformer import *


def parse_method(cfg, c: int, d: int, device: torch.device):
    if cfg.method == 'gcn':
        model = GCN(in_channels=d,
                    hidden_channels=cfg.hidden_channels,
                    out_channels=c,
                    num_layers=cfg.num_layers,
                    dropout=cfg.dropout,
                    use_bn=cfg.use_bn).to(device)
    if cfg.method == 'gcnbin':
        model = GCN_bin(in_channels=d,
                    hidden_channels=cfg.hidden_channels,
                    out_channels=c,
                    num_layers=cfg.num_layers,
                    dropout=cfg.dropout,
                    use_bn=cfg.use_bn).to(device)
    elif cfg.method == 'gat':
        model = GAT(d, cfg.hidden_channels, c,
                    num_layers=cfg.num_layers,
                    dropout=cfg.dropout,
                    use_bn=cfg.use_bn,
                    heads=cfg.gat_heads,
                    out_heads=cfg.out_heads).to(device)
    elif cfg.method == 'gatbin':
        model = GAT_bin(d, cfg.hidden_channels, c,
                    num_layers=cfg.num_layers,
                    dropout=cfg.dropout,
                    use_bn=cfg.use_bn,
                    heads=cfg.gat_heads,
                    out_heads=cfg.out_heads).to(device)
    elif cfg.method in ['mlp', 'manireg']:
        model = MLP(in_channels=d,
                    hidden_channels=cfg.hidden_channels,
                    out_channels=c,
                    num_layers=cfg.num_layers,
                    dropout=cfg.dropout).to(device)
    elif cfg.method == 'heat':
        model = GCN(in_channels=d,
                    hidden_channels=cfg.hidden_channels,
                    out_channels=c,
                    num_layers=cfg.num_layers,
                    dropout=cfg.dropout,
                    use_bn=cfg.use_bn).to(device)
    elif cfg.method == 'sgc':
        model = SGCMem(in_channels=d,
                       out_channels=c,
                       hops=cfg.hops,
                       use_bn=cfg.use_bn).to(device)
    elif cfg.method == 'sgcbin':
        model = SGCMem_bin(in_channels=d,
                       out_channels=c,
                       hops=cfg.hops,
                       use_bn=cfg.use_bn).to(device)
    elif cfg.method == 'sgc2':
        model = SGC2(d, cfg.hidden_channels, c,
                     cfg.hops, cfg.num_layers,
                     cfg.dropout, use_bn=cfg.use_bn).to(device)
    elif cfg.method == 'sgc2bin':
        model = SGC2_bin(d, cfg.hidden_channels, c,
                     cfg.hops, cfg.num_layers,
                     cfg.dropout, use_bn=cfg.use_bn).to(device)
    elif cfg.method == 'sign':
        model = SIGN(in_channels=d,
                     hidden_channels=cfg.hidden_channels,
                     out_channels=c,
                     hops=cfg.hops,
                     num_layers=cfg.num_layers,
                     dropout=cfg.dropout,
                     use_bn=cfg.use_bn).to(device)
    elif cfg.method == 'signbin':
        model = SIGN_bin(in_channels=d,
                     hidden_channels=cfg.hidden_channels,
                     out_channels=c,
                     hops=cfg.hops,
                     num_layers=cfg.num_layers,
                     dropout=cfg.dropout,
                     use_bn=cfg.use_bn).to(device)
    elif cfg.method == 'sgformer':
        model = SGFormer(
            d, cfg.hidden_channels, c,
            graph_weight=cfg.graph_weight,
            aggregate=cfg.aggregate,
            trans_num_layers=cfg.trans_num_layers,
            trans_dropout=cfg.trans_dropout,
            trans_num_heads=cfg.trans_num_heads,
            trans_use_bn=cfg.trans_use_bn,
            trans_use_residual=cfg.trans_use_residual,
            trans_use_weight=cfg.trans_use_weight,
            trans_use_act=cfg.trans_use_act,
            gnn_num_layers=cfg.gnn_num_layers,
            gnn_dropout=cfg.gnn_dropout,
            gnn_use_bn=cfg.gnn_use_bn,
            gnn_use_residual=cfg.gnn_use_residual,
            gnn_use_weight=cfg.gnn_use_weight,
            gnn_use_init=cfg.gnn_use_init,
            gnn_use_act=cfg.gnn_use_act,
        ).to(device)
    elif cfg.method == 'sgformerbin':
        model = SGFormer_bin(
            d, cfg.hidden_channels, c,
            graph_weight=cfg.graph_weight,
            aggregate=cfg.aggregate,
            trans_num_layers=cfg.trans_num_layers,
            trans_dropout=cfg.trans_dropout,
            trans_num_heads=cfg.trans_num_heads,
            trans_use_bn=cfg.trans_use_bn,
            trans_use_residual=cfg.trans_use_residual,
            trans_use_weight=cfg.trans_use_weight,
            trans_use_act=cfg.trans_use_act,
            gnn_num_layers=cfg.gnn_num_layers,
            gnn_dropout=cfg.gnn_dropout,
            gnn_use_bn=cfg.gnn_use_bn,
            gnn_use_residual=cfg.gnn_use_residual,
            gnn_use_weight=cfg.gnn_use_weight,
            gnn_use_init=cfg.gnn_use_init,
            gnn_use_act=cfg.gnn_use_act,
        ).to(device)
    elif cfg.method == 'nodeformer':
        model = NodeFormer(
            in_channels=d,
            hidden_channels=cfg.hidden_channels,
            out_channels=c,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            num_heads=cfg.num_heads,
            use_bn=cfg.use_bn
        ).to(device)
    elif cfg.method == 'nodeformerbin':
        model = NodeFormer_bin(
            in_channels=d,
            hidden_channels=cfg.hidden_channels,
            out_channels=c,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            num_heads=cfg.num_heads,
            use_bn=cfg.use_bn
        ).to(device)
    elif cfg.method == 'difformer':
        model = DIFFormer(
            in_channels=d,
            hidden_channels=cfg.hidden_channels,
            out_channels=c,
            num_layers=cfg.num_layers,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            num_heads=cfg.num_heads
        ).to(device)
    elif cfg.method == 'difformerbin':
        model = DIFFormer_bin(
            in_channels=d,
            hidden_channels=cfg.hidden_channels,
            out_channels=c,
            num_layers=cfg.num_layers,
            alpha=cfg.alpha,
            dropout=cfg.dropout,
            num_heads=cfg.num_heads
        ).to(device)
    else:
        raise ValueError(f"Invalid method: {cfg.method}")
    return model







