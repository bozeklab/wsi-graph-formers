
<div align="center">

[![tests](https://github.com/bozeklab/wsi-graph-formers/actions/workflows/test.yml/badge.svg)](https://github.com/ORG/wsi-graph-formers/actions/workflows/test.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2602.15783-b31b1b.svg)](https://arxiv.org/abs/2602.15783)
[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

</div>


# Skin Cancer Epithelial Cell Classification with Scalable Graph Transformers

<div align="center">

[Project presentation](#project-presentation) • [Project structure](#project-structure) • [Installation](#installation) •  [Datasets](#datasets) • [Node classification](#node-classification) • [Ablation study](#ablation-study) • [Generate your own cell graphs](#generate-your-own-cell-graphs) • [Visualize cell graphs](#visualize-cell-graphs)  • [Image Baseline](#image-baseline)  • [References and citation](#references-and-citation)

</div>

<br>

This repository contains the code for ["Context-aware Skin Cancer Epithelial Cell Classification with Scalable Graph Transformers"](https://arxiv.org/abs/2602.15783)  paper.

## Project presentation 

In this project, we generate cell graphs from Whole Slide Images of patients with cancer (cSCC here). Each node of the graph represents a nucleus and contain node features that encode nucleus morphology and texture. Each node also has a label corresponding to its cell type. Once such graphs are generated, tumor detection become a binary node classification on epithelial nodes, to classify such nodes as healthy or tumor.  Generated graphs reach half to several millions of nodes, so standard Graph Transformers with quadratic complexity could not be trained on them. We then benchmark scalable Graph Transformers (SGFormer, NodeFormer, DIFFormer) against classic GNNs (GCN, GAT, SGC, SGC-MLP, SIGN) on our two released datasets, WSI-Graph and TILE-Graphs. 


## Project structure

```bash
├── configs/                          # all Hydra configs
│   ├── config.yaml                   # for dataset_tools/ and baseline_tools/
│   ├── orders/                       # file orderings to reproduce the paper splits
│   └── Graph_Transformers/
│       ├── config_largewsi.yaml      # main config for models/
│       └── experiments/              # one yaml per paper experiment
├── dataset_tools/                    # build, convert, simplify, split and visualize cell graphs
├── models/                           # models, training, evaluation, inference
├── utils/                            # features preprocessing, cross-validation splits
├── baseline_tools/                   # prepare the image baseline datasets
│   ├── cellvit/
│   └── hovernet/
├── docs/                             # figures of this README
├── environment.yaml                  # conda environment
├── pyproject.toml                    # makes the repo importable with `pip install -e .`
├── download_data_classification.sh
├── download_data_experiments.sh
├── Baseline.md                       # image baselines documentation
├── LICENSE.md
└── README.md
```

## Installation

- [Prerequisites](#prerequisites) 
- [Installation commands](#installation-commands) 
- [Tested on](#tested-on) 

### Prerequisites

- Linux x86-64
- conda or miniconda
- For GPU use:
  - An NVIDIA driver supporting CUDA 11.1 (450.80.02+). No CUDA toolkit is required; the torch wheels bundle their own runtime.
  - A GPU of compute capability 8.6 or lower (Ampere or older: A100, V100, RTX 30-series, etc.). Torch 1.9 will be installed and ships no kernels for Hopper (H100, sm_90) or newer.

Python 3.8, torch 1.9.0+cu111 and all other packages are installed by
`environment.yaml`.

### Installation commands

To install simply run:

```bash
conda env create -f environment.yaml
conda activate wsi-graph-formers
pip install -e .
```

To Verify:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expected: 1.9.0+cu111 True   (on a GPU node)
```

### Tested on

Linux x86-64 (Ubuntu 20.04)

GPU: CUDA 11.1 on an NVIDIA A100 (sm_80) — imports and CUDA ops.
CPU-only: imports only, on a GPU-less node.
Other platforms have not been tested.

## Datasets

The datasets used in this work are publicly available on Zenodo following this link:

* [WSI-Graph and TILE-Graphs datasets](https://doi.org/10.5281/zenodo.21415205) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21415205.svg)](https://doi.org/10.5281/zenodo.21415205)

## Node classification 

- [Download datasets](#download-datasets) 
- [With WSI-Graph](#with-wsi-graph)
	- [Scalable Graph Transformers eval](#scalable-graph-transformers-eval) 
	- [Classic GNNs eval](#classic-gnns-eval) 
	- [Random nodes eval](#random-nodes-eval) 
- [With TILE-Graphs](#with-tile-graphs) 
	- [Scalable Graph Transformers eval](#scalable-graph-transformers-eval-1) 
	- [Classic GNNs eval](#classic-gnns-eval-1) 
- [Save cv result in a file](#save-cross-validation-result-in-a-file-rather-than-display-in-terminal) 

Reproduce the node classification cross-validations from the paper. 

### Download datasets 

Inside `wsi-graph-former` run:

```bash
chmod +x download_data_classification.sh
./download_data_classification.sh
```

This will create a folder `data` inside your local `wsi-graph-formers` folder and include  extracted folders `TILE-Graphs` and `WSI-Graph-100splits` as well as the file `TILE_patients_ID.csv` from the Zenodo dataset. 

Remove all unnecessary MAC OS files (step needed):
```bash
cd data
find . -type f \( -name '._*' -o -name '.DS_Store' \) -delete
cd ..
```

### With WSI-Graph

#### Scalable Graph Transformers eval

To **evaluate scalable Graph Transformers models** with subgraph evaluation  method for binary node classification, on WSI-Graph dataset with 3-fold cross-validation, run the following command within this repository:

```bash
conda activate wsi-graph-formers
python models/main-batch.py experiments=<chosen_experiment> data_dir=data/WSI-Graph-100splits/3-max-hops-simplification/ 
```

<details>
	<summary>[Unfold] With chosen_experiment being one of the following:</summary>

- WSI-Graphs_subgraphs_crossval_DIFFormer ; to evaluate DIFFormer model with 3-fold cross-validation on WSI-Graph dataset using subgraphs
- WSI-Graphs_subgraphs_crossval_NodeFormer ; to evaluate NodeFormer model with 3-fold cross-validation on WSI-Graph dataset using subgraphs
- WSI-Graphs_subgraphs_crossval_SGFormer ; to evaluate SGFormer model with 3-fold cross-validation on WSI-Graph dataset using subgraphs     

</details>

#### Classic GNNs eval

To **evaluate other GNNs models** with subgraph evaluation  method for binary node classification, on WSI-Graph dataset with 3-fold cross-validation, run the following command within this repository:

```bash
conda activate wsi-graph-formers
python models/main-batch.py experiments=WSI-Graphs_subgraphs_crossval_generalgnn data_dir=data/WSI-Graph-100splits/3-max-hops-simplification/ method=<chosen_method>
```

<details>
	<summary>[Unfold] With chosen_method being one of the following:</summary>

-  gcnbin ; to evaluate GCN model with 3-fold cross-validation on WSI-Graph dataset using subgraphs 
-  gatbin ; to evaluate GAT model with 3-fold cross-validation on WSI-Graph dataset using subgraphs 
-  sgcbin ; to evaluate SGC model with 3-fold cross-validation on WSI-Graph dataset using subgraphs 
-  sgc2bin ; to evaluate SGC-MLP model with 3-fold cross-validation on WSI-Graph dataset using subgraphs
-  signbin ; to evaluate SIGN model with 3-fold cross-validation on WSI-Graph dataset using subgraphs

</details>

#### Random nodes eval

To evaluate with Random Nodes method follow the same logic and use experiments files from `configs/Graph_Transformers/experiments/` and data from `WSI-Graph` folder (to download) instead of ``WSI-Graph-100splits`` folder.


### With TILE-Graphs


#### Scalable Graph Transformers eval

To **evaluate scalable Graph Transformers models** for binary node classification on TILE-Graphs dataset with 3-fold cross-validation, run the following command within this repository:

```bash
conda activate wsi-graph-formers
python models/main-batch.py experiments=<chosen_experiment> data_dir=data/TILE-Graphs/ patient_csv=data/TILE_patients_ID.csv
```

<details>
	<summary>[Unfold] With chosen_experiment being one of the following:</summary>

- TILE-Graphs_crossval_DIFFormer ; to evaluate DIFFormer model with 3-fold cross-validation on TILE-Graphs dataset
- TILE-Graphs_crossval_NodeFormer ; to evaluate NodeFormer model with 3-fold cross-validation on TILE-Graphs dataset
- TILE-Graphs_crossval_SGFormer ; to evaluate SGFormer model with 3-fold cross-validation on TILE-Graphs dataset

</details>

#### Classic GNNs eval

To **evaluate other GNNs models** for binary node classification on TILE-Graphs dataset with 3-fold cross-validation, run the following command within this repository:

```bash
conda activate wsi-graph-formers
python models/main-batch.py experiments=TILE-Graphs_crossval_generalgnn data_dir=data/TILE-Graphs/ patient_csv=data/TILE_patients_ID.csv method=<chosen_method> 
```

<details>
	<summary>[Unfold] With chosen_method being one of the following:</summary>

-  gcnbin ; to evaluate GCN model with 3-fold cross-validation on TILE-Graphs dataset
-  gatbin ; to evaluate GAT model with 3-fold cross-validation on TILE-Graphs dataset
-  sgcbin ; to evaluate SGC model with 3-fold cross-validation on TILE-Graphs dataset 
-  sgc2bin ; to evaluate SGC-MLP model with 3-fold cross-validation on TILE-Graphs dataset 
-  signbin ; to evaluate SIGN model with 3-fold cross-validation on TILE-Graphs dataset

</details>

### Save cross-validation result in a file rather than display in terminal

Add to the previous commands:

```bash
 > eval.txt 2>&1 
```

To generate a text file eval.txt in current folder rather than generating evaluation in terminal. 

## Ablation study

- [Download corresponding data](#download-corresponding-data) 
- [Feature ablation study](#feature-ablation-study) 

To reproduce feature ablation study experiments from the paper.

### Download corresponding data

Inside `wsi-graph-former` run:

```bash
chmod +x download_data_experiments.sh
./download_data_experiments.sh
```

This will create a folder `data` inside your local `wsi-graph-formers` folder (if not already done) and include extracted folders `experiments`  and `WSI-Graph-100splits` folders from the Zenodo dataset (if not already done). 

Remove all unnecessary MAC OS files (step needed):

```bash
cd data
find . -type f \( -name '._*' -o -name '.DS_Store' \) -delete
cd ..
```

### Feature ablation study

To reproduce feature ablation study for WSI-Graph **containing texture features** within each node:

```bash
conda activate wsi-graph-formers
python models/main-batch.py experiments=experiments_subgraphs_crossval_feature_ablation data_dir=data/WSI-Graph-100splits/3-max-hops-simplification/ zscore_normalization=<True/False> celltype_asfeature=<True/False>
```

Choose False or True depending on which experiement (line of the table) you want to reproduce. To reproduce everything run all configurations.

To reproduce feature ablation study for WSI-Graph **without texture feature** within nodes:

```bash
conda activate wsi-graph-formers
python models/main-batch.py experiments=experiments_subgraphs_crossval_feature_ablation data_dir=data/experiments/feature-ablations/subgraphs/ order_file=data/experiments/feature-ablations/orders/notexture_order.txt zscore_normalization=<True/False> celltype_asfeature=<True/False>
```

Choose False or True depending on which experiement (line of the table) you want to reproduce. To reproduce everything run all configurations.

Random nodes evaluation follows the same logic with `main.py` , `experiments_randomnodes_crossval_feature_ablation` config and _data/WSI-Graph/_  `data_dir` (WSI-Graph folder is to download from Zenodo, it is not the same as WSI-Graph-100splits ; for nodes without texture feature corresponding data is in experiment folder). No needs of `order_file` as there is only one input file.

## Generate your own cell graphs

- [Requirements](#requirements) 
- [Build](#build) 
- [Convert](#convert) 
- [Simplify](#simplify) 
- [Split](#split) 

<p align="center">
  <img src="docs/generate-cell-graphs.png">
</p>

### Requirements

Need to run histo-miner inferences on the WSI / patch you want to generate graph from.  Either run SCC Segmenter if no tumor annotation exists or create your own annotations. 

See [histo-miner](https://github.com/bozeklab/histo-miner/tree/master) / Models inference: nucleus segmentation and classification. 

It will then generate the prediction as a json file that you can use for the following steps.

### Build

Variable to update in the config:

```yaml
### all paths
wsi_input_folder: "/path/to/data/"
json_folder: "/path/to/data/"
pickle_output_folder: "/path/to/output/folder/"
```

Then inside `wsi-graph-formers`:

```bash
conda activate wsi-graph-formers
python dataset_tools/build_cell_graph.py
```

### Convert 

Fill `pickle_output_folder` and `conversion_output_folder` in the `config.yaml` file, then:

```bash
python dataset_tools/convert_graph.py
```

### Simplify

```bash
python dataset_tools/simplify_graph.py \
max_hops=3 \
conversion_output_folder=/path/to/conversion_output_folder/ \
simplifiedgraph_folder=/path/to/simplifiedgraph_folder/ 
```

### Split

```bash
python dataset_tools/split_graph.py nbr_subgraphs=100 simplifiedgraph_folder=/path/to/simplifiedgraph_folder/ split_output_folder=/path/to/split_output_folder/ 
```

## Visualize cell graphs

The visualization might have to be done locally, to avoid activating GUI on remote clusters.

**Visualize output of inference from binary classification**

```bash
conda activate wsi-graph-formers
python dataset_tools/visualize_cell_graph.py backend=pyg graphtype=binary visualization_path=/path/to/file/to/visualize.pt
```

**Visualize dataset graph with pyg**

```bash
conda activate wsi-graph-formers
python dataset_tools/visualize_cell_graph.py backend=pyg graphtype=dataset visualization_path=/path/to/file/to/visualize.pt
```

## Image Baseline

Information about the generation and training of image baseline datasets is available on `Baseline.md` file.

## References and citation

The model and training code is partially built upon [SGFormer repository](https://github.com/qitianwu/SGFormer) (NeurIPS 2023). The main updates performed are the cell graph datasets and loaders,  the main additions are the masked binary node classification task, subgraph mini-batch training, patient-aware cross-validation, and the switch from `argparse` to Hydra.

To cite our paper: 

```
@misc{sancéré2026contextawareskincancerepithelial,
      title={Context-aware Skin Cancer Epithelial Cell Classification with Scalable Graph Transformers}, 
      author={Lucas Sancéré and Noémie Moreau and Katarzyna Bozek},
      year={2026},
      eprint={2602.15783},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.15783}, 
}
```
If you use this code or the dataset link please also consider starring the repo to increase its visibility! Thanks 💫

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg