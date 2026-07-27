# Skin Cancer Epithelial Cell Classification with Scalable Graph Transformers

<div align="center">
[Project presentation](#todo) • [Project structure](#project-structure) •  [Installation](#installation) •  [Datasets](#datasets) •   [Node classification](#node-classification) • [Generate cell graphs](#generate-cell-graphs) • [Visualize cell graphs](#visualize-cell-graphs)  • [Image Baseline](#image-baseline)  • [Citation](#citation) 
</div>
## TO DO

NOT TO KEEP

- once the installation works: remove setup.py, env_starting.yaml
- once the installation works:seee if we rename and keep reauirements_starting.yaml or if we simply remove it 

## Project presentation 

To write 

Add the fact it is highly inspired from SGFormer repository!


## Project sructure

To write 


## Installation

To write  Lucas


## Datasets

The datasets used in this work are publicly available on Zenodo following this link:

* [All datasets]  (link from zenodo)  [![DOI] (badgefromzenodo)] (link from zenodo) 

## Node classification

To write Lucas 


## Generate cell graphs

<p align="center">
  <img src="docs/generate-cell-graphs.png" width="650">
</p>

### Requirements

Need to run histo-miner before on the WSI / patch you want to predict from.  Either run SCC Segmenter if no tumor annotation exists or create your own annotations. 

### Build

Variable to update in the config:

```yaml
### all paths
wsi_input_folder: "/path/to/data/"

json_folder: "/path/to/data/"

pickle_output_folder: "/path/to/data/"
```

Then:

```bash
cd /path/to/wsi-graph-formers/
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
python dataset_tools/simplify_graph.py max_hops=3 conversion_output_folder=/path/to/conversion_output_folder/ simplifiedgraph_folder=/path/to/simplifiedgraph_folder/ 
```

### Split

```bash
python dataset_tools/split_graph.py  nbr_subgraphs=100 	simplifiedgraph_folder=/path/to/simplifiedgraph_folder/ 	split_output_folder=/path/to/split_output_folder/ 
```

## Visualize cell graphs

The visualization might have to be done locally, to avoid activating GUI on remote clusters if you work on cluster.


**Visualize output of inference from binary classification**

```bash
cd /path/to/wsi-graph-formers/
conda activate wsi-graph-formers
python dataset_tools/visualize_cell_graph.py backend=pyg graphtype=binary visualization_path=/path/to/file/to/visualize.pt
```

**Visualize dataset graph with pyg**

```bash
cd /path/to/wsi-graph-formers/
conda activate wsi-graph-formers
python dataset_tools/visualize_cell_graph.py backend=pyg graphtype=dataset visualization_path=/path/to/file/to/visualize.pt
```

## Image Baseline

Information about the generation and training of image baseline datasets is available on `Baseline.md` file.


## Citation

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


Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg