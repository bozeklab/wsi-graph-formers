
# Image Baseline datasets information 

## Generate baseline datasets

### Requirement 

Download the baseline WSI and annotation from the public dataset (see README.md). Then update config file as it will be detailed in following sections.

### CellVit format for baseline 

**This dataset is already available in the public dataset**. There is no need to reproduce it following the explanation below. Nevertheless, to generate it, here are the commands:

- Update `config.yaml`, more precisely `baselinedata_dir` variable and `wsijson_to_trainset.py` section. You don't need to change by default parameters.
- Run:

```bash
cd /path/to/wsi-graph-formers/
conda activate wsi-graph-formers
python baseline_tools/json_to_trainset.py
```

- Update `config.yaml`, more precisely `concat binaryformat ` section. You don't need to change by default parameters.
- Run:

```bash
python baseline_tools/concat_binaryformat.py 
```

- Then, use the cellvit repository and copy paste the file `prepare_onegraphskinwsi_binary.py ` in `/CellViT/cell_segmentation/datasets/`

- Then run:

```bash
cd ~/CellViT/cell_segmentation/datasets
conda activate cellvit_env
python prepare_onegraphskinwsi_binary.py  --input_path /path/to/concat_binaryformat.py/output/ --output_path /path/to/new-output/ --input_format onehot --class1_channel 4 --class2_channel 5
```

- Coming back to _wsi-graph-formers_ repository,  run create_types_csv function (edit `baselinedata_dir` of  `config.yaml` if necessary ):

```bash
cd /path/to/wsi-graph-formers/
conda activate wsi-graph-formers
python baseline_tools/cellvit/generate_typescsv.py
```

- Copy `baseline_tools/cellvit/dataset_config.yaml` and  `baseline_tools/cellvit/weight_config.yaml` to the folder where the baseline dataset was generated.

### Hovernet format for baseline

To generate this format here are the commands:

- Update `config.yaml`, more precisely `baselinedata_dir` variable and `wsijson_to_trainset.py` section. You don't need to change by default parameters.
- Run:

```bash
cd /path/to/wsi-graph-formers/
conda activate wsi-graph-formers
python baseline_tools/json_to_trainset.py
```

- Update `config.yaml`, more precisely `concat binaryformat ` section. You don't need to change by default parameters.
- Run:

```bash
python baseline_tools/concat_binaryformat.py 
```

- Update `config.yaml`, more precisely `generate hovernet format` section. You don't need to change by default parameters.
- Run:

```bash
python baseline_tools/hovernet/generate_hovernetbinary.py
```

## Train baseline datasets

To train the baseline datasets and evaluate with cross-validaton, refer to their corresponding repositories with the newly generated data:

- [CellViT](https://github.com/TIO-IKIM/CellViT)
- [Hovernet](https://github.com/vqdang/hover_net)