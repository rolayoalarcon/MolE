# MolE: Molecular representations through redundancy reduction of Embeddings  

This repository contains the implementation of the MolE pre-training framework. We provide the scripts to pre-train, finetune, and extract the static representation learned after pre-training. 


## Pre-print:
For more details on MolE, you can check out our pre-print on bioarxiv:
[**Pre-trained molecular representations enable antimicrobial discovery**](https://www.biorxiv.org/content/10.1101/2024.03.11.584456v1)

## Installation
To use MolE, it is convenient to create a conda environment and install the necessary dependencies.  

```
# Create the conda environment
conda create --name mole python=3.8
conda activate mole

# Install pytorch and pytorch geometric
pip install torch==2.2.1 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric==2.5.0

# Install rdkit
pip install rdkit==2022.3.3

# Other dependencies
pip install pandas==2.0.3 xgboost==1.6.2
```

Once this is done, you can clone this repository.

## Pre-training
To pre-train the MolE framework you can run `pretrain.py`

```
python pretrain.py
```

These hyperparameters for pre-training can be found and modified at `config.yaml`. There, you will also find a more detailed explanation of each parameter and how to modify it. Keep in mind that pre-training can take several hours.  

The pre-trained models will be saved in subdirectories of `ckpt`. We provide a pre-trained model [here](https://zenodo.org/records/10803099?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI3NTg0OTU0LTI5YWItNDgxZS04OGYyLTU5MmM1MjcwYzJjZiIsImRhdGEiOnt9LCJyYW5kb20iOiIzNzgyNTE5ZGU5N2MzZWI3YjZiZjkwYTIzZjFiMmEwZSJ9.oL6G0WZKxIowSb-2qdP55cPhef1W4yG5iF4PFlsWPpuPROmzRhutJtySzs9q02ACltl0qy9YPJjzB7NvzRMyaw). You can download the `model.pth` file and place it in the `ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints` subdirectory.   


## Finetuning  
To fine-tune a pre-trained representation to a specific task one can run `finetune.py`

```
python finetune.py
```

The hyperparameters for fine-tuning can be found and modified in `config_finetune.yaml`. The output will be a subdirectory within the `ckpt` directory. 

## Static-representation  
After pre-training, one can use the static representation as input molecular features to a more accessible machine-learning model such as XGBoost. For this, we've made available the `representation.py` script.   
```
python representation.py
```
Hyperparameters can be found and modified in `config_representation.yaml`.  

It is also  possible to obtain the static representation for a collection of SMILES using the `process_dataset()` function described in `dataset/dataset_representation.py`. An example of how this can be done is shown in `examine_representation.ipynb`

```
bbbp_split, bbbp_mole_2k = process_dataset("./data/benchmark_data/bbbp/BBBP.csv",
                                                         "gin_concat",
                                                         PRETRAINED_MODEL,
                                                         "scaffold",
                                                         0.1,
                                                         0.1)
```

Here, the `PRETRAINED_MODEL` model should be substituted by a model present in the `ckpt` directory. For example, you can write `gin_concat_R1000_E8000_lambda0.0001` to use the provided pre-trained model.  

## Datasets  
Benchmark datasets are provided by the authors of [MolCLR](https://github.com/yuyangw/MolCLR), and can also be collected from [MoleculeNet](https://moleculenet.org/). You can download the provided zip file ([here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view)) and extract it in the `data/benchmark_data` directory. 

In the same link, one can find the original set of 10 million compounds from PubChem. We sample 100,000 molecules from this set and can be found in `data/pubchem_data` directory. 


# Acknowledgements  

A big acknowledgment to the authors of [MolCLR](https://github.com/yuyangw/MolCLR) for providing publicly available code.
  
