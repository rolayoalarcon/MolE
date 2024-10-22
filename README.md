# MolE: Molecular representations through redundancy reduction of Embeddings  

This repository contains the implementation of the MolE pre-training framework. We provide the scripts to pre-train, finetune, train ML models, and extract the static molecular representation learned after pre-training. 


## Pre-print:
For more details on MolE, you can check out our pre-print on bioarxiv:  

[**Pre-trained molecular representations enable antimicrobial discovery**](https://www.biorxiv.org/content/10.1101/2024.03.11.584456v2)

## Installation
To use MolE, it is convenient to create a conda environment and install the necessary dependencies. To make things easier, we provide an `environment.yaml` file, that you can use to set up the environment with all the necessary dependencies. Keep in mind, that we install [pytorch assuming a CUDA 11.8 compute platform](https://pytorch.org/).

```
# Create the conda environment with name mole
conda env create -f environment.yaml

# Afterwards activate the environment
conda activate mole
```

Once this is done, you can clone this repository.

## How to use MolE 
  
In this repository, we provide the necessary scripts to: 
 - [Pre-train](#pre-training) a molecular representation using the MolE framework.  
 - [Fine-tune](#finetuning) the pre-trained weights on a specific benchmark molecular property prediction task. 
 - [Train a Machine Learning model](#training-an-ml-model) (XGBoost or Random Forest) using the static pre-trained representation on a specific benchmark task.  
 - [Gather the static pre-trained representation](#gather-static-representation) for any set of molecules.  

In all instances, arguments can be specified in a `.yaml` file or as command line arguments. Whenever a `.yaml` is provided command line arguments are ignored. 


### Pre-trained MolE
To help you get started we provide a pre-trained model [here](https://zenodo.org/records/10803099?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI3NTg0OTU0LTI5YWItNDgxZS04OGYyLTU5MmM1MjcwYzJjZiIsImRhdGEiOnt9LCJyYW5kb20iOiIzNzgyNTE5ZGU5N2MzZWI3YjZiZjkwYTIzZjFiMmEwZSJ9.oL6G0WZKxIowSb-2qdP55cPhef1W4yG5iF4PFlsWPpuPROmzRhutJtySzs9q02ACltl0qy9YPJjzB7NvzRMyaw). 
You can download the `model.pth` file and place it in the `ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints` subdirectory. 
This model was used for our task of antimicrobial prediction (see [mole_antimicrobial_potential](https://github.com/rolayoalarcon/mole_antimicrobial_potential)).

## Pre-training
To pre-train the MolE framework you can use `pretrain.py` script. This script receives the hyperparamters to pre-train molecular representation using the MolE framework. Hyperparameters can be given either in `.yaml` file or as command line arguments. If a `.yaml` file is provided, command line arguments are ignored. The program will return a model.pth file with the pre-trained weights and config.yaml file with the indicated hyperparameters.
  
A description of the options is avaible.

```
$ python pretrain.py -h
usage: python pretrain.py -y config.yaml | python pretrain.py [options]

This program receives the hyperparamters to train a pre-trained molecular representation using the MolE framework. Hyperparameters can be given either in yaml file or as command line arguments. If a yaml file is
provided, command line arguments are ignored. The program will return a model.pth file with the pre-trained weights and config.yaml file with the indicated hyperparameters.

optional arguments:
  -h, --help            show this help message and exit
  --config_yaml CONFIG_YAML
                        Complete path to yaml file that contains the parameters for model pre-training (default: None)

Pre-training parameters:
  Parameters for model pre-training.

  --batch_size BATCH_SIZE
                        Number of compounds in each batch during pre-training. (default: 1000)
  --warm_up WARM_UP     Number of warm up epochs before cosine annealing begins. (default: 10)
  --epochs EPOCHS       Total number of epochs to pre-train. (default: 1000)
  --eval_every_n_epochs EVAL_EVERY_N_EPOCHS
                        Validation frequency. (default: 1)
  --save_every_n_epochs SAVE_EVERY_N_EPOCHS
                        Automatically save model every n epochs. (default: 5)
  --init_lr INIT_LR     Inital learning rate for the ADAM optimizer. (default: 0.0005)
  --weight_decay WEIGHT_DECAY
                        Weight decay for ADAM. (default: 1e-05)
  --gpu GPU             Name of the cuda device to run pre-training. Can also be 'cpu'. (default: cuda:0)
  --model_type {gin_concat,gin_noconcat,gcn_concat,gcn_noconcat}
                        Pre-training architechture consisting of GNN backbone (gin or gcn) and representation building (concat or noconcat). (default: gin_concat)
  --load_model LOAD_MODEL
                        Name of model in ckpt to resume pre-training. (default: None)

Model parameters:
  Parameters for the GNN model backbone.

  --num_layer NUM_LAYER
                        Number of GNN layers. (default: 5)
  --emb_dim EMB_DIM     Dimensionality of the graph representation. If representation building is '*_concat', then the graph representation of each GNN layer is emb_dim dimensional and the final molecular
                        representation (r) will have dimension = num_layer * emb_dim. Else representation will be emb_dim dimensional. (default: 200)
  --feat_dim FEAT_DIM   Dimensionality of the embedding vector (z) that is the input to the barlow-twins loss. (default: 8000)
  --drop_ratio DROP_RATIO
                        Dropout ratio (default: 0.0)
  --pool {max,add,avg}  Readout pooling function to great graph-layer representations. (default: add)

Dataset parameters:
  Parameters for the dataset of unlabled molecules.

  --num_workers NUM_WORKERS
                        Dataloader number of workers. (default: 100)
  --valid_size VALID_SIZE
                        Ratio of the total of molecules provided that are used as validation set. (default: 0.1)
  --data_path DATA_PATH
                        Path to pre-training dataset of unlabeled molecules. (default: data/pubchem_data/pubchem_100k_random.txt)
  --lambda_val LAMBDA_VAL
                        Value of lambda value for BT loss (default: 0.0001)

```
  
For convinience, all arguments have default values. Therefore, one can start pre-training a model by running the following command.  

```
# Pre-train with default arguments.
$ python pretrain.py
```
 
If you wish to change the value of one or more paramters, this can be specified:  
  
```
# Change the number of epochs we pre-train.
$ python pretrain.py --epochs 50
```
  
Alternatively, if the arguments are specified in a `.yaml` file, this can be passed to the `--config_yaml` argument. If a config file is specified then all command line arguments are ignored. An example of the expected input can be see in `config.yaml`

```
# Pass arguments as config file
$ python pretrain.py --config_file config.yaml
```

The pre-trained models will be saved in subdirectories of `ckpt`, with the corresponding `.ckpt` binary model file, as well as a record of the parameters used in a `config.yaml` file. 
We provide a pre-trained model [here](https://zenodo.org/records/10803099?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI3NTg0OTU0LTI5YWItNDgxZS04OGYyLTU5MmM1MjcwYzJjZiIsImRhdGEiOnt9LCJyYW5kb20iOiIzNzgyNTE5ZGU5N2MzZWI3YjZiZjkwYTIzZjFiMmEwZSJ9.oL6G0WZKxIowSb-2qdP55cPhef1W4yG5iF4PFlsWPpuPROmzRhutJtySzs9q02ACltl0qy9YPJjzB7NvzRMyaw). You can download the `model.pth` file and place it in the `ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints` subdirectory. This model was used for our task of antimicrobial prediction (see [mole_antimicrobial_potential](https://github.com/rolayoalarcon/mole_antimicrobial_potential)).


## Finetuning  
To fine-tune a pre-trained representation to a specific task one can use the `finetune.py` script.
We provide a pre-trained model that you can fine-tune [here](https://zenodo.org/records/10803099?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI3NTg0OTU0LTI5YWItNDgxZS04OGYyLTU5MmM1MjcwYzJjZiIsImRhdGEiOnt9LCJyYW5kb20iOiIzNzgyNTE5ZGU5N2MzZWI3YjZiZjkwYTIzZjFiMmEwZSJ9.oL6G0WZKxIowSb-2qdP55cPhef1W4yG5iF4PFlsWPpuPROmzRhutJtySzs9q02ACltl0qy9YPJjzB7NvzRMyaw). You can download the `model.pth` file and place it in the `ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints` subdirectory.

```
$ python finetune.py -h

usage: python finetune.py -y config_finetune.yaml | python finetune.py [options]

FINE-TUNE A PRE-TRAINED MODEL. This program receives the hyperparamters to fine-tune a pre-trained molecular representation for a given benchmark prediction task. Hyperparameters can be given either in yaml file
or as command line arguments. If a yaml file is provided, command line arguments are ignored. Several values can be given to some hyperparamters to perform a random search. The program will return a model.pth
file with the fine-tuned weights and config.yaml file with the indicated hyperparameters. Addionally, it will return a file with the performance metrics.

optional arguments:
  -h, --help            show this help message and exit
  --config_yaml CONFIG_YAML
                        Complete path to yaml file that contains the parameters for model pre-training (default: None)

Training parameters:
  Parameters for model finetuning.

  --batch_size [BATCH_SIZE [BATCH_SIZE ...]]
                        Possible number of compounds in each batch during pre-training. Several options can be given for random search. (default: [32, 101, 512, 800])
  --epochs EPOCHS       Total number of epochs to pre-train. (default: 1000)
  --eval_every_n_epochs EVAL_EVERY_N_EPOCHS
                        Validation frequency. (default: 1)
  --log_every_n_steps LOG_EVERY_N_STEPS
                        Print training log frequency. (default: 50)
  --fine_tune_from FINE_TUNE_FROM
                        Name of the pre-trained weights that should be fine-tuned. Should be a sub-directory of ./ckpt. (default: gin_concat_R1000_E8000_lambda0.0001)
  --init_lr [INIT_LR [INIT_LR ...]]
                        Possible inital learning rate for the PREDICTION head. Several can be given for random search. (default: [0.0005, 0.001])
  --init_base_lr [INIT_BASE_LR [INIT_BASE_LR ...]]
                        Possible initial learning rate for the PRE-TRAINED GNN encoder. Several can be given for random search. (default: [5e-05, 0.0001, 0.0002, 0.0005])
  --weight_decay WEIGHT_DECAY
                        Weight decay for ADAM. (default: 1e-05)
  --gpu GPU             Name of the cuda device to run pre-training. Can also be 'cpu'. (default: cuda:0)
  --task_name {BBBP,BACE,ClinTox,Tox21,HIV,SIDER,FreeSolv,ESOL,Lipo,qm7,qm8}
                        Name of benchmark tasks for fine-tuning. (default: BBBP)

Prediction head:
  Parameters for the prediction head.

  --drop_ratio [DROP_RATIO [DROP_RATIO ...]]
                        Possible dropout rate. Several can be given for random search. (default: [0, 0.1, 0.3, 0.5])
  --pred_n_layer [PRED_N_LAYER [PRED_N_LAYER ...]]
                        Possible number of layers on prediction head. Several can be given for random search. (default: [1, 2])
  --pred_act [PRED_ACT [PRED_ACT ...]]
                        Possible activation functions on the prediction head. At the moment, only softplus and relu are supported. (default: ['softplus', 'relu'])

Dataset:
  Arguments for how to handle benchmark task.

  --num_workers NUM_WORKERS
                        Dataloader number of workers. (default: 10)
  --valid_size VALID_SIZE
                        Fraction of molecules to use for validation. (default: 0.1)
  --test_size TEST_SIZE
                        Fraction of molecules to use for testing. (default: 0.1)
  --splitting {random,scaffold}
                        Method to split the dataset into training, validation, and testing sets. (default: scaffold)

Random search:
  Arguments for random search.

  --n_models N_MODELS   Number of model configurations to train. (default: 5)
  --n_trains N_TRAINS   Number of training iterations for each model configuration (default: 3)

Output:
  Arguments for output directories.

  --model_outdir MODEL_OUTDIR
                        Directory where finetuned models are written (default: finetune)
  --metrics_outdir METRICS_OUTDIR
                        Directory where performance metrics are written (default: output_metrics)

```

As with our pre-training script, all arguments have default values. Therefore, one can fine-tune a model by running the following command.  

```
# Pre-train with default arguments.
$ python finetune.py
```
 
If you wish to change the value of one or more parameters, this can be specified. Furthermore, some arguments accept several possible values. These values are then considered during a random search.  
  
```
# Give the possible values for batch size to choose from during random search. Will be 30, 50 or 100
$ python pretrain.py --batch_size 30 50 100
```
  
Alternatively, if the arguments are specified in a `.yaml` file, this can be passed to the `--config_yaml` argument. If a config file is specified then all command line arguments are ignored. An example of the expected input can be see in `config_finetune.yaml`

```
# Pass arguments as config file
$ python finetune.py --config_file config_experiment.yaml
```

The hyperparameters for fine-tuning can be found and modified in `config_finetune.yaml`. The finetuned model is output to the directory passed to `--model_outdir`. Performance metrics are written to the directory passed to `--metrics_outdir`.  


## Training an ML model
After pre-training, one can use the static representation as input molecular features to a more accessible machine-learning model. For this, we've made available the `train_ml.py` script.   
We provide a pre-trained model whose features can be extracted [here](https://zenodo.org/records/10803099?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI3NTg0OTU0LTI5YWItNDgxZS04OGYyLTU5MmM1MjcwYzJjZiIsImRhdGEiOnt9LCJyYW5kb20iOiIzNzgyNTE5ZGU5N2MzZWI3YjZiZjkwYTIzZjFiMmEwZSJ9.oL6G0WZKxIowSb-2qdP55cPhef1W4yG5iF4PFlsWPpuPROmzRhutJtySzs9q02ACltl0qy9YPJjzB7NvzRMyaw). You can download the `model.pth` file and place it in the `ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints` subdirectory.

```
$ python train_ml.py -h 
  
usage: python train_ml.py -y config_ml.yaml | python train_ml.py [options]

TRAIN ML MODEL WITH PRE-TRAINED MOLECULAR REPRESENTATIONS. This program receives the hyperparamters to train a machine learning model on a specific benchmark task. Hyperparameters can be given either in yaml file
or as command line arguments. The program will return a TSV file with the performance metrics of the experiments.

optional arguments:
  -h, --help            show this help message and exit
  --config_yaml CONFIG_YAML
                        Complete path to yaml file that contains the parameters for model training. (default: None)
  --metrics_outdir METRICS_OUTDIR
                        Directory to save the experiment performance metrics. (default: output_metrics)

Training parameters:
  Parameters for training the machine learning model.

  --task_name {BBBP,BACE,ClinTox,Tox21,HIV,SIDER,FreeSolv,ESOL,Lipo,qm7,qm8}
                        Name of the benchmark task to train the model on. (default: BBBP)
  --ml_model {RandomForest,XGBoost}
                        Type of machine learning model to train. Can be either RandomForest or XGBoost. Depending the benchmark task, a classifier or a regressor model is trained. Depending on the model,
                        different hyperparameters are explored during random search (see models/ml_params.py). (default: XGBoost)
  --pretrained_model PRETRAINED_MODEL
                        Name of the pre-trained model to use for the molecular representation. Should be a sub-directory of ./ckpt. Alternatively can also be 'MolCLR' or 'ECFP4'. (default:
                        gin_concat_R1000_E8000_lambda0.0001)
  --gpu GPU             Name of the cuda device to run the training. Can also be 'cpu'. (default: cuda:0)

Dataset parameters:
  Parameters for the dataset.

  --splitting {random,scaffold}
                        Type of splitting to use to build training, validation and testing sets. (default: scaffold)
  --validation_size VALIDATION_SIZE
                        Proportion of the dataset to include in the validation set. (default: 0.1)
  --test_size TEST_SIZE
                        Proportion of the dataset to include in the test set. (default: 0.1)

Random search parameters:
  Parameters for the random search of hyperparameters.

  --n_models N_MODELS   Number of models configurations to train for each target. (default: 5)
  --n_trains N_TRAINS   Number of training iterations for each model configuration. (default: 3)
```

As before, all arguments have default values. Therefore, one can start training a model by running the following command.  

```
# Pre-train with default arguments.
$ python train_ml.py
```
 
If you wish to change the value of one or more parameters, this can be specified.  
  
```
# Change the benchmark task we want to train
$ python train_ml.py --task_name ClinTox
```
  
Alternatively, if the arguments are specified in a `.yaml` file, this can be passed to the `--config_yaml` argument. If a config file is specified then all command line arguments are ignored. An example of the expected input can be see in `config_ml.yaml`
```
# Pass arguments as config file
$ python train_ml.py --config_file config_ml.yaml
```
  
The hyperparamters over which a random search is done can be found in `models/ml_params.py`. The program will return a file with the performance metrics of the experiments in the directory specified in `--metrics_outdir`.
  
## Gather static representation.  
  
It is also  possible to obtain the static representation for a collection of SMILES using the `gather_representation.py` script. You can obtain the pre-trained representation for any pre-trained MolE model in `.ckpt` as well as the ECFP4 representation.  
  
We provide a pre-trained model whose features can be extracted [here](https://zenodo.org/records/10803099?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImI3NTg0OTU0LTI5YWItNDgxZS04OGYyLTU5MmM1MjcwYzJjZiIsImRhdGEiOnt9LCJyYW5kb20iOiIzNzgyNTE5ZGU5N2MzZWI3YjZiZjkwYTIzZjFiMmEwZSJ9.oL6G0WZKxIowSb-2qdP55cPhef1W4yG5iF4PFlsWPpuPROmzRhutJtySzs9q02ACltl0qy9YPJjzB7NvzRMyaw). You can download the `model.pth` file and place it in the `ckpt/gin_concat_R1000_E8000_lambda0.0001/checkpoints` subdirectory.

```
$ python gather_representation.py -h

usage: python gather_representation.py --config_file config_representation.yaml | python gather_representation.py [options]

From a file with SMILES, gather the indicated molecular representation

optional arguments:
  -h, --help            show this help message and exit
  --config_yaml CONFIG_YAML
                        Complete path to yaml file that contains the parameters for model pre-training (default: None)
  --gpu GPU             Name of the cuda device to run pre-training. Can also be 'cpu'. (default: cuda:0)
  --output_filepath OUTPUT_FILEPATH
                        Complete path to the file where the molecular representation will be saved. Outputs a tsv file. (default:
                        representation_dir/output_representation.tsv.gz)

Input arguments:
  Arguments about the input data.

  --smiles_filepath SMILES_FILEPATH
                        Complete path to the file with the SMILES of the molecules. Must be a .csv file. (default: ./data/benchmark_data/bbbp/BBBP.csv)
  --smiles_colname SMILES_COLNAME
                        Name of the column in --smiles_filepath that contains SMILES of the molecules. (default: smiles)
  --chemid_colname CHEMID_COLNAME
                        Name of the column in --smiles_filepath that contains the identifier of the molecules. If None, the index of the dataframe will be used. (default: None)

Representation arguments:
  Arguments about the molecular representation.

  --representation REPRESENTATION
                        Type of molecular representation to be gathered. Can be 'MolCLR', 'ECFP4', or one of the pre-trained models in ./ckpt (only subfolder name is necessary).
                        (default: gin_concat_R1000_E8000_lambda0.0001)
```
  
As before, all arguments have default values. Meaning you can gather representations with:
  
```
$ python gather_representation.py
```

If you wish to change the value of one or more parameters, this can be specified.  

```
# Gather the ECFP4 representation
python gather_representation.py --representation ECFP4
```

Alternatively, if the arguments are specified in a `.yaml` file, this can be passed to the `--config_yaml` argument. If a config file is specified then all command line arguments are ignored. An example of the expected input can be see in `config_representation.yaml`
```
# Pass arguments as config file
$ python gather_representation.py --config_file config_representation.yaml
```

The output is a `.tsv` file with each row being a molecule.


## Datasets  
Benchmark datasets are provided by the authors of [MolCLR](https://github.com/yuyangw/MolCLR), and can also be collected from [MoleculeNet](https://moleculenet.org/). You can download the provided zip file ([here](https://drive.google.com/file/d/1aDtN6Qqddwwn2x612kWz9g0xQcuAtzDE/view)) and extract it in the `data/benchmark_data` directory. 

In the same link, one can find the original set of 10 million compounds from PubChem. We sample 100,000 molecules from this set and can be found in `data/pubchem_data` directory. 


# Acknowledgements  

A big acknowledgment to the authors of [MolCLR](https://github.com/yuyangw/MolCLR) for providing publicly available code.
  
