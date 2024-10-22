import os
import yaml
import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from dataset.dataset_representation import process_dataset




def get_args():

    parser = ArgumentParser(prog="Gather molecular representation",
                            description='From a file with SMILES, gather the indicated molecular representation', 
                            formatter_class=ArgumentDefaultsHelpFormatter,
                            
                            usage="python gather_representation.py --config_file config_representation.yaml | python gather_representation.py [options]")
    
    parser.add_argument("--config_yaml", dest="config_yaml",
                        help="Complete path to yaml file that contains the parameters for model pre-training")
    
    inputargs = parser.add_argument_group("Input arguments", "Arguments about the input data.")
    inputargs.add_argument("--smiles_filepath", dest="smiles_filepath",
                           type=str,
                           default="./data/benchmark_data/bbbp/BBBP.csv",
                           help="Complete path to the file with the SMILES of the molecules. Must be a .csv file.")
    
    inputargs.add_argument("--smiles_colname", dest="smiles_colname",
                           type=str,
                           default="smiles",
                           help="Name of the column in --smiles_filepath that contains SMILES of the molecules.")
    
    inputargs.add_argument("--chemid_colname", dest="chemid_colname",
                           type=str,
                           default=None,
                           help="Name of the column in --smiles_filepath that contains the identifier of the molecules. If None, the index of the dataframe will be used.")
    
    repargs = parser.add_argument_group("Representation arguments", "Arguments about the molecular representation.")
    repargs.add_argument("--representation", dest="representation",
                         type=str,
                         default="gin_concat_R1000_E8000_lambda0.0001",
                         help="Type of molecular representation to be gathered. Can be 'MolCLR', 'ECFP4', or one of the pre-trained models in ./ckpt (only subfolder name is necessary).")
    
    parser.add_argument("--gpu", 
                        dest="gpu",
                        type=str,
                        default="cuda:0",
                        help="Name of the cuda device to run pre-training. Can also be 'cpu'.")
    
    parser.add_argument("--output_filepath", dest="output_filepath",
                        type=str,
                        default="representation_dir/output_representation.tsv.gz",
                        help="Complete path to the file where the molecular representation will be saved. Outputs a tsv file.")
    
    args = parser.parse_args()

    return args


def gather_pretraining_config(pretrain_name):
    """
    This function returns a config dictionary that contains pre-training hyperparameters.
    """

    with open(os.path.join("ckpt", pretrain_name, "checkpoints", "config.yaml"), "r") as file:
        pretrain_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    # We are only interested in the model_type
    pretrain_config_dict = {key: value for key, value in pretrain_config_dict.items() if key in ["model_type"]}

    return pretrain_config_dict 

def gather_config():
    """
    This function returns a config dictionary to be used to gather the pre-trained representation.
    """

    args = get_args()

    if args.config_yaml != None:
        print(f"Reading pre-training arguments from {args.config_yaml}. Ignoring command line arguments.")
        with open(args.config_yaml, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

            if config_dict["chemid_colname"] == "None":
                config_dict["chemid_colname"] = None
    else:
        config_dict = vars(args)


    # If the representation is a pre-trained model, we need to gather the model_type
    if config_dict["representation"] not in ["MolCLR", "ECFP4"]:
        pretrain_config_dict = gather_pretraining_config(config_dict["representation"])
        config_dict["model_type"] = pretrain_config_dict["model_type"]

    else:
        config_dict["model_type"] = None

    return config_dict


def main():

    config = gather_config()
    print(config)


    # Gather the molecular representation requested
    mol_representation = process_dataset(dataset_path=config["smiles_filepath"],
                                         smiles_colnname=config["smiles_colname"],
                                         id_colname=config["chemid_colname"],
                                         
                                         # Parameters for splitting
                                         dataset_split=False,
                                         
                                         # Details about pre-trained representation
                                         pretrain_architecture=config["model_type"],
                                         pretrained_model=config["representation"])
    

    # Save the molecular representation
    mol_representation.to_csv(config["output_filepath"], sep="\t")

if __name__ == "__main__":
    main()

    
    