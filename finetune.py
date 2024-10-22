import os
import yaml
import random
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from finetune_object import finetune
from dataset.dataset_test import MolTestDatasetWrapper


def determine_dataset(config_dict):
    # Create copy of input
    out_dict = config_dict.copy()


    # Some preparation for specific tasks. Not elegant. But it's what we got. 
    if out_dict['task_name'] == 'BBBP':
        out_dict['dataset']['task'] = 'classification'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/bbbp/BBBP.csv'
        target_list = ["p_np"]

    elif out_dict['task_name'] == 'Tox21':
        out_dict['dataset']['task'] = 'classification'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/tox21/tox21.csv'
        target_list = [
            "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", 
            "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
        ]

    elif out_dict['task_name'] == 'ClinTox':
        out_dict['dataset']['task'] = 'classification'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/clintox/clintox.csv'
        target_list = ['CT_TOX', 'FDA_APPROVED']

    elif out_dict['task_name'] == 'HIV':
        out_dict['dataset']['task'] = 'classification'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/hiv/HIV.csv'
        target_list = ["HIV_active"]

    elif out_dict['task_name'] == 'BACE':
        out_dict['dataset']['task'] = 'classification'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/bace/bace.csv'
        target_list = ["Class"]

    elif out_dict['task_name'] == 'SIDER':
        out_dict['dataset']['task'] = 'classification'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/sider/sider.csv'
        target_list = [
            "Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", 
            "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", 
            "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", 
            "Reproductive system and breast disorders", 
            "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", 
            "General disorders and administration site conditions", "Endocrine disorders", 
            "Surgical and medical procedures", "Vascular disorders", 
            "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", 
            "Congenital, familial and genetic disorders", "Infections and infestations", 
            "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", 
            "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", 
            "Ear and labyrinth disorders", "Cardiac disorders", 
            "Nervous system disorders", "Injury, poisoning and procedural complications"
        ]
    
    elif out_dict['task_name'] == 'MUV':
        out_dict['dataset']['task'] = 'classification'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/muv/muv.csv'
        target_list = [
            'MUV-692', 'MUV-689', 'MUV-846', 'MUV-859', 'MUV-644', 'MUV-548', 'MUV-852',
            'MUV-600', 'MUV-810', 'MUV-712', 'MUV-737', 'MUV-858', 'MUV-713', 'MUV-733',
            'MUV-652', 'MUV-466', 'MUV-832'
        ]

    elif out_dict['task_name'] == 'FreeSolv':
        out_dict['dataset']['task'] = 'regression'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/freesolv/freesolv.csv'
        target_list = ["expt"]
    
    elif out_dict["task_name"] == 'ESOL':
        out_dict['dataset']['task'] = 'regression'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/esol/esol.csv'
        target_list = ["measured log solubility in mols per litre"]

    elif out_dict["task_name"] == 'Lipo':
        out_dict['dataset']['task'] = 'regression'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/lipophilicity/Lipophilicity.csv'
        target_list = ["exp"]
    
    elif out_dict["task_name"] == 'qm7':
        out_dict['dataset']['task'] = 'regression'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/qm7/qm7.csv'
        target_list = ["u0_atom"]

    elif out_dict["task_name"] == 'qm8':
        out_dict['dataset']['task'] = 'regression'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/qm8/qm8.csv'
        target_list = [
            "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", 
            "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"
        ]
    
    elif out_dict["task_name"] == 'qm9':
        out_dict['dataset']['task'] = 'regression'
        out_dict['dataset']['data_path'] = 'data/benchmark_data/qm9/qm9.csv'
        target_list = ['mu', 'alpha', 'homo', 'lumo', 'gap', 'r2', 'zpve', 'cv']

    else:
        raise ValueError('Undefined downstream task!')
    
    return out_dict, target_list




def select_params(original_config):
    """
    Receives a dictionary, where any values consisting of lists will be 
    reduced to a single value by random choice
    """
    model_config = original_config.copy()

    for key, value in model_config.items():

        if type(value) == list:
            model_config[key] = random.choice(value)
        elif type(value) == dict:
            model_config[key] = select_params(value)
        
    return model_config

def get_args():
    """
    Parses command line arguments for finetuning
    """
    parser=ArgumentParser(prog="Fine-tune a pre-trained model for a specific prediction task.",
                            description="FINE-TUNE A PRE-TRAINED MODEL. This program receives the hyperparamters to fine-tune a pre-trained molecular representation \
                                for a given benchmark prediction task. Hyperparameters can be given either in yaml file or as command line arguments. \
                                If a yaml file is provided, command line arguments are ignored. Several values can be given to some hyperparamters to perform a random search.  \
                            The program will return a model.pth file with the fine-tuned weights and config.yaml file with the indicated hyperparameters. Addionally, it will return a file with the performance metrics.",
                            usage="python finetune.py -y config_finetune.yaml | python finetune.py [options]",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    # YAML config
    parser.add_argument("--config_yaml", dest="config_yaml",
                        help="Complete path to yaml file that contains the parameters for model pre-training")

    # Training arguments
    trainargs = parser.add_argument_group("Training parameters", "Parameters for model finetuning.")
    trainargs.add_argument("--batch_size", dest="batch_size", nargs="*",
                        default=[32, 101, 512, 800],
                        help="Possible number of compounds in each batch during pre-training. Several options can be given for random search.")
    
    trainargs.add_argument("--epochs", dest="epochs",
                        default=1000, type=int,
                        help="Total number of epochs to pre-train.")
    
    trainargs.add_argument("--eval_every_n_epochs", dest="eval_every_n_epochs",
                        default=1, type=int,
                        help="Validation frequency.")
    
    trainargs.add_argument("--log_every_n_steps", dest="log_every_n_steps",
                        default=50, type=int,
                        help="Print training log frequency.")
    
    trainargs.add_argument("--fine_tune_from", dest="fine_tune_from",
                        default="gin_concat_R1000_E8000_lambda0.0001",
                        help="Name of the pre-trained weights that should be fine-tuned. Should be a sub-directory of ./ckpt.")
    
    trainargs.add_argument("--init_lr", 
                        nargs="*",
                        dest="init_lr",
                        type=float,
                        default=[0.0005, 0.001], 
                        help="Possible inital learning rate for the PREDICTION head. Several can be given for random search.")
    
    trainargs.add_argument("--init_base_lr", 
                        nargs="*",
                        dest="init_base_lr",
                        type=float,
                        default=[0.00005, 0.0001, 0.0002, 0.0005],
                        help="Possible initial learning rate for the PRE-TRAINED GNN encoder. Several can be given for random search.")
    
    trainargs.add_argument("--weight_decay", 
                        dest="weight_decay",
                        type=float,
                        default=0.00001,
                        help="Weight decay for ADAM.")
    
    trainargs.add_argument("--gpu", 
                        dest="gpu",
                        type=str,
                        default="cuda:0",
                        help="Name of the cuda device to run pre-training. Can also be 'cpu'.")

    trainargs.add_argument("--task_name", dest="task_name", 
                        type=str,
                        default="BBBP",
                        choices=["BBBP", "BACE", "ClinTox", "Tox21", "HIV", "SIDER",
                                 "FreeSolv", "ESOL", "Lipo", "qm7", "qm8"],
                        help="Name of benchmark tasks for fine-tuning.")

    # Prediction head parameters
    modelargs = parser.add_argument_group("Prediction head", "Parameters for the prediction head.")
    modelargs.add_argument("--drop_ratio", 
                        dest="drop_ratio",
                        type=float,
                        nargs="*",
                        default=[0, 0.1, 0.3, 0.5],
                        help="Possible dropout rate. Several can be given for random search.")
    
    modelargs.add_argument("--pred_n_layer", dest="pred_n_layer",
                        type=int,
                        nargs="*",
                        default=[1,2],
                        help="Possible number of layers on prediction head. Several can be given for random search.")
    
    modelargs.add_argument("--pred_act", dest="pred_act",
                           type=str,
                           default=["softplus", "relu"],
                           nargs="*",
                           help="Possible activation functions on the prediction head. At the moment, only softplus and relu are supported.")
    
    dataargs = parser.add_argument_group("Dataset", "Arguments for how to handle benchmark task.")
    dataargs.add_argument("--num_workers",
                          type=int,
                          default=10,
                          dest="num_workers",
                          help="Dataloader number of workers.")
    
    dataargs.add_argument("--valid_size", dest="valid_size",
                          type=float,
                          default=0.1,
                          help="Fraction of molecules to use for validation.")
    
    dataargs.add_argument("--test_size", dest="test_size",
                            type=float,
                            default=0.1,
                            help="Fraction of molecules to use for testing.")
    
    dataargs.add_argument("--splitting", dest="splitting",
                          type=str,
                          default="scaffold",
                          choices=["random", "scaffold"],
                          help="Method to split the dataset into training, validation, and testing sets.")
    
    randargs = parser.add_argument_group("Random search", "Arguments for random search.")
    randargs.add_argument("--n_models", dest="n_models",
                          type=int,
                          default=5,
                          help="Number of model configurations to train.")
    randargs.add_argument("--n_trains", dest="n_trains",
                            type=int,
                            default=3,
                            help="Number of training iterations for each model configuration")


    oarrgs = parser.add_argument_group("Output", "Arguments for output directories.")
   
    oarrgs.add_argument("--model_outdir", dest="model_outdir", default="finetune",
                        help="Directory where finetuned models are written")
    
    oarrgs.add_argument("--metrics_outdir", dest="metrics_outdir", default="output_metrics",
                        help="Directory where performance metrics are written")
   
    args = parser.parse_args()

    return args


def gather_pretraining_config(pretrain_name):
    """
    This function returns a config dictionary that contains pre-training hyperparameters.
    """

    with open(os.path.join("ckpt", pretrain_name, "checkpoints", "config.yaml"), "r") as file:
        pretrain_config_dict = yaml.load(file, Loader=yaml.FullLoader)

    # We are only interested in the model and model_type keys
    pretrain_config_dict = {key: value for key, value in pretrain_config_dict.items() if key in ["model", "model_type"]}

    # Delete drop_ratio
    del pretrain_config_dict["model"]["drop_ratio"]
    
    return pretrain_config_dict


def gather_config():
    """
    This function returns a config dictionary to be used during fine-tuning.
    """

    # Parse command line arguments
    args = get_args()

    # If a config file is given, then parameters are read from there
    if args.config_yaml != None:
        print(f"Reading fine-tuning arguments from {args.config_yaml}. Ignoring command line arguments.")

        with open(args.config_yaml, "r") as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)

    else:        
        # Else we have to build a dictionary with the same config structure
        config_dict = vars(args)

        # Build model param dictionary
        config_subdicts = {"model": {k: config_dict[k] for k in ["drop_ratio", "pred_n_layer", "pred_act"]},
                           "dataset": {k: config_dict[k] for k in ["num_workers", "valid_size", "test_size", "splitting"]},
                           "random_search": {k: config_dict[k] for k in ["n_models", "n_trains"]}}
        
        # Update config_dict
        for key in ["drop_ratio", "pred_n_layer", "pred_act", "num_workers", "valid_size", "test_size", "splitting", "n_models", "n_trains"]:
            del config_dict[key]
        
        config_dict.update(config_subdicts)
    
    # Read pre-training config
    pretrain_config = gather_pretraining_config(config_dict["fine_tune_from"])

    # Add model_type and model to config_dict
    config_dict["model_type"] = pretrain_config["model_type"]
    config_dict["model"].update(pretrain_config["model"]) 

    return config_dict


def determine_outdir(label_name, m_name, t_name, m_type):
    base_model_str = "{model_type}_{label_name}_{model_name}_{training_name}_{datetime_str}"

    directory_name = base_model_str.format(model_type=m_type,
                                           label_name = label_name,
                                           model_name = m_name,
                                           training_name = t_name,
                                           datetime_str=datetime.now().strftime("%d.%b.%Y_%H.%M.%S"))
    
    return directory_name


def main():

    """
    Main function for conducting a random search and fine-tuning multiple models on a dataset.

    Reads command-line arguments, reads the configuration file, modifies it if necessary,
    gathers parameters for random search, determines dataset configuration, and iterates over
    targets, models, and training iterations to conduct fine-tuning. Results are collected
    and stored in a dataframe, then written to an output file.

    Returns:
        None
    """

    # Read the arguments
    config_originial = gather_config()

    # Gather the number of models and the number of training iterations for each model
    rsearch_params = config_originial.pop("random_search")

    # Pop the model and experiment output directory
    model_odir = config_originial.pop("model_outdir")
    experiment_odir = config_originial.pop("metrics_outdir")

    # Determine the dataset configuration
    task_config, label_list = determine_dataset(config_originial)

    # Iterate over targets
    results_list = []
    for target in label_list:
        # Iterate over models
        for m in range(rsearch_params["n_models"]):
            model_name = f"model_{m}"
            model_config = select_params(task_config)
            # Iterate over train

            for t in range(rsearch_params["n_trains"]):
                training_name = f"train_{t}"

                model_config["dataset"]["target"] = target
                model_config["model"]["task"] = model_config["dataset"]["task"]

                model_directory = determine_outdir(target, model_name, training_name, model_config["model_type"])

                result_dict = finetune(model_config, model_directory, model_odir)

                for key, value in result_dict.items():
                    results_list.append([target, key, value, model_name, training_name, model_directory])

    # Create a dataframe
    df = pd.DataFrame(results_list)

    # Write the output
    os.makedirs(experiment_odir, exist_ok=True)

    outfile = f'{config_originial["task_name"]}_{config_originial["fine_tune_from"]}.tsv.gz'


    df.to_csv(os.path.join(experiment_odir, outfile), mode='a', index=False, header=False, sep='\t')

if __name__ == "__main__":
    main()