import os 
import random
import yaml
import pandas as pd
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from dataset.dataset_representation import process_dataset
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from models.ml_params import TASK_LABELS, TASK_DATAPATH, RF_CLASSIFIER_PARAMS, RF_REGRESSOR_PARAMS, XGB_REGRESSOR_PARAMS, XGB_CLASSIFIER_PARAMS

def determine_dataset(config_dict, task_target_dict=TASK_LABELS, task_path_dict = TASK_DATAPATH):

    target_list = task_target_dict[config_dict["task_name"]]
    task_datapath = task_path_dict[config_dict["task_name"]]

    return task_datapath, target_list

def determine_predictor_hyper(config_dict):
    """
    This function returns the hyperparameters for the model to be used during training depending on the task. Also returns task type: regression or classification.
    """

    if config_dict["task_name"] in ["ClinTox", "BBBP", "BACE", "Tox21", "HIV", "SIDER"]:
        if config_dict["ml_model"] == "RandomForest":
            return RF_CLASSIFIER_PARAMS, "classification"
        
        elif config_dict["ml_model"] == "XGBoost":
            return XGB_CLASSIFIER_PARAMS, "classification"
    
    elif config_dict["task_name"] in ["FreeSolv", "ESOL", "Lipo", "qm8", "qm7"]:
        if config_dict["ml_model"] == "RandomForest":
            return RF_REGRESSOR_PARAMS, "regression"
        
        elif config_dict["ml_model"] == "XGBoost":
            return XGB_REGRESSOR_PARAMS, "regression"
    
def select_params(original_config, dname, model_name, task_type):
    """
    Receives a dictionary, where any values consisting of lists will be 
    reduced to a single value by random choice
    """
    model_config = original_config.copy()

    # Make sure that if we are doing regression, we use the correct eval metric during XGB
    if task_type == "regression" and dname in ["qm7", "qm8", "qm9"] and model_name == "XGBoost":
        model_config["objective"] = "reg:squarederror"
        model_config["eval_metric"] = "mae"

    elif task_type == "regression" and dname not in ["qm7", "qm8", "qm9"] and model_name == "XGBoost":
        model_config["objective"] = "reg:squarederror"
        model_config["eval_metric"] = "rmse"
    

    for key, value in model_config.items():

        if type(value) == list:
            model_config[key] = random.choice(value)
        elif type(value) == dict:
            model_config[key] = select_params(value)
        
    return model_config

def get_predictor(model_params, model_type, task_type):

    """Initialize models with the chosen hyperparameters configuration"""

    if task_type == "classification":
        if model_type == "RandomForest":
            model_with_hyperparam = RandomForestClassifier(**model_params)
        
        elif model_type == "XGBoost":
            model_params["seed"] = np.random.randint(1_000_000, size=1)[0]
            model_with_hyperparam = XGBClassifier(**model_params)
    
    elif task_type == "regression":
        if model_type == "RandomForest":
            model_with_hyperparam = RandomForestRegressor(**model_params)
        
        elif model_type == "XGBoost":
            model_params["seed"] = np.random.randint(1_000_000, size=1)[0]
            model_with_hyperparam = XGBRegressor(**model_params)

    return model_with_hyperparam

def evaluate_model(fitted_model, data_df, representation_df, config_dict, target_name,split_category, task_type):

    """
    Evaluates a fitted model on a given dataset split

    Args:
        fitted_model: The trained model to evaluate.
        data_df (DataFrame): DataFrame containing chemical IDs, target values, and split information.
        representation_df (DataFrame): DataFrame containing feature representations for the chemical data.
        config_dict (dict): Dictionary containing configuration parameters.
        target_name (str): Name of the target variable.
        split_category (str): Category of the dataset split to evaluate ('train', 'valid', or 'test').
        task_type (str): Type of task ('classification' or 'regression').

    Returns:
        float: The evaluation score for the model on the specified split category.
    """

    # Gather the X and y values
    ids = data_df.loc[data_df["split"] == split_category, "chem_id"].values
    X_eval = representation_df.loc[ids]
    y_eval = data_df.loc[data_df["split"] == split_category, target_name].values.ravel()

    # Classification error
    if task_type == "classification":
        y_pred = fitted_model.predict_proba(X_eval)
        eval_score = roc_auc_score(y_score=y_pred[:, 1], y_true=y_eval)
    
    elif task_type == "regression":
        y_pred = fitted_model.predict(X_eval)
        
        if config_dict["task_name"] in ["qm7", "qm8"]:
            eval_score = mean_absolute_error(y_true=y_eval, y_pred=y_pred)
        
        else:
            eval_score = mean_squared_error(y_true=y_eval, y_pred=y_pred, squared=False) 

    return eval_score


def get_args():

    """
    Parses command line arguments for finetuning
    """

    parser =ArgumentParser(prog="Train a machine learning model on a specific benchmark task using pre-trained molecular features.",
                            description="TRAIN ML MODEL WITH PRE-TRAINED MOLECULAR REPRESENTATIONS. This program receives the hyperparamters \
                                to train a machine learning model on a specific benchmark task. Hyperparameters can be given either in yaml file or as command line arguments. \
                                    The program will return a TSV file with the performance metrics of the experiments.",
                            usage="python train_ml.py -y config_ml.yaml | python train_ml.py [options]",
                            formatter_class=ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("--config_yaml", dest="config_yaml",
                        type=str,
                        help="Complete path to yaml file that contains the parameters for model training.")
    
    trainargs = parser.add_argument_group("Training parameters", "Parameters for training the machine learning model.")
    trainargs.add_argument("--task_name", dest="task_name",
                        type=str,
                        default="BBBP",
                        choices=["BBBP", "BACE", "ClinTox", "Tox21", "HIV", "SIDER",
                                 "FreeSolv", "ESOL", "Lipo", "qm7", "qm8"],
                        help="Name of the benchmark task to train the model on.")
    
    trainargs.add_argument("--ml_model", dest="ml_model",
                           type=str,
                           default="XGBoost",
                           choices=["RandomForest", "XGBoost"],
                            help="Type of machine learning model to train. Can be either RandomForest or XGBoost. Depending the benchmark task, a classifier or a regressor model is trained. Depending on the model, different hyperparameters are explored during random search (see models/ml_params.py).")
    
    trainargs.add_argument("--pretrained_model", dest="pretrained_model",
                           type=str,
                           default="gin_concat_R1000_E8000_lambda0.0001",
                            help="Name of the pre-trained model to use for the molecular representation. Should be a sub-directory of ./ckpt. Alternatively can also be 'MolCLR' or 'ECFP4'.")
    trainargs.add_argument("--gpu", dest="gpu",
                            type=str,
                            default="cuda:0",
                            help="Name of the cuda device to run the training. Can also be 'cpu'.")
    
    datasetargs = parser.add_argument_group("Dataset parameters", "Parameters for the dataset.")
    datasetargs.add_argument("--splitting", dest="splitting",
                            choices=["random", "scaffold"],
                            default="scaffold",
                            help="Type of splitting to use to build training, validation and testing sets.")
    datasetargs.add_argument("--validation_size", dest="validation_size",
                             type=float,
                             default=0.1,
                                help="Proportion of the dataset to include in the validation set.")
    datasetargs.add_argument("--test_size", dest="test_size",
                                type=float,
                                default=0.1,
                                    help="Proportion of the dataset to include in the test set.")
    
    randomsearchargs = parser.add_argument_group("Random search parameters", "Parameters for the random search of hyperparameters.")
    randomsearchargs.add_argument("--n_models", dest="n_models",
                                  type=int,
                                    default=5,
                                        help="Number of models configurations to train for each target.")
    randomsearchargs.add_argument("--n_trains", dest="n_trains",
                                  type=int,
                                  default=3,
                                  help="Number of training iterations for each model configuration.")
    
    parser.add_argument("--metrics_outdir", dest="metrics_outdir",
                      type=str,
                      default="output_metrics",
                      help="Directory to save the experiment performance metrics.")
    
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
    This function returns a config dictionary to be used during training.
    """

    # Parse command line arguments
    args = get_args()

    # If a config file is given, then parameters are read from there
    if args.config_yaml != None:
        print(f"Reading training arguments from {args.config_yaml}. Ignoring command line arguments.")

        with open(args.config_yaml, "r") as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)

    else:        
        # Else we have to build a dictionary with the same config structure
        config_dict = vars(args)

        # Build model param dictionary
        config_subdicts = {"dataset": {k: config_dict[k] for k in ["validation_size", "test_size", "splitting"]},
                           "random_search": {k: config_dict[k] for k in ["n_models", "n_trains"]}}
        
        # Update config_dict
        for key in ["validation_size", "test_size", "splitting", "n_models", "n_trains"]:
            del config_dict[key]
        
        config_dict.update(config_subdicts)
    
    # Read pre-training config
    pretrain_config = gather_pretraining_config(config_dict["pretrained_model"])

    # Add model_type and model to config_dict
    config_dict["pretrain_architecture"] = pretrain_config["model_type"]

    return config_dict


def main():
    """
    Main function for conducting experiments with various models on chemical representation and dataset.

    Reads the configuration file and extracts parameters for random search. Determines the task,
    labels, dataset splits, and model hyperparameters. Iterates over targets, models, and training iterations
    to train and evaluate models. Outputs the results to a TSV file.

    Returns:
        None
    """

    # Read config file
    config_original = gather_config()
    print(config_original)

    # Gather the number of models and the number of training iterations for each model
    rsearch_params = config_original.pop("random_search")

    # Determine the task and the labels
    smile_datapath, label_list = determine_dataset(config_original)

    # Gather output directory
    outdir = config_original.pop("metrics_outdir")

    # Determine the split and representation parameters
    split_df, features_df = process_dataset(smile_datapath,
                                            dataset_split = True,
                                            pretrain_architecture=config_original["pretrain_architecture"],
                                            pretrained_model=config_original["pretrained_model"],
                                            split_approach=config_original["dataset"]["splitting"],
                                            validation_proportion=config_original["dataset"]["validation_size"],
                                            test_proportion=config_original["dataset"]["test_size"],
                                            device=config_original["gpu"]) 
    
    # Determine the model hyperparameters to be used and task type
    model_param_dict_original, task_type = determine_predictor_hyper(config_original)

    results_list = []
    # Iterate over targets
    for target in label_list:

        # Iterate over models
        for m in range(rsearch_params["n_models"]):
            model_name = f"model_{m}"

            # Select the hyperparameters for the model
            model_param_dict = select_params(model_param_dict_original, config_original["task_name"], config_original["ml_model"], task_type)
            model_config_str = str(model_param_dict)

            # Iterate over training iterations
            
            for t in range(rsearch_params["n_trains"]):
                training_name = f"train_{t}"

                # Get the model with the chosen hyperparameters
                model = get_predictor(model_param_dict, config_original["ml_model"], task_type)

                # Train the model
                ids_train = split_df.loc[split_df["split"] == "train", "chem_id"].values
                X_train = features_df.loc[ids_train]
                y_train = split_df.loc[split_df["split"] == "train", target].values

                model.fit(X_train, y_train)

                # Validation score
                validation_score = evaluate_model(fitted_model=model, 
                                                  data_df=split_df, 
                                                  representation_df=features_df, 
                                                  config_dict=config_original, 
                                                  target_name=target,
                                                  split_category="valid",
                                                  task_type=task_type)
                
                # Test score
                test_score = evaluate_model(fitted_model=model, 
                                                  data_df=split_df, 
                                                  representation_df=features_df, 
                                                  config_dict=config_original, 
                                                  target_name=target,
                                                  split_category="test",
                                                  task_type=task_type)
                
                experiment_info = [config_original["task_name"], target, config_original["ml_model"], 
                                   config_original["pretrained_model"], model_name, training_name, 
                                   validation_score, test_score, model_config_str]
                
                results_list.append(experiment_info)
    
    # Output the results
    results_df = pd.DataFrame(results_list, columns=["DATASET", "TARGET", "MODEL", "FEATURES", 
                                                     "MODEL_N", "TRAIN_N", "VALIDATION_SCORE", 
                                                     "TEST_SCORE", "MODEL_CONFIG"])
    

    os.makedirs(outdir, exist_ok=True)
    outfile_name =f"{config_original['task_name']}_{config_original['ml_model']}_{config_original['pretrained_model']}.tsv.gz"
    outpath = os.path.join(outdir, outfile_name)
    results_df.to_csv(outpath, index=False) 
                


if __name__ == "__main__":
    main()