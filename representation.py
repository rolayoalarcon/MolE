import os 
import random
import yaml
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

from dataset.dataset_representation import process_dataset
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
from models.ml_params import TASK_LABELS, TASK_DATAPATH, RF_CLASSIFIER_PARAMS, RF_REGRESSOR_PARAMS, XGB_REGRESSOR_PARAMS, XGB_CLASSIFIER_PARAMS

def determine_dataset(config_dict, task_target_dict=TASK_LABELS, task_path_dict = TASK_DATAPATH):

    target_list = task_target_dict[config_dict["task_name"]]
    task_datapath = task_path_dict[config_dict["task_name"]]

    return task_datapath, target_list

def determine_predictor(config_dict):

    if config_dict["model_type"] == "RandomForestClassifier":
        return RF_CLASSIFIER_PARAMS
    
    elif config_dict["model_type"] == "RandomForestRegressor":
        return RF_REGRESSOR_PARAMS
    
    elif config_dict["model_type"] == "XGBClassifier":
        return XGB_CLASSIFIER_PARAMS
    
    elif config_dict["model_type"] == "XGBRegressor":
        return XGB_REGRESSOR_PARAMS
    
def select_params(original_config, dname, model_name):
    """
    Receives a dictionary, where any values consisting of lists will be 
    reduced to a single value by random choice
    """
    model_config = original_config.copy()

    # Make sure that if we are doing regression, we use the correct eval metric during XGB
    if dname in ["qm7", "qm8", "qm9"] and model_name == "XGBRegressor":
        model_config["objective"] = "reg:squarederror"
        model_config["eval_metric"] = "mae"

    elif dname not in ["qm7", "qm8", "qm9"] and model_name == "XGBRegressor":
        model_config["objective"] = "reg:squarederror"
        model_config["eval_metric"] = "rmse"
    

    for key, value in model_config.items():

        if type(value) == list:
            model_config[key] = random.choice(value)
        elif type(value) == dict:
            model_config[key] = select_params(value)
        
    return model_config

def get_predictor(model_params, model_type):

    """Initialize models with the chosen hyperparameters configuration"""

     # Classifiers
    if model_type == "RandomForestClassifier":
        model_with_hyperparam = RandomForestClassifier(**model_params)
    
    elif model_type == "XGBClassifier":
        model_params["seed"] = np.random.randint(1_000_000, size=1)[0]
        model_with_hyperparam = XGBClassifier(**model_params)
    
    # Regression
    elif model_type == "RandomForestRegressor":
        model_with_hyperparam = RandomForestRegressor(**model_params)
    
    elif model_type == "XGBRegressor":
        model_params["seed"] = np.random.randint(1_000_000, size=1)[0]
        model_with_hyperparam = XGBRegressor(**model_params)

    return model_with_hyperparam

def evaluate_model(fitted_model, data_df, representation_df, config_dict, target_name,split_category):

    """
    Evaluates a fitted model on a given dataset split

    Args:
        fitted_model: The trained model to evaluate.
        data_df (DataFrame): DataFrame containing chemical IDs, target values, and split information.
        representation_df (DataFrame): DataFrame containing feature representations for the chemical data.
        config_dict (dict): Dictionary containing configuration parameters.
        target_name (str): Name of the target variable.
        split_category (str): Category of the dataset split to evaluate ('train', 'valid', or 'test').

    Returns:
        float: The evaluation score for the model on the specified split category.
    """

    # Gather the X and y values
    ids = data_df.loc[data_df["split"] == split_category, "chem_id"].values
    X_eval = representation_df.loc[ids]
    y_eval = data_df.loc[data_df["split"] == split_category, target_name].values.ravel()

    # Classification error
    if config_dict["model_type"] in ["RandomForestClassifier", "XGBClassifier"]:
        y_pred = fitted_model.predict_proba(X_eval)
        eval_score = roc_auc_score(y_score=y_pred[:, 1], y_true=y_eval)
    
    elif config_dict["model_type"] in ["RandomForestRegressor", "XGBRegressor"]:
        y_pred = fitted_model.predict(X_eval)
        
        if config_dict["task_name"] in ["qm7", "qm8"]:
            eval_score = mean_absolute_error(y_true=y_eval, y_pred=y_pred)
        
        else:
            eval_score = mean_squared_error(y_true=y_eval, y_pred=y_pred, squared=False) 

    return eval_score

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
    config_original = yaml.load(open("config_representation.yaml", "r"), Loader=yaml.FullLoader)

    # Gather the number of models and the number of training iterations for each model
    rsearch_params = config_original.pop("random_search")

    # Determine the task and the labels
    smile_datapath, label_list = determine_dataset(config_original)

    # Determine the split and representation parameters
    split_df, features_df = process_dataset(smile_datapath,
                                            dataset_split = True,
                                            pretrain_architecture=config_original["pretrain_architecture"],
                                            pretrained_model=config_original["pretrained_model"],
                                            split_approach=config_original["dataset"]["splitting"],
                                            validation_proportion=config_original["dataset"]["validation_size"],
                                            test_proportion=config_original["dataset"]["test_size"],
                                            device=config_original["device"]) 
    
    # Determine the model hyperparameters to be used
    model_param_dict_original = determine_predictor(config_original)

    results_list = []
    # Iterate over targets
    for target in label_list:

        # Iterate over models
        for m in range(rsearch_params["n_models"]):
            model_name = f"model_{m}"

            # Select the hyperparameters for the model
            model_param_dict = select_params(model_param_dict_original, config_original["task_name"], config_original["model_type"])
            model_config_str = str(model_param_dict)

            # Iterate over training iterations
            
            for t in range(rsearch_params["n_trains"]):
                training_name = f"train_{t}"

                # Get the model with the chosen hyperparameters
                model = get_predictor(model_param_dict, config_original["model_type"])

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
                                                  split_category="valid")
                
                # Test score
                test_score = evaluate_model(fitted_model=model, 
                                                  data_df=split_df, 
                                                  representation_df=features_df, 
                                                  config_dict=config_original, 
                                                  target_name=target,
                                                  split_category="test")
                
                experiment_info = [config_original["task_name"], target, config_original["model_type"], 
                                   config_original["pretrained_model"], model_name, training_name, 
                                   validation_score, test_score, model_config_str]
                
                results_list.append(experiment_info)
    
    # Output the results
    results_df = pd.DataFrame(results_list, columns=["DATASET", "TARGET", "MODEL", "FEATURES", 
                                                     "MODEL_N", "TRAIN_N", "VALIDATION_SCORE", 
                                                     "TEST_SCORE", "MODEL_CONFIG"])
    
    outfile_name =f"{config_original['task_name']}_{config_original['model_type']}_{config_original['pretrained_model']}.tsv.gz"
    outpath = os.path.join("experiments", outfile_name)
    results_df.to_csv(outpath, index=False, sep='\t') 
                


if __name__ == "__main__":
    main()