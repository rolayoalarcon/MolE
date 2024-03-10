import os
import yaml
import random
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser
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
    parser = ArgumentParser()
    parser.add_argument("-c", "--config_filepath", dest="config_filepath",
                        default="config_finetune.yaml",
                        help="Path to the finetune config file", metavar="FILE")
    
    parser.add_argument("-t", "--task_name", dest="task_name", 
                        default="None",
                        help="benchmark task to use")
   
    parser.add_argument("-m", "--model_outdir", dest="model_odir", default="finetune",
                        help="Directory where finetuned models are written")
    
    parser.add_argument("-e", "--experiment_odir", dest="experiment_odir", default="experiments",
                        help="Directory where performance metrics are written")
   
    args = parser.parse_args()

    return args

def determine_outdir(label_name, m_name, t_name, m_type):
    base_model_str = "{model_type}_{label_name}_{model_name}_{training_name}_{datetime_str}"

    directory_name = base_model_str.format(model_type=m_type,
                                           label_name = label_name,
                                           model_name = m_name,
                                           training_name = t_name,
                                           datetime_str=datetime.now().strftime("%d.%b.%Y_%H.%M.%S"))
    
    return directory_name


def main():

    # Read the arguments
    given_args = get_args()
    config_file = given_args.config_filepath

    # Read the random search parameters
    config_originial = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)

    # Determine whether the confing file has to be modified
    if given_args.task_name != "None":
        config_originial["task_name"] = given_args.task_name
    

    # Gather the number of models and the number of training iterations for each model
    rsearch_params = config_originial.pop("random_search")

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

                result_dict = finetune(model_config, model_directory, given_args.model_odir)

                for key, value in result_dict.items():
                    results_list.append([target, key, value, model_name, training_name, model_directory])

    # Create a dataframe
    df = pd.DataFrame(results_list)

    # Write the output
    os.makedirs(given_args.experiment_odir, exist_ok=True)

    outfile = f'{config_originial["task_name"]}_{config_originial["fine_tune_from"]}.tsv.gz'


    df.to_csv(os.path.join(given_args.experiment_odir, outfile), mode='a', index=False, header=False, sep='\t')




if __name__ == "__main__":
    main()