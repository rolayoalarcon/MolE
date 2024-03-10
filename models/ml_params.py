# This file contains the parameters used for the random search of hyperparameters for the different models
# Roberto Olayo Alarcon 31.01.2024


# Task names and labels by dataset
TASK_LABELS = {"clintox": ["CT_TOX", "FDA_APPROVED"],
               "BBBP": ["p_np"],
               "bace": ["Class"],
               "tox21":["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"],
               "sider":["Hepatobiliary disorders", "Metabolism and nutrition disorders", "Product issues", "Eye disorders", "Investigations", "Musculoskeletal and connective tissue disorders", "Gastrointestinal disorders", "Social circumstances", "Immune system disorders", "Reproductive system and breast disorders", "Neoplasms benign, malignant and unspecified (incl cysts and polyps)", "General disorders and administration site conditions", "Endocrine disorders", "Surgical and medical procedures", "Vascular disorders", "Blood and lymphatic system disorders", "Skin and subcutaneous tissue disorders", "Congenital, familial and genetic disorders", "Infections and infestations", "Respiratory, thoracic and mediastinal disorders", "Psychiatric disorders", "Renal and urinary disorders", "Pregnancy, puerperium and perinatal conditions", "Ear and labyrinth disorders", "Cardiac disorders", "Nervous system disorders", "Injury, poisoning and procedural complications"],
               "HIV": ["HIV_active"],
               "Lipophilicity": ["exp"],
               "esol": ["measured log solubility in mols per litre"],
               "freesolv": ["expt"],
               "qm7": ["u0_atom"],
               "qm8": ["E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM", "f1-CAM","f2-CAM"]}

TASK_DATAPATH = {"clintox": "data/benchmark_data/clintox/clintox.csv",
                 "BBBP": "data/benchmark_data/bbbp/BBBP.csv",
                 "bace": "data/benchmark_data/bace/bace.csv",
                 "tox21": "data/benchmark_data/tox21/tox21.csv",
                 "sider": "data/benchmark_data/sider/sider.csv",
                 "HIV": "data/benchmark_data/hiv/HIV.csv",
                 "Lipophilicity": "data/benchmark_data/lipophilicity/Lipophilicity.csv",
                 "esol": "data/benchmark_data/esol/esol.csv",
                 "freesolv": "data/benchmark_data/freesolv/freesolv.csv",
                 "qm7": "data/benchmark_data/qm7/qm7.csv",
                 "qm8": "data/benchmark_data/qm8/qm8.csv"
}

## Parameters for our classifiers
RF_CLASSIFIER_PARAMS = {"criterion": ["gini", "entropy"],
             "max_depth": [None],
             "max_features": ["sqrt", "log2", None],
             "n_jobs": 10,
             "class_weight": ["balanced", None]}

RF_REGRESSOR_PARAMS = {"n_jobs": [20],
                "n_estimators": [100, 300, 500,700,1000],
                "criterion": ["squared_error"],
                "max_features": ["sqrt", "log2"]}

XGB_CLASSIFIER_PARAMS = {"nthread": 20,
                         "n_estimators":[30, 100, 300, 500, 1000],
                         "max_depth": [5, 10, 50, 100],
                         "eta":[0.3, 0.1, 0.05, 1],
                         "subsample": [0.3, 0.5, 0.8, 1.0],
                         "objective": "binary:logistic"}

XGB_REGRESSOR_PARAMS = {"nthread": 20,
                         "n_estimators":[30, 100, 300, 500, 1000],
                         "max_depth": [5, 10, 50, 100],
                         "eta":[0.3, 0.1, 0.05, 1],
                         "subsample": [0.3, 0.5, 0.8, 1.0],
                         "objective": ["reg:squarederror", "reg:absoluteerror"]}