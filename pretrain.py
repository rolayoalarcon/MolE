######################################################
# Script for pre-trainig a model using Barlow Twins loss.
# The script is based on the scripts provided by MolCLR 
#
# Modified by: Roberto Olayo Alarcon
###################################################### 

import os
import shutil
import sys
import torch
import yaml
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from loss_function.btwins import BarlowTwinsObjective
from dataset.dataset_subgraph import MoleculeDatasetWrapper

# Write the current config file to the model folder
def save_config_file(model_checkpoints_folder, config_dict):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)

        # Change this to reflect the current params
        with open(os.path.join(model_checkpoints_folder, 'config.yaml'), "w") as outfile:
            yaml.dump(config_dict, outfile, default_flow_style=False)

# Write the indices of the data used for training and validation
def write_indices(odir, tr_loader, va_loader):
    # Write train inidces
    pd.DataFrame(tr_loader.sampler.indices).to_csv(os.path.join(odir, "train_idx.csv.gz"), index=False, header=False)
    # Validation
    pd.DataFrame(va_loader.sampler.indices).to_csv(os.path.join(odir, "validation_idx.csv.gz"), index=False, header=False)

# Determine the name of the directory where the model will be saved
def determine_dirname(config_dict):
    # Model param names
    if config_dict["model_type"] in ["gin_concat", "gcn_concat"]:
        representation_dimensionality = config_dict["model"]["emb_dim"] * config_dict["model"]["num_layer"]
    else:
        representation_dimensionality = config_dict["model"]["emb_dim"]
    
    base_model_str = "{model_type}_R{representation_dim}_E{embedding_dim}_lambda{lambdaval}_{datetime_str}"

    directory_name = base_model_str.format(model_type=config_dict["model_type"],
                                          representation_dim=representation_dimensionality ,
                                          embedding_dim=config_dict["model"]["feat_dim"],
                                         lambdaval=config_dict["loss"]["l"],
                                         datetime_str=datetime.now().strftime("%d.%b.%Y_%H.%M.%S"))
    
    return directory_name



class PretrainChemRep(object):

    """
    A class for pretraining a chemical representation model with MolE strategy.

    Args:
        dataset (object): A dataset object containing the dataset splits and augmented graphs.
        config (dict): A dictionary containing pre-training hyperparameters.

    Attributes:
        config (dict): A dictionary containing pre-training hyperparameters.
        device (str): The device (CPU or GPU) where the model will be trained.
        log_dir (str): The directory path for storing model checkpoints and config.
        dataset (object):  dataset object containing the dataset splits and augmented graphs.
        btwin_objective (object): An object calculating the Barlow Twins objective.

    Methods:
        __init__(self, dataset, config): Initializes the PretrainChemRep object.
        _get_device(self): Determines and returns the device for training.
        _step(self, model, xis, xjs): Performs a single optimization step.
        train(self): Trains the chemical representation model.
        _load_pre_trained_weights(self, model): Loads pre-trained weights for the model if available.
        _validate(self, model, valid_loader): Validates the model on the validation set.

    Example Usage:
        # Initialize PretrainChemRep object
        pretrainer = PretrainChemRep(dataset, config)
        
        # Train the model
        pretrainer.train()
    """

    def __init__(self, dataset, config):
        self.config = config
        self.device = self._get_device()
        
        dir_name = determine_dirname(config)
        self.log_dir = os.path.join('ckpt', dir_name)

        #log_dir = os.path.join('ckpt', dir_name)
        #self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.btwin_objective = BarlowTwinsObjective(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, xis, xjs):
        """
        Performs a single optimization step.

        Args:
            model (object): The chemical representation model.
            xis (tensor): Input data for the first augmentation.
            xjs (tensor): Input data for the second augmentation.

        Returns:
            loss (tensor): The calculated loss for the optimization step.
        """
       
        # get the representations and the embeddings
        _, zis = model(xis)

        # get the representations and the embeddings
        _, zjs = model(xjs) 

        loss = self.btwin_objective(zis, zjs)

        return loss

    def train(self):
        """
        Trains the chemical representation model.
        """

        # Establish where the model will be saved
        model_checkpoints_folder = os.path.join(self.log_dir, 'checkpoints')

        # Save config file
        save_config_file(model_checkpoints_folder, self.config)
        
        # Gather dataloaders
        train_loader, valid_loader = self.dataset.get_data_loaders()

        # Write data indices
        write_indices(self.log_dir, train_loader, valid_loader)

        
        # Load the pre-training config
        if self.config['model_type'] == 'gin_concat':
            from models.ginet_concat import GINet
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        
        elif self.config['model_type'] == 'gin_noconcat':
            from models.ginet_noconcat import GINet
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        elif self.config["model_type"] == "gcn_concat":
            from models.gcn_concat import GCN
            model = GCN(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        
        elif self.config["model_type"] == "gcn_noconcat":
            from models.gcn_noconcat import GCN
            model = GCN(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)

        else:
            raise ValueError('Undefined GNN model.')
        print(model)
        
        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'], 
            weight_decay=self.config['weight_decay']
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']-self.config['warm_up'], 
            eta_min=0, last_epoch=-1
        )

        # Train the model
        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf 

        for epoch_counter in range(self.config['epochs']):
            for bn, (xis, xjs) in enumerate(train_loader):
                optimizer.zero_grad()

                # Move data to device
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                # Forward pass
                loss = self._step(model, xis, xjs)

                # Backward pass
                loss.backward()
                optimizer.step()

                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                print(epoch_counter, bn, valid_loss, '(validation)')
                if valid_loss < best_valid_loss:
                    # save the model weights fds
                    best_valid_loss = valid_loss 
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
            
                #self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()

    def _load_pre_trained_weights(self, model):
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            model.load_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            for (xis, xjs) in valid_loader:
                xis = xis.to(self.device)
                xjs = xjs.to(self.device)

                loss = self._step(model, xis, xjs)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        
        model.train()
        return valid_loss

def get_args():
    """
    Parses command line arguments for finetuning
    """
    parser = ArgumentParser(prog="Pre-train a molecular representation using the MolE framework.",
                            description="This program receives the hyperparamters to train a pre-trained molecular representation using the MolE framework. Hyperparameters can be given either in yaml file or as command line arguments. If a yaml file is provided, command line arguments are ignored. The program will return a model.pth file with the pre-trained weights and config.yaml file with the indicated hyperparameters.",
                            usage="python pretrain.py -y config.yaml | python pretrain.py [options]",
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("--config_yaml", dest="config_yaml",
                        help="Complete path to yaml file that contains the parameters for model pre-training")
    
    # General arguments for trainig
    pretargs = parser.add_argument_group("Pre-training parameters", "Parameters for model pre-training.")
    pretargs.add_argument("--batch_size", dest="batch_size", type=int,
                        default=1000,
                        help="Number of compounds in each batch during pre-training.")
    
    pretargs.add_argument("--warm_up", dest="warm_up", type=int,
                        default=10,
                        help="Number of warm up epochs before cosine annealing begins.")
    
    pretargs.add_argument("--epochs", dest="epochs",
                        default=1000, type=int,
                        help="Total number of epochs to pre-train.")
    
    pretargs.add_argument("--eval_every_n_epochs", dest="eval_every_n_epochs",
                        default=1, type=int,
                        help="Validation frequency.")
    
    pretargs.add_argument("--save_every_n_epochs", dest="save_every_n_epochs",
                        default=5, type=int,
                        help="Automatically save model every n epochs.")
    
    pretargs.add_argument("--init_lr", 
                        dest="init_lr",
                        type=float,
                        default=0.0005, 
                        help="Inital learning rate for the ADAM optimizer.")
    
    pretargs.add_argument("--weight_decay", 
                        dest="weight_decay",
                        type=float,
                        default=0.00001,
                        help="Weight decay for ADAM.")
    
    pretargs.add_argument("--gpu", 
                        dest="gpu",
                        type=str,
                        default="cuda:0",
                        help="Name of the cuda device to run pre-training. Can also be 'cpu'.")
    
    pretargs.add_argument("--model_type", 
                        dest="model_type",
                        type=str,
                        default="gin_concat", 
                        choices=["gin_concat", "gin_noconcat", "gcn_concat", "gcn_noconcat"],
                        help="Pre-training architechture consisting of GNN backbone (gin or gcn) and representation building (concat or noconcat).")

    pretargs.add_argument("--load_model", 
                        dest="load_model",
                        default="None",
                        type=str,
                        help="Name of model in ckpt to resume pre-training.")
    
    # Model architechture
    modelargs = parser.add_argument_group("Model parameters", "Parameters for the GNN model backbone.")
    modelargs.add_argument("--num_layer", 
                        dest="num_layer",
                        type=int,
                        default=5, 
                        help="Number of GNN layers.")
    
    modelargs.add_argument("--emb_dim", 
                        dest="emb_dim",
                        type=int,
                        default=200,
                        help="Dimensionality of the graph representation. If representation building is '*_concat', then the graph representation of each GNN layer is emb_dim dimensional and the final molecular representation (r) will have dimension = num_layer * emb_dim. Else representation will be emb_dim dimensional.")
    
    modelargs.add_argument("--feat_dim", 
                        dest="feat_dim",
                        type=int,
                        default=8000,
                        help="Dimensionality of the embedding vector (z) that is the input to the barlow-twins loss.")
    
    modelargs.add_argument("--drop_ratio", 
                        dest="drop_ratio",
                        type=float,
                        default=0.0, 
                        help="Dropout ratio")
    
    modelargs.add_argument("--pool", 
                        dest="pool",
                        type=str,
                        default="add",
                        choices=["max", "add", "avg"],
                        help="Readout pooling function to great graph-layer representations.")

    # Dataset parameters
    datasetargs = parser.add_argument_group("Dataset parameters", "Parameters for the dataset of unlabled molecules.")
    datasetargs.add_argument("--num_workers", 
                        dest="num_workers",
                        type=int,
                        default=100,
                        help="Dataloader number of workers.")

    datasetargs.add_argument("--valid_size", 
                        dest="valid_size",
                        type=float, 
                        default=0.1,
                        help="Ratio of the total of molecules provided that are used as validation set.")   
    
    datasetargs.add_argument("--data_path", 
                        dest="data_path",
                        type=str,
                        default="data/pubchem_data/pubchem_100k_random.txt",
                        help="Path to pre-training dataset of unlabeled molecules.")
    
    # Loss function parameters
    datasetargs.add_argument("--lambda_val", dest="lambda_val", 
                        default=1e-4, type=float,
                        help="Value of lambda value for BT loss")

    args = parser.parse_args()

    return args

def gather_config():
    """
    This function returns a config dictionary to be used during pre-training
    """

    # Parse command line arguments
    args = get_args()

    # If a config file is given, then parameters are read from there
    if args.config_yaml != None:
        print(f"Reading pre-training arguments from {args.config_yaml}. Ignoring command line arguments.")

        with open(args.config_yaml, "r") as file:
            config_dict = yaml.load(file, Loader=yaml.FullLoader)
        
        return config_dict
    
    # Else we have to build a dictionary with the same config structure
    arg_dict = vars(args)
    
    # Build model param dictionary
    config_subdicts = {"model": {k: arg_dict[k] for k in ["num_layer", "emb_dim", "feat_dim", "drop_ratio", "pool"]},
                       "loss": {"l":arg_dict["lambda_val"]},
                       "dataset": {k: arg_dict[k] for k in ["num_workers", "valid_size", "data_path"]}}
    
    # Update arg_dict
    for key in ["num_layer", "emb_dim", "feat_dim", "drop_ratio", "pool", "lambda_val", "num_workers", "valid_size", "data_path"]:
        del arg_dict[key]
    
    arg_dict.update(config_subdicts)

    return arg_dict

def main():

    """
    Main function for running pre-training of a chemical representation model.

    Reads command-line arguments and configuration file, modifies the configuration if necessary,
    initializes dataset and pretraining objects, and executes the pretraining process.

    Returns:
        None
    """

    # Argument readings
    config = gather_config()

    print(config)

    # Run pre-training
    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    chemprep = PretrainChemRep(dataset, config)
    chemprep.train()


if __name__ == "__main__":
    main()