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
from argparse import ArgumentParser

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
       
        # get the representations and the projections
        _, zis = model(xis)  # [N,C]

        # get the representations and the projections
        _, zjs = model(xjs)  # [N,C]

        loss = self.btwin_objective(zis, zjs)

        return loss

    def train(self):

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
            weight_decay=eval(self.config['weight_decay'])
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
    parser = ArgumentParser()
    
    parser.add_argument("-m", "--model_type", dest="model_type",
                        default="None", 
                        help="model_type")

    parser.add_argument("-l", "--lambda_val", dest="lambda_val", 
                        default=-1, type=float,
                        help="value of lambda value for BT loss")
   
    args = parser.parse_args()

    return args

def main():
    # Argument readings
    given_args = get_args()

    # Read config file
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)

    # Check if we have to modify the config file
    if given_args.model_type != "None":
        config["model_type"] = given_args.model_type
    
    if given_args.lambda_val != -1:
        config["loss"]["l"] = given_args.lambda_val

    print(config)

    # Run pre-training
    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    chemprep = PretrainChemRep(dataset, config)
    chemprep.train()


if __name__ == "__main__":
    main()