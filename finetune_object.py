import os
import shutil
import sys
import yaml
import numpy as np
import pandas as pd
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error, precision_recall_curve, auc

from dataset.dataset_test import MolTestDatasetWrapper


def _save_config_file(model_checkpoints_folder, param_dictionary):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, "model_config.yaml"), "w") as file:
            documents = yaml.dump([param_dictionary], file)


class Normalizer(object):
    """Center and scale tensor. Save parameters to denormalize later."""

    def __init__(self, tensor):
        """calculate mean and stdv"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {'mean': self.mean,
                'std': self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict['mean']
        self.std = state_dict['std']


class FineTune(object):

    """
    A class for fine-tuning a chemical representation model on a specific task.

    Args:
        dataset (object): An object representing the dataset used for fine-tuning.
        config (dict): A dictionary containing hyperparameter configuration.
        parent_odir (str): The parent output directory path.
        model_sdir (str): The subdirectory name for the model.

    Attributes:
        config (dict): A dictionary hyperparameter configuration.
        device (str): The device (CPU or GPU) where the model will be trained.
        log_dir (str): The directory path for storing logs and checkpoints.
        dataset (object): An object representing the dataset used for fine-tuning.
        criterion (object): The loss criterion for the fine-tuning task.

    Methods:
        __init__(self, dataset, config, parent_odir, model_sdir): Initializes the FineTune object.
        _get_device(self): Determines and returns the device for training.
        _step(self, model, data, n_iter): Performs a single optimization step.
        train(self): Trains the chemical representation model.
        _load_pre_trained_weights(self, model): Loads pre-trained weights for the model if available.
        _validate(self, model, valid_loader): Validates the model on the validation set.
        _test(self, model, test_loader): Evaluates the model on the test set.

    Example Usage:
        # Initialize FineTune object
        finetuner = FineTune(dataset, config, parent_odir, model_sdir)
        
        # Fine-tune the model
        finetuner.train()
    """

    def __init__(self, dataset, config, parent_odir, model_sdir):
        self.config = config
        self.device = self._get_device()

        dir_name = model_sdir
        self.log_dir = os.path.join(parent_odir, dir_name)
        
        
        
        
        self.dataset = dataset
        if config['dataset']['task'] == 'classification':
            self.criterion = nn.CrossEntropyLoss()

        elif config['dataset']['task'] == 'regression':
            if self.config["task_name"] in ['qm7', 'qm8', 'qm9']:
                self.criterion = nn.L1Loss()
            else:
                self.criterion = nn.MSELoss()

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    def _step(self, model, data, n_iter):
        # get the prediction
        __, pred = model(data) 

        if self.config['dataset']['task'] == 'classification':
            loss = self.criterion(pred, data.y.flatten())

        elif self.config['dataset']['task'] == 'regression':
            if self.normalizer:
                loss = self.criterion(pred, self.normalizer.norm(data.y))
            else:
                loss = self.criterion(pred, data.y)

        return loss

    def train(self):
        train_loader, valid_loader, test_loader = self.dataset.get_data_loaders()

        self.normalizer = None
        if self.config["task_name"] in ['qm7', 'qm9']:
            labels = []
            for d in train_loader:
                labels.append(d.y)
            labels = torch.cat(labels)
            self.normalizer = Normalizer(labels)
            print(self.normalizer.mean, self.normalizer.std, labels.shape)

        if self.config['model_type'] == 'gin_concat':
            from models.ginet_concat_finetune import GINet
            model = GINet(**self.config["model"]).to(self.device)
            if self.config["fine_tune_from"] != "no_pretrain":
                model = self._load_pre_trained_weights(model)
            
        elif self.config["model_type"] == "gin_noconcat":
            from models.ginet_noconcat_finetune import GINet
            model = GINet(**self.config["model"]).to(self.device)
            if self.config["fine_tune_from"] != "no_pretrain":
                model = self._load_pre_trained_weights(model)

        elif self.config["model_type"] == "gcn_concat":
            from models.gcn_concat_finetune import GCN
            model = GCN(**self.config["model"]).to(self.device)
            if self.config["fine_tune_from"] != "no_pretrain":
                model = self._load_pre_trained_weights(model)
        
        elif self.config["model_type"] == "gcn_noconcat":
            from models.gcn_noconcat_finetune import GCN
            model = GCN(**self.config["model"]).to(self.device)
            if self.config["fine_tune_from"] != "no_pretrain":
                model = self._load_pre_trained_weights(model)
        
        print(model)

        layer_list = []
        for name, param in model.named_parameters():
            if 'pred_head' in name:
                print(name, param.requires_grad)
                layer_list.append(name)

        params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] in layer_list, model.named_parameters()))))
        base_params = list(map(lambda x: x[1],list(filter(lambda kv: kv[0] not in layer_list, model.named_parameters()))))

        optimizer = torch.optim.Adam(
            [{'params': base_params, 'lr': self.config['init_base_lr']}, {'params': params}],
            self.config['init_lr'], weight_decay=self.config['weight_decay']
        )

        model_checkpoints_folder = os.path.join(self.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder, self.config)

        n_iter = 0
        valid_n_iter = 0
        best_valid_loss = np.inf
        best_valid_rgr = np.inf
        best_valid_cls = 0

        for epoch_counter in range(self.config['epochs']):
            for bn, data in enumerate(train_loader):
                optimizer.zero_grad()

                data = data.to(self.device)
                loss = self._step(model, data, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    #self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    print(epoch_counter, bn, loss.item())

               
                loss.backward()
                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                if self.config['dataset']['task'] == 'classification': 
                    valid_loss, valid_cls = self._validate(model, valid_loader)
                    if valid_cls > best_valid_cls:
                        # save the model weights
                        best_valid_cls = valid_cls
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
                elif self.config['dataset']['task'] == 'regression': 
                    valid_loss, valid_rgr = self._validate(model, valid_loader)
                    if valid_rgr < best_valid_rgr:
                        # save the model weights
                        best_valid_rgr = valid_rgr
                        torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))

                valid_n_iter += 1
        
        self._test(model, test_loader)

    def _load_pre_trained_weights(self, model):
        """
        Loads pre-trained weights for the model if available.

        Args:
            model (object): The chemical representation model.

        Returns:
            model (object): The model with loaded pre-trained weights if available, else the original model.
        """
        try:
            checkpoints_folder = os.path.join('./ckpt', self.config['fine_tune_from'], 'checkpoints')
            state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'), map_location=self.device)
            # model.load_state_dict(state_dict)
            model.load_my_state_dict(state_dict)
            print("Loaded pre-trained model with success.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            num_data = 0
            for bn, data in enumerate(valid_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                valid_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            valid_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                mae = mean_absolute_error(labels, predictions)
                print('Validation loss:', valid_loss, 'MAE:', mae)
                return valid_loss, mae
            else:
                rmse = mean_squared_error(labels, predictions, squared=False)
                print('Validation loss:', valid_loss, 'RMSE:', rmse)
                return valid_loss, rmse

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            roc_auc = roc_auc_score(labels, predictions[:,1])
            print('Validation loss:', valid_loss, 'ROC AUC:', roc_auc)
            return valid_loss, roc_auc

    def _test(self, model, test_loader):
        model_path = os.path.join(self.log_dir, 'checkpoints', 'model.pth')
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        print("Loaded trained model with success.")

        # test steps
        predictions = []
        labels = []
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            num_data = 0
            for bn, data in enumerate(test_loader):
                data = data.to(self.device)

                __, pred = model(data)
                loss = self._step(model, data, bn)

                test_loss += loss.item() * data.y.size(0)
                num_data += data.y.size(0)

                if self.normalizer:
                    pred = self.normalizer.denorm(pred)

                if self.config['dataset']['task'] == 'classification':
                    pred = F.softmax(pred, dim=-1)

                if self.device == 'cpu':
                    predictions.extend(pred.detach().numpy())
                    labels.extend(data.y.flatten().numpy())
                else:
                    predictions.extend(pred.cpu().detach().numpy())
                    labels.extend(data.y.cpu().flatten().numpy())

            test_loss /= num_data
        
        model.train()

        if self.config['dataset']['task'] == 'regression':
            predictions = np.array(predictions)
            labels = np.array(labels)
            if self.config['task_name'] in ['qm7', 'qm8', 'qm9']:
                self.mae = mean_absolute_error(labels, predictions)
                print('Test loss:', test_loss, 'Test MAE:', self.mae)
            else:
                self.rmse = mean_squared_error(labels, predictions, squared=False)
                print('Test loss:', test_loss, 'Test RMSE:', self.rmse)

        elif self.config['dataset']['task'] == 'classification': 
            predictions = np.array(predictions)
            labels = np.array(labels)
            
            # ROC AUC
            self.roc_auc = roc_auc_score(labels, predictions[:,1])
            
            # PRC AUC
            precision, recall, _ = precision_recall_curve(labels, predictions[:,1])
            self.prc_auc = auc(recall, precision)

            print('Test loss:', test_loss, 'Test ROC AUC:', self.roc_auc, "Test PRC AUC:", self.prc_auc)


def finetune(config, model_subdir, finetune_dir):

    """
    Fine-tunes a chemical representation model based on the given hyperparameter configuration.

    Args:
        config (dict): A dictionary containing hyperparamter configuration.
        model_subdir (str): The subdirectory name for the model.
        finetune_dir (str): The directory path for fine-tuning outputs.

    Returns:
        dict: A dictionary containing evaluation metrics based on the fine-tuning task.
            For classification tasks: {"ROC_AUC": roc_auc_score, "PRC_AUC": prc_auc_score}
            For regression tasks:
                If task_name is ['qm7', 'qm8', 'qm9']: {"MAE": mean_absolute_error}
                Otherwise: {"RMSE": root_mean_squared_error}
    """

    dataset = MolTestDatasetWrapper(config['batch_size'], **config['dataset'])

    fine_tune = FineTune(dataset, config, finetune_dir, model_subdir)
    fine_tune.train()
    
    if config['dataset']['task'] == 'classification':
        return {"ROC_AUC": fine_tune.roc_auc, "PRC_AUC":fine_tune.prc_auc}

    if config['dataset']['task'] == 'regression':
        if config['task_name'] in ['qm7', 'qm8', 'qm9']:
            return {"MAE": fine_tune.mae}
        else:
            return {"RMSE": fine_tune.rmse}