"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

import logging
from collections import OrderedDict
from typing import Any

import csv
import pytorch_lightning as pl
import torch
import numpy as np
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch import nn
from torch.nn import functional as func
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from scipy.io import arff
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)


## Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._classification")

## Class definition
class FLIIRModel(pl.LightningModule):
    '''
    Class for the neural network for project 2.
    '''
    def __init__(self, num_channels=1, num_classes=2, learning_rate=0.001) -> None:
        '''
        Create the building blocks of the neural network.
        '''
        super(FLIIRModel, self).__init__()
        self.learning_rate = learning_rate

        self.pool = nn.MaxPool1d(2)
        self.conv1 = nn.Conv1d(num_channels, 32, 3)
        self.bn1 = nn.BatchNorm1d(32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=64, batch_first=True, num_layers=2, dropout=0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        '''
        Function called when training or inferring.
        Represents the structure of the NN.
        '''
        x = self.pool(func.relu(self.bn1(self.conv1(x))))
        lstm_out, _ = self.lstm(x.permute(0, 2, 1))
        x = lstm_out.mean(dim=1)
        x = self.flatten(x)
        x = func.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, _):
        '''
        Function called by Pytorch Lightning on a training step.

        Calculates loss, as well as training accuracy. Saves both
        to Pytorch Lightning logs for later TensorBoard viewing.
        '''
        data, labels = batch
        outputs = self(data)
        loss = func.cross_entropy(outputs, labels)

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        acc = accuracy_score(labels.cpu().numpy(), preds)
        self.log('train_acc', acc, on_step=True, on_epoch=True)

        return loss
    
    def validation_step(self, batch, _):
        '''
        Function called by Pytorch Lightning on a validation step.

        Calculates loss, as well as training accuracy. Saves both
        to Pytorch Lightning logs for later TensorBoard viewing.
        '''
        data, labels = batch
        outputs = self(data)
        loss = func.cross_entropy(outputs, labels)

        self.log('val_loss', loss, on_step=True, on_epoch=True)
        
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        acc = accuracy_score(labels.cpu().numpy(), preds)
        self.log('val_acc', acc, on_step=True, on_epoch=True)

        return loss
    
    def test_step(self, batch, _):
        '''
        Function called by Pytorch Lightning on a test step.

        Calculates loss, as well as training accuracy. Saves both
        to Pytorch Lightning logs for later TensorBoard viewing.

        Uses balanced accuracy instead of regular accuracy.
        '''
        data, labels = batch
        outputs = self(data)
        loss = func.cross_entropy(outputs, labels)

        self.log('test_loss', loss, on_step=False, on_epoch=True)
        
        preds = torch.argmax(outputs, dim=1)
        acc = balanced_accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
        self.log('test_bal_acc', acc, on_step=False, on_epoch=True)

        return {'test_loss': loss, 'test_bal_acc': acc, 'truth_labels': labels, 'pred_labels': preds}
    
    def configure_optimizers(self):
        '''
        Function called by Pytorch Lightning to configure the optimizer.
        '''
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def apply_transforms(batch):
    """Apply transforms to the partition from FederatedDataset."""
    batch["image"] = [transforms.functional.to_tensor(img) for img in batch["image"]]
    return batch

def load_csv_dataset(file_path, transform=None):

    # Load file
    csv_data = []

    with open(file_path, 'r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            csv_data.append(row)

    # Cut out header
    dataset_data = csv_data[1:]

    # Transpose and cut
    data_transpose = np.transpose(dataset_data)
    data_no_labels, data_transpose_labels = data_transpose[:-1], data_transpose[-1]
    dataset_labels_list = list(set(data_transpose_labels))

    # Encode labels
    encoder = LabelEncoder()
    encoder.fit(dataset_labels_list)
    data_labels_encoded = encoder.transform(data_transpose_labels)

    # Data conversions
    encoded_data_array = np.array(data_no_labels, dtype=np.float32).transpose()
    data_labels_encoded = np.array(data_labels_encoded, dtype=np.int64)

    # Convert to tensors
    x_tensor = torch.tensor(encoded_data_array).unsqueeze(1)
    y_tensor = torch.tensor(data_labels_encoded)

    # Apply transform
    if transform:
        x_tensor = transform(x_tensor)

    # Return
    return x_tensor, y_tensor, data_transpose_labels

fds_dataset = None  # Cache FederatedDataset

def load_data(partition_id, num_partitions):

    # Download latest version of model
    #dataset_path = kagglehub.dataset_download("hassan06/nslkdd")
    dataset_path = 'datasets/'
    
    global fds_dataset
    if fds_dataset is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)

        x_full, y_full, _ = load_csv_dataset(f'{dataset_path}/WSN-DS.csv')
        full_dataset = TensorDataset(x_full, y_full)
        fds_dataset = FederatedDataset(full_dataset, partitioners={"train": partitioner}, balance=True)
    
    partition_full = fds_dataset.load_partition(partition_id, "train")

    partition_test_train = partition_full.train_test_split(test_size=0.2, seed=42)

    partition_train_val = partition_test_train["train"].train_test_split(test_size=0.2, seed=42)

    trainloader = DataLoader(
        partition_train_val["train"],
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
    valloader = DataLoader(
        partition_train_val["test"],
        batch_size=32,
        num_workers=2,
    )
    testloader = DataLoader(partition_test_train["test"], batch_size=32, num_workers=1)
    return trainloader, valloader, testloader
