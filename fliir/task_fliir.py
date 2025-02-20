"""pytorchlightning_example: A Flower / PyTorch Lightning app."""

import logging
from collections import OrderedDict
from typing import Any

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
import warnings
import kagglehub

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
        self.conv1 = nn.Conv1d(num_channels, 32, 5)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.bn2 = nn.BatchNorm1d(64)
        self.lstm = nn.LSTM(input_size=64, hidden_size=64, batch_first=True, num_layers=2, dropout=0.2)
        self.conv3 = nn.Conv1d(64, 64, 3)
        self.conv4 = nn.Conv1d(64, 64, 3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        '''
        Function called when training or inferring.
        Represents the structure of the NN.
        '''
        x = self.pool(func.relu(self.bn1(self.conv1(x))))

        x = self.pool(func.relu(self.bn2(self.conv2(x))))

        lstm_out, _ = self.lstm(x.permute(0, 2, 1))
        x = lstm_out.mean(dim=1)
        x = self.pool(func.relu(self.conv3(x)))
        x = self.pool(func.relu(self.conv4(x)))
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

def load_arff_file(file_path, transform=None):
    data, meta = arff.loadarff(file_path)
    np_data = np.array(data.tolist(), dtype=np.float32)
    X, y = np_data[:, :-1], np_data[:, -1]
    y = y.astype(np.int64)  # Convert to integer labels if needed
    X_tensor = torch.tensor(X)
    if transform:
        X_tensor = transform(X_tensor)
    return X_tensor, torch.tensor(y)

fds_train = None  # Cache FederatedDataset
fds_test = None  # Cache FederatedDataset


def load_data(partition_id, num_partitions):

    # Download latest version of model
    #dataset_path = kagglehub.dataset_download("hassan06/nslkdd")
    dataset_path = 'datasets/'
    
    global fds_train
    if fds_train is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)

        x_train, y_train = load_arff_file(f'{dataset_path}/KDDTrain+.arff')
        train_dataset = TensorDataset(x_train, y_train)
        fds_train = FederatedDataset(train_dataset, partitioners={"train": partitioner}, balance=True)
    
    partition_train = fds_train.load_partition(partition_id, "train")

    global fds_test
    if fds_test is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)

        x_test, y_test = load_arff_file(f'{dataset_path}/KDDTest+.arff')
        test_dataset = TensorDataset(x_test, y_test)
        fds_test = FederatedDataset(test_dataset, partitioners={"train": partitioner}, balance=True)
    
    partition_test = fds_test.load_partition(partition_id, "train")

    partition_train_valid = partition_train.train_test_split(test_size=0.2, seed=42)
    trainloader = DataLoader(
        partition_train_valid["train"],
        batch_size=32,
        shuffle=True,
        num_workers=2,
    )
    valloader = DataLoader(
        partition_train_valid["test"],
        batch_size=32,
        num_workers=2,
    )
    testloader = DataLoader(partition_test["test"], batch_size=32, num_workers=1)
    return trainloader, valloader, testloader
