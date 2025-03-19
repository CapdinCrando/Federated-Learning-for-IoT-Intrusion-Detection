import torch
from task_fliir import load_csv_dataset, FLIIRModel
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split, TensorDataset
from pytorch_lightning import LightningModule, Trainer, callbacks, Callback

## Parameters
batch_size = 256            # How many data samples in each training step
learning_rate = 0.0779591   # How fast the model learns
max_epochs = 10             # Number of training epochs
training_patience = 10      # How many epochs to wait for no improvement
num_workers = 2             # Number of worker threads per dataloader
test_train_split = 0.2      # How much of overall data goes to test as opposed to train
train_val_split = 0.8       # How much of train data goes to train as opposed to val
matmul_precision = 'medium' # Matmul precision
dataset_path = 'datasets/WSN-DS.csv'

## Set matmul precision
torch.set_float32_matmul_precision(matmul_precision)


## Main
if __name__ == '__main__':

    # Check for device
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    pre_split_data, pre_split_labels, labels = load_csv_dataset(dataset_path)

    formatted_data = [(x, y.item()) for x, y in zip(pre_split_data, pre_split_labels)]

    # Do test train split
    whole_data_size = len(formatted_data)
    test_data_size = int(test_train_split*whole_data_size)
    pre_train_data_size = whole_data_size - test_data_size
    pre_train_dataset, test_dataset = random_split(formatted_data, [pre_train_data_size, test_data_size])

    # Split train into train and val data
    pre_train_data_size = len(pre_train_dataset)
    train_data_size = int(train_val_split*pre_train_data_size)
    val_data_size = pre_train_data_size - train_data_size
    train_dataset, val_dataset = random_split(pre_train_dataset, [train_data_size, val_data_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        persistent_workers=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True
    )

    # Get number of classes
    num_classes = len(labels)

    # Make model
    model = FLIIRModel(
        num_channels=1,
        num_classes=num_classes, 
        learning_rate=learning_rate
    )

    # Send model to GPU
    model.to(device)

    # Get GPU id
    gpu_id = (1 if torch.cuda.is_available() else 0)

    # Set up early stopping (prevent overfitting)
    early_stopping = callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=training_patience,
        mode='min'
    )

    trainer = Trainer(max_epochs=max_epochs, devices=gpu_id, callbacks=[early_stopping])

    trainer.fit(model, train_loader, val_loader)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    trainer.test(model, test_loader)