"""
    This file defines the ways of fine tuning a pretrained Model which is MobileNet_V2 on custom image datasets to 
    detect drowsiness on the images
"""

import torch
import torch.nn as nn
from torch.optim import SGD, lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
import time
import os
import copy
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter




def checkpoint(model, filepath):
    """Save the model state

    Args:
        model (nn.Module): The pytorch model to be saved
        filepath (str): Filepath where model to be saved to

    """
    torch.save(model.state_dict(), filepath)



def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for 1 epoch
    Args:
        model: nn.Module
        train_loader: train DataLoader
        criterion: callable loss function
        optimizer: pytorch optimizer
        device: torch.device
    Returns
    -------
    Tuple[Float, Float]
        average train loss and average train accuracy for current epoch
    """

    train_losses = []
    train_corrects = []
    model.train()

    # Iterate over data.
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # prediction
        outputs = model(inputs)

        # calculate loss
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        train_losses.append(loss.item())
        train_corrects.append(torch.sum(preds == labels.data).item())

    return sum(train_losses)/len(train_losses), sum(train_corrects)/len(train_loader.dataset)


def val_epoch(model, val_loader, criterion, device):
    """Validate the model for 1 epoch
    Args:
        model: nn.Module
        val_loader: val DataLoader
        criterion: callable loss function
        device: torch.device

    Returns
    -------
    Tuple[Float, Float]
        average val loss and average val accuracy for current epoch
    """

    val_losses = []
    val_corrects = []
    model.eval()

    # Iterate over data
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # prediction
            outputs = model(inputs)

            # calculate loss
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            val_losses.append(loss.item())
            val_corrects.append(torch.sum(preds == labels.data).item())

    return sum(val_losses)/len(val_losses), sum(val_corrects)/len(val_loader.dataset)


def main():
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    
    # dir paths
    DATA_DIR = "./data/"
    MODEL_DIR = './model/'
    LOG_DIR = os.path.join('runs', current_time) 

    # Hyperparameters
    BATCH_SIZE = 4
    NUM_WORKERS = 0
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    
    # constants
    CHECKPOINT_STEPS = 10  # number of epochs after which to checkpoint the model
    TRAIN_SIZE_RATIO = 0.7

    # for logging
    writer = SummaryWriter(LOG_DIR)

    # Define image preprocessing transformations
    preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.Grayscale(3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load data
    dataset = datasets.ImageFolder(DATA_DIR, preprocess)
    classes = dataset.classes 
    # print(f"Classes: {classes}") # Classes: ['awake', 'background', 'drowsy']


    # Random splitting datasets
    train_size = int(len(dataset) * TRAIN_SIZE_RATIO)
    val_size = len(dataset) - train_size
    train_data, val_data = random_split(dataset, [train_size, val_size])
    
    # Dataloaders
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)


    # load pretrained model
    model = models.mobilenet.mobilenet_v2(pretrained=True)

    # freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(num_features, len(classes))  # 2 classess

    # transfer to cuda device if any
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # loss, optimizer and scheduler 
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='max') #StepLR(optimizer, step_size=10, gamma=0.1)


    # START TRAINING MODEL
    best_model_state = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()
    for epoch in range(1, NUM_EPOCHS + 1):
        # train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        message = f'Epoch: {epoch}/{NUM_EPOCHS} \tTrainLoss: {train_loss:.4f} \tTrainAcc: {train_acc:.4f}'
        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_accuracy", train_acc, epoch)


        # validation
        if len(val_data) > 0:  
            val_loss, val_acc = val_epoch(model, val_loader, criterion, device)
            message += f'\tValLoss: {val_loss:.4f} \tValAcc: {val_acc:.4f}'
            writer.add_scalar("val_loss", val_loss, epoch)
            writer.add_scalar("val_accuracy", val_acc, epoch)

            # tracking the best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_model_state = copy.deepcopy(model.state_dict())

        print(message)

        # save model checkpoint for every CHECKPOINT_STEPS
        if epoch % CHECKPOINT_STEPS == 0:
            print('Checkpointing model...')
            checkpoint(model, os.path.join(LOG_DIR, f'model_{epoch}.pt'))

        # schedule lr
        # scheduler.step()
        scheduler.step(val_acc)

    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print("Best val Acc: {:4f}".format(best_acc))
    
    print(f"Saving best model as best_model_{current_time}.pt to {MODEL_DIR}")
    torch.save(best_model_state, os.path.join(LOG_DIR, 'best_model.pt'))
    torch.save(best_model_state, os.path.join(MODEL_DIR, f'best_model_{current_time}.pt'))


if __name__ == "__main__":
    main()
