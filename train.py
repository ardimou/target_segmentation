# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 18:09:59 2023

@author: argdi
"""
import argparse
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from models import get_model_instance_segmentation
import preprocessing
from tqdm.auto import tqdm
import math
from pathlib import Path
import datetime
import json



# # construct an optimizer
# params = [p for p in model.parameters() if p.requires_grad]

# # let's train it for 5 epochs
# num_epochs = 1

# for epoch in range(num_epochs):
#     for batch in train_loader:
#         images, targets = batch
#         images = images.unsqueeze(1).to(device)
#         targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
        
#         optimizer.zero_grad()
#         loss_dict = model(images, targets)
#         losses = sum(loss for loss in loss_dict.values())
   
#         losses.backward()
#         optimizer.step()
        
def run_epoch(model, dataloader, optimizer, lr_scheduler, device, scaler, is_training):
    """
    Function to run a single training or evaluation epoch.
    
    Args:
        model: A PyTorch model to train or evaluate.
        dataloader: A PyTorch DataLoader providing the data.
        optimizer: The optimizer to use for training the model.
        loss_func: The loss function used for training.
        device: The device (CPU or GPU) to run the model on.
        scaler: Gradient scaler for mixed-precision training.
        is_training: Boolean flag indicating whether the model is in training or evaluation mode.
    
    Returns:
        The average loss for the epoch.
    """
    # Set the model to training mode
    model.train()
    
    epoch_loss = 0  # Initialize the total loss for this epoch
    progress_bar = tqdm(total=len(dataloader), desc="Train" if is_training else "Eval")  # Initialize a progress bar
    
    # Loop over the data
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # Move inputs and targets to the specified device
        #inputs = torch.stack(inputs).to(device)
        
        # Forward pass with Automatic Mixed Precision (AMP) context manager
        with torch.autocast(torch.device(device).type):
            #images, targets
            inputs = inputs.unsqueeze(1).to(device)
            targets=[{k: v.to(device) for k,v in t.items()} for t in targets]
            if is_training:
                losses = model(inputs, targets)
            else:
                with torch.no_grad():
                    losses = model(inputs, targets)
        
            # Compute the loss
            loss = sum([loss for loss in losses.values()])  # Sum up the losses

        # If in training mode, backpropagate the error and update the weights
        if is_training:
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                old_scaler = scaler.get_scale()
                scaler.update()
                new_scaler = scaler.get_scale()
                if new_scaler >= old_scaler:
                    lr_scheduler.step()
            else:
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
                
            optimizer.zero_grad()

        # Update the total loss
        loss_item = loss.item()
        epoch_loss += loss_item
        
        # Update the progress bar
        progress_bar_dict = dict(loss=loss_item, avg_loss=epoch_loss/(batch_id+1))
        if is_training:
            progress_bar_dict.update(lr=lr_scheduler.get_last_lr()[0])
        progress_bar.set_postfix(progress_bar_dict)
        progress_bar.update()

        # If the loss is NaN or infinite, stop the training/evaluation process
        if math.isnan(loss_item) or math.isinf(loss_item):
            print(f"Loss is NaN or infinite at batch {batch_id}. Stopping {'training' if is_training else 'evaluation'}.")
            break

    # Cleanup and close the progress bar 
    progress_bar.close()
    
    # Return the average loss for this epoch
    return epoch_loss / (batch_id + 1)

def train_loop(model, 
               train_dataloader, 
               valid_dataloader, 
               optimizer,  
               lr_scheduler, 
               device, 
               epochs, 
               checkpoint_path, 
               use_scaler=False):
    """
    Main training loop.
    
    Args:
        model: A PyTorch model to train.
        train_dataloader: A PyTorch DataLoader providing the training data.
        valid_dataloader: A PyTorch DataLoader providing the validation data.
        optimizer: The optimizer to use for training the model.
        lr_scheduler: The learning rate scheduler.
        device: The device (CPU or GPU) to run the model on.
        epochs: The number of epochs to train for.
        checkpoint_path: The path where to save the best model checkpoint.
        use_scaler: Whether to scale graidents when using a CUDA device
    
    Returns:
        None
    """
    # Initialize a gradient scaler for mixed-precision training if the device is a CUDA GPU
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' and use_scaler else None
    best_loss = float('inf')  # Initialize the best validation loss

    # Loop over the epochs
    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Run a training epoch and get the training loss
        train_loss = run_epoch(model, train_dataloader, optimizer, lr_scheduler, device, scaler, is_training=True)
        # Run an evaluation epoch and get the validation loss
        with torch.no_grad():
            valid_loss = run_epoch(model, valid_dataloader, None, None, device, scaler, is_training=False)

        # If the validation loss is lower than the best validation loss seen so far, save the model checkpoint
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), f"{checkpoint_path}/model.pth")

            # Save metadata about the training process
            training_metadata = {
                'epoch': epoch,
                'train_loss': train_loss,
                'valid_loss': valid_loss, 
                'learning_rate': lr_scheduler.get_last_lr()[0],
                #'model_architecture': model.name
            }
            with open(Path(f"{checkpoint_path}/training_metadata.json"), 'w') as f:
                json.dump(training_metadata, f)

        # If the training or validation loss is NaN or infinite, stop the training process
        if any(math.isnan(loss) or math.isinf(loss) for loss in [train_loss, valid_loss]):
            print(f"Loss is NaN or infinite at epoch {epoch}. Stopping training.")
            break

    # If the device is a GPU, empty the cache
    if device.type != 'cpu':
        getattr(torch, device.type).empty_cache()

def get_args():
    parser = argparse.ArgumentParser(description='Train the mask RCNN on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        help='Learning rate', dest='lr')
    #parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    #parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    #parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        #help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    #parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()
        
if __name__ == '__main__':
    
    args = get_args()
        # Generate timestamp for the training session (Year-Month-Day_Hour_Minute_Second)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create a directory to store the checkpoints if it does not already exist
    checkpoint_dir = Path(f"./{timestamp}")
    
    # Create the checkpoint directory if it does not already exist
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    folder_dir = '../hybrid-pushing/image_data'
    # our dataset has two classes only - background and person
    num_classes = args.classes
    model = get_model_instance_segmentation(num_classes)
    # use our dataset and defined transformations
    dataset = preprocessing.Preprocess(folder_dir, 0, 200)
    # move model to the right device
    model.to(device)
    # Define the sizes for training and validation sets
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Use random_split to split the dataset
    train_set, val_set = random_split(dataset, [train_size, val_size])
    batch_size = 4
    # 3. Create data loaders
    loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args, collate_fn=preprocessing.collate_fn)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args, collate_fn=preprocessing.collate_fn)
    # The model checkpoint path
    checkpoint_path = f"{checkpoint_dir}"
    
    print(checkpoint_path)
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)
    # define training and validation data loaders
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                   max_lr=args.lr, 
                                                   total_steps=args.epochs*len(train_loader))
    train_loop(model=model, 
           train_dataloader=train_loader,
           valid_dataloader=val_loader,
           optimizer=optimizer, 
           lr_scheduler=lr_scheduler, 
           device=torch.device(device), 
           epochs=args.epochs, 
           checkpoint_path=checkpoint_path,
           use_scaler=True)

    
    
    
        
