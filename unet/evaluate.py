import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from utils.utils import *
import pickle
from utils.dice_score import multiclass_dice_coeff, dice_coeff


@torch.inference_mode()
def evaluate(net, dataloader, device, amp, iou, viz_file):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iouVector = torch.zeros(net.n_classes)
    global_step = 0
    # iterate over the validation set
    cnt = 0
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true = batch
            
            
            #image = image.unsqueeze(1)
            # # move images and labels to correct device and type
            # image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            # mask_true = mask_true.to(device=device, dtype=torch.uint8)

            # # predict the mask
            # mask_pred = net(image)
             
            # mask_pred = F.softmax(mask_pred, dim=1).float()
            # mask_pred = (mask_pred > 0.5).float()
            # dice_score += multiclass_dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            
            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            if net.n_classes == 1:
                assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                #print(f'pred: {mask_pred.shape}, true: {mask_true.shape}')
                dice_score += dice_coeff(mask_pred.squeeze(1), mask_true, reduce_batch_first=False)
                if viz_file!=None:
                    if global_step%20 == 0:
                      viz_img = mask_true[0]
                      viz_pred = mask_pred[0]
                      #with open(f'{viz_file}/pkl_{cnt}.pkl', 'wb') as file:
                          #pickle.dump([mask_true, mask_pred], file)
                      plot_img_and_mask(viz_img.cpu().numpy(), viz_pred.cpu().numpy().squeeze(0), f'{viz_file}/{cnt}', True)
                      cnt += 1
                    global_step+=1
                if iou:
                    iouVector += calculate_iou_per_class(mask_pred.squeeze(1), mask_true, net.n_classes)
            else:
                assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)
                
                if viz_file!=None:
                    if global_step % 40 == 0:
                      viz_img = mask_true[0].argmax(dim=0)
                      viz_pred = mask_pred[0].argmax(dim=0)
                      #with open(f'{viz_file}/pkl_{cnt}.pkl', 'wb') as file:
                          #pickle.dump([mask_true, mask_pred], file)
                      plot_img_and_mask(viz_img.cpu().numpy(), viz_pred.cpu().numpy(), f'{viz_file}/{cnt}')
                      cnt += 1
                    global_step+=1
                if iou:
                    iouVector += calculate_iou_per_class(mask_pred, mask_true, net.n_classes)      
    net.train()
    if iou == False:
        return dice_score / max(num_val_batches, 1)
    else:       
        return dice_score / max(num_val_batches, 1), iouVector/max(num_val_batches, 1)
    
        


def calculate_iou_per_class(predicted_masks, true_masks, num_classes):
    """
    Calculate Intersection over Union (IoU) for each class.

    Parameters:
    - predicted_masks (torch.Tensor): Predicted segmentation masks with shape (batch_size, num_classes, H, W).
    - true_masks (torch.Tensor): Ground truth segmentation masks with shape (batch_size, num_classes, H, W).
    - num_classes (int): Number of classes.

    Returns:
    - iou_per_class (torch.Tensor): IoU for each class with shape (num_classes,).
    """
    iou_per_class = torch.zeros(num_classes)
    if num_classes == 1:
      predicted_masks = predicted_masks.unsqueeze(1)
      true_masks = true_masks.unsqueeze(1)

    for i in range(num_classes):
        # Compute True Positive (TP), False Positive (FP), and False Negative (FN)
        tp = torch.sum((predicted_masks[:, i, :, :] == 1) & (true_masks[:, i, :, :] == 1)).float()
        fp = torch.sum((predicted_masks[:, i, :, :] == 1) & (true_masks[:, i, :, :] == 0)).float()
        fn = torch.sum((predicted_masks[:, i, :, :] == 0) & (true_masks[:, i, :, :] == 1)).float()

        # Calculate IoU for each class
        iouV = tp / (tp + fp + fn + 1e-6)  # Add a small epsilon to avoid division by zero
        iou_per_class[i] = iouV

    return iou_per_class