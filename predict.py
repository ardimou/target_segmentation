# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 13:00:07 2023

@author: argdi
"""
import argparse
import torch
from torch.utils.data import DataLoader, Dataset
import preprocessing
from tqdm.auto import tqdm
import math
from pathlib import Path
import datetime
import json
from models import get_model_instance_segmentation
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle

def predict(model, dataloader, batch_size, save_folder):
    model.eval()
    cnt = 0
    iou = 0
    progress_bar = tqdm(total=len(dataloader), desc="Test")  # Initialize a progress bar
    
    for batch_id, (inputs, targets) in enumerate(dataloader):
        # Move inputs and targets to the specified device
        #inputs = torch.stack(inputs).to(device)
        # Forward pass with Automatic Mixed Precision (AMP) context manager
       
        with torch.autocast(torch.device(device).type):
            #images, targets
            inputs = inputs.unsqueeze(1).to(device)      
            
            with torch.no_grad():
                predictions = model(inputs)  
                
                pred_masks, _ = get_outputs(predictions, 0.8)
                save_seg(pred_masks, batch_id, batch_size, save_folder)
                input_masks = postprocess_input_target(targets)
                iou += calculate_iou(input_masks, pred_masks)
                
                #if batch_id % 50 == 0: 
                    #plot(input_masks[0], pred_masks[0], f"./images/{batch_id}")   
        progress_bar.update()   
                
    iou /= batch_id + 1
    print(f"iou = {iou}")

def calculate_iou(matrix1, matrix2):
    iou = 0
    for i in range(matrix1.shape[0]):
         
        intersection = np.sum((matrix1[i] > 0) & (matrix1[i] == matrix2[i]))
        union = np.sum((matrix1[i] > 0) | (matrix2[i] > 0))
    
        #iou += np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0.0
        iou += intersection / union if union>0 else 0
    return iou/matrix1.shape[0]  

def plot(target_mask, pred_mask, output_file):
    fig, ax = plt.subplots(1, 3, figsize = (10, 8))  
    #plt.axis('off')
    
    ax[0].set_title('target mask') 
    ax[0].imshow(target_mask)  
    ax[0].axis("off")
    
    ax[1].set_title('prediction mask') 
    ax[1].imshow(pred_mask)   
    ax[1].axis("off")
    
    ax[2].set_title('intersection') 
    ax[2].imshow((target_mask > 0) & (target_mask == pred_mask))   
    ax[2].axis("off")
    
    #os.makedirs(f"{output_file}/..", exist_ok=True)
    fig.savefig(f"{output_file}.png")
    
def save_seg(pred_masks, batch_id, batch_len, save_folder):
    
    #save_folder = './seg_pred'
    #os.mkdir(save_folder, exist_ok=True)
    for i in range(len(pred_masks)):
        gl_id = batch_id * batch_len + i
        with open(f'{save_folder}/S_{gl_id}.pkl', 'wb') as file:
            pickle.dump(pred_masks[i], file)
    
    
            
def postprocess_input_target(targets):
    input_masks = np.zeros((len(targets), 200, 200), dtype = np.uint8)
    for i in range(len(targets)):
        masks = targets[i]['masks']
        sorted_masks = sort_masks(masks)
        input_masks[i, :, :] = one_hot_to_mask(sorted_masks)
    
    return input_masks
        
def one_hot_to_mask(mask):
    
    out_mask = np.zeros((mask.shape[1], mask.shape[2]), dtype = np.uint8)
    for i in range(mask.shape[0]):
        ind = mask[i] > 0
        out_mask[ind] = i + 1
    
    return out_mask

def sort_masks(mask):
    objects = list(range(mask.shape[0]))
    masks = np.zeros(mask.shape, dtype = np.uint8)
    distances = np.zeros(len(objects))
    
    x, y = np.indices((mask.shape[1], mask.shape[2]), dtype=np.uint8)
    
    for ind, obj in enumerate(objects):
        mask_ind = mask[ind] > 0 
        #masks[ind, :, :] = mask_ind.astype(np.int8)
    
        # Calculate distances from the center of mass for sorting
        distances[ind] = np.sqrt(np.mean(x[mask_ind])**2 + np.mean(y[mask_ind])**2)
    
    # Sort objects based on distances
    sorted_indices = np.argsort(distances)
    #objects_sorted = objects[sorted_indices]
    
    # Accessing masks using sorted order
    sorted_masks = mask[sorted_indices, :, :]
    return sorted_masks
        
# def get_outputs(outputs, threshold):
    
#     final_masks = np.zeros((len(outputs), 200, 200), np.uint8)
#     # get all the scores
#     for i in range(len(outputs)):
#         scores = list(outputs[i]['scores'].detach().cpu().numpy())
#         # index of those scores which are above a certain threshold
#         thresholded_preds_indices = [scores.index(i) for i in scores if i > threshold]
#         thresholded_preds_count = len(thresholded_preds_indices)
#         # get the masks
#         masks = (outputs[i]['masks']>0.5).squeeze(1).detach().cpu().numpy()
#         # discard masks for objects which are below threshold
#         masks = masks[thresholded_preds_indices]
#         mask_sorted = sort_masks(masks)
#         final_masks[i, :, :] = one_hot_to_mask(mask_sorted)
#         # get the bounding boxes, in (x1, y1), (x2, y2) format
#         boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))]  for i in outputs[i]['boxes'].detach().cpu()]
#         # discard bounding boxes below threshold value
#         boxes = boxes[thresholded_preds_indices]
#         # get the classes labels
#         #labels = [coco_names[i] for i in outputs[0]['labels']]
#     return final_masks, boxes

def get_outputs(outputs, threshold):
    
    final_masks = np.zeros((len(outputs), 200, 200), np.uint8)
    # get all the scores
    for i in range(len(outputs)):
        scores = list(outputs[i]['scores'].detach().cpu().numpy())
        # index of those scores which are above a certain threshold
        thresholded_preds_indices = np.where(np.array(scores) > threshold)[0]
        #print(thresholded_preds_indices)
        thresholded_preds_count = len(thresholded_preds_indices)
        # get the masks
        masks = (outputs[i]['masks']>0.5).squeeze(1).detach().cpu().numpy()
        #print(masks.shape)
        # discard masks for objects which are below threshold
        #masks = masks[:thresholded_preds_count]
        masks = masks[thresholded_preds_indices]
        #print(f'after : {masks.shape}')
        mask_sorted = sort_masks(masks)
        final_masks[i, :, :] = one_hot_to_mask(mask_sorted)
        # get the bounding boxes, in (x1, y1), (x2, y2) format
        boxes = np.array([(int(j[0]), int(j[1]), int(j[2]), int(j[3])) for j in outputs[i]['boxes'].detach().cpu()])
        # discard bounding boxes below threshold value
        #boxes = boxes[:thresholded_preds_count]
        boxes = boxes[thresholded_preds_indices]
        # get the classes labels
        #labels = [coco_names[i] for i in outputs[0]['labels']]
    return final_masks, boxes
        
def get_args():
    parser = argparse.ArgumentParser(description='test the mask RCNN on images and target masks')
    #parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    #parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-5,
                        #help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    #parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    #parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        #help='Percent of the data that is used as validation (0-100)')
    #parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    #parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    
    folder_dir = '../hybrid-pushing/push_data'
    save_folder = './seg_pred'
    os.makedirs(save_folder, exist_ok=True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    dataset = preprocessing.Preprocess(folder_dir, 300, 999, save_folder)   
    
    num_classes = args.classes
    model = get_model_instance_segmentation(num_classes)
  
    if args.load:
        state_dict = torch.load(args.load, map_location=device)
        model.load_state_dict(state_dict)
        #logging.info(f'Model loaded from {args.load}')
        
    # move model to the right device
    model.to(device)
    
    loader_args = dict(batch_size=args.batch_size, num_workers=0, pin_memory=True)
    val_loader = DataLoader(dataset, shuffle=False, drop_last=True, **loader_args, collate_fn=preprocessing.collate_fn)
    
    predict(model, val_loader, args.batch_size, save_folder)