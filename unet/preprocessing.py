 # -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 17:40:33 2024

@author: argdi
"""


import pickle 
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.ops.boxes import masks_to_boxes
#from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
import cv2

def collate_fn(batch):
    imgs, targets = zip(*batch)
    return torch.stack(imgs, 0), targets

class Preprocess(Dataset):
    def __init__(self, folder_dir, ep_start, ep_end, save_file=False):
        
        self.folder_dir = folder_dir
        #self.seg_files = [file for file in os.listdir(self.folder_dir)]
        self.seg_files = [f'{i}.pkl' for i in range(ep_start, ep_end)]
        
    def __len__(self):
        return len(self.seg_files)
              
    def segm_masks(self, seg):
        
        objects = np.unique(seg)[1:]
        masks = np.zeros((len(objects), *seg.shape), dtype = np.uint8)
        
        segPross = np.zeros((seg.shape[0], seg.shape[1]), dtype = np.uint8)

        
        for ind, obj in enumerate(objects):
            mask = (seg == obj)
            segPross[mask] = ind + 1
        
        return segPross

    def rgb_images(self, rgb):
        rgb_image = (rgb/255.0)
        return np.float32(rgb_image)
    
    def heightmap_images(self, heightmap):
        epsilon = 1e-8
        heightmap_image = (heightmap - np.mean(heightmap))/(np.std(heightmap) + epsilon)
        
        return heightmap_image
    
    def load(self, dim = 200, nr_objects = 6, start_idx = 0, batch = 5000):
        image_files = ([file for file in os.listdir(self.folder_dir)])
        N = batch
        
        segV = np.zeros((N, nr_objects, dim, dim), dtype = np.uint8)
        rgbV = np.zeros((N, 3, dim, dim), dtype = np.float32)
        heightmapV = np.zeros((N, dim, dim),  dtype = np.float32)
        for indN in range(start_idx, start_idx + batch):
            with open(f'{self.folder_dir}/{image_files[indN]}', 'rb') as file:
               data =  pickle.load(file)
            
            #seg, rgb, heightmap = data['seg'], data['rgb'], data['heightmap']
            
            rgbV[indN, :, :, :] = np.transpose(self.rgb_images(data['rgb']), (2, 0, 1))
            #segV[indN, :, :, :] = self.segm_masks(seg)
            segm_masks = self.segm_masks(data['seg'])            
            segV[indN, :segm_masks.shape[0], :, :] = segm_masks     
            heightmapV[indN, :, :] = self.heightmap_images(data['heightmap'])
            
        
        input_data = torch.cat((torch.from_numpy(rgbV), torch.from_numpy(heightmapV).unsqueeze(0)), dim=0)
        output_data = torch.from_numpy(segV)
        #print(f'input: {input_data.shape}')
        #print(f'output: {output_data.shape}')
        
        return [input_data, output_data]
    
    def __getitem__(self, ind):
        #image_files = ([file for file in os.listdir(self.folder_dir)])
        dim, nr_objects = 200, 6
        #print(ind)
        with open(f'{self.folder_dir}/{self.seg_files[ind]}', 'rb') as file:
            data = pickle.load(file)            
            
        #seg1 = self.preprocess_seg(data['S_t1'], data['target_t1'])
        seg1 = torch.zeros((200, 200), dtype=torch.uint8)
        seg2 = torch.zeros((200, 200), dtype=torch.uint8)
        
        seg1[data['S_t1']!=0] = 1
        #seg2 = data['S_t2']
        seg2[data['S_t2']!=0] = 1
        
        
        push = data['push'].astype(np.int16)
        target1 = data['target_t1']
        target1[target1!=0] = 1
        
        target2 = data['target_t2']
        target2[target2!=0] = 1
         
        width = 5
        push_img = self.draw_line(push[0,:-1], push[1,:-1], width, seg1.shape)
        input_data = torch.cat(((seg1).unsqueeze(0), (seg2).unsqueeze(0), torch.from_numpy(target1).unsqueeze(0), torch.from_numpy(push_img).unsqueeze(0)), dim=0)
        output_data = torch.from_numpy(target2)
        
        return input_data, output_data
    
    def preprocess_seg(self, seg, target):
        """
        create a segmentation matrix, where every mask that belongs to the target 
        is equal to 0.5, or 1 if it belongs to any other object
        """
        
        final_seg = np.zeros(seg.shape)
        
        final_seg[seg!=0] = 0.5
        final_seg[target!=0] = 1
        
        return final_seg

    def process_push(self, p1p2):
        width = 2
        image_size = (200, 200)
        push = self.scale_value(p1p2, -0.25, 0.25, 0, 199)
        push_img = self.draw_line(push[0], push[1], width, image_size)
        
        return push_img       
        
    def scale_value(self, value, min_original, max_original, min_scaled, max_scaled):
        scaled_value = ((value - min_original) / (max_original - min_original)) * (max_scaled - min_scaled) + min_scaled
        return scaled_value
    
    def draw_line(self, p1, p2, width, image_size):
        img = np.zeros(image_size, dtype=np.uint8)

        # Bresenham's line algorithm
        x1, y1 = p1
        x2, y2 = p2
        dx = np.abs(x2 - x1)
        dy = np.abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            # Set pixel value to 1 within the line width
            img[max(0, y1 - width // 2): min(image_size[0], y1 + (width + 1) // 2),
                max(0, x1 - width // 2): min(image_size[1], x1 + (width + 1) // 2)] = 1

            if x1 == x2 and y1 == y2:
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy

        return img
    
    
        
    
 
if __name__ == '__main__':

    dataset = Preprocess('../dataset', 0, 100)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    cnt = 0
    # for batch in dataloader:
    #     input_data, output_data = batch
    #     cnt += 1
    #     print(cnt*256/13187)
            
        
    for batch_id, (inputs, targets) in enumerate(dataloader):
        
        cnt += 1
        print(cnt*256/13187)
        
        
