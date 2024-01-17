# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 14:48:03 2023

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
        self.image_files_temp = [file for file in os.listdir(self.folder_dir)]
        self.image_files = []
        ann = []
        for ep in range(ep_start, ep_end):
            for step in range(0, 31):
                if f'ep{ep}_{step}.pkl' in self.image_files_temp:
                    self.image_files.append(f'ep{ep}_{step}.pkl')
                    if save_file:
                        ann.append({'file_name': f'{folder_dir}/ep{ep}_{step}',
                        'video': ep,
                        'frame': step})
        if save_file:
            with open(f'{save_file}/valid.pkl', 'wb') as file:
                pickle.dump(ann, file)
        
    def __len__(self):
        return len(self.image_files)
              
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
            
        
        input_data = torch.cat((torch.from_numpy(rgbV), torch.from_numpy(heightmapV).unsqueeze(1)), dim=1)
        output_data = torch.from_numpy(segV)
        #print(f'input: {input_data.shape}')
        #print(f'output: {output_data.shape}')
        
        return [input_data, output_data]
    
    def __getitem__(self, ind):
        #image_files = ([file for file in os.listdir(self.folder_dir)])
        dim, nr_objects = 200, 6
        segm_masks = np.zeros((dim, dim), dtype = np.uint8)
        heightmapV1 = np.zeros((dim, dim),  dtype = np.float32)
        heightmapV2 = np.zeros((dim, dim),  dtype = np.float32)
        
        with open(f'{self.folder_dir}/{self.image_files[ind]}', 'rb') as file:
            data2 =  pickle.load(file)
        
        heightmapV2 = self.heightmap_images(data2['heightmap'])
        segm_masks = self.segm_masks(data2['seg']) 
        
        obj_ids = np.unique(segm_masks)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        num_objs = len(obj_ids)
        
        masks = (segm_masks == obj_ids[:, None, None]).astype(np.uint8)
        boxes = torch.zeros([num_objs, 4], dtype=torch.float32)
        for i in range(num_objs):
            x,y,w,h = cv2.boundingRect(masks[i])
            boxes[i] = torch.tensor([x, y, x+w, y+h])
        
        img = torch.as_tensor(heightmapV2, dtype=torch.float32)

        data = {}
        data["boxes"] =  boxes
        data["labels"] =  torch.ones((num_objs,), dtype=torch.int64)   
        data["masks"] = torch.from_numpy(masks)
        
        return img, data
    
def create_dataset(folder_dir, anns_file, save_folder):
    seg_files_temp = [file for file in os.listdir(folder_dir)]
    with open(f'{folder_dir}/{anns_file}.pkl', 'rb') as file:
        anns = pickle.load(file)
    seg_files = []
    image_files = []
    data = {}
    os.makedirs(save_folder, exist_ok = True)
    file_name = 0 
    for ind in range(len(seg_files_temp) - 2):
        item1 = anns[ind]
        item2 = anns[ind + 1]
        if item2['video'] == item1['video']:
            if item1['frame'] == 0:
                obj_ids = 6
            #self.seg_files.append([f'{folder_dir}/S_{ind}.pkl', f'{self.folder_dir}/S_{ind + 1}.pkl', anns[ind]])
            with open(f'{folder_dir}/S_{ind}.pkl', 'rb') as file:
                S_t1 = pickle.load(file)
            data['S_t1'] =  S_t1
            #print(S_t1.shape)
            
            with open(f'{folder_dir}/S_{ind + 1}.pkl', 'rb') as file:
                S_t2 = pickle.load(file)
            data['S_t2'] =  S_t2
            #print(S_t2.shape)
            file_name_t = item1['file_name']
            file_name_t2 = item2['file_name']
            
            with open(f'{file_name_t}.pkl', 'rb') as file:
                original_data_t = pickle.load(file)
                
            with open(f'{file_name_t2}.pkl', 'rb') as file:
                original_data_t2 = pickle.load(file)
                
            
            ids_t = len(np.unique(S_t1)) - 1
            ids_t2 = len(np.unique(S_t2)) - 1
            
            if ids_t2 < obj_ids:
                obj_ids = ids_t2
                continue
            
            data['push'] = original_data_t['p1_p2']
            data['target_t1'] = original_data_t['target_mask']
            data['target_t2'] = original_data_t2['target_mask']
            
            with open(f'{save_folder}/{file_name}.pkl', 'wb') as file: 
                pickle.dump(data, file)
            file_name += 1

def visualize_samples(folder, item):
    with open(f'{folder}/{item}.pkl', 'rb') as file:
      data = pickle.load(file)
    
    plt.imshow(data['S_t1'])
    plt.title('S_t')
    plt.show()   
    
    plt.imshow(data['target_t1'])
    plt.title('M_t')
    plt.show()
    
    plt.imshow(data['S_t2'])
    plt.title('S_t+1')
    plt.show()   
    
    plt.imshow(data['target_t2'])
    plt.title('M_t+1')
    plt.show()
    
    
    
    
            
if __name__ == '__main__':   
        
    #create_dataset('./seg_pred', 'valid', './dataset')   
    visualize_samples(your_folder, item)
