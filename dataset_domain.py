import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import SimpleITK as sitk
from skimage.measure import label, regionprops
import math
import pdb

class CMRDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', domain='', crop_size=256, scale=0.1, rotate=10, debug=False):

        self.mode = mode
        self.dataset_dir = dataset_dir
        self.crop_size = crop_size
        self.scale = scale
        self.rotate = rotate

        if self.mode == 'train':
            pre_face = 'Training'

        elif self.mode == 'test':
            if 'A' in domain:
                pre_face = 'Testing/A' 
            elif 'B' in domain:
                pre_face = 'Testing/B'
            elif 'C' in domain:
                pre_face = 'Testing/C'

        else:
            print('Wrong mode')
            raise StandardError
        if debug:
            # validation set is the smallest, need the shortest time for load data.
           pre_face = 'Testing'

        path = self.dataset_dir + pre_face + '/'
        print('start loading data')
        
        name_list = []

        df = pd.read_csv(path+"name.csv")
        for index, row in df.iterrows():
            name = row["vendor"] + '_' + row["image_name"] + '_' + str(row["slice"]).zfill(3) + ".tiff"
            name_list.append(name)
        img_list = []
        lab_list = []
        spacing_list = []
        for name in name_list:
            # print(name)
            itk_img = sitk.ReadImage(self.dataset_dir+"pre_processed/oct_imgs/"+name)
            itk_lab = sitk.ReadImage(self.dataset_dir+"pre_processed/oct_masks/"+name)
            spacing = np.array(itk_lab.GetSpacing()).tolist()
            spacing_list.append(spacing[::-1])

            assert itk_img.GetSize() == itk_lab.GetSize()
            img = sitk.GetArrayFromImage(itk_img)
            lab = sitk.GetArrayFromImage(itk_lab)
            img_t = torch.from_numpy(img).float()
            lab_t = torch.from_numpy(lab).short()
            img_list.append(img_t)
            lab_list.append(lab_t)
        self.img_slice_list = img_list
        self.lab_slice_list = lab_list
        self.spacing_list = spacing_list
        self.name_list = name_list

        print('load done, length of dataset:', len(self.spacing_list))
        
    def __len__(self):
        return len(self.spacing_list)


    def __getitem__(self, idx):
        # name = self.name_list[idx]
        # itk_img = sitk.ReadImage(self.dataset_dir+"pre_processed/oct_imgs/"+name)
        # itk_lab = sitk.ReadImage(self.dataset_dir+"pre_processed/oct_masks/"+name)
        # img = sitk.GetArrayFromImage(itk_img)
        # lab = sitk.GetArrayFromImage(itk_lab)
        # tensor_image = torch.from_numpy(img).float()
        # tensor_label = torch.from_numpy(lab).long()
        tensor_image = self.img_slice_list[idx]
        tensor_label = self.lab_slice_list[idx]
        # 裁剪图片，使用中间的图像进行训练
        if self.mode == 'train':
            # 增加两个维度
            tensor_image = tensor_image.unsqueeze(0).unsqueeze(0)
            tensor_label = tensor_label.unsqueeze(0).unsqueeze(0)
            
            # Gaussian Noise 增加了一些随机噪声
            tensor_image += torch.randn(tensor_image.shape) * 0.02
            # Additive brightness 增加了随机亮度
            rnd_bn = np.random.normal(0, 0.7)#0.03
            tensor_image += rnd_bn
            # gamma
            minm = tensor_image.min()
            rng = tensor_image.max() - minm
            gamma = np.random.uniform(0.5, 1.6)
            tensor_image = torch.pow((tensor_image-minm)/rng, gamma)*rng + minm

            tensor_image, tensor_label = self.random_zoom_rotate(tensor_image, tensor_label)
            tensor_image, tensor_label = self.randcrop(tensor_image, tensor_label)
        else:
            tensor_image, tensor_label = self.center_crop(tensor_image, tensor_label)
        
        assert tensor_image.shape == tensor_label.shape
        
        if self.mode == 'train':
            return tensor_image, tensor_label
        else:
            return tensor_image, tensor_label
            # return tensor_image, tensor_label, np.array(self.spacing_list[idx])

    def randcrop(self, img, label):
        _, _, H, W = img.shape
        
        # center_H = H - 400
        # diff_H = 400 - self.crop_size
        diff_H = H - self.crop_size
        diff_W = W - self.crop_size
        
        rand_x = np.random.randint(0, diff_H) # + (center_H >> 1)
        rand_y = np.random.randint(0, diff_W)
        
        croped_img = img[0, :, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]
        croped_lab = label[0, :, rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]

        return croped_img, croped_lab


    def center_crop(self, img, label):
        croped_img = torch.zeros([1,256,256]).to(img.device)
        croped_lab = torch.zeros([1,256,256]).to(img.device)
        H, W = img.shape
        
        diff_H = H - self.crop_size
        diff_W = W - self.crop_size
        
        rand_x = diff_H // 2
        rand_y = diff_W // 2
        
        croped_img[0,...] = img[rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]
        croped_lab[0,...] = label[rand_x:rand_x+self.crop_size, rand_y:rand_y+self.crop_size]

        return croped_img, croped_lab

    def random_zoom_rotate(self, img, label):
        scale_x = np.random.random() * 2 * self.scale + (1 - self.scale)
        scale_y = np.random.random() * 2 * self.scale + (1 - self.scale)


        theta_scale = torch.tensor([[scale_x, 0, 0],
                                    [0, scale_y, 0],
                                    [0, 0, 1]]).float()
        angle = (float(np.random.randint(-self.rotate, self.rotate)) / 180.) * math.pi

        theta_rotate = torch.tensor( [  [math.cos(angle), -math.sin(angle), 0], 
                                        [math.sin(angle), math.cos(angle), 0], 
                                        ]).float()
        
    
        theta_rotate = theta_rotate.unsqueeze(0)
        grid = F.affine_grid(theta_rotate, img.size())
        img = F.grid_sample(img, grid, mode='bilinear')
        label = F.grid_sample(label.float(), grid, mode='nearest').long()
    
        return img, label


