# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import PIL.Image as Image
from data_aug import *
import cv2
import numpy as np
import glob
class dataset(data.Dataset):
    def __init__(self,lcz42_data, transforms=None,test_mode=False):
        self.transforms = transforms
        self.test_mode =test_mode
        if not self.test_mode:
            self.label_data = lcz42_data[:, -17:]
            self.s2_data = lcz42_data[:, :-17]
        else:
            self.s2_data = lcz42_data[:, :]
    def __len__(self):
        return self.s2_data.shape[0]

    def __getitem__(self, item):

        img = self.s2_data[item]
        img=np.reshape(img,(32,32,10))

        if self.transforms is not None:
            img = self.transforms(img)
        if not self.test_mode:
            label = self.label_data[item]
            label =np.argmax(label)
        else:
            label=0
        # img=img/255
        return torch.from_numpy(img).permute(2,0,1).float(), label

def collate_fn(batch):
    imgs = []
    label = []

    for sample in batch:
        imgs.append(sample[0])
        label.append(sample[1])

    return torch.stack(imgs, 0), \
           label

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import pandas as pd
    import sys
    from sklearn.model_selection import train_test_split
    reload(sys)
    import h5py
    sys.setdefaultencoding('utf8')
    data_path="/home/detao/Desktop/pytorch_classification/data_shuffle.npy"
    data_all = np.load(data_path)
    train_data = data_all[:49000, 8192:]
    val_data = data_all[49000:, 8192:]

    print('The shape of train is ', train_data.shape)
    print('The shape of vali is ', val_data.shape)
    val_label =val_data[:,-17:]
    jl = np.sum(val_label,axis=0)
    print(jl)
    # data_set = {}
    # data_set['val'] = dataset(val_data,transforms=None)
    # print(len(data_set['val']))
    # for data in data_set["val"]:
    #     image, label =data
    #     print(label)
    #     print(image.size())
