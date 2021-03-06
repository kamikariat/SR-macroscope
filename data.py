from PIL import Image
import glob
import torch
import torch.utils.data as data
# torch.utils.data.dataset is an abstract class representing a dataset
from torch.utils.data.dataset import Dataset
# https://pytorch.org/docs/stable/_modules/torch/utils/data/dataset.html
import os
import torch
import numpy as np
import pandas as pd
import sys
import csv


'''
Pytorch uses datasets and has a very handy way of creating dataloaders in your main.py
Make sure you read enough documentation.
'''


class DIV2K(Dataset):
    """
    CIFAR dataset
    Implements Dataset (torch.utils.data.dataset)
    """

    def __init__(self, hr_dir, lr_dir, transHR, transLR):
        """
        Args:
            data_dir (string): Directory with all the images
        """
        # gets the data from the directory
        self.lr_image_list = glob.glob(lr_dir + '*')
        self.hr_image_list = glob.glob(hr_dir + '*')

        # calculates the length of image_list
        # print(self.hr_image_list)
        # print(self.lr_image_list)
        assert (len(self.hr_image_list) == len(self.lr_image_list))

        self.data_len = len(self.hr_image_list)

        self.transLR = transLR
        self.transHR = transHR

    def __getitem__(self, index):
        """
        Lazily get the item at the index.

        """
        LR_single_image_path = self.lr_image_list[index]
        HR_single_image_path = self.hr_image_list[index]

        LR_image = Image.open(LR_single_image_path)
        HR_image = Image.open(HR_single_image_path)

        LR_trans = self.transLR(LR_image)
        HR_trans = self.transHR(HR_image)

        return (LR_trans, HR_trans)

        # crop
        #LR_image_cropped = LR_image.crop((0, 0, 256, 256))
        #HR_image_cropped = HR_image.crop((0, 0, 1024, 1024))

        # convert both into np
        #LR_image_np = np.asarray(LR_image_cropped)/255
        #HR_image_np = np.asarray(HR_image_cropped)/255
        # convert both into tensor
        #LR_image_tensor = torch.from_numpy(LR_image_np).float()
        #HR_image_tensor = torch.from_numpy(HR_image_np).float()

        # return (LR, HR)
        #return (LR_image_tensor, HR_image_tensor)

    def __len__(self):
        return self.data_len
