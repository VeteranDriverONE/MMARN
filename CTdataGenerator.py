import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import torch.utils.data as Data
import warnings
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms


def adjustWW(image, width=350, level=40):
    # 腹部窗宽350，窗位40
    v_min = level - (width / 2)
    v_max = level + (width / 2)

    img = image.copy()
    img[image < v_min] = v_min
    img[image > v_max] = v_max

    img = (img - v_min) / (v_max - v_min)
    # img = (img-img.mean()) / img.std()

    return img


class CTdataGenerator(Data.Dataset):
    def __init__(self, root, resize):
        self.root, resize
        
        

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        

        # 返回值自动转换为torch的tensor类型
        return img_arr
    
    def read_dcm(file_path):
        img_names = self.reader.GetGDCMSeriesFileNames(str(self.origin_path / seq / 'Image' / name.split('.')[0]))
        self.reader.SetFileNames(img_names)
        origin = self.reader.Execute()
        self.moving_origin.append(origin)

    def read_nii(file_path):
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))
        return 


