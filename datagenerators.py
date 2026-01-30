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
from collections import defaultdict
'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''

class Dataset(Data.Dataset):
    def __init__(self, files):
        # 初始化
        self.files = files

    def __len__(self):
        # 返回数据集的大小
        return len(self.files)

    def __getitem__(self, index):
        # 索引数据集中的某个数据，还可以对数据进行预处理
        # 下标index参数是必须有的，名字任意
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(self.files[index]))[np.newaxis, ...]
        # 返回值自动转换为torch的tensor类型
        return img_arr

class AbdomenDataset1(Data.Dataset):
    def __init__(self, path, fixed_seqs=['CMP','UP','NP'], moving_seqs=['CMP','UP','NP'],same_skip=False):
        self.path = path
        self.fixeds = []
        self.movings = []
        self.p_types_movings = [] # 字符串：CMP，NP，UP
        self.p_types_fixeds = []  # 字符串：CMP，NP，UP
        
        fixeds = []
        seqs = list(set(fixed_seqs).union(set(moving_seqs)))
        seqs_dict = {}
        self.trans_str2int = {}
        for i, seq in  enumerate(seqs):
            seqs_dict[seq] = []
            self.trans_str2int[seq] = i

        for file in path.rglob("*.nii.gz"):
            if file.is_file():
                seq_name = file.name.split("-")[0]
                if seq_name  in seqs:
                    seqs_dict[seq_name].append(file)
        
        for k in seqs_dict.keys():
            seqs_dict[k].sort()

        for moving_seq in moving_seqs:
            for fixed_seq in fixed_seqs:
                if same_skip and moving_seq == fixed_seq:
                    continue
                moving = seqs_dict[moving_seq]
                fixed = seqs_dict[fixed_seq]
                self.movings = self.movings + moving
                self.fixeds = self.fixeds + fixed

                self.p_types_movings = self.p_types_movings + [moving_seq] * len(moving)
                self.p_types_fixeds = self.p_types_fixeds + [fixed_seq] * len(fixed)
        
        self.check_file()

        self.moving_imgs = []
        self.fixed_imgs = []
        for moving_path, fixed_path in zip(self.movings, self.fixeds):
            moving_img = sitk.GetArrayFromImage(sitk.ReadImage(str(moving_path)))[np.newaxis,...]
            fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_path)))[np.newaxis, ...]
            self.moving_imgs.append(moving_img)
            self.fixed_imgs.append(fixed_img)
        # print("movings:"+str(len(self.movings)),"fixed:"+str(len(self.fixeds)),"p_type:"+str(len(self.p_type)))
        
    def __len__(self):
        return len(self.movings)

    def __getitem__(self, index):
        # fixed_path = self.fixeds[index]
        # moving_path = self.movings[index]
        p_type_moving = self.p_types_movings[index]
        p_type_fixed = self.p_types_fixeds[index]
        moving_img = self.moving_imgs[index]
        fixed_img = self.fixed_imgs[index]
        
        # 读取fixed和moving
        # f_img = sitk.ReadImage(str(fixed_path))
        # print("fixed_path:"+str(fixed_path)+'\n',"moving_path:"+str(moving_path)+'\n',"p_type:"+str(p_type))
        # fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_path)))[np.newaxis, ...]
        # moving_img = sitk.GetArrayFromImage(sitk.ReadImage(str(moving_path)))[np.newaxis,...]
        # pil_image=Image.fromarray(fixed_img[0,40,:,:]).convert('L')
        # pil_image.show()
        # print(moving_path)
        # 预处理
        return moving_img/255, fixed_img/255, p_type_moving, p_type_fixed

    def check_file(self):
        assert len(self.movings) == len(self.fixeds), "moving和fixed的数量不匹配"
        assert len(self.movings) == len(self.p_types_movings), "moving和p_type的数量不匹配"
        for i in range(len(self.movings)):
            moving_type, moving_name = self.movings[i].name.split('-')
            fixed_type,fixed_name = self.fixeds[i].name.split('-')
            if moving_name != fixed_name:
                raise RuntimeError("存在患者名字匹配错误")
            if moving_type != self.p_types_movings[i] or fixed_type != self.p_types_fixeds[i]:
                raise RuntimeError("影像类型不匹配")


class AbdomenDataset2(Data.Dataset):
    def __init__(self, img_path, label_path, fixed_seqs=['CMP','UP','NP'], moving_seqs=['CMP','UP','NP'], same_skip=False):
        self.img_path = img_path
        self.label_path = label_path
        self.fixed_imgs = []
        self.moving_imgs = []
        self.fixed_labs = []
        self.moving_labs = []
        self.p_types_movings = [] # CMP,NP等字符串
        self.p_types_fixeds = [] 
            
        seqs = list(set(fixed_seqs).union(set(moving_seqs)))
        img_seqs_dict = {}
        lab_seqs_dict = {}
        self.trans_str2int = {}
        for i, seq in enumerate(seqs):
            img_seqs_dict[seq] = []
            lab_seqs_dict[seq] = []
            self.trans_str2int[seq] = i

        # 读取图片
        for file in self.img_path.glob("*.nii.gz"):
            seq_name = file.name.split("-")[0]
            if seq_name in seqs:
                img_seqs_dict[seq_name].append(file)
        for k in img_seqs_dict.keys():
            img_seqs_dict[k].sort()

        # 读取标签
        for file in self.label_path.glob("*.nii.gz"):
            seq_name = file.name.split("-")[0]
            if seq_name in seqs:
                lab_seqs_dict[seq_name].append(file)
        for k in lab_seqs_dict.keys():
            lab_seqs_dict[k].sort()
        
        self.moving_imgs_path = []
        self.fixed_imgs_path = []
        self.moving_labs_path = []
        self.fixed_labs_path = []
        for moving_seq in moving_seqs:
            for fixed_seq in fixed_seqs:
                if same_skip and moving_seq == fixed_seq:
                    continue
                moving = img_seqs_dict[moving_seq]
                fixed = img_seqs_dict[fixed_seq]
                moving_lab = lab_seqs_dict[moving_seq]
                fixed_lab = lab_seqs_dict[fixed_seq]

                self.moving_imgs_path = self.moving_imgs_path + moving
                self.fixed_imgs_path = self.fixed_imgs_path + fixed
                self.moving_labs_path = self.moving_labs_path + moving_lab
                self.fixed_labs_path = self.fixed_labs_path + fixed_lab

                self.p_types_movings = self.p_types_movings + [moving_seq] * len(moving)
                self.p_types_fixeds = self.p_types_fixeds + [fixed_seq] * len(fixed)

        self.check_file()

        for moving_path, fixed_path, m_lab_path, f_lab_path in zip(self.moving_imgs_path, self.fixed_imgs_path, self.moving_labs_path, self.fixed_labs_path):
            moving_img = sitk.GetArrayFromImage(sitk.ReadImage(str(moving_path)))[np.newaxis,...]
            fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_path)))[np.newaxis, ...]
            moving_lab = sitk.GetArrayFromImage(sitk.ReadImage(str(m_lab_path)))[np.newaxis, ...].astype('int16')
            fixed_lab = sitk.GetArrayFromImage(sitk.ReadImage(str(f_lab_path)))[np.newaxis, ...].astype('int16')
            self.moving_imgs.append(moving_img)
            self.fixed_imgs.append(fixed_img)
            self.moving_labs.append(moving_lab)
            self.fixed_labs.append(fixed_lab)
        

    def __len__(self):
        return len(self.moving_imgs)

    def __getitem__(self, index):
        p_type_moving = self.p_types_movings[index]
        p_type_fixed = self.p_types_fixeds[index]
        moving_img = self.moving_imgs[index]
        fixed_img = self.fixed_imgs[index]
        moving_lab = self.moving_labs[index]
        fixed_lab = self.fixed_labs[index]
        return moving_img/255, fixed_img/255, moving_lab, fixed_lab, p_type_moving, p_type_fixed

    def check_file(self):
        assert len(self.moving_imgs_path) == len(self.fixed_imgs_path), "moving图像和fixed的数量不匹配"
        assert len(self.moving_imgs_path) == len(self.moving_labs_path), "moving图像和标签数量不匹配"
        assert len(self.moving_labs_path) == len(self.fixed_labs_path), "moving的label和fixed的数量不匹配"
        assert len(self.moving_imgs_path) == len(self.p_types_movings), "moving图像和moving的p_type数量不匹配"
        assert len(self.p_types_movings) == len(self.p_types_fixeds), "moving和fixeds类型的数量不匹配"
        if len(self.moving_imgs_path) == 0:
            warnings.warn('测试集数量为0')
        for i in range(len(self.moving_imgs)):
            moving_type, moving_name = self.moving_imgs[i].name.split('-')
            fixed_type,fixed_name = self.fixed_imgs[i].name.split('-')
            moving_lab_type, moving_type_name = self.moving_labs[i].name.split('-')
            fixed_lab_type, fixed_type_name = self.fixed_labs[i].name.split('-')
            if moving_name != fixed_name:
                raise RuntimeError("img中患者名字匹配错误")
            if moving_type != self.p_types_movings[i] or fixed_type != self.p_types_fixeds[i]:
                raise RuntimeError("影像类型不匹配")
            if moving_lab_type != self.p_types_movings[i] or fixed_lab_type != self.p_types_fixeds[i]:
                raise RuntimeError("标签类型不一致")
            if moving_type_name != fixed_type_name:
                raise RuntimeError('type中名字匹配错误')

class AbdomenDataset3(Data.Dataset):
    def __init__(self, img_path, label_path, origin_path, fixed_seqs=['CMP','UP','NP'], moving_seqs=['CMP','UP','NP'], same_skip=False):
        self.img_path = img_path
        self.label_path = label_path
        self.origin_path = origin_path
        self.fixed_imgs = []
        self.moving_imgs = []
        self.fixed_labs = []
        self.moving_labs = []
        self.moving_origin = []
        self.fixed_origin = []
        self.p_types_movings = [] # CMP,NP等字符串
        self.p_types_fixeds = []
        self.reader = sitk.ImageSeriesReader() 
        
        seqs = list(set(fixed_seqs).union(set(moving_seqs)))
        img_seqs_dict = {}
        lab_seqs_dict = {}
        for seq in seqs:
            img_seqs_dict[seq] = []
            lab_seqs_dict[seq] = []

        # 读取图片
        for file in self.img_path.glob("*.nii.gz"):
            seq_name = file.name.split("-")[0]
            if seq_name in seqs:
                img_seqs_dict[seq_name].append(file)
        for k in img_seqs_dict.keys():
            img_seqs_dict[k].sort()

        # 读取标签
        for file in self.label_path.glob("*.nii.gz"):
            seq_name = file.name.split("-")[0]
            if seq_name in seqs:
                lab_seqs_dict[seq_name].append(file)
        for k in lab_seqs_dict.keys():
            lab_seqs_dict[k].sort()
        
        self.moving_imgs_path = []
        self.fixed_imgs_path = []
        self.moving_labs_path = []
        self.fixed_labs_path = []
        for moving_seq in moving_seqs:
            for fixed_seq in fixed_seqs:
                if same_skip and moving_seq == fixed_seq:
                    continue
                moving = img_seqs_dict[moving_seq]
                fixed = img_seqs_dict[fixed_seq]
                moving_lab = lab_seqs_dict[moving_seq]
                fixed_lab = lab_seqs_dict[fixed_seq]

                self.moving_imgs_path = self.moving_imgs_path + moving
                self.fixed_imgs_path = self.fixed_imgs_path + fixed
                self.moving_labs_path = self.moving_labs_path + moving_lab
                self.fixed_labs_path = self.fixed_labs_path + fixed_lab

                self.p_types_movings = self.p_types_movings + [moving_seq] * len(moving)
                self.p_types_fixeds = self.p_types_fixeds + [fixed_seq] * len(fixed)

        self.check_file()

        for moving_path, fixed_path, m_lab_path, f_lab_path in zip(self.moving_imgs_path, self.fixed_imgs_path, self.moving_labs_path, self.fixed_labs_path):
            moving_img = sitk.GetArrayFromImage(sitk.ReadImage(str(moving_path)))[np.newaxis,...]
            fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_path)))[np.newaxis, ...]
            moving_lab = sitk.GetArrayFromImage(sitk.ReadImage(str(m_lab_path)))[np.newaxis, ...].astype('int16')
            fixed_lab = sitk.GetArrayFromImage(sitk.ReadImage(str(f_lab_path)))[np.newaxis, ...].astype('int16')
            self.moving_imgs.append(moving_img)
            self.fixed_imgs.append(fixed_img)
            self.moving_labs.append(moving_lab)
            self.fixed_labs.append(fixed_lab)
            
            seq, name = moving_path.stem.split('-')
            img_names = self.reader.GetGDCMSeriesFileNames(str(self.origin_path / seq / 'Image' / name.split('.')[0]))
            self.reader.SetFileNames(img_names)
            origin = self.reader.Execute()
            self.moving_origin.append(origin)

            seq, name = fixed_path.stem.split('-')
            img_names = self.reader.GetGDCMSeriesFileNames(str(self.origin_path / seq / 'Image' / name.split('.')[0]))
            self.reader.SetFileNames(img_names)
            origin = self.reader.Execute()
            self.fixed_origin.append(origin)
        
    def __len__(self):
        return len(self.moving_imgs)

    def __getitem__(self, index):
        p_type_moving = self.p_types_movings[index]
        p_type_fixed = self.p_types_fixeds[index]
        moving_img = self.moving_imgs[index]
        fixed_img = self.fixed_imgs[index]
        moving_lab = self.moving_labs[index]
        fixed_lab = self.fixed_labs[index]
        moving_sp = self.moving_origin[index].GetSpacing()
        moving_sp = torch.tensor([moving_sp[2],moving_sp[1],moving_sp[0]])
        fixed_sp = self.fixed_origin[index].GetSpacing()
        fixed_sp = torch.tensor([fixed_sp[2],fixed_sp[1],fixed_sp[0]])
        moving_np = sitk.GetArrayFromImage(self.moving_origin[index]).astype("int16")
        fixed_np = sitk.GetArrayFromImage(self.fixed_origin[index]).astype("int16")
        
        return moving_img, fixed_img, moving_lab, fixed_lab, \
            p_type_moving, p_type_fixed, moving_np, fixed_np, moving_sp, fixed_sp

    def check_file(self):
        assert len(self.moving_imgs_path) == len(self.fixed_imgs_path), "moving图像和fixed的数量不匹配"
        assert len(self.moving_imgs_path) == len(self.moving_labs_path), "moving图像和标签数量不匹配"
        assert len(self.moving_labs_path) == len(self.fixed_labs_path), "moving的label和fixed的数量不匹配"
        assert len(self.moving_imgs_path) == len(self.p_types_movings), "moving图像和moving的p_type数量不匹配"
        assert len(self.p_types_movings) == len(self.p_types_fixeds), "moving和fixeds类型的数量不匹配"
        if len(self.moving_imgs_path) == 0:
            warnings.warn('测试集数量为0')
        for i in range(len(self.moving_imgs)):
            moving_type, moving_name = self.moving_imgs[i].name.split('-')
            fixed_type,fixed_name = self.fixed_imgs[i].name.split('-')
            moving_lab_type, moving_type_name = self.moving_labs[i].name.split('-')
            fixed_lab_type, fixed_type_name = self.fixed_labs[i].name.split('-')
            if moving_name != fixed_name:
                raise RuntimeError("img中患者名字匹配错误")
            if moving_type != self.p_types_movings[i] or fixed_type != self.p_types_fixeds[i]:
                raise RuntimeError("影像类型不匹配")
            if moving_lab_type != self.p_types_movings[i] or fixed_lab_type != self.p_types_fixeds[i]:
                raise RuntimeError("标签类型不一致")
            if moving_type_name != fixed_type_name:
                raise RuntimeError('type中名字匹配错误')
 
class AbdomenDataset4(Data.Dataset):
    def __init__(self, img_path, label_path, origin_path, fixed_seqs=['CMP','UP','NP'], moving_seqs=['CMP','UP','NP'], same_skip=True):
        self.img_path = img_path
        self.label_path = label_path
        self.origin_path = origin_path
        self.fixed_imgs = []
        self.moving_imgs = []
        self.fixed_labs = []
        self.moving_labs = []
        self.moving_origin = []
        self.fixed_origin = []
        self.p_types_movings = [] # CMP,NP等字符串
        self.p_types_fixeds = []
        self.movings_name = []
        self.fixeds_name =[]
        self.reader = sitk.ImageSeriesReader() 
        
        seqs = list(set(fixed_seqs).union(set(moving_seqs)))
        img_seqs_dict = {}
        lab_seqs_dict = {}
        name_dict = {}
        for seq in seqs:
            img_seqs_dict[seq] = []
            lab_seqs_dict[seq] = []
            name_dict[seq] = []

        # 读取图片
        for file in self.img_path.glob("*.nii.gz"):
            seq_name, patient_name = file.stem.split("-")
            patient_name = patient_name.split(".")[0]
            if seq_name in seqs:
                img_seqs_dict[seq_name].append(file)
                name_dict[seq_name].append(patient_name)
        for k in img_seqs_dict.keys():
            img_seqs_dict[k].sort(key = lambda x: str(x))
            name_dict[k].sort(key = lambda x: str(x))

        # 读取标签
        for file in self.label_path.glob("*.nii.gz"):
            seq_name = file.name.split("-")[0]
            if seq_name in seqs:
                lab_seqs_dict[seq_name].append(file)
        for k in lab_seqs_dict.keys():
            lab_seqs_dict[k].sort(key = lambda x: str(x))
        
        self.moving_imgs_path = []
        self.fixed_imgs_path = []
        self.moving_labs_path = []
        self.fixed_labs_path = []
        for moving_seq in moving_seqs:
            for fixed_seq in fixed_seqs:
                if moving_seq == fixed_seq:
                    continue
                moving = img_seqs_dict[moving_seq]
                fixed = img_seqs_dict[fixed_seq]
                moving_lab = lab_seqs_dict[moving_seq]
                fixed_lab = lab_seqs_dict[fixed_seq]
                moving_name = name_dict[moving_seq]
                fixed_name = name_dict[fixed_seq]

                self.moving_imgs_path = self.moving_imgs_path + moving
                self.fixed_imgs_path = self.fixed_imgs_path + fixed
                self.moving_labs_path = self.moving_labs_path + moving_lab
                self.fixed_labs_path = self.fixed_labs_path + fixed_lab
                self.movings_name = self.movings_name + moving_name
                self.fixeds_name = self.fixeds_name + fixed_name

                self.p_types_movings = self.p_types_movings + [moving_seq] * len(moving)
                self.p_types_fixeds = self.p_types_fixeds + [fixed_seq] * len(fixed)

        self.check_file()

        for moving_path, fixed_path, m_lab_path, f_lab_path in zip(self.moving_imgs_path, self.fixed_imgs_path, self.moving_labs_path, self.fixed_labs_path):
            moving_img = sitk.GetArrayFromImage(sitk.ReadImage(str(moving_path)))[np.newaxis,...]
            fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_path)))[np.newaxis, ...]
            moving_lab = sitk.GetArrayFromImage(sitk.ReadImage(str(m_lab_path)))[np.newaxis, ...].astype('int16')
            fixed_lab = sitk.GetArrayFromImage(sitk.ReadImage(str(f_lab_path)))[np.newaxis, ...].astype('int16')
            self.moving_imgs.append(moving_img)
            self.fixed_imgs.append(fixed_img)
            self.moving_labs.append(moving_lab)
            self.fixed_labs.append(fixed_lab)
            
            seq, name = moving_path.stem.split('-')
            img_names = self.reader.GetGDCMSeriesFileNames(str(self.origin_path / seq / 'Image' / name.split('.')[0]))
            self.reader.SetFileNames(img_names)
            origin = self.reader.Execute()
            self.moving_origin.append(origin)

            seq, name = fixed_path.stem.split('-')
            img_names = self.reader.GetGDCMSeriesFileNames(str(self.origin_path / seq / 'Image' / name.split('.')[0]))
            self.reader.SetFileNames(img_names)
            origin = self.reader.Execute()
            self.fixed_origin.append(origin)
        
    def __len__(self):
        return len(self.moving_imgs)

    def __getitem__(self, index):
        p_type_moving = self.p_types_movings[index]
        p_type_fixed = self.p_types_fixeds[index]
        moving_img = self.moving_imgs[index]
        fixed_img = self.fixed_imgs[index]
        moving_lab = self.moving_labs[index]
        fixed_lab = self.fixed_labs[index]
        moving_sp = self.moving_origin[index].GetSpacing()
        moving_sp = torch.tensor([moving_sp[2],moving_sp[1],moving_sp[0]])
        fixed_sp = self.fixed_origin[index].GetSpacing()
        fixed_sp = torch.tensor([fixed_sp[2],fixed_sp[1],fixed_sp[0]])
        moving_ori = sitk.GetArrayFromImage(self.moving_origin[index]).astype("int16")
        fixed_ori = sitk.GetArrayFromImage(self.fixed_origin[index]).astype("int16")
        moving_name = self.movings_name[index]
        fixed_name = self.fixeds_name[index]
        
        return moving_img/255, fixed_img/255, moving_lab, fixed_lab,  p_type_moving, p_type_fixed, \
            moving_name, fixed_name, moving_ori, fixed_ori, moving_sp, fixed_sp

    def check_file(self):
        assert len(self.moving_imgs_path) == len(self.fixed_imgs_path), "moving图像和fixed的数量不匹配"
        assert len(self.moving_imgs_path) == len(self.moving_labs_path), "moving图像和标签数量不匹配"
        assert len(self.moving_labs_path) == len(self.fixed_labs_path), "moving的label和fixed的数量不匹配"
        assert len(self.moving_imgs_path) == len(self.p_types_movings), "moving图像和moving的p_type数量不匹配"
        assert len(self.p_types_movings) == len(self.p_types_fixeds), "moving和fixeds类型的数量不匹配"
        assert len(self.moving_imgs_path) == len(self.movings_name), "moving和患者数量数量不匹配"

        if len(self.moving_imgs_path) == 0:
            warnings.warn('测试集数量为0')
        for i in range(len(self.moving_imgs)):
            moving_type, moving_name = self.moving_imgs[i].name.split('-')
            fixed_type,fixed_name = self.fixed_imgs[i].name.split('-')
            moving_lab_type, moving_type_name = self.moving_labs[i].name.split('-')
            fixed_lab_type, fixed_type_name = self.fixed_labs[i].name.split('-')
            m_name = self.moving_names[i]
            f_name = self.fixed_names[i]
            if moving_name != fixed_name:
                raise RuntimeError("img中患者名字匹配错误")
            if moving_type != self.p_types_movings[i] or fixed_type != self.p_types_fixeds[i]:
                raise RuntimeError("影像类型不匹配")
            if moving_lab_type != self.p_types_movings[i] or fixed_lab_type != self.p_types_fixeds[i]:
                raise RuntimeError("标签类型不一致")
            if moving_type_name != fixed_type_name:
                raise RuntimeError('type中名字匹配错误')
            if m_name != f_name:
                raise RuntimeError('moving和fixed的患者姓名不匹配')
            

class AbdomenDataset5(Data.Dataset):
    def __init__(self, path, fixed_seqs=['CMP','UP','NP'], moving_seqs=['CMP','UP','NP'],same_skip=False):
        self.path = path
        self.fixeds = []
        self.movings = []
        self.p_types_movings = [] # 字符串：CMP，NP，UP
        self.p_types_fixeds = []  # 字符串：CMP，NP，UP
        
        fixeds = []
        seqs = list(set(fixed_seqs).union(set(moving_seqs)))
        seqs_dict = {}
        self.trans_str2int = {}
        for i, seq in  enumerate(seqs):
            seqs_dict[seq] = []
            self.trans_str2int[seq] = i

        for file in path.rglob("*.nii.gz"):
            if file.is_file():
                seq_name = file.name.split("-")[0]
                if seq_name  in seqs:
                    seqs_dict[seq_name].append(file)
        
        for k in seqs_dict.keys():
            seqs_dict[k].sort()

        for moving_seq in moving_seqs:
            for fixed_seq in fixed_seqs:
                if same_skip and moving_seq == fixed_seq:
                    continue
                moving = seqs_dict[moving_seq]
                fixed = seqs_dict[fixed_seq]
                self.movings = self.movings + moving
                self.fixeds = self.fixeds + fixed

                self.p_types_movings = self.p_types_movings + [moving_seq] * len(moving)
                self.p_types_fixeds = self.p_types_fixeds + [fixed_seq] * len(fixed)
        
        self.check_file()

        self.moving_imgs = []
        self.fixed_imgs = []
        for moving_path, fixed_path in zip(self.movings, self.fixeds):
            moving_img = sitk.GetArrayFromImage(sitk.ReadImage(str(moving_path)))[np.newaxis,...]
            fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_path)))[np.newaxis, ...]
            self.moving_imgs.append(moving_img)
            self.fixed_imgs.append(fixed_img)
        # print("movings:"+str(len(self.movings)),"fixed:"+str(len(self.fixeds)),"p_type:"+str(len(self.p_type)))
        
        self.dataset_length = len(self.movings)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, index):
        # fixed_path = self.fixeds[index]
        # moving_path = self.movings[index]
        p_type_moving = self.p_types_movings[index]
        p_type_fixed = self.p_types_fixeds[index]
        pat1 = self.moving_imgs[index]
        pat2 = self.fixed_imgs[index]
        new_index = (index + round(random.random() * (self.dataset_length-2)) + 1) % self.dataset_length
        pat3 = self.moving_imgs[new_index]
        
        # 读取fixed和moving
        # f_img = sitk.ReadImage(str(fixed_path))
        # print("fixed_path:"+str(fixed_path)+'\n',"moving_path:"+str(moving_path)+'\n',"p_type:"+str(p_type))
        # fixed_img = sitk.GetArrayFromImage(sitk.ReadImage(str(fixed_path)))[np.newaxis, ...]
        # moving_img = sitk.GetArrayFromImage(sitk.ReadImage(str(moving_path)))[np.newaxis,...]
        # pil_image=Image.fromarray(fixed_img[0,40,:,:]).convert('L')
        # pil_image.show()
        # print(moving_path)
        # 预处理
        return pat1, pat2, pat3

    def check_file(self):
        assert len(self.movings) == len(self.fixeds), "moving和fixed的数量不匹配"
        assert len(self.movings) == len(self.p_types_movings), "moving和p_type的数量不匹配"
        for i in range(len(self.movings)):
            moving_type, moving_name = self.movings[i].name.split('-')
            fixed_type,fixed_name = self.fixeds[i].name.split('-')
            if moving_name != fixed_name:
                raise RuntimeError("存在患者名字匹配错误")
            if moving_type != self.p_types_movings[i] or fixed_type != self.p_types_fixeds[i]:
                raise RuntimeError("影像类型不匹配")

class AbdomenDataset6(Dataset):
    def __init__(self, img_path, label_path, origin_path, fixed_seqs=['CMP','UP','NP'], moving_seqs=['CMP','UP','NP'], resize=(64,128,128), same_skip=True):
        self.img_path = img_path
        self.label_path = label_path
        self.origin_path = origin_path
        self.fixed_imgs = [] # 图像
        self.moving_imgs = []
        self.fixed_labs = [] # label
        self.moving_labs = []
        self.moving_imgs_origin = [] # 原始图像
        self.fixed_imgs_origin = []
        self.moving_labs_origin =[] # 原始图像label
        self.fixed_labs_origin = []
        self.p_types_movings = [] # CMP,NP等字符串
        self.p_types_fixeds = []
        self.movings_name = []
        self.fixeds_name =[]
        self.moving_sp = []
        self.fixed_sp = []
        self.reader = sitk.ImageSeriesReader() 
        
        seqs = list(set(fixed_seqs).union(set(moving_seqs)))
        img_seqs_dict = {}
        lab_seqs_dict = {}
        name_dict = {}
        for seq in seqs:
            img_seqs_dict[seq] = []
            lab_seqs_dict[seq] = []
            name_dict[seq] = []

        # 读取图片
        for file in self.img_path.glob("*.nii.gz"):
            seq_name, patient_name = file.stem.split("-")
            patient_name = patient_name.split(".")[0]
            if seq_name in seqs:
                img_seqs_dict[seq_name].append(file)
                name_dict[seq_name].append(patient_name)
        for k in img_seqs_dict.keys():
            img_seqs_dict[k].sort(key = lambda x: str(x))
            name_dict[k].sort(key = lambda x: str(x))

        # 读取标签
        for file in self.label_path.glob("*.nii.gz"):
            seq_name = file.name.split("-")[0]
            if seq_name in seqs:
                lab_seqs_dict[seq_name].append(file)
        for k in lab_seqs_dict.keys():
            lab_seqs_dict[k].sort(key = lambda x: str(x))
        
        # read to itk
        img_itk_dict_origin = defaultdict(list)
        lab_itk_dict_origin = defaultdict(list)
        img_itk_dict = defaultdict(list)
        lab_itk_dict = defaultdict(list)
        for key, val in img_seqs_dict.items():
            for i in range(len(val)):
                img_itk = sitk.ReadImage(img_seqs_dict[key][i]) 
                lab_itk = sitk.ReadImage(lab_seqs_dict[key][i])

                img_itk_dict_origin[key].append(img_itk)
                lab_itk_dict_origin[key].append(lab_itk)
                
                # assert (np.array(img_itk.GetSpacing()).round(4) == np.array(lab_itk.GetSpacing()).round(4)).all(), f'{img_seqs_dict[key][i]}'
                if resize is not None:
                    img_itk, lab_itk = self.__resize__(resize, img_itk, lab_itk)

                img_itk_dict[key].append(img_itk)
                lab_itk_dict[key].append(lab_itk)

        # pairs
        for moving_seq in moving_seqs:
            for fixed_seq in fixed_seqs:
                if moving_seq == fixed_seq:
                    continue
                
                for i in range(len(img_itk_dict[moving_seq])):
                    
                    moving_itk = img_itk_dict[moving_seq][i]
                    fixed_itk = img_itk_dict[fixed_seq][i]
                    moving_lab_itk = lab_itk_dict[moving_seq][i]
                    fixed_lab_itk = lab_itk_dict[fixed_seq][i]

                    moving_itk_ori = img_itk_dict_origin[moving_seq][i]
                    fixed_itk_ori = img_itk_dict_origin[fixed_seq][i]
                    moving_lab_itk_ori = lab_itk_dict_origin[moving_seq][i]
                    fixed_lab_itk_ori = lab_itk_dict_origin[fixed_seq][i]
                    
                    self.moving_imgs.append(sitk.GetArrayFromImage(moving_itk).astype('uint8'))
                    self.fixed_imgs.append(sitk.GetArrayFromImage(fixed_itk).astype('uint8'))
                    self.moving_labs.append(sitk.GetArrayFromImage(moving_lab_itk).astype('uint8'))
                    self.fixed_labs.append(sitk.GetArrayFromImage(fixed_lab_itk).astype('uint8'))

                    self.moving_imgs_origin.append(sitk.GetArrayFromImage(moving_itk_ori).astype('uint8'))
                    self.fixed_imgs_origin.append(sitk.GetArrayFromImage(fixed_itk_ori).astype('uint8'))
                    self.moving_labs_origin.append(sitk.GetArrayFromImage(moving_lab_itk_ori).astype('uint8'))
                    self.fixed_labs_origin.append(sitk.GetArrayFromImage(fixed_lab_itk_ori).astype('uint8'))

                    m_sp = moving_itk_ori.GetSpacing()
                    f_sp = fixed_itk_ori.GetSpacing()
                    self.moving_sp.append(torch.tensor([m_sp[2],m_sp[1],m_sp[0]]))
                    self.fixed_sp.append(torch.tensor([f_sp[2],f_sp[1],f_sp[0]]))
                    self.movings_name.append(name_dict[moving_seq][i])
                    self.fixeds_name.append(name_dict[fixed_seq][i])
                    self.p_types_movings.append(moving_seq)
                    self.p_types_fixeds.append(fixed_seq)

        # self.check_file()
        
    def __len__(self):
        return len(self.moving_imgs)

    def __getitem__(self, index):
        p_type_moving = self.p_types_movings[index]
        p_type_fixed = self.p_types_fixeds[index]
        
        moving_img = (self.moving_imgs[index])[np.newaxis, ...] /255
        fixed_img = (self.fixed_imgs[index])[np.newaxis, ...] /255
        moving_img_ori = (self.moving_imgs_origin[index])[np.newaxis, ...] /255
        fixed_img_ori = (self.fixed_imgs_origin[index])[np.newaxis, ...] /255
        moving_lab = self.moving_labs[index][np.newaxis, ...]
        fixed_lab = self.fixed_labs[index][np.newaxis, ...]
        moving_lab_ori = self.moving_labs_origin[index][np.newaxis, ...]
        fixed_lab_ori = self.fixed_labs_origin[index][np.newaxis, ...]
        
        moving_sp = self.moving_sp[index]
        fixed_sp = self.fixed_sp[index]
        
        moving_name = self.movings_name[index]
        fixed_name = self.fixeds_name[index]
        
        return moving_img, fixed_img, moving_lab, fixed_lab,  p_type_moving, p_type_fixed, \
            moving_name, fixed_name, moving_img_ori, fixed_img_ori, moving_sp, fixed_sp

    def __resize__(self, new_size, img_itk, lab_itk=None):

        img_origin_size = np.array(img_itk.GetSize()) # w,h,z
        img_origin_spacing = np.array(img_itk.GetSpacing()) # z,h,w -> w,h,z

        new_size = np.array((new_size[2],new_size[1],new_size[0]))
        new_spacing = (img_origin_size * img_origin_spacing) / new_size

        # 图像缩放            
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_itk)
        resampler.SetSize(new_size.tolist())
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        img_itk_resample = resampler.Execute(img_itk)
        
        # img = sitk.GetArrayFromImage(img_itk_resample) # w,h,z
        # new_img = sitk.GetImageFromArray(img)
        # # new_img = sitk.Cast(sitk.RescaleIntensity(new_img), sitk.sitkUInt8)
        # new_img.SetDirection(img_itk_resample.GetDirection())
        # new_img.SetOrigin(img_itk_resample.GetOrigin())
        # new_img.SetSpacing(img_itk_resample.GetSpacing())
        if lab_itk is None:
            return img_itk_resample
        
        lab_origin_size = np.array(lab_itk.GetSize()) # w,h,z
        lab_origin_spacing = np.array(lab_itk.GetSpacing()) # z,h,w -> w,h,z
        
        assert (img_origin_size == lab_origin_size).all() and (img_origin_spacing.round(4) == lab_origin_spacing.round(4)).all(), '图像和标签的origing和spacing不一致'
        # 标签缩放
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(lab_itk)
        resampler.SetSize(new_size.tolist())
        resampler.SetOutputSpacing(new_spacing.tolist())
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        lab_itk_resample = resampler.Execute(lab_itk)

        # lab = sitk.GetArrayFromImage(lab_itk_resample) # w,h,z
        # new_lab = sitk.GetImageFromArray(lab)
        # new_lab.SetDirection(lab_itk_resample.GetDirection())
        # new_lab.SetOrigin(lab_itk_resample.GetOrigin())
        # new_lab.SetSpacing(lab_itk_resample.GetSpacing())

        return img_itk_resample, lab_itk_resample


class AbdomenDatasetTSNE(Data.Dataset):
    def __init__(self, img_path, seqs=['CMP','UP','NP']):
        self.images = []
        self.images_type = [] # CMP,NP等字符串

        self.reader = sitk.ImageSeriesReader() 
        
        # 读取图片
        for file in img_path.glob("*.nii.gz"):
            seq_name, patient_name = file.stem.split("-")
            patient_name = patient_name.split(".")[0]
            image = sitk.GetArrayFromImage(sitk.ReadImage(str(file)))[np.newaxis,...]
            self.images.append(image)
            self.images_type.append(seqs.index(seq_name))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        images_type = self.images_type[index]
        return image, images_type

