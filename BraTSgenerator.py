import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import torchio as tio
import torch.utils.data as Data
import warnings
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms
from collections import defaultdict

from monai.transforms import Rand3DElastic, create_grid
from monai.networks.layers import GaussianFilter
from monai.utils import fall_back_tuple

from scipy.ndimage import gaussian_filter, map_coordinates

'''
通过继承Data.Dataset，实现将一组Tensor数据对封装成Tensor数据集
至少要重载__init__，__len__和__getitem__方法
'''

def adjustWW(image, width=None, level=None):
    # 腹部窗宽350，窗位40

    if width is None or level is None:
        max_v = image.max()
        min_v = image.min()
        voxel_num = np.prod(image.shape)
        width = max_v
        for i in range(int(max_v), int(min_v), -1):
            if (image > i).sum() / voxel_num > 0.001:
                width = i
                break

        level = width // 2

    v_min = level - (width / 2)
    v_max = level + (width / 2)

    img = image.copy()
    img[image < v_min] = v_min
    img[image > v_max] = v_max

    img = (img - v_min) / (v_max - v_min)
    # img = (img-img.mean()) / img.std()

    return img


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

class BraTSRegDataset1(Data.Dataset):
    def __init__(self, root_path, fixed_seqs=['t1','t1ce','t21','flair'], moving_seqs=['t1','t1ce','t21','flair'], resize=None, label='WT', flag=0):
        
        # WT(whole tumor) = ED(浮肿区域，标签2) + ET(增强肿瘤区域，标签4) + NET(坏疽,标签1)
        # TC(tumor core) = ET(增强肿瘤区域，标签4) + NET(坏疽,标签1)
        # ET(enhancing tumor) = 标签4

        # 弹性变换，插值过程中可能出现标签3，将其归入ET
        if label == 'WT':
            self.label_flag = [1,2,3,4]
        elif label == 'TC':
            self.label_flag = [1,3,4]
        elif label == 'ET':
            self.label_flag = [3,4]
        else:
            assert False, '无效的肿瘤标签'

        self.flag = flag

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
        
        self.name = []
        self.type = []
        self.origin = []
        self.spacing = []
        
        seqs = list(set(fixed_seqs).union(set(moving_seqs)))
        img_seqs_dict = {}
        lab_seqs_dict = {}
        name_dict = {}
        for seq in seqs:
            img_seqs_dict[seq] = []
            lab_seqs_dict[seq] = []
            name_dict[seq] = []

        # 读文件
        for file in root_path.rglob("*_seg.nii"):
            p_name = file.parent.name
            for tp in seqs:
                image_path = file.parent / f'{p_name}_{tp}.nii'
                if tp == 't1' or tp=='t2' or tp=='t1ce' or tp=='flair':
                    label_path = file.parent / f'{p_name}_seg.nii'
                else:
                    label_path = file.parent / f'{p_name}_seg{tp}.nii'
                
                lab_seqs_dict[tp].append(label_path)
                img_seqs_dict[tp].append(image_path)
                name_dict[tp].append(p_name)

        # # 按照字典序重排
        # for k in img_seqs_dict.keys():
        #     img_seqs_dict[k].sort(key = lambda x: x.name)
        #     name_dict[k].sort(key = lambda x: x)
        #     lab_seqs_dict[k].sort(key = lambda x: x.name)
        
        # 排列moving
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

                # self.movings_name = self.movings_name + moving_name
                # self.fixeds_name = self.fixeds_name + fixed_name

                # self.p_types_movings = self.p_types_movings + [moving_seq] * len(moving)
                # self.p_types_fixeds = self.p_types_fixeds + [fixed_seq] * len(fixed)

                self.type = self.type + list(zip([moving_seq] * len(moving), [fixed_seq] * len(fixed)))
                self.name = self.name + list(zip(moving_name, fixed_name))
        
        self.check_file()

        for moving_path, fixed_path, m_lab_path, f_lab_path in zip(self.moving_imgs_path, self.fixed_imgs_path, self.moving_labs_path, self.fixed_labs_path):
            
            m_image_itk = sitk.ReadImage(str(moving_path))
            f_image_itk = sitk.ReadImage(str(fixed_path))
            m_label_itk = sitk.ReadImage(str(m_lab_path))
            f_label_itk = sitk.ReadImage(str(f_lab_path))

            if resize is not None:
                m_image_itk, m_label_itk = self.__resize__(m_image_itk, m_label_itk, resize)
                f_image_itk, f_label_itk = self.__resize__(f_image_itk, f_label_itk, resize)

            moving_img = sitk.GetArrayFromImage(m_image_itk)
            fixed_img = sitk.GetArrayFromImage(f_image_itk)

            moving_img = adjustWW(moving_img, width=1000, level=400)
            fixed_img = adjustWW(fixed_img, width=1000, level=400)

            moving_lab = sitk.GetArrayFromImage(m_label_itk).astype('uint8')
            fixed_lab = sitk.GetArrayFromImage(f_label_itk).astype('uint8')
            
            self.moving_imgs.append(moving_img)
            self.fixed_imgs.append(fixed_img)
            self.moving_labs.append(moving_lab)
            self.fixed_labs.append(fixed_lab)

            m_o = m_image_itk.GetOrigin()
            f_o = f_image_itk.GetOrigin()
            m_sp = m_image_itk.GetSpacing()
            f_sp = f_image_itk.GetSpacing()
            self.origin.append([np.array((m_o[2], m_o[0], m_o[1])), np.array((f_o[2], f_o[0], f_o[1]))])
            self.spacing.append([np.array((m_sp[2],m_sp[0],m_sp[1])), np.array((f_sp[2],f_sp[0],f_sp[1]))])

        print('load finished')

    def __len__(self):
        return len(self.moving_imgs)

    def __getitem__(self, index):

        if self.flag == 0:
            moving_img = torch.tensor(self.moving_imgs[index])
            fixed_img = torch.tensor(self.fixed_imgs[index])
            m_type, f_type = self.type[index]
            return moving_img.unsqueeze(0), fixed_img.unsqueeze(0), m_type, f_type
        elif self.flag == 1:
            m_img = torch.tensor(self.moving_imgs[index]) 
            f_img = torch.tensor(self.fixed_imgs[index]) 
            m_lab = torch.tensor(self.moving_labs[index])
            f_lab = torch.tensor(self.fixed_labs[index])
            m_type, f_type = self.type[index]
            return m_img.unsqueeze(0), f_img.unsqueeze(0), m_lab.unsqueeze(0), f_lab.unsqueeze(0), m_type, f_type
        elif self.flag == 2 :
            m_img = torch.tensor(self.moving_imgs[index]) 
            f_img = torch.tensor(self.fixed_imgs[index]) 
            m_lab = torch.tensor(self.moving_labs[index])
            f_lab = torch.tensor(self.fixed_labs[index])
            m_name, f_name = self.name[index]
            m_type, f_type = self.type[index]
            m_origin, f_origin = self.origin[index]
            m_spacing, f_spacing = self.spacing[index]
            return m_img.unsqueeze(0), f_img.unsqueeze(0), m_lab.unsqueeze(0), f_lab.unsqueeze(0), \
                m_type, f_type, m_name, f_name, m_origin, f_origin, m_spacing, f_spacing
        else:
            warnings.warn(f'不存在的flag选项{self.flag}')
    

    def check_file(self):
        assert len(self.moving_imgs_path) == len(self.fixed_imgs_path), "moving图像和fixed的数量不匹配"
        assert len(self.moving_imgs_path) == len(self.moving_labs_path), "moving图像和标签数量不匹配"
        assert len(self.moving_labs_path) == len(self.fixed_labs_path), "moving的label和fixed的数量不匹配"
        assert len(self.moving_imgs_path) == len(self.type), "moving图像和moving的p_type数量不匹配"
        assert len(self.moving_imgs_path) == len(self.name), "moving和患者数量数量不匹配"

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

    def __resize__(self, img_itk, lab_itk, new_size):

        img_origin_size = np.array(img_itk.GetSize()) # w,h,z
        img_origin_spacing = np.array(img_itk.GetSpacing()) # z,h,w -> w,h,z

        lab_origin_size = np.array(lab_itk.GetSize()) # w,h,z
        lab_origin_spacing = np.array(lab_itk.GetSpacing()) # z,h,w -> w,h,z
        
        assert (img_origin_size == lab_origin_size).all() and (img_origin_spacing == lab_origin_spacing).all(), '图像和标签的origing和spacing不一致'

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

class BraTSRegDataset2(Data.Dataset):
    def __init__(self, root_path, fixed_seqs=['t1','t1ce','t21','flair'], moving_seqs=['t1','t1ce','t21','flair'], resize=None, label='WT', flag=0):
        # WT(whole tumor) = ED(浮肿区域，标签2) + ET(增强肿瘤区域，标签4) + NET(坏疽,标签1)
        # TC(tumor core) = ET(增强肿瘤区域，标签4) + NET(坏疽,标签1)
        # ET(enhancing tumor) = 标签4

        # 弹性变换，插值过程中可能出现标签3，将其归入ET
        if label == 'WT':
            self.label_flag = [1,2,3,4]
        elif label == 'TC':
            self.label_flag = [1,3,4]
        elif label == 'ET':
            self.label_flag = [3,4]
        else:
            assert False, '无效的肿瘤标签'
        
        self.flag = flag

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
        
        self.name = []
        self.type = []
        self.origin = []
        self.spacing = []
        
        seqs = list(set(fixed_seqs).union(set(moving_seqs)))
        img_seqs_dict = {}
        lab_seqs_dict = {}
        name_dict = {}
        for seq in seqs:
            img_seqs_dict[seq] = []
            lab_seqs_dict[seq] = []
            name_dict[seq] = []

        # 读文件
        for file in root_path.rglob("*_seg.nii"):
            p_name = file.parent.name
            for tp in seqs:
                image_path = file.parent / f'{p_name}_{tp}.nii'
                if tp == 't1' or tp=='t2' or tp=='t1ce' or tp=='flair':
                    label_path = file.parent / f'{p_name}_seg.nii'
                else:
                    label_path = file.parent / f'{p_name}_seg{tp}.nii'
                
                lab_seqs_dict[tp].append(label_path)
                img_seqs_dict[tp].append(image_path)
                name_dict[tp].append(p_name)

        # # 按照字典序重排
        # for k in img_seqs_dict.keys():
        #     img_seqs_dict[k].sort(key = lambda x: x.name)
        #     name_dict[k].sort(key = lambda x: x)
        #     lab_seqs_dict[k].sort(key = lambda x: x.name)
        
        img_seqs_dict_itk = defaultdict(list)
        lab_seqs_dict_itk = defaultdict(list)
        for key in img_seqs_dict.keys():
            img_ls = img_seqs_dict[key]
            lab_ls = lab_seqs_dict[key]
            
            tmp_img_ls = []
            tmp_lab_ls = []
            for i in range(len(img_ls)):
                img_itk = sitk.ReadImage(str(img_ls[i]))
                lab_itk = sitk.ReadImage(str(lab_ls[i]))
                
                if resize is not None:
                    img_itk, lab_itk = self.__resize__(img_itk, lab_itk, resize)
                
                tmp_img_ls.append(img_itk)
                tmp_lab_ls.append(lab_itk)

            img_seqs_dict_itk[key] = tmp_img_ls
            lab_seqs_dict_itk[key] = tmp_lab_ls

            
        # 配对moving和fixed
        for moving_seq in moving_seqs:
            for fixed_seq in fixed_seqs:
                if moving_seq == fixed_seq:
                    continue
                
                for i in range(len(img_seqs_dict_itk[moving_seq])):

                    m_image_itk = img_seqs_dict_itk[moving_seq][i]
                    f_image_itk = img_seqs_dict_itk[fixed_seq][i]

                    moving_img = sitk.GetArrayFromImage(m_image_itk)
                    fixed_img = sitk.GetArrayFromImage(f_image_itk)

                    moving_img = adjustWW(moving_img)
                    fixed_img = adjustWW(fixed_img)

                    self.moving_imgs.append(moving_img)
                    self.fixed_imgs.append(fixed_img)
                    
                    m_o = m_image_itk.GetOrigin()
                    f_o = f_image_itk.GetOrigin()
                    m_sp = m_image_itk.GetSpacing()
                    f_sp = f_image_itk.GetSpacing()
                    self.origin.append([np.array((m_o[2], m_o[0], m_o[1])), np.array((f_o[2], f_o[0], f_o[1]))])
                    self.spacing.append([np.array((m_sp[2],m_sp[0],m_sp[1])), np.array((f_sp[2],f_sp[0],f_sp[1]))])
                    self.type.append([moving_seq, fixed_seq])
                    self.name.append([name_dict[moving_seq][i],name_dict[fixed_seq][i]])
                    # print(m_sp, f_sp)
                    if flag > 0:
                        m_label_itk = lab_seqs_dict_itk[moving_seq][i]
                        f_label_itk = lab_seqs_dict_itk[fixed_seq][i]

                        moving_lab = sitk.GetArrayFromImage(m_label_itk).astype('uint8')
                        fixed_lab = sitk.GetArrayFromImage(f_label_itk).astype('uint8')

                        # mask[mask==self.label_flag] = 1
                        moving_lab = np.in1d(moving_lab, self.label_flag).reshape(moving_lab.shape).astype('uint8')
                        fixed_lab = np.in1d(fixed_lab, self.label_flag).reshape(fixed_lab.shape).astype('uint8')

                        self.moving_labs.append(moving_lab)
                        self.fixed_labs.append(fixed_lab)

        print('load finished')

        
    def __len__(self):
        return len(self.moving_imgs)

    def __getitem__(self, index):

        if self.flag == 0:
            moving_img = torch.tensor(self.moving_imgs[index])
            fixed_img = torch.tensor(self.fixed_imgs[index])
            m_type, f_type = self.type[index]
            return moving_img.unsqueeze(0), fixed_img.unsqueeze(0), m_type, f_type
        elif self.flag == 1:
            m_img = torch.tensor(self.moving_imgs[index])
            f_img = torch.tensor(self.fixed_imgs[index])
            m_lab = torch.tensor(self.moving_labs[index])
            f_lab = torch.tensor(self.fixed_labs[index])
            m_type, f_type = self.type[index]
            return m_img.unsqueeze(0), f_img.unsqueeze(0), m_lab.unsqueeze(0), f_lab.unsqueeze(0), m_type, f_type
        elif self.flag == 2 :
            m_img = torch.tensor(self.moving_imgs[index])
            f_img = torch.tensor(self.fixed_imgs[index])
            m_lab = torch.tensor(self.moving_labs[index])
            f_lab = torch.tensor(self.fixed_labs[index])
            m_name, f_name = self.name[index]
            m_type, f_type = self.type[index]
            m_origin, f_origin = self.origin[index]
            m_spacing, f_spacing = self.spacing[index]
            return m_img.unsqueeze(0), f_img.unsqueeze(0), m_lab.unsqueeze(0), f_lab.unsqueeze(0), \
                m_type, f_type, m_name, f_name, m_origin, f_origin, m_spacing, f_spacing
        else:
            warnings.warn(f'不存在的flag选项{self.flag}')

    def __resize__(self, img_itk, lab_itk, new_size):
        img_origin_size = np.array(img_itk.GetSize()) # w,h,z
        img_origin_spacing = np.array(img_itk.GetSpacing()) # z,h,w -> w,h,z

        lab_origin_size = np.array(lab_itk.GetSize()) # w,h,z
        lab_origin_spacing = np.array(lab_itk.GetSpacing()) # z,h,w -> w,h,z
        
        assert (img_origin_size == lab_origin_size).all() and (img_origin_spacing == lab_origin_spacing).all(), '图像和标签的origing和spacing不一致'

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

class BraTSTSNE(Data.Dataset):
    def __init__(self, img_path, seqs=['t1','t1ce1','t21','flair1'], resize=None):

        self.images = []
        self.images_type = [] # CMP,NP等字符串
        
        # 读文件
        for file in img_path.rglob("*_seg.nii"):
            p_name = file.parent.name

            for tp in seqs:
                image_path = file.parent / f'{p_name}_{tp}.nii'
                image_itk = sitk.ReadImage(image_path)

                if resize is not None:
                    image_itk = self.__resize__(resize, image_itk)

                image = sitk.GetArrayFromImage(image_itk)

                image = adjustWW(image, width=1000, level=400)
                
                self.images.append(image[np.newaxis,:])
                self.images_type.append(seqs.index(tp))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        images_type = self.images_type[index]
        return image, images_type

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
        
        assert (img_origin_size == lab_origin_size).all() and (img_origin_spacing == lab_origin_spacing).all(), '图像和标签的origing和spacing不一致'
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

class Rand3DElasticPair(Rand3DElastic):
    def __call__(
        self,
        sample: dict,
        spatial_size: tuple[int, int, int] | int | None = None,
        mode: str | int | None = None,
        padding_mode: str | None = None,
        randomize: bool = True,
    ) -> torch.Tensor:
        """
        Args:
            img: shape must be (num_channels, H, W, D),
            spatial_size: specifying spatial 3D output image spatial size [h, w, d].
                if `spatial_size` and `self.spatial_size` are not defined, or smaller than 1,
                the transform will use the spatial size of `img`.
            mode: {``"bilinear"``, ``"nearest"``} or spline interpolation order 0-5 (integers).
                Interpolation mode to calculate output values. Defaults to ``self.mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When it's an integer, the numpy (cpu tensor)/cupy (cuda tensor) backends will be used
                and the value represents the order of the spline interpolation.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            padding_mode: {``"zeros"``, ``"border"``, ``"reflection"``}
                Padding mode for outside grid values. Defaults to ``self.padding_mode``.
                See also: https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
                When `mode` is an integer, using numpy/cupy backends, this argument accepts
                {'reflect', 'grid-mirror', 'constant', 'grid-constant', 'nearest', 'mirror', 'grid-wrap', 'wrap'}.
                See also: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
            randomize: whether to execute `randomize()` function first, default to True.
        """
        img = sample['image']
        lab = sample['label']
        sp_size = fall_back_tuple(self.spatial_size if spatial_size is None else spatial_size, img.shape[1:])
        if randomize:
            self.randomize(grid_size=sp_size)

        _device = img.device if isinstance(img, torch.Tensor) else self.device
        grid = create_grid(spatial_size=sp_size, device=_device, backend="torch")
        if self._do_transform:
            if self.rand_offset is None:
                raise RuntimeError("rand_offset is not initialized.")
            gaussian = GaussianFilter(3, self.sigma, 3.0).to(device=_device)
            offset = torch.as_tensor(self.rand_offset, device=_device).unsqueeze(0)
            grid[:3] += gaussian(offset)[0] * self.magnitude
            grid = self.rand_affine_grid(grid=grid)
        
        out = {}
        out['image'] = self.resampler(
            img,
            grid,  # type: ignore
            mode=mode if mode is not None else self.mode,
            padding_mode=padding_mode if padding_mode is not None else self.padding_mode,
        )
        out['label'] = self.resampler(
            lab,
            grid,  # type: ignore
            mode=mode if mode is not None else self.mode,
            padding_mode=padding_mode if padding_mode is not None else self.padding_mode,
        )
        return out

def BraTS_3Dtrans(brats_root, types=[]):
    image_shape = [240,240,155] # H,W,Z
    # take H,W,Z
    # elastic3d = Rand3DElasticPair(sigma_range=[8,20], magnitude_range=[10,150], prob=1, padding_mode="zeros")
    affine_trans = tio.RandomAffine()
    elastix_trans = tio.RandomElasticDeformation(num_control_points=(7,7,7))

    for file in brats_root.rglob("*_seg.nii"):
        p_id = file.parent.name
        for tp in types:
        
            img_path = file.parent / (p_id + f'_{tp}.nii')
            # original_mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
            img_itk = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img_itk) # ZHW
            img_tensor = torch.tensor(img_np).permute(1,2,0).unsqueeze(0)  # C,H,W,Z

            lab_itk = sitk.ReadImage(file)
            lab_np = sitk.GetArrayFromImage(lab_itk)
            lab_tensor = torch.tensor(lab_np).permute(1,2,0).unsqueeze(0)  # C,H,W,Z

            # new_img, new_lab = random_gen_flows(img_np, lab_np)
            sample = {'image':img_tensor, 'label':lab_tensor}
            sample = elastix_trans(sample)

            new_img = sample['image'][0].permute(2,0,1).numpy()
            new_lab = sample['label'][0].permute(2,0,1).numpy()
            
            new_img_itk = sitk.GetImageFromArray(new_img.astype('int16'))
            new_img_itk.SetDirection(img_itk.GetDirection())
            new_img_itk.SetOrigin(img_itk.GetOrigin())
            new_img_itk.SetSpacing(img_itk.GetSpacing())

            new_lab_itk = sitk.GetImageFromArray(new_lab.astype('int16'))
            new_lab_itk.SetDirection(lab_itk.GetDirection())
            new_lab_itk.SetOrigin(lab_itk.GetOrigin())
            new_lab_itk.SetSpacing(lab_itk.GetSpacing())

            sitk.WriteImage(new_img_itk, file.parent / f'{file.parent.name}_{tp}1.nii')
            sitk.WriteImage(new_lab_itk, file.parent / f'{file.parent.name}_seg{tp}1.nii')
        print(f'{p_id} finished')

def BraTS_3Dtrans_torchio(brats_root, types=[]):

    affine_trans = tio.RandomAffine(scales=(0.9, 1.1), degrees=5, translation=0)
    elastix_trans = tio.RandomElasticDeformation(num_control_points=(14, 14, 14))

    for file in brats_root.rglob("*_seg.nii"):
        p_id = file.parent.name
        for tp in types:
        
            img_path = file.parent / (p_id + f'_{tp}.nii')
            # original_mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
            img_itk = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img_itk) # ZHW
            img_tensor = torch.tensor(img_np).permute(1,2,0).unsqueeze(0)  # C,H,W,Z

            lab_itk = sitk.ReadImage(file)
            lab_np = sitk.GetArrayFromImage(lab_itk)
            lab_tensor = torch.tensor(lab_np).permute(1,2,0).unsqueeze(0)  # C,H,W,Z

            sample = tio.Subject(one_image=tio.ScalarImage(str(img_path)),
                                a_segmentation=tio.LabelMap(str(file)) )
            sample = affine_trans(sample)
            sample = elastix_trans(sample)

            new_img = sample['one_image'].data[0].permute(2,1,0).numpy()
            new_lab = sample['a_segmentation'].data[0].permute(2,1,0).numpy()
            
            new_img_itk = sitk.GetImageFromArray(new_img.astype('int16'))
            new_img_itk.SetDirection(img_itk.GetDirection())
            new_img_itk.SetOrigin(img_itk.GetOrigin())
            new_img_itk.SetSpacing(img_itk.GetSpacing())

            new_lab_itk = sitk.GetImageFromArray(new_lab.astype('int16'))
            new_lab_itk.SetDirection(lab_itk.GetDirection())
            new_lab_itk.SetOrigin(lab_itk.GetOrigin())
            new_lab_itk.SetSpacing(lab_itk.GetSpacing())

            sitk.WriteImage(new_img_itk, file.parent / f'{file.parent.name}_{tp}1.nii')
            sitk.WriteImage(new_lab_itk, file.parent / f'{file.parent.name}_seg{tp}1.nii')
        print(f'{p_id} finished')


def FFD_BraTS(brats_root, types=[]):
    
    for file in brats_root.rglob("*_seg.nii"):
        p_id = file.parent.name
        for tp in types:
        
            img_path = file.parent / (p_id + f'_{tp}.nii')
            # original_mesh_points = np.load('tests/test_datasets/meshpoints_sphere_orig.npy')
            img_itk = sitk.ReadImage(img_path)
            img_np = sitk.GetArrayFromImage(img_itk)

            lab_itk = sitk.ReadImage(file)
            lab_np = sitk.GetArrayFromImage(lab_itk)

            # new_img, new_lab = random_gen_flows(img_np, lab_np)
            new_img_ls = []
            new_lab_ls = []
            for i in range(img_np.shape[0]):
                img_slicer, lab_slicer = elastic_transform(img_np[i], lab_np[i], img_np.shape[1]*0.16, img_np.shape[1]*0.08, img_np.shape[1]*0.01)
                new_img_ls.append(img_slicer)
                new_lab_ls.append(lab_slicer)

            new_img = np.stack(new_img_ls)
            new_lab = np.stack(new_lab_ls)
            
            new_img_itk = sitk.GetImageFromArray(new_img.astype('int16'))
            new_img_itk.SetDirection(img_itk.GetDirection())
            new_img_itk.SetOrigin(img_itk.GetOrigin())
            new_img_itk.SetSpacing(img_itk.GetSpacing())

            new_lab_itk = sitk.GetImageFromArray(new_lab.astype('int16'))
            new_lab_itk.SetDirection(lab_itk.GetDirection())
            new_lab_itk.SetOrigin(lab_itk.GetOrigin())
            new_lab_itk.SetSpacing(lab_itk.GetSpacing())

            sitk.WriteImage(new_img_itk, file.parent / f'{file.parent.name}_{tp}1.nii')
            sitk.WriteImage(new_lab_itk, file.parent / f'{file.parent.name}_seg{tp}1.nii')
        print(f'{p_id} finished')

def FFD_one_file(img:np.array, seg:np.array):
    ffd = FFD()
    # ffd.read_parameters('tests/test_datasets/parameters_test_ffd_sphere.prm')
    nz, ny, nx = img.shape
    xv = np.linspace(0, nx, nx)
    yv = np.linspace(0, ny, ny)
    zv = np.linspace(0, nz, nz)
    z, y, x = np.meshgrid(zv, yv, xv)
    meshgrid = np.array([z.ravel(), y.ravel(), x.ravel()]) #3,z,y,x
    mesh = meshgrid.T # x,y,z,3

    new_locs =  torch.tensor(ffd(mesh).reshape((nx,ny,nz,3)).T).unsqueeze(0) 
    
    for i in range(len(img.shape)):
        new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (img.shape[i] - 1) - 0.5)
    new_locs = new_locs.permute(0, 2, 3, 4, 1)
    # new_locs = new_locs[..., [2, 1, 0]]
    
    new_img =  F.grid_sample(torch.tensor(img).double().unsqueeze(0).unsqueeze(0), torch.tensor(new_locs),  mode='bilinear', align_corners=False)
    new_lab =  F.grid_sample(torch.tensor(seg).double().unsqueeze(0).unsqueeze(0), torch.tensor(new_locs),  mode='bilinear', align_corners=False)

    return new_img[0,0].numpy(), new_lab[0,0].numpy()

def random_gen_flows(img:np.array, seg:np.array):
    stn = SpatialTransformer(size = img.shape)
    z, y, x = img.shape
    sizes = [z,y,x]
    sigma = 1

    flows = []
    for i in range(3):
        flow = np.random.normal(0, sigma, size=(sizes)) * (sizes[i]/(3*sigma)*np.random.random()*0.2) #z,y,w
        # flow = np.random.normal(0, sigma, size=(sizes)) #z,y,w
        flows.append(torch.tensor(gaussian_filter_3d(flow.transpose((2,1,0)), K_size=7, sigma=1)).permute(2,1,0))
    flows = torch.stack(flows).unsqueeze(0)

    # flows = [torch.randn(sizes) * size for size in sizes]
    # flows = torch.stack(flows).unsqueeze(0)
    
    new_img = stn(torch.tensor(img).float().unsqueeze(0).unsqueeze(0), flows)
    new_lab = stn(torch.tensor(seg).float().unsqueeze(0).unsqueeze(0), flows)

    return new_img[0,0].numpy(), new_lab[0,0].numpy()

def gaussian_filter_3d(img, K_size=3, sigma=1.5):
    if len(img.shape) == 3:
        H, W, C = img.shape
    else:
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape
 
    ## Zero padding
    pad = K_size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float32)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float32)
 
    ## prepare Kernel
    K = np.zeros((K_size, K_size), dtype=np.float32)
    for x in range(-pad, -pad + K_size):
        for y in range(-pad, -pad + K_size):
            K[y + pad, x + pad] = np.exp( -(x ** 2 + y ** 2) / (2 * (sigma ** 2)))
 
    K /= (2 * np.pi * sigma * sigma)
 
    K /= K.sum()
    tmp = out.copy()
 
    # filtering
    for y in range(H):
        for x in range(W):
            for c in range(C):
                out[pad + y, pad + x, c] = np.sum(K * tmp[y: y + K_size, x: x + K_size, c])
 
    out = np.clip(out, 0, 255)
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8)
    return out

def elastic_transform(image, label, alpha, sigma, alpha_affine, random_state=None):
    # alpha (float): 扭曲变换参数。默认值1，值越大扭曲效果越明显（如alpha=500，sigma=50）
    # sigma (float): 高斯滤波参数。默认值50，值越小扭曲效果越明显（如alpha=100，sigma=20）
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape
    shape_size = shape

    # # Random affine
    center_square = (np.float32(shape_size) / 10).astype('int')  #2
    square_size = min(shape_size) // 30   #6

    # # pts1: 仿射变换前的点（3个点）
    pts1 = np.float32([center_square+square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square-square_size])

    # pts1: 仿射变换前的点（铺满图像）
    # point_ls = []
    # for i in range(center_square[-2], image.shape[-2], center_square[-2]):
    #     for j in range(center_square[-1], image.shape[-1], center_square[-1]):
    #         point_ls.append([i, j])
    # pts1 = np.array(point_ls).astype(np.float32)

    # pts2: 仿射变换后的点
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    # pts2 = pts1 + np.random.normal(0, alpha_affine, size=pts1.shape).astype(np.float32)

    # # 仿射变换矩阵
    M = cv2.getAffineTransform(pts1, pts2)
    
    # # 对image进行仿射变换
    imageB = cv2.warpAffine(image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101) 
    labelB = cv2.warpAffine(label, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

    # generate random displacement fields
    # random_state.rand(*shape)会产生一个和shape一样的服从[0,1]均匀分布的矩阵
    dx = gaussian_filter( (random_state.rand(*shape)*2-1), sigma)* alpha
    dy = gaussian_filter( (random_state.rand(*shape)*2-1), sigma)* alpha

    #generate meshgrid
    x, y, = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

    indices = np.reshape(y+dy, (-1,1)), np.reshape(x+dx, (-1,1))

    # bilinear interpolation
    imageC = map_coordinates(imageB, indices, order=1, mode='constant').reshape(shape)
    labelC = map_coordinates(labelB, indices, order=1, mode='constant').reshape(shape)

    return imageC, labelC

if __name__ == '__main__':
    # create_grid(out_path='grid_pic.jpg')
    # FFD_BraTS(Path('E:\\datasets\\BRAST2018\\MICCAI_BraTS_2018_Data_Training\\LGG'), types=['t1','t2','t1ce','flair'])
    BraTS_3Dtrans_torchio(Path('E:\\datasets\\BRAST2018\\MICCAI_BraTS_2018_Data_Training_origin\\LGG'), types=['t1','t2','t1ce','flair'])