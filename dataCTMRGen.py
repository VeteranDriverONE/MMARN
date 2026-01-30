import cv2
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import torch
import torch.utils.data as Data
import warnings
import random
from pathlib import Path
from collections import defaultdict
import json
from  torchvision import utils as vutils
import re

class AbdomenMRCT1(Data.Dataset):
    # Learn2reg2023 MRCT
    def __init__(self, josn_path, image_path, mask_path, label_path=Path(''), paired=False):
        moving_imgs_path = {}
        fixed_imgs_path = {}
        moving_labs_path = {}
        fixed_labs_path = {}
        moving_masks_path = {}
        fixed_masks_path = {}
        self.label_path = label_path
        
        with open(josn_path, encoding="utf-8") as file:
            file_json = json.loads(file.read())
        
        reg_direct = file_json['registration_direction']
        index_m, index_f = str(reg_direct['moving']), str(reg_direct['fixed'])
        
        image_shape = file_json['tensorImageShape']

        img_dict={}
        for image in image_path.glob('*.nii.gz'):
            filename = image.stem.split('.')[0]
            filename_array = filename.split('_')
            name = filename_array[1]
            id = int(filename_array[2])
            if id == 0:
                moving_imgs_path[name] = image
            else:
                fixed_imgs_path[name] = image
            img = sitk.ReadImage(str(image))
            img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
            img = sitk.GetArrayFromImage(img)
            img_dict[image.name] = img

        lab_dict = {}
        for label in label_path.glob('*.nii.gz'):
            filename = label.stem.split('.')[0]
            filename_array = filename.split('_')
            name = filename_array[1]
            id = int(filename_array[2])
            if id == 0:
                moving_labs_path[name] = label
            else:
                fixed_labs_path[name] = label
            lab = sitk.ReadImage(str(label))
            lab = sitk.Cast(sitk.RescaleIntensity(lab), sitk.sitkUInt8)
            lab = sitk.GetArrayFromImage(lab)
            lab_dict[label.name] = lab

        mask_dict = {}
        for mask in mask_path.glob('*.nii.gz'):
            filename = mask.stem.split('.')[0]
            filename_array = filename.split('_')
            name = filename_array[1]
            id = int(filename_array[2])
            if id == 0:
                moving_masks_path[name] = mask
            else:
                fixed_masks_path[name] = mask
            msk = sitk.ReadImage(str(mask))
            msk = sitk.Cast(sitk.RescaleIntensity(msk), sitk.sitkUInt8)
            msk = sitk.GetArrayFromImage(msk)
            mask_dict[mask.name] = msk
        
        fixed_img_list = []
        fixed_lab_list = []
        fixed_mask_list = []
        for k in fixed_imgs_path:
            if k not in moving_imgs_path:
                fixed_img_list.append(fixed_imgs_path[k])
                fixed_lab_list.append(fixed_labs_path[k])
                fixed_mask_list.append(fixed_masks_path[k])

        fixed_num = len(fixed_img_list)

        unpair_moving_img = []
        unpair_fixed_img = []
        unpair_moving_lab = []
        unpair_fixed_lab = []
        unpair_moving_mask = []
        unpair_fixed_mask = []
        
        pair_moving_img = []
        pair_fixed_img = []
        pair_moving_lab = []
        pair_fixed_lab = []
        pair_moving_mask = []
        pair_fixed_mask = []

        for k in moving_imgs_path.keys():
            if k in fixed_imgs_path:
                pair_moving_img.append(moving_imgs_path[k])
                pair_fixed_img.append(fixed_imgs_path[k])
                pair_moving_lab.append(moving_labs_path.get(k,[]))
                pair_fixed_lab.append(fixed_labs_path.get(k,[]))
                pair_moving_mask.append(moving_masks_path[k])
                pair_fixed_mask.append(fixed_masks_path[k])
            else:
                unpair_moving_img = unpair_moving_img + [moving_imgs_path[k]] * fixed_num
                unpair_fixed_img = unpair_fixed_img + fixed_img_list
                unpair_moving_lab = unpair_moving_lab + [moving_labs_path[k]] * fixed_num
                unpair_fixed_lab = unpair_fixed_lab + fixed_lab_list
                unpair_moving_mask = unpair_moving_mask + [moving_masks_path[k]] * fixed_num
                unpair_fixed_mask = unpair_fixed_mask + fixed_mask_list

        self.unpair_moving_img = unpair_moving_img 
        self.unpair_fixed_img = unpair_fixed_img
        self.unpair_moving_lab = unpair_moving_lab
        self.unpair_fixed_lab = unpair_fixed_lab 
        self.unpair_moving_mask = unpair_moving_mask
        self.unpair_fixed_mask = unpair_fixed_mask

        self.pair_moving_img = pair_moving_img
        self.pair_fixed_img = pair_fixed_img
        self.pair_moving_lab = pair_moving_lab
        self.pair_fixed_lab = pair_fixed_lab
        self.pair_moving_mask = pair_moving_mask
        self.pair_fixed_mask = pair_fixed_mask

        self.check_list()

        if paired == 0:
            image = {'pair_moving_imgs':unpair_moving_img+pair_moving_img,'pair_fixed_imgs':unpair_fixed_img+pair_fixed_img, 'pair_moving_labs':unpair_moving_lab+pair_moving_lab, 
                    'pair_fixed_labs':unpair_fixed_lab+pair_fixed_lab, 'pair_moving_masks':unpair_moving_mask+pair_moving_mask, 'pair_fixed_masks':unpair_fixed_mask+pair_fixed_mask}
        elif paired == 1:
            image = {'pair_moving_imgs':unpair_moving_img,'pair_fixed_imgs':unpair_fixed_img, 'pair_moving_labs':unpair_moving_lab, 'pair_fixed_labs':unpair_fixed_lab, 
                    'pair_moving_masks':unpair_moving_mask, 'pair_fixed_masks':unpair_fixed_mask}
        elif paired == 2:
            image = {'pair_moving_imgs':pair_moving_img,'pair_fixed_imgs':pair_fixed_img, 'pair_moving_labs':pair_moving_lab, 'pair_fixed_labs':pair_fixed_lab, 
                    'pair_moving_masks':pair_moving_mask, 'pair_fixed_masks':pair_fixed_mask}
        else:
            warnings.warn('无效paired')

        for k, v in image.items():
            tmp = []
            for img in v:
                if k[-4:] == 'imgs':
                    tmp.append(torch.tensor(img_dict[img.name]))
                elif k[-4:] == 'labs'and self.label_path.name != '':
                    tmp.append(torch.tensor(lab_dict[img.name]))
                elif k[-5:] == 'masks':
                    tmp.append(torch.tensor(mask_dict[img.name]))
            setattr(self, k, tmp)

    def __len__(self):
        return len(self.pair_moving_imgs)
        
    def __getitem__(self, index):
        if self.label_path.name == '':
            moving = self.pair_moving_imgs[index]
            fixed = self.pair_fixed_imgs[index]
            moving_mask = self.pair_moving_masks[index]
            fixed_mask = self.pair_fixed_masks[index]
            return moving.unsqueeze(0), fixed.unsqueeze(0), moving_mask.unsqueeze(0), fixed_mask.unsqueeze(0), 'CT', 'T1'
        else:
            moving = self.pair_moving_imgs[index]
            fixed = self.pair_fixed_imgs[index]
            moving_mask = self.pair_moving_masks[index]
            fixed_mask = self.pair_fixed_masks[index]
            moving_lab = self.pair_moving_labs[index]
            fixed_lab = self.pair_fixed_labs[index]
            return moving.unsqueeze(0), fixed.unsqueeze(0), 'CT', 'T1', moving_mask.unsqueeze(0), fixed_mask.unsqueeze(0), moving_lab.unsqueeze(0), fixed_lab.unsqueeze(0)


    def check_list(self):    
        assert len(self.pair_moving_img) == len(self.pair_fixed_img), '配对病例间"数量"不匹配'
        assert len(self.pair_moving_img) == len(self.pair_moving_lab) and len(self.pair_moving_img) == len(self.pair_moving_mask), '配对病例间img,lab,mask"数量"不匹配'
        assert len(self.pair_moving_lab) == len(self.pair_fixed_lab), '配对病例lab"数量"不匹配'
        assert len(self.pair_moving_mask) == len(self.pair_fixed_mask), '配对病例mask"数量"不匹配'

        assert len(self.unpair_moving_img) == len(self.unpair_fixed_img), '非配对病例间"数量"不匹配'
        assert len(self.unpair_moving_img) == len(self.unpair_moving_lab) and len(self.unpair_moving_img) == len(self.unpair_moving_mask), '非配对病例间img,lab,mask"数量"不匹配'
        assert len(self.unpair_moving_lab) == len(self.unpair_fixed_lab), '非配对病例间lab"数量"不匹配'
        assert len(self.unpair_moving_mask) == len(self.unpair_fixed_mask), '非配对病例间mask"数量"不匹配'

        for i in range(len(self.pair_moving_img)):
            assert self.pair_moving_img[i].name.split('_')[1] == self.pair_fixed_img[i].name.split('_')[1], '配对病例，病例名不匹配'
            assert self.pair_moving_mask[i].name.split('_')[1] == self.pair_fixed_mask[i].name.split('_')[1], '配对病例，病例名不匹配'
            if self.label_path.name != '':
                assert self.pair_moving_lab[i].name.split('_')[1] == self.pair_fixed_lab[i].name.split('_')[1], '配对病例，病例名不匹配'

        for i in range(len(self.unpair_moving_img)):
            assert self.unpair_moving_img[i].name == self.unpair_moving_lab[i].name and self.unpair_moving_img[i].name == self.unpair_moving_mask[i].name, '非配对病例,moving,img、lab、mask之间不匹配'
        for i in range(len(self.unpair_fixed_img)):
            assert self.unpair_fixed_img[i].name == self.unpair_fixed_lab[i].name and self.unpair_fixed_img[i].name == self.unpair_fixed_mask[i].name, '非配对病例,moving,img、lab、mask之间不匹配'
        
        if len(self.pair_moving_img)==0:
            warnings.warn('pair数据集为0')
        if len(self.unpair_moving_img)==0:
            warnings.warn('unpair数据集为0')

class AbdomenMRCT3(Data.Dataset):
    # Learn2reg2021, fixed:MR, moving:CT
    def __init__(self, unpair_moving_path=None, unpair_fixed_path=None, pair_path=None, test_path=None):
        assert (unpair_moving_path is not None and unpair_fixed_path is not None) or pair_path is not None or test_path is not None, '所使用的数据集不正确'
        self.test_path = test_path
        if  unpair_moving_path is not None and unpair_fixed_path is not None:
            images_m = []
            images_f = []
            masks_m = []
            masks_f = []
            labels_m = []
            labels_f = []
            for image in unpair_moving_path.glob("img*CT.nii.gz"):
                name = image.stem.split('.')[0]
                name_arr = name.split('_')
                modal = name_arr[2]
                id = re.findall('\d+', name)[0]
                mask_path = image.parent / ('mask'+id+'_'+name_arr[1]+'_'+name_arr[2]+'.nii.gz')
                label_path = image.parent / ('seg'+id+'_'+name_arr[1]+'_'+name_arr[2]+'.nii.gz')

                img = sitk.ReadImage(str(image))
                img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                img = sitk.GetArrayFromImage(img)
                images_m.append(img)

                lab = sitk.ReadImage(str(label_path))
                lab = sitk.Cast(sitk.RescaleIntensity(lab), sitk.sitkUInt8)
                lab = sitk.GetArrayFromImage(lab)
                labels_m.append(lab)

                msk = sitk.ReadImage(str(mask_path))
                msk = sitk.Cast(sitk.RescaleIntensity(msk), sitk.sitkUInt8)
                msk = sitk.GetArrayFromImage(msk)
                masks_m.append(msk)
                
            for image in unpair_fixed_path.glob("img*MR.nii.gz"):
                name = image.stem.split('.')[0]
                name_arr = name.split('_')
                modal = name_arr[2]
                id = re.findall('\d+', name)[0]
                mask_path = image.parent / ('mask'+id+'_'+name_arr[1]+'_'+name_arr[2]+'.nii.gz')
                label_path = image.parent / ('seg'+id+'_'+name_arr[1]+'_'+name_arr[2]+'.nii.gz')

                img = sitk.ReadImage(str(image))
                img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                img = sitk.GetArrayFromImage(img)
                images_f.append(img)

                lab = sitk.ReadImage(str(label_path))
                lab = sitk.Cast(sitk.RescaleIntensity(lab), sitk.sitkUInt8)
                lab = sitk.GetArrayFromImage(lab)
                labels_f.append(lab)

                msk = sitk.ReadImage(str(mask_path))
                msk = sitk.Cast(sitk.RescaleIntensity(msk), sitk.sitkUInt8)
                msk = sitk.GetArrayFromImage(msk)
                masks_f.append(msk)

            self.movings_img = []
            self.fixeds_img = []
            self.movings_lab = []
            self.fixeds_lab = []
            self.movings_msk = []
            self.fixeds_msk = []
            for i in range(len(images_m)):
                self.movings_img = self.movings_img + [images_m[i]] * len(images_f)
                self.fixeds_img = self.fixeds_img +  images_f
                self.movings_lab = self.movings_lab + [labels_m[i]] * len(images_f)
                self.fixeds_lab = self.fixeds_lab + labels_f
                self.movings_msk = self.movings_msk + [masks_m[i]] * len(images_f)
                self.fixeds_msk = self.fixeds_msk + masks_f

        if pair_path is not None:
            images_m = []
            images_f = []
            masks_m = []
            masks_f = []
            labels_m = []
            labels_f = []
            for image in pair_path.glob("img*.nii.gz"):
                name = image.stem.split('.')[0]
                name_arr = name.split('_')
                modal = name_arr[2]
                id = re.findall('\d+', name)[0]
                mask_path = image.parent / ('mask'+id+'_'+name_arr[1]+'_'+name_arr[2]+'.nii.gz')
                label_path = image.parent / ('seg'+id+'_'+name_arr[1]+'_'+name_arr[2]+'.nii.gz')
                if modal == 'CT':
                    img = sitk.ReadImage(str(image))
                    img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                    img = sitk.GetArrayFromImage(img)
                    images_m.append(img)

                    lab = sitk.ReadImage(str(label_path))
                    lab = sitk.Cast(sitk.RescaleIntensity(lab), sitk.sitkUInt8)
                    lab = sitk.GetArrayFromImage(lab)
                    labels_m.append(lab)

                    msk = sitk.ReadImage(str(mask_path))
                    msk = sitk.Cast(sitk.RescaleIntensity(msk), sitk.sitkUInt8)
                    msk = sitk.GetArrayFromImage(msk)
                    masks_m.append(msk)

                elif modal == 'MR':
                    img = sitk.ReadImage(str(image))
                    img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                    img = sitk.GetArrayFromImage(img)
                    images_f.append(img)

                    lab = sitk.ReadImage(str(label_path))
                    lab = sitk.Cast(sitk.RescaleIntensity(lab), sitk.sitkUInt8)
                    lab = sitk.GetArrayFromImage(lab)
                    labels_f.append(lab)

                    msk = sitk.ReadImage(str(mask_path))
                    msk = sitk.Cast(sitk.RescaleIntensity(msk), sitk.sitkUInt8)
                    msk = sitk.GetArrayFromImage(msk)
                    masks_f.append(msk)
                else:
                    warnings.warn('无此模态')
            
            self.movings_img = []
            self.fixeds_img = []
            self.movings_lab = []
            self.fixeds_lab = []
            self.movings_msk = []
            self.fixeds_msk = []
            for i in range(len(images_m)):
                self.movings_img = self.movings_img + [images_m[i]] * len(images_f)
                self.fixeds_img = self.fixeds_img +  images_f
                self.movings_lab = self.movings_lab + [labels_m[i]] * len(images_f)
                self.fixeds_lab = self.fixeds_lab + labels_f
                self.movings_msk = self.movings_msk + [masks_m[i]] * len(images_f)
                self.fixeds_msk = self.fixeds_msk + masks_f

        if test_path is not None:
            images_m = []
            images_f = []
            masks_m = []
            masks_f = []
            for image in test_path.glob("img*.nii.gz"):
                name = image.stem.split('.')[0]
                name_arr = name.split('_')
                modal = name_arr[2]
                id = re.findall('\d+', name)[0]
                mask_path = image.parent / ('mask'+id+'_'+name_arr[1]+'_'+name_arr[2]+'.nii.gz')
                label_path = image.parent / ('seg'+id+'_'+name_arr[1]+'_'+name_arr[2]+'.nii.gz')
                if modal == 'CT':
                    img = sitk.ReadImage(str(image))
                    img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                    img = sitk.GetArrayFromImage(img)
                    images_m.append(img)

                    msk = sitk.ReadImage(str(mask_path))
                    msk = sitk.Cast(sitk.RescaleIntensity(msk), sitk.sitkUInt8)
                    msk = sitk.GetArrayFromImage(msk)
                    masks_m.append(msk)

                elif modal == 'MR':
                    img = sitk.ReadImage(str(image))
                    img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                    img = sitk.GetArrayFromImage(img)
                    images_f.append(img)

                    msk = sitk.ReadImage(str(mask_path))
                    msk = sitk.Cast(sitk.RescaleIntensity(msk), sitk.sitkUInt8)
                    msk = sitk.GetArrayFromImage(msk)
                    masks_f.append(msk)

                else:
                    warnings.warn('无此模态')
            
            self.movings_img = []
            self.fixeds_img = []
            self.movings_msk = []
            self.fixeds_msk = []
            for i in range(len(images_m)):
                self.movings_img = self.movings_img + [images_m[i]] * len(images_f)
                self.fixeds_img = self.fixeds_img +  images_f
                self.movings_msk = self.movings_msk + [masks_m[i]] * len(images_f)
                self.fixeds_msk = self.fixeds_msk + masks_f

    def __len__(self):
        return len(self.movings_img)
        
    def __getitem__(self, index):
        if self.test_path is not None:
            moving = torch.tensor(self.movings_img[index])
            fixed = torch.tensor(self.fixeds_img[index])
            moving_mask = torch.tensor(self.movings_msk[index])
            fixed_mask = torch.tensor(self.fixeds_msk[index])
            return moving.unsqueeze(0), fixed.unsqueeze(0), moving_mask.unsqueeze(0), fixed_mask.unsqueeze(0), 'CT', 'MR'
        else:
            moving = torch.tensor(self.movings_img[index])
            fixed = torch.tensor(self.fixeds_img[index])
            moving_mask = torch.tensor(self.movings_msk[index])
            fixed_mask = torch.tensor(self.fixeds_msk[index])
            moving_lab = torch.tensor(self.movings_lab[index])
            fixed_lab = torch.tensor(self.fixeds_lab[index])
            return moving.unsqueeze(0), fixed.unsqueeze(0), moving_mask.unsqueeze(0), fixed_mask.unsqueeze(0), 'CT', 'MR', moving_lab.unsqueeze(0), fixed_lab.unsqueeze(0)


class Learn2RegBase(Data.Dataset):
    def __init__(self, json_path, image_root, dataset_type:int):
        # 1:unpair训练，2：测试，3：pair训练，4：pair测试，5：验证，6：测试        
        with open(json_path, encoding="utf-8") as file:
            file_json = json.loads(file.read())
        
        index_m, index_f = file_json['registration_direction']['moving'], file_json['registration_direction']['moving']['fixed']
        modality_name = file_json['modality']
        provided_data = file_json['provided_data']

        image_shape = file_json['tensorImageShape']
        label_index_dict = file_json['labels']

        unpair_len = file_json['numTraining']
        unpair_train_path_dict = file_json['training']
        unpair_test_path_dict = file_json['test']

        pair_len = file_json['numPairedTraining']
        pair_train_path_list = file_json['training_paired_imgs']
        pair_test_path_list = file_json['test_paired_images']

        val_len = file_json['numRegistration_val']
        val_path_list = file_json['registration_val']

        test_len = file_json['numRegistration_test']
        test_path_list = file_json['registration_test']

        if dataset_type == 1:
            assert unpair_len>0, '无配对数据集为0'
            path_list = unpair_train_path_dict
            if len(modality_name)>1:
                all_dict = {}
                for modality_index in modality_name.keys(): # 模态
                    pat_dict = {}
                    for data_type in provided_data[modality_index]: #各模态提供的数据类型
                        tmp_img = []
                        for pat in path_list: # 遍历病例
                            tmp_img.append(image_root / pat[data_type])
                        pat_dict[data_type] = tmp_img
                    all_dict[modality_index] = pat_dict
                for data_type in provided_data[modality_index]:
                    moving = []
                    fixed = []
                    for i in range(unpair_len):
                        for j in range(unpair_len):
                            moving.append(all_dict[index_m][data_type][i])
                            fixed.append(all_dict[index_f][data_type][j])
                    setattr(self, f'moving_{data_type}', moving)
                    setattr(self, f'fixed_{data_type}', fixed)
            else:
                pat_dict = {}
                for data_type in provided_data[modality_index]: #各模态提供的数据类型
                    tmp_img = []
                    for pat in path_list: # 遍历病例
                        tmp_img.append(image_root / pat[data_type])
                    pat_dict[data_type] = tmp_img
                all_dict[modality_index] = pat_dict
                for data_type in provided_data[modality_index]:
                    moving = []
                    fixed = []
                    for i in range(unpair_len):
                        for j in range(unpair_len):
                            moving.append(all_dict[data_type][i])
                            fixed.append(all_dict[data_type][j])
                    setattr(self, f'moving_{data_type}', moving)
                    setattr(self, f'fixed_{data_type}', fixed)
        
        elif dataset_type == 2:
            assert unpair_len>0, '五配对数据集为0'
            path_list = unpair_test_path_dict
            if len(modality_name)>1: # 无配对 多模态
                all_dict = {}
                for modality_index in modality_name.keys(): # 模态
                    pat_dict = {}
                    for data_type in provided_data[modality_index]: #各模态提供的数据类型
                        tmp_img = []
                        for pat in path_list: # 遍历病例
                            tmp_img.append(image_root / pat[data_type])
                        pat_dict[data_type] = tmp_img
                    all_dict[modality_index] = pat_dict
                for data_type in provided_data[modality_index]:
                    moving = []
                    fixed = []
                    for i in range(unpair_len):
                        for j in range(unpair_len):
                            img = sitk.ReadImage(str(image_root / pat[str(index_m)]))
                            img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                            img = sitk.GetArrayFromImage(img)
                            moving.append(all_dict[index_m][data_type][i])
                            img = sitk.ReadImage(str(image_root / pat[str(index_m)]))
                            img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                            img = sitk.GetArrayFromImage(img)
                            fixed.append(all_dict[index_f][data_type][j])
                    setattr(self, f'moving_{data_type}', moving)
                    setattr(self, f'fixed_{data_type}', fixed)
            else: # 无配对，单模态
                pat_dict = {}
                for data_type in provided_data[modality_index]: #各模态提供的数据类型
                    tmp_img = []
                    for pat in path_list: # 遍历病例
                        tmp_img.append(image_root / pat[data_type])
                    pat_dict[data_type] = tmp_img
                all_dict[modality_index] = pat_dict
                for data_type in provided_data[modality_index]:
                    moving = []
                    fixed = []
                    for i in range(unpair_len):
                        for j in range(unpair_len):
                            moving.append(all_dict[data_type][i])
                            fixed.append(all_dict[data_type][j])
                    setattr(self, f'moving_{data_type}', moving)
                    setattr(self, f'fixed_{data_type}', fixed)

        elif dataset_type == 3:
            assert pair_len>0, '配对数据集为0'
            moving = []
            fixed = []
            for pat in pair_train_path_list:
                img = sitk.ReadImage(str(image_root / pat[str(index_m)]))
                img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                img = sitk.GetArrayFromImage(img)
                moving.append(img)
                img = sitk.ReadImage(str(image_root / pat[str(index_f)]))
                img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                img = sitk.GetArrayFromImage(img)
                fixed.append(img)
            self.moving_image = moving
            self.fixed_image = fixed

        elif dataset_type == 4:
            assert pair_len>0, '配对数据集为0'
            moving = []
            fixed = []
            for pat in pair_test_path_list:
                img = sitk.ReadImage(str(image_root / pat[str(index_m)]))
                img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                img = sitk.GetArrayFromImage(img)
                moving.append(img)
                img = sitk.ReadImage(str(image_root / pat[str(index_f)]))
                img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                img = sitk.GetArrayFromImage(img)
                fixed.append(img)
            self.moving_image = moving
            self.fixed_image = fixed
        
        elif dataset_type == 5:
            assert val_len>0, '验证'
            moving = []
            fixed = []
            for pat in val_path_list:
                img = sitk.ReadImage(str(image_root / pat['moving']))
                img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                img = sitk.GetArrayFromImage(img)
                moving.append(img)
                img = sitk.ReadImage(str(image_root / pat['fixed']))
                img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                img = sitk.GetArrayFromImage(img)
                fixed.append(img)
            self.moving_image = moving
            self.fixed_image = fixed

        elif dataset_type == 6:
            assert test_len>0, '测试'
            moving = []
            fixed = []
            for pat in test_path_list:
                img = sitk.ReadImage(str(image_root / pat['moving']))
                img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                img = sitk.GetArrayFromImage(img)
                moving.append(img)
                img = sitk.ReadImage(str(image_root / pat['fixed']))
                img = sitk.Cast(sitk.RescaleIntensity(img), sitk.sitkUInt8)
                img = sitk.GetArrayFromImage(img)
                fixed.append(img)
            self.moving_image = moving
            self.fixed_image = fixed

        self.check_list()

    def __len__(self):
        return len(self.moving_image)
     
    def __getitem__(self, index):
        raise NotImplementedError


class AbdomenMRCT(Learn2RegBase):
    def __init__(self, json_path, image_root, dataset_type:int):
        super(self, AbdomenMRCT).__init__(json_path, image_root, dataset_type)

    def __getitem__(self, index):
        if self.dataset_type == 1 or self.dataset_type == 3:
            moving = self.moving_image[index]
            fixed = self.fixed_image[index]
            moving_label = self.moving_label[index]
            fixed_label = self.fixed_label[index]
            moving_mask = self.moving_mask[index]
            fixed_mask = self.fixed_mask[index]
            return moving, fixed, moving_label, fixed_label, moving_mask, fixed_mask
        elif self.dataset_type == 2 or self.dataset_type == 4:
            moving = self.moving_image[index]
            fixed = self.fixed_image[index]
            moving_label = self.moving_label[index]
            fixed_label = self.fixed_label[index]
            moving_mask = self.moving_mask[index]
            fixed_mask = self.fixed_mask[index]
            return moving, fixed, moving_mask, fixed_mask


def resize(root_path: Path, out_root: Path, resize=(128,100,128)):
    # z，h, w
    for dir in root_path.iterdir():
        if dir.is_file():
            continue
        out_path = out_root / dir.name
        out_path.mkdir(exist_ok=True)

        for file in dir.glob('*.nii.gz'):    
            img_itk = sitk.ReadImage(str(file))
            origin_size = np.array(img_itk.GetSize()) # w,h,z
            origin_spacing = np.array(img_itk.GetSpacing()) # z,h,w -> w,h,z
            
            new_size = np.array((resize[2],resize[1],resize[0]))
            new_spacing = (origin_size * origin_spacing)/resize
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(img_itk)
            resampler.SetSize(new_size.tolist())
            resampler.SetOutputSpacing(new_spacing.tolist())
            resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            img_itk_resample = resampler.Execute(img_itk)
            img = sitk.GetArrayFromImage(img_itk_resample) # w,h,z

            out = sitk.GetImageFromArray(img)
            sitk.WriteImage(out, str(out_path / file.name))


def nii2jpg(root):
    out_root = root.parent / (root.name + '-jpg')
    out_root.mkdir(exist_ok=True)
    for file in root.rglob('*.nii.gz'):
        name = file.stem.split('.')[0]
        dir_name = out_root / f'{name}-{file.parent.stem}'
        dir_name.mkdir(exist_ok=True)
        image_itk = sitk.ReadImage(str(file))
        image_itk = sitk.Cast(sitk.RescaleIntensity(image_itk), sitk.sitkUInt8)
        image = sitk.GetArrayFromImage(image_itk)
        for i in range(image.shape[0]):
            cv2.imwrite(str(dir_name / f'{i}.jpg'),image[i])


if __name__ == '__main__':
    # trainset = AbdomenMRCT3(Path('E:\datasets\Learn2reg2021Task1\L2R_Task1_CT'), Path('E:\datasets\Learn2reg2021Task1\L2R_Task1_MR'))
    # valset = AbdomenMRCT3(pair_path=Path('E:\\datasets\\Learn2reg2021Task1\\Train-Validation'))
    # testset = AbdomenMRCT3(test_path=Path('E:\\datasets\\Learn2reg2021Task1\\Test'))
    # bad = AbdomenMRCT('E:\\datasets\\learn2reg\\AbdomenMRCT','E:\\datasets\\learn2reg\\AbdomenMRCT\\AbdomenMRCT_dataset.json')
    # nii2jpg(Path('E:\\datasets\\learn2reg\\OASIS'))
    resize(Path('E:\\datasets\\Learn2reg2021Task1'), Path('E:\\datasets\\Learn2reg2021Task1-'), (96,80,96))
    train_dataset = AbdomenMRCT1('E:\\datasets\\learn2reg\\AbdomenMRCT-968096\\AbdomenMRCT_dataset.json',
                                Path('E:\\datasets\\learn2reg\\AbdomenMRCT-968096\\imagesTr'),
                                Path('E:\\datasets\\learn2reg\\AbdomenMRCT-968096\\masksTr'),
                                Path('E:\\datasets\\learn2reg\\AbdomenMRCT-968096\\labelsTr'))
    test_dataset = AbdomenMRCT1('E:\\datasets\\learn2reg\\AbdomenMRCT-968096\\AbdomenMRCT_dataset.json',
                               Path('E:\\datasets\\learn2reg\\AbdomenMRCT-968096\\imagesTs'),
                                Path('E:\\datasets\\learn2reg\\AbdomenMRCT-968096\\masksTs'))
    
    data_loader = Data.DataLoader(train_dataset,batch_size=1,shuffle=False,num_workers=0, drop_last=True)

    test_dataloader = Data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0,drop_last=True)

    out_dir_name = Path('E:\\workspace\\Python\\PARNet6\\test_jpg')
    for i, (movings, fixeds, m_type, f_type, moving_mask, fixed_mask, moving_lab, fixed_lab) in enumerate(data_loader):
        for z in range(movings.size(2)):
            vutils.save_image(movings[0,0,z].float(), str(out_dir_name/ str(i) / f'{z}_m.jpg'), normalize=True)
            vutils.save_image(fixeds[0,0,z].float(), str(out_dir_name / str(i) / f'{z}_f.jpg'), normalize=True)
            vutils.save_image(moving_mask[0,0,z].float(), str(out_dir_name/ str(i) / f'{z}_m_mask.jpg'), normalize=True)
            vutils.save_image(fixed_mask[0,0,z].float(), str(out_dir_name / str(i) / f'{z}_f_mask.jpg'), normalize=True)
            vutils.save_image(moving_lab[0,0,z].float(), str(out_dir_name / str(i) / f'{z}_m_lab.jpg'), normalize=True)
            vutils.save_image(fixed_lab[0,0,z].float(), str(out_dir_name / str(i) / f'{z}_f_lab.jpg'), normalize=True)
        print('-')

    for i, (movings, fixeds, moving_lab, fixed_lab, p_type_m, p_type_f) in enumerate(test_dataloader):
        for z in range(movings.size(2)):
            vutils.save_image(movings[0,0,z].float(), str(out_dir_name / f'{z}_m.jpg'), normalize=True)
            vutils.save_image(fixeds[0,0,z].float(), str(out_dir_name / f'{z}_f.jpg'), normalize=True)
            vutils.save_image(moving_lab[0,0,z].float(), str(out_dir_name / f'{z}_m_lab.jpg'), normalize=True)
            vutils.save_image(fixed_lab[0,0,z].float(), str(out_dir_name / f'{z}_f_lab.jpg'), normalize=True)
            print('+')
    