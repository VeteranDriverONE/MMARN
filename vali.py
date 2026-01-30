import os,sys
import glob
import torch
import numpy as np
import argparse as arg
import torchvision
from config import args
import SimpleITK as sitk
from datagenerators import Dataset,AbdomenDataset1
import torch.utils.data as Data
import models.losses as losses 
from pathlib import Path
from PIL import Image
import time

def dice_coeff(pred, target):
    smooth = 1e-6
    num = pred.shape[0]
    m1 = pred.reshape(num, -1)  # Flatten
    m2 = target.reshape(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection ) / (m1.sum() + m2.sum()+smooth)

def dice(array1, target, threshold, labels=None, include_zero=False,):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    assert len(target.unique()) <= 2, 'label非二值对象'
    array1 = array1.cpu().numpy()
    target = target.cpu().numpy()
    bs = target.shape[0]
    array1 = array1.reshape(bs,-1)
    target = target.reshape(bs,-1)
    intersection = np.sum((target == 1)*(array1 > threshold), -1)
    union = np.sum(array1 > threshold, -1)+np.sum(target == 1, -1)
    
    return (2*intersection/union).mean()

def dice2(array1, target, threshold, labels=None, include_zero=False,):
    """
    Computes the dice overlap between two arrays for a given set of integer labels.
    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    assert len(target.unique()) <= 2, 'label非二值对象'
    array1 = array1.cpu().numpy()
    target = target.cpu().numpy()
    # array2 = array2..numpy()
    label = target.shape[0]
    dice = []
    for i in range(label):
        intersection = np.sum((target[i] == 1)*(array1[i] > threshold))
        union = np.sum(array1[i] > threshold)+np.sum(target[i] == 1)
        dice.append(2*intersection/union)
    
    return dice

def vali(flow,moving_path,fixed_path,device,size):
    stn = SpatialTransformer(size).to(device)
    dices = []
    moving_label = glob.glob(os.path.join(moving_path,'*.nii'))
    moving_label.sort()
    fixed_label = glob.glob(os.path.join(fixed_path,'*.nii'))
    fixed_label.sort()
    for i in range(len(moving_label)):
        m_label = sitk.GetArrayFromImage(sitk.ReadImage(moving_label[i]))[np.newaxis,np.newaxis, ...]
        f_label = sitk.GetArrayFromImage(sitk.ReadImage(fixed_label[i]))[np.newaxis,np.newaxis, ...]
        src = torch.from_numpy(m_label).to(device)
        # f_label = torch.from_numpy(f_label)
        for j in flow:
            warped = stn(src,j)
            src = warped
        dice_ = dice(src,f_label,0.4)
        dices.append(dice_)
        
        # print('Dice: %.4f' % np.mean(dices))
    return dices

def save_volfile(array, filename, affine=None):
    """
    Saves an array to nii, nii.gz, or npz format.

    Parameters:
        array: The array to save.
        filename: Filename to save to.
        affine: Affine vox-to-ras matrix. Saves LIA matrix if None (default).
    """
    if filename.endswith(('.nii', '.nii.gz')):
        import nibabel as nib
        if affine is None and array.ndim >= 3:
            # use LIA transform as default affine
            affine = np.array([[-1, 0, 0, 0],  # nopep8
                               [0, 0, 1, 0],  # nopep8
                               [0, -1, 0, 0],  # nopep8
                               [0, 0, 0, 1]], dtype=float)  # nopep8
            pcrs = np.append(np.array(array.shape[:3]) / 2, 1)
            affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        nib.save(nib.Nifti1Image(array, affine), filename)
    elif filename.endswith('.npz'):
        np.savez_compressed(filename, vol=array)
    else:
        raise ValueError('unknown filetype for %s' % filename)

    
    