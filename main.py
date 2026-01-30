import torch
import numpy as np
from pathlib import Path
from torchvision import utils as vutils
from models.PMAR import PMARNet
from models.PMAR2 import PMARNet2, PMARNet21
from models.PMAR3 import PMARNet3, PMARNet31
from config import args
from test_trans import test, test_ssim, infer_trans_jpg
from datagenerators import AbdomenDataset1, AbdomenDataset2, AbdomenDataset4
from test import tsne
from collections import defaultdict
from models.transform import Encoder, Decoder
import time
import vali

def train():
    # Abdomen（原）
    args.w_dis = 1
    args.w_gan = args.w_dis
    args.w_style = 1e1
    args.w_content = 1
    args.w_proto = 1
    args.w_contra = 1e1
    args.w_morp = 1
    args.w_sim = 1e3
    args.w_smooth = args.w_sim
    args.dis_gap = 5

    # Abdomen CMP,UP->NP
    # args.w_dis = 1
    # args.w_gan = args.w_dis
    # args.w_style = 1e1
    # args.w_content = 1e-1
    # args.w_proto = 1
    # args.w_contra = 1e1
    # args.w_morp = 1
    # args.w_sim = 1e1
    # args.w_smooth = args.w_sim
    # args.dis_gap = 5
    
    # Abdomen CMP,NP->UP
    # args.w_dis = 1
    # args.w_gan = args.w_dis
    # args.w_style = 1e1
    # args.w_content = 1e-1
    # args.w_proto = 1
    # args.w_contra = 1e1
    # args.w_morp = 1
    # args.w_sim = 1e2
    # args.w_smooth = args.w_sim
    # args.dis_gap = 5
    
    # BraTS
    # args.w_dis = 1
    # args.w_gan = args.w_dis
    # args.w_style = 1e1
    # args.w_content = 1e-1
    # args.w_proto = 1
    # args.w_contra = 1e1
    # args.w_morp = 1
    # args.w_sim = 1e1
    # args.w_smooth = 1e1
    # args.dis_gap = 5

    pmar = PMARNet3(args)
    pmar.train_model()
    


if __name__ == '__main__':
   
    train()
    
    



