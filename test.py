import numpy as np
import cv2
import torch
import torch.utils.data as Data
import vali
import SimpleITK as sitk
import argparse
import surface_distance as surfdist
import warnings
import time

from models.UNet7 import SpatialTransformer, Reg, Trans, Reg2
from models.PMAR2 import PMARNet2
from models.PMAR3 import PMARNet3, PMARNet31
from pathlib import Path
from datagenerators import AbdomenDataset2, AbdomenDataset4, AbdomenDataset6, AbdomenDatasetTSNE
from BraTSgenerator import BraTSRegDataset1, BraTSRegDataset2, BraTSTSNE
from models.utils import FlowShow, jacobian_determinant, NJD, NMI, save_deformation_field, visualize_registration_error, visualize_registration_results
from medpy import metric
from torchvision import transforms
from models.base_networks import VTNAffineStem
from config import args
from collections import defaultdict
from torchvision import utils as vutils
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.pytorch_ssim import SSIM
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics.simple_metrics import normalized_mutual_information as snmi
from models.losses import NCC, MINDSSCLoss
from models.calcParam import AdvancedModelAnalyzer, ModelParameterCounter, quick_param_count, quick_trainable_param_count
from torch.utils.tensorboard import SummaryWriter


# def psnr(img1, img2, max_v):
#     mse = ((img1-img2)**2).mean()
#     if mse == 0:
#         return float('inf')
#     else:
#         return 20*(max_v/mse.sqrt()).log10()

def test_detail(model, data_loader, args, jpg_path=None, model_size=None):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')  

    model.modal_encoder.eval()
    model.morph_encoder.eval()
    model.gen.eval()
    model.reg.eval()
    
    p_dice = defaultdict(list)
    asd = defaultdict(list)
    hd_95 = defaultdict(list)
    so = defaultdict(list)
    sd = defaultdict(list)
    v_dice = defaultdict(list)
    j_phis = defaultdict(list)
    psnr_dict = defaultdict(list)
    nmi_dict = defaultdict(list)
    rmse_dict = defaultdict(list)
    ssim_dict = defaultdict(list)    
    ncc_dict = defaultdict(list)
    mind_dict = defaultdict(list)
    jc_dict = defaultdict(list)
    time_ls = []
    
    ssim =  SSIM()
    ncc = NCC()
    mind = MINDSSCLoss()

    thresholds = np.linspace(0,1,101)
    stn = SpatialTransformer(args.img_shape).to(device)

    model_size = None
    model_size = (64,64,64)
    stn = SpatialTransformer((64,128,128)).to(device)

    with torch.no_grad():
        for movings, fixeds, moving_lab, fixed_lab,  m_type, f_type, \
            moving_name, fixed_name, moving_origin, fixed_origin, m_sp, f_sp in data_loader:
            movings = movings.to(device).float()
            fixeds = fixeds.to(device).float()
            moving_lab = moving_lab.to(device).float()
            fixed_lab = fixed_lab.to(device).float()
            
            if model_size is not None:
                image_shape = movings.shape[2:]
                tmp_movings = torch.nn.functional.interpolate(movings, size=model_size, mode='trilinear')
                tmp_fixeds = torch.nn.functional.interpolate(fixeds, size=model_size, mode='trilinear')
                time1 = time.time()
                gen_mf, _, _ = model.test2(tmp_movings, tmp_fixeds, m_type, f_type)  # 包括模态转换模块
                flows, warpeds = model.infer(tmp_movings, tmp_fixeds)  # 直接推断，不包括模态转换模块
                time2 = time.time()
                flows = [torch.nn.functional.interpolate(flow, size=image_shape, mode='trilinear') for flow in flows]
            else:
                # gen_mf, flows, warpeds = model.test2(movings, fixeds, m_type, f_type)  # 包括模态转换模块
                time1 = time.time()
                gen_mf, _, _ = model.test2(movings, fixeds, m_type, f_type)  # 包括模态转换模块
                flows, warpeds = model.infer(movings, fixeds) # 直接推断，不包括模态转换模块
                time2 = time.time()
            
            time_ls.append( (time2-time1)*1000 )
                
            warped_lab = moving_lab
            warped_img = movings
            warped_gen = gen_mf
            if isinstance(flows, list):
                warped_labs = []
                warped_imgs = []
                val = []
                for flow in flows:
                    warped_lab = stn(warped_lab, flow)
                    warped_img = stn(warped_img, flow)
                    warped_gen = stn(warped_gen, flow)
                #     warped_labs.append(warped_lab)
                #     warped_imgs.append(warped_img)
                #     val.append(-ncc.loss(warped_img, fixeds, reduction='batch'))
                #     # val.append(mind(warped_img, fixeds))
                # ind = torch.stack(val).argmax(dim=0,keepdim=True).to(device)
                # print(ind)
                # tmp_warped_labs = []
                # tmp_warped_imgs = []
                # for b in range(len(m_type)):
                #     tmp_warped_labs.append(warped_labs[ind[b]][b])
                #     tmp_warped_imgs.append(warped_imgs[ind[b]][b])
                # warped_lab = torch.stack(tmp_warped_labs)
                # warped_img = torch.stack(tmp_warped_imgs)
            else:
                warped_lab = stn(warped_lab, flow)
                warped_img = stn(warped_img, flow)
                warped_gen = stn(warped_gen, flow)
            
            for b in range(len(m_type)):
                j_phi_perc = 0
                tmp_flow = torch.zeros_like(flow[b])
                for flow in flows:
                    j_phi_perc += NJD(flow[b].permute(1,2,3,0).cpu().numpy()) / np.prod(flow[b].shape)
                    tmp_flow += flow[b]
                    
                # j_phi = torch.tensor(jacobian_determinant(flows[0][i].permute(1,2,3,0).cpu().numpy()))
                # j_phi_perc = (j_phi<=0).sum() / j_phi.numel()
                # j_phi = torch.tensor(jacobian_determinant(flows[1][i].permute(1,2,3,0).cpu().numpy()))
                # j_phi_perc = (j_phi_perc + (j_phi<=0).sum() / j_phi.numel()) / 2
                j_phi_perc_2 = NJD(tmp_flow.permute(1,2,3,0).cpu().numpy()) / np.prod(tmp_flow.shape)
                
                ssim_val = 0
                ncc_val = -ncc.loss(warped_img[b:b+1], fixeds[b:b+1])
                mind_val = mind(warped_img[b:b+1], fixeds[b:b+1])
                for z in range(fixeds.shape[2]):
                    ssim_val += ssim.forward(warped_img[b,0,z].unsqueeze(0).unsqueeze(0), fixeds[b,0,z].unsqueeze(0).unsqueeze(0))
                ssim_val = ssim_val / fixeds.shape[2]
                psnr_val = psnr(fixeds[b:b+1].cpu().numpy(), warped_img[b:b+1].cpu().numpy())
                rmse_val = torch.nn.functional.mse_loss(warped_img[b:b+1], fixeds[b:b+1]).sqrt()
                nmi_val = snmi(warped_img[b:b+1].cpu().numpy(), fixeds[b:b+1].cpu().numpy())
                # nmi_val = NMI(warped_img.cpu().numpy().reshape(-1), fixeds.cpu().numpy().reshape(-1))

                key = m_type[b] + '-' + f_type[b]

                dice_ = []
                for threshold in thresholds:
                    dice_.append(vali.dice2(warped_lab[b:b+1], fixed_lab[b:b+1], threshold))
                index = np.array(dice_).argmax()
                warped_lab = warped_lab[b:b+1]>thresholds[index]
                jc_val = metric.jc(warped_lab[b:b+1].cpu().numpy(), fixed_lab[b:b+1].cpu().numpy()>0.5)

                if True and jpg_path is not None and moving_name[b] in ['XIE_DAN_ZS21138163', 'LIU_BAIGEN_ZS21148271', 'YE_GAOZHANG_XM21007902', 
                                          'MA_WENJUN_ZS21111530', 'PAN_ZHIYU_ZS21125512', 'YE_GAOZHANG_XM21007902',
                                          'Brats18_TCIA13_630_1', 'Brats18_TCIA13_653_1', 'Brats18_TCIA12_470_1',
                                          'Brats18_TCIA13_630_1', 'Brats18_TCIA13_634_1', 'Brats18_TCIA13_630_1']:
                    moving_lab_np = moving_lab.cpu().numpy().astype('uint8')
                    fixed_lab_np = fixed_lab.cpu().numpy().astype('uint8')
                    warped_lab_np = warped_lab.cpu().numpy().astype('uint8')
                    movings_np = (movings.permute(0,2,3,4,1).repeat(1,1,1,1,3).cpu().numpy()*255).astype('uint8')
                    fixeds_np = (fixeds.permute(0,2,3,4,1).repeat(1,1,1,1,3).cpu().numpy()*255).astype('uint8')
                    warped_img_np = (warped_img.permute(0,2,3,4,1).repeat(1,1,1,1,3).cpu().numpy()*255).astype('uint8')

                    out_dir = Path(jpg_path) / (m_type[b] + '-' + f_type[b])
                    out_dir.mkdir(exist_ok=True)
                    out_dir_name = out_dir / moving_name[b]
                    out_dir_name.mkdir(exist_ok=True)
                    
                    save_deformation_field(
                        flow = tmp_flow.unsqueeze(0).permute(0,2,3,4,1),
                        save_dir = out_dir_name,
                        slice_indices = list(range(movings.shape[2])),
                    )
                    
                    visualize_registration_error(
                        fixed=fixeds[b:b+1],
                        warped=warped_gen[b:b+1],
                        save_dir = out_dir_name,
                        slice_indices = list(range(movings.shape[2])),
                        show_heat = False,
                        show_mae = True,
                        show_ssim = False,
                        show_ncc = True,
                    )
                    
                    # visualize_registration_results(
                    #     flow = tmp_flow.unsqueeze(0).permute(0,2,3,4,1),
                    #     fixed = fixeds[b:b+1],
                    #     warped = warped_img[b:b+1],
                    #     save_dir =  out_dir_name,
                    #     slice_indices = list(range(movings.shape[2])),
                    #     show_mae = True,
                    #     show_ssim = True,
                    #     show_ncc = True,
                    #     dpi= 300,
                    # )
                    
                    # for z in range(movings.shape[2]):
                    #     if True:
                    #         # 绘制边线
                    #         m_slice = movings_np[b,z]
                    #         f_slice = fixeds_np[b,z]
                    #         w_slice = warped_img_np[b,z]
                    #         if moving_lab_np[b,0,z].sum() > np.prod(moving_lab_np.shape[-2:])*0.001:
                    #             # _, binary = cv2.threshold(moving_lab_np[b,0,z]*255, 127, 255, cv2.THRESH_BINARY)
                    #             # b1, b2= cv2.findContours( binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #             m_contours, m_hierarchy = cv2.findContours(moving_lab_np[b,0,z], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #             # m_slice = cv2.drawContours(m_slice, m_contours, -1, (0， 255, 255), 1)  # 青色RGB
                    #             m_slice = cv2.drawContours(m_slice, m_contours, -1, (255, 255, 0), 1)  # 青色BGR
                    #         cv2.imwrite(str(out_dir_name / f'{z}_m.jpg'), m_slice)
                            
                    #         f_contours = None
                    #         if fixed_lab_np[b,0,z].sum() > np.prod(fixed_lab_np.shape[-2:])*0.001:
                    #             f_contours, f_hierarchy = cv2.findContours(fixed_lab_np[b,0,z], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #             f_slice =  cv2.drawContours(f_slice, f_contours, -1, (0, 255, 0), 1)  # 绿色RGB
                    #         cv2.imwrite(str(out_dir_name / f'{z}_f.jpg'), f_slice)

                    #         if warped_lab_np[b,0,z].sum() > np.prod(warped_lab_np.shape[-2:])*0.001:
                    #             w_contours, w_hierarchy = cv2.findContours(warped_lab_np[b,0,z], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    #             # w_slice = cv2.drawContours(w_slice, w_contours, -1, (255, 0, 0), 1)  # 红色RGB
                    #             w_slice = cv2.drawContours(w_slice, w_contours, -1, (0, 0, 255), 1)  # 红色BGR
                    #         if f_contours is not None:
                    #             w_slice =  cv2.drawContours(w_slice, f_contours, -1, (0, 255, 0), 1)  # 绿色RGB=BGRW
                    #         cv2.imwrite(str(out_dir_name / f'{z}_w.jpg'), w_slice)
                    #     else:
                    #         vutils.save_image(movings[b,0,z], str(out_dir_name / f'{z}_m.jpg'), normalize=True)
                    #         vutils.save_image(fixeds[b,0,z], str(out_dir_name / f'{z}_f.jpg'), normalize=True)
                    #         # vutils.save_image(gen_mf[b,0,z], str(out_dir_name / f'{z}_t.jpg'), normalize=True)
                    #         vutils.save_image(warped_img[b,0,z], str(out_dir_name / f'{z}_w.jpg'), normalize=True)

                    #     vutils.save_image(moving_lab[b,0,z], str(out_dir_name / f'{z}_m_l.jpg'), normalize=True)
                    #     vutils.save_image(fixed_lab[b,0,z], str(out_dir_name / f'{z}_f_l.jpg'), normalize=True)
                    #     # vutils.save_image(warped_lab[b,0,z], str(out_dir_name / f'{z}_w_l.jpg'), normalize=True
                    #     tmp_dice = vali.dice_coeff(warped_lab[b,0,z:z+1], fixed_lab[b,0,z:z+1])
                    #     cv2.imwrite(str(out_dir_name / '{}_w_l_{:.4f}.jpg'.format(z, tmp_dice)), warped_lab[b,0,z].cpu().numpy()*255)
                        
                warped_lab_np = warped_lab[b,0].cpu().numpy()
                fixed_lab_np = fixed_lab[b,0].cpu().numpy().astype('bool')
                # print('type:{},msp:{}——type:{},fsp:{}'.format(p_type[0],m_sp,p_type_fixed[0],f_sp))
                # spacing_mm = torch.tensor(moving_origin.shape[-3:])*m_sp/torch.tensor([64,128,128])
                surface_distances = surfdist.compute_surface_distances(fixed_lab_np, warped_lab_np, spacing_mm=m_sp[b])
                avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances) # len=2
                hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95) # len=1
                surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1) # len=2
                surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1) # len=1
                volume_dice = surfdist.compute_dice_coefficient(fixed_lab_np.astype('int'), warped_lab_np.astype('int')) # len=1

                p_dice[key].append(dice_)
                asd[key].append(avg_surf_dist)
                hd_95[key].append(hd_dist_95)
                so[key].append(surface_overlap)
                sd[key].append(surface_dice)
                v_dice[key].append(volume_dice)
                j_phis[key].append(j_phi_perc_2)
                psnr_dict[key].append(psnr_val)
                nmi_dict[key].append(nmi_val)
                ncc_dict[key].append(ncc_val.cpu().numpy())
                mind_dict[key].append(mind_val.cpu().numpy())
                ssim_dict[key].append(ssim_val.cpu().numpy())
                rmse_dict[key].append(rmse_val.cpu().numpy())
                jc_dict[key].append(jc_val)
        
        time_np = np.array(time_ls)
        print(f"Average inference time: {time_np.mean():.4f} ms")
        print(f"Average FPS: {1000 / time_np.mean():.4f}")
        
        all_info = ''
        dice_info = ''
        psnr_info = ''
        nmi_info = ''
        ssim_info = ''
        ncc_info = ''
        mind_info = ''
        rmse_info = ''
        jc_info = ''
        
        sorted_key_ls = sorted(v_dice)
        sorted_v_dice = {key:v_dice[key] for key in sorted_key_ls}
        for k in sorted_v_dice.keys():
            print(f'\n{k}:')

            p_dice_np = np.array(p_dice[k])
            asd_np = np.array(asd[k])
            hd_95_np = np.array(hd_95[k])
            so_np = np.array(so[k])
            sd_np = np.array(sd[k])
            v_dice_np = np.array(v_dice[k])
            j_phis_np = np.array(j_phis[k])
            psnr_np = np.array(psnr_dict[k])
            nmi_np = np.array(nmi_dict[k])
            ssim_np = np.array(ssim_dict[k])
            rmse_np = np.array(rmse_dict[k])
            ncc_np = np.array(ncc_dict[k])
            mind_np = np.array(mind_dict[k])
            jc_np = np.array(jc_dict[k])

            static_p_dice = [p_dice_np.mean(0), p_dice_np.std(0)]
            static_asd = [asd_np[:,0].mean(), asd_np[:,0].std(),asd_np[:,1].mean(), asd_np[:,1].std()]
            static_hd_95 = [hd_95_np.mean(), hd_95_np.std()]
            static_so = [so_np[:,0].mean(), so_np[:,0].std(),so_np[:,1].mean(), so_np[:,1].std()]
            static_sd = [sd_np.mean(), sd_np.std()]
            static_v_dice = [v_dice_np.mean(), v_dice_np.std()]
            static_jc = [jc_np.mean(),jc_np.std()]
            static_j_phis = [j_phis_np.mean(), j_phis_np.std()]
            static_psnr = psnr_np.mean()
            static_nmi = nmi_np.mean()
            static_ssim = ssim_np.mean()
            static_rmse = rmse_np.mean()
            static_ncc = ncc_np.mean()
            static_mind = mind_np.mean()

            print(static_asd)
            print(static_hd_95)
            print(static_so)
            print(static_sd)
            print(static_v_dice)
            print(static_jc)
            print(static_j_phis)
            print(static_psnr)
            print(static_nmi)
            print(static_ssim)
            print(static_rmse)
            print(static_ncc)
            print(static_mind)

            dice_info = dice_info + k + "," + str(static_p_dice[0].squeeze(-1))[1:-1].replace('\n','') .replace(' ',',').replace(',,',',').replace(',,',',')+ '\n'
            all_info = all_info + '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(k,
                        static_asd[0],static_asd[1],static_asd[2],static_asd[3], static_hd_95[0], static_hd_95[1], static_so[0], static_so[1], static_so[2], static_so[3], 
                        static_sd[0], static_sd[1],static_v_dice[0], static_v_dice[1], static_jc[0],static_jc[1], static_j_phis[0], static_j_phis[1], static_psnr, 
                        static_ssim, static_rmse, static_nmi, static_ncc, static_mind)
            
            psnr_info += k
            nmi_info += k
            ssim_info += k
            ncc_info += k
            mind_info += k
            rmse_info += k
            jc_info += k
            for i in range(len(asd[k])):
                psnr_info = psnr_info +  ',' + str(psnr_dict[k][i])
                nmi_info = nmi_info + ',' + str(nmi_dict[k][i])
                ssim_info = ssim_info + ',' + str(ssim_dict[k][i])
                rmse_info = rmse_info + ',' + str(rmse_dict[k][i])
                ncc_info = ncc_info + ',' + str(ncc_dict[k][i])
                mind_info = mind_info + ',' + str(mind_dict[k][i])
                jc_info = jc_info + ',' + str(jc_dict[k][i])
            
            psnr_info += '\n'
            nmi_info += '\n'
            ssim_info += '\n'
            ncc_info += '\n'
            mind_info += '\n'
            rmse_info += '\n'
            jc_info += '\n'
            
        with open('log_test_detail.csv', 'a') as f:  # 设置文件对象
            print(all_info, flush=True, file = f)
        with open('log_test_dice_detail.csv', 'a') as f:  # 设置文件对象
            print(dice_info, flush=True, file = f)  
        with open('log_test_psrn_detail.csv', 'a') as f:  # 设置文件对象
            print(psnr_info, flush=True, file = f)
        with open('log_test_ssim_detail.csv', 'a') as f:  # 设置文件对象
            print(ssim_info, flush=True, file = f)
        with open('log_test_rmse_detail.csv', 'a') as f:  # 设置文件对象
            print(rmse_info, flush=True, file = f)
        with open('log_test_nmi_detail.csv', 'a') as f:  # 设置文件对象
            print(nmi_info, flush=True, file = f)  
        with open('log_test_ncc_detail.csv', 'a') as f:  # 设置文件对象
            print(psnr_info, flush=True, file = f)
        with open('log_test_mind_detail.csv', 'a') as f:  # 设置文件对象
            print(psnr_info, flush=True, file = f)

def plot_dvf(model, data_loader, args, jpg_path=None, model_size=None):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')  

    model.modal_encoder.eval()
    model.morph_encoder.eval()
    model.gen.eval()
    model.reg.eval()

    thresholds = np.linspace(0,1,101)
    stn = SpatialTransformer(args.img_shape).to(device)
    model_size = (64,128,128)
    # model_size = (64,64,64)
    stn = SpatialTransformer((64,256,256)).to(device)
    flowshow = FlowShow((64,64,64), 'grid_pic.jpg', device)
    dice_dict = defaultdict(list)
    ncc = NCC([9]*3).loss
    with torch.no_grad():
        for movings, fixeds, moving_lab, fixed_lab,  m_type, f_type, \
            moving_name, fixed_name, moving_origin, fixed_origin, m_sp, f_sp in data_loader:
            movings = movings.to(device).float()
            fixeds = fixeds.to(device).float()
            moving_lab = moving_lab.to(device).float()
            fixed_lab = fixed_lab.to(device).float()
            
            if model_size is not None:
                image_shape = movings.shape[2:]
                tmp_movings = torch.nn.functional.interpolate(movings, size=model_size, mode='trilinear')
                tmp_fixeds = torch.nn.functional.interpolate(fixeds, size=model_size, mode='trilinear')
                flows, warpeds, gen_mf = model(tmp_movings, tmp_fixeds, m_type, f_type)
                # flows, warpeds = model.infer(tmp_movings, tmp_fixeds)  # 直接推断，不包括模态转换模块
                gen_mf = torch.nn.functional.interpolate(gen_mf, size=image_shape, mode='trilinear')
                flows = [torch.nn.functional.interpolate(flow, size=image_shape, mode='trilinear') for flow in flows]
            else:
                flows, warpeds, gen_mf = model(movings, fixeds, m_type, f_type)
                # flows, warpeds = model.infer(movings, fixeds) # 直接推断，不包括模态转换模块
            
            warped_lab = moving_lab
            warped_img = movings
            warped_labs = []
            warped_imgs = []
            
            # tmp = torch.round(gen_mf, decimals=2)
            # gen_mf = (gen_mf-gen_mf.min()) / (gen_mf.max()-gen_mf.min())
            # gen_mf[tmp==0] = 0
            
            warped_img_gens = [gen_mf]
            err = fixeds-gen_mf
            # mean, var = err.mean(), err.std()
            # err = torch.max(mean-3*var, err)
            # err = torch.min(mean+3*var, err)
            err = abs(err)
            # err = err / err.max()

            errors = [err]
            errors_val = [err.sum(dim=[-1,-2])]
            
            for flow in flows:
                warped_lab = stn(warped_lab, flow)
                warped_img = stn(warped_img, flow)
                gen = stn(warped_img_gens[-1], flow)

                # gen_mf = (gen_mf-gen_mf.min()) / (gen_mf.max()-gen_mf.min())
                
                err = fixeds-gen
                # mean, var = err.mean(), err.std()
                # err = torch.max(mean-3*var, err)
                # err = torch.min(mean+3*var, err)
                err = abs(err)
                # err = err / err.max()

                warped_labs.append(warped_lab)
                warped_imgs.append(warped_img)
                warped_img_gens.append(gen)
                errors.append(err)
                errors_val.append(err.sum(dim=[-1,-2]))

            for b in range(len(m_type)):

                key = m_type[b] + '-' + f_type[b]

                if True and jpg_path is not None:
                    moving_lab_np = moving_lab.cpu().numpy().astype('uint8')
                    fixed_lab_np = fixed_lab.cpu().numpy().astype('uint8')
                    warped_lab_np = warped_lab.cpu().numpy().astype('uint8')
                    movings_np = (movings.permute(0,2,3,4,1).repeat(1,1,1,1,3).cpu().numpy()*255).astype('uint8')
                    fixeds_np = (fixeds.permute(0,2,3,4,1).repeat(1,1,1,1,3).cpu().numpy()*255).astype('uint8')
                    warped_img_np = (warped_img.permute(0,2,3,4,1).repeat(1,1,1,1,3).cpu().numpy()*255).astype('uint8')
                    warped_img_gens_np = [(gen.permute(0,2,3,4,1).repeat(1,1,1,1,3).cpu().numpy()*255).astype('uint8') for gen in warped_img_gens]
                    errors_np = [(err.permute(0,2,3,4,1).repeat(1,1,1,1,3).cpu().numpy()*255).astype('uint8') for err in errors]
                    warped_labs_np = [lab.cpu().numpy().astype('uint8') for lab in warped_labs]

                    out_dir = Path(jpg_path) / (m_type[b] + '-' + f_type[b])
                    out_dir.mkdir(exist_ok=True)
                    out_dir_name = out_dir / moving_name[b]
                    out_dir_name.mkdir(exist_ok=True)
                    for z in range(movings.shape[2]):
                        vutils.save_image(moving_lab[b,0,z], str(out_dir_name / f'{z}_m_l.jpg'), normalize=True)
                        vutils.save_image(fixed_lab[b,0,z], str(out_dir_name / f'{z}_f_l.jpg'), normalize=True)
                        tmp_dice = vali.dice_coeff(warped_lab[b,0,z:z+1], fixed_lab[b,0,z:z+1])
                        cv2.imwrite(str(out_dir_name / '{}_w_l_{:.4f}.jpg'.format(z, tmp_dice)), warped_lab[b,0,z].cpu().numpy()*255)

                        if True:
                            # 绘制边线
                            m_slice = movings_np[b,z]
                            f_slice = fixeds_np[b,z]
                            
                            #moving_image
                            if moving_lab_np[b,0,z].sum() > np.prod(moving_lab_np.shape[-2:])*0.001:
                                # _, binary = cv2.threshold(moving_lab_np[b,0,z]*255, 127, 255, cv2.THRESH_BINARY)
                                # b1, b2= cv2.findContours( binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                m_contours, m_hierarchy = cv2.findContours(moving_lab_np[b,0,z], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                # m_slice = cv2.drawContours(m_slice, m_contours, -1, (0， 255, 255), 1)  # 青色RGB
                                m_slice = cv2.drawContours(m_slice, m_contours, -1, (255, 255, 0), 1)  # 青色BGR
                            cv2.imwrite(str(out_dir_name / f'{z}_m.jpg'), m_slice)
                            # fixed image
                            f_contours = None
                            if fixed_lab_np[b,0,z].sum() > np.prod(fixed_lab_np.shape[-2:])*0.001:
                                f_contours, f_hierarchy = cv2.findContours(fixed_lab_np[b,0,z], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                f_slice =  cv2.drawContours(f_slice, f_contours, -1, (0, 255, 0), 1)  # 绿色RGB
                            cv2.imwrite(str(out_dir_name / f'{z}_f.jpg'), f_slice)

                            for i in range(len(warped_img_gens)):
                                t_slice = warped_img_gens_np[i][b,z]
                                e_slice = errors_np[i][b,z]
                                err_val = errors_val[i][b,0,z]
                                
                                # label
                                # w_contours = None
                                # if i > 0:
                                #     lab_slice = warped_labs_np[i-1][b,0,z]                                
                                #     if lab_slice.sum() > np.prod(lab_slice.shape[-2:])*0.001:
                                #         w_contours, w_hierarchy = cv2.findContours(lab_slice, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                                #         tmp_dice = round(vali.dice_coeff(lab_slice, fixed_lab_np[b,0,z]), 4)
                                #         cv2.imwrite(str(out_dir_name / f'{z}_w_l_{i}_{tmp_dice}.jpg'), lab_slice*255)
                                
                                # if w_contours is not None:
                                #     t_slice = cv2.drawContours(t_slice, w_contours, -1, (0, 0, 255), 1)  # 红色BGR
                                #     e_slice = cv2.drawContours(e_slice, w_contours, -1, (0, 0, 255), 1)  # 红色BGR
                                if f_contours is not None:
                                    t_slice =  cv2.drawContours(t_slice, f_contours, -1, (0, 255, 0), 1)  # 绿色RGB=BGR
                                    e_slice = cv2.drawContours(e_slice, f_contours, -1, (0, 255, 0), 1)  # 绿色RGB=BGR

                                cv2.imwrite(str(out_dir_name / f'{z}_t_{i}.jpg'), t_slice)
                                cv2.imwrite(str(out_dir_name / f'{z}_e_{i}_{err_val:.4f}.jpg'), e_slice)                   


def tsne(model, args, epoch=0):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu') 
    dictX = defaultdict(list)
    dicty = defaultdict(list)
    test_dataset = AbdomenDatasetTSNE(Path(args.test_image_dir), seqs=['CMP','UP','NP'])
    # test_dataset = BraTSTSNE(Path('E:\\datasets\\BRAST2018\\MICCAI_BraTS_2018_Data_LGG_Test'), seqs=['t1','t21','t1ce1'], resize=(64, 64, 64))
    data_loader = Data.DataLoader(test_dataset,batch_size=1,shuffle=False,num_workers=0)

    with torch.no_grad():
        for movings, img_type in data_loader:
            x = movings.float().to(device)
            for d in range(model.stage_num):
                x = model.encode_blocks[d](x)
                dictX[d].append(x.flatten().cpu().numpy())
                dicty[d].append(img_type)
        
        for d in dictX.keys():
            if d < 2:
                continue
            tmpX = np.stack(dictX[d]).squeeze()
            tmpy = np.stack(dicty[d])
            tmpX.squeeze()
            ts = TSNE(n_components=2)
            # 训练模型
            X_tsne = ts.fit_transform(tmpX)
            plt.figure()
            handle = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=tmpy, label="t-SNE")
            # plt.legend(handles=handle.legend_elements()[0],labels=['t1','t21','t1ce1'],title="Classes")
            plt.legend(handles=handle.legend_elements()[0],labels=['CMP','T1w','NP'],title="Classes")
            plt.savefig(Path(args.root_path)/'tsne'/f'{epoch}_{d}.png', dpi=300)
            plt.close()
            # 显示图像
            # plt.show()
            

def infer_trans_jpg(trans, dataloader, args, epoch=0):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu' )    
    trans.eval()

    jpg_path = Path(args.root_path) / args.test_jpg_outpath

    with torch.no_grad():
       for movings, fixeds, moving_lab, fixed_lab,  p_type_m, p_type_f, \
            moving_name, fixed_name, moving_origin, fixed_origin, m_sp, f_sp in dataloader:
            movings = movings.to(device).float()
            fixeds = fixeds.to(device).float()

            mf_trans = trans(movings, fixeds, p_type_m, p_type_f)[0]

            for b in range(len(p_type_m)):
                out_dir = Path(jpg_path) / (p_type_m[b] + '-' + p_type_f[b])
                out_dir.mkdir(exist_ok=True)
                out_dir_name = out_dir / moving_name[b]
                out_dir_name.mkdir(exist_ok=True)
                for z in range(movings.shape[2]):
                    vutils.save_image(movings[b,0,z], str(out_dir_name / f'{z}_m.jpg'), normalize=True)
                    vutils.save_image(fixeds[b,0,z], str(out_dir_name / f'{z}_f.jpg'), normalize=True)
                    vutils.save_image(mf_trans[b,0,z], str(out_dir_name / f'{z}_t.jpg'), normalize=True)


def test_ssim(model, dataloader, args):
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu' )    

    model.eval()
    ssim = SSIM()
    jpg_path = Path(args.root_path) / args.test_trans_jpg
    lum_dict=defaultdict(list)
    cont_dict=defaultdict(list)
    stru_dict=defaultdict(list)
    with torch.no_grad():
        for movings, fixeds, moving_lab, fixed_lab,  p_type_m, p_type_f, \
                moving_name, fixed_name, moving_origin, fixed_origin, m_sp, f_sp in dataloader:
            input_img = movings.to(device).float()
            input_fixed = fixeds.to(device).float()
            moving_lab = moving_lab.to(device).float()
            fixed_lab = fixed_lab.to(device).float()

            mf_trans, _, _ = model.test2(input_img, input_fixed, p_type_m, p_type_f)
            
            for b in range(len(p_type_m)):
                key = p_type_m[b] + '-' + p_type_f[b]
                out_dir = Path(jpg_path) / key
                out_dir.mkdir(exist_ok=True)
                out_dir_name = out_dir / moving_name[b]
                out_dir_name.mkdir(exist_ok=True)
                lum_acc = 0
                cont_acc = 0
                stru_acc = 0
                for z in range(movings.shape[2]):
                    vutils.save_image(input_img[b,0,z], str(out_dir_name / f'{z}_m.jpg'), normalize=True)
                    vutils.save_image(input_fixed[b,0,z], str(out_dir_name / f'{z}_f.jpg'), normalize=True)
                    vutils.save_image(mf_trans[b,0,z], str(out_dir_name / f'{z}_t.jpg'), normalize=True)
                    lum, cont, stru = ssim.ssim2(mf_trans[b,0:1,z:z+1], input_fixed[b,0:1,z:z+1], input_img[b,0:1,z:z+1])
                    lum_acc = lum_acc + lum
                    cont_acc = cont_acc + cont
                    stru_acc =stru_acc + stru
                
                lum = lum_acc / movings.shape[2]
                cont = cont_acc / movings.shape[2]
                stru = stru_acc / movings.shape[2]
                lum_dict[key].append(lum.cpu())
                cont_dict[key].append(cont.cpu())
                stru_dict[key].append(stru.cpu())

        info = ''
        exp_str = ''
        for key in lum_dict.keys():
            lum_np = np.array(lum_dict[key])
            cont_np = np.array(cont_dict[key])
            stru_np = np.array(stru_dict[key])
            info = info + '{},lum:{:.4e},cont:{:.4e},stru:{:.4e}\n'.format(key, lum_np.mean(), cont_np.mean(), stru_np.mean())
            exp_str = exp_str + f'{key},{lum_np.mean()},{cont_np.mean()},{stru_np.mean()}\n'
        print(info)
        with open(str(Path(args.root_path) / 'ssim_log.csv'), 'a')as f :
            print(exp_str, flush=True, file = f)

def calc_param():
    model = PMARNet3(args)
    counter = ModelParameterCounter()
    print("=== 模型参数分析演示 ===")
    
    # 详细统计
    counter.count_parameters(model, detailed=True)
    
    # 快速统计
    print(f"\n快速统计:")
    print(f"总参数量: {quick_param_count(model):,}")
    print(f"可训练参数: {quick_trainable_param_count(model):,}")
    
    # 按层统计
    layer_stats = counter.count_parameters_by_layer(model)
    print(f"\n按层类型统计:")
    for layer_type, stats in sorted(layer_stats.items(), key=lambda x: x[1]['params'], reverse=True):
        percentage = (stats['params'] / quick_param_count(model)) * 100
        print(f"  {layer_type:>15}: {stats['count']:>3} 层, {stats['params']:>8,} 参数 ({percentage:>5.1f}%)")
        
    # 进行模型比较
    models_to_compare ={'model1':None, 'model2':None}
    analyzer = AdvancedModelAnalyzer()
    comparison_results = analyzer.compare_models(models_to_compare)
    
    # 分析特定模型的内存使用
    sample_model = list(models_to_compare.values())[0]
    memory_stats = analyzer.analyze_memory_usage(sample_model, (1, 3, 32, 32))
    
def test_plot(model, data_loader, args, jpg_path=None):
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')  

    # pmar.modal_encoder.eval()
    # pmar.morph_encoder.eval()
    pmar.encoder.eval()
    pmar.gen.eval()
    pmar.reg.eval()
    
    p_dice = defaultdict(list)
    asd = defaultdict(list)
    hd_95 = defaultdict(list)
    so = defaultdict(list)
    sd = defaultdict(list)
    v_dice = defaultdict(list)
    j_phis = defaultdict(list)

    thresholds = np.linspace(0,1,101)
    stn = SpatialTransformer(args.img_shape).to(device)
    with torch.no_grad():
        for movings, fixeds, moving_lab, fixed_lab,  m_type, f_type, \
            moving_name, fixed_name, moving_origin, fixed_origin, m_sp, f_sp in data_loader:
            movings = movings.to(device).float()
            fixeds = fixeds.to(device).float()
            moving_lab = moving_lab.to(device).float()
            fixed_lab = fixed_lab.to(device).float()
            
            gen_mf, flows, warpeds = model.test2(movings, fixeds, m_type, f_type)

            warped_lab = moving_lab
            warped_img = gen_mf
            if isinstance(flows, list):
                for flow in flows:
                    warped_lab = stn(warped_lab, flow)
                    warped_img = stn(warped_img, flow)
            else:
                warped_lab = stn(warped_lab, flow)
                warped_img = stn(warped_img, flow)
                        
            for b in range(len(m_type)):
                j_phi = torch.tensor(jacobian_determinant(flows[0][b].permute(1,2,3,0).cpu().numpy()))
                j_phi_perc = (j_phi<=0).sum() / j_phi.numel()
                # j_phi = torch.tensor(jacobian_determinant(flows[1][i].permute(1,2,3,0).cpu().numpy()))
                # j_phi_perc = (j_phi_perc + (j_phi<=0).sum() / j_phi.numel()) / 2
                key = m_type[b] + '-' + f_type[b]

                dice_ = []
                for threshold in thresholds:
                    dice_.append(vali.dice2(warped_lab[b], fixed_lab[b], threshold))
                index = np.array(dice_).argmax()
                moving_lab = warped_lab>thresholds[index]

                moving_lab_np = moving_lab.squeeze(0).squeeze(0).cpu().numpy()
                fixed_lab_np = fixed_lab.squeeze(0).squeeze(0).cpu().numpy().astype('bool')
                # print('type:{},msp:{}——type:{},fsp:{}'.format(p_type[0],m_sp,p_type_fixed[0],f_sp))
                # spacing_mm = torch.tensor(moving_origin.shape[-3:])*m_sp/torch.tensor([64,128,128])
                surface_distances = surfdist.compute_surface_distances(fixed_lab_np, moving_lab_np, spacing_mm=m_sp[b])
                avg_surf_dist = surfdist.compute_average_surface_distance(surface_distances) # len=2
                hd_dist_95 = surfdist.compute_robust_hausdorff(surface_distances, 95) # len=1
                surface_overlap = surfdist.compute_surface_overlap_at_tolerance(surface_distances, 1) # len=2
                surface_dice = surfdist.compute_surface_dice_at_tolerance(surface_distances, 1) # len=1
                volume_dice = surfdist.compute_dice_coefficient(fixed_lab_np.astype('int'), moving_lab_np.astype('int')) # len=1

                p_dice[key].append(dice_)
                asd[key].append(avg_surf_dist)
                hd_95[key].append(hd_dist_95)
                so[key].append(surface_overlap)
                sd[key].append(surface_dice)
                v_dice[key].append(volume_dice)
                j_phis[key].append(j_phi_perc)
        
        asd_info = ''
        hd_info = ''
        for k in v_dice.keys():
            asd_info += k
            hd_info += k

            p_dice_np = np.array(p_dice[k])
            asd_np = np.array(asd[k])
            hd_95_np = np.array(hd_95[k])
            so_np = np.array(so[k])
            sd_np = np.array(sd[k])
            v_dice_np = np.array(v_dice[k])
            j_phis_np = np.array(j_phis[k])

            # static_p_dice = [p_dice_np.mean(0), p_dice_np.std(0)]
            # static_asd = [asd_np[:,0].mean(), asd_np[:,0].std(),asd_np[:,1].mean(), asd_np[:,1].std()]
            # static_hd_95 = [hd_95_np.mean(), hd_95_np.std()]
            # static_so = [so_np[:,0].mean(), so_np[:,0].std(),so_np[:,1].mean(), so_np[:,1].std()]
            # static_sd = [sd_np.mean(), sd_np.std()]
            # static_v_dice = [v_dice_np.mean(), v_dice_np.std()]
            # static_j_phis = [j_phis_np.mean(), j_phis_np.std()]

            if len(asd[k]) != len(hd_95[k]):
                print(len(asd[k]))
                print(len(hd_95[k]))
                warnings.warn('样本数量不一致')
            for i in range(len(asd[k])):
                asd_info = asd_info +  ',' + str(asd[k][i][0])
                hd_info = hd_info + ',' + str(hd_95[k][i])

            asd_info += '\n'
            hd_info += '\n'    
            
        with open('log_asd_test.csv', 'a') as f:  # 设置文件对象
            print(asd_info, flush=True, file = f)
        with open('log_hd95_test.csv', 'a') as f:  # 设置文件对象
            print(hd_info, flush=True, file = f)


if __name__ == "__main__":

    # calc_param()
    
    device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(Path(args.root_path) / args.test_tb)

    pmar = PMARNet3(args)
    pmar.load_model("weights/Brats_646464_mmarn_best.pt", is_best=True)
    
    # test_detail_dataset = AbdomenDataset4(Path(args.test_image_dir), Path(args.test_label_dir), Path(args.origin_dir), fixed_seqs=['CMP','UP','NP'], moving_seqs=['CMP','UP','NP'], same_skip=True)
    # test_detail_dataset = AbdomenDataset6(Path("kidney_processed/test-PARNet-new/moving_img"), 
    #                                       Path("kidney_processed/test-PARNet-new/moving_label"), 
    #                                       Path( "kidney_2021-01-06_WKQ/reg_origin_test"), 
    #                                       resize=(64,128,128), fixed_seqs=['CMP','NP','UP'], moving_seqs=['CMP','NP','UP'], same_skip=True)


    test_detail_dataset = BraTSRegDataset2(Path('BRAST2018/MICCAI_BraTS_2018_Data_LGG_Test'), fixed_seqs=['t1','t21','t1ce1'], moving_seqs=['t1','t21','t1ce1'],  
                                           resize=(64, 64, 64), label='WT', flag=2)
    
    test_detail_loader = Data.DataLoader(test_detail_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    jpg_path = Path(args.root_path) / args.test_jpg_outpath
    dvf_path = Path(args.root_path) / 'test_dvf_img'
    # tsne(pmar.modal_encoder, args)
    test_detail(pmar, test_detail_loader, args, jpg_path)
