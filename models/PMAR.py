from models.UNet7 import Trans, Reg2, SpatialTransformer
from models.base_networks import VTNAffineStem
from models.discriminator import Discriminator
from datagenerators import AbdomenDataset1, AbdomenDataset2
import models.losses as losses
import vali

import torch
import torch.utils.data as Data
import numpy as np
import time

from config import args
from collections import defaultdict
from pathlib import Path
from  torchvision import utils as vutils
from models.utils import FlowShow
from models.voxelmorph.networks import VxmDense

from torch.utils.tensorboard import SummaryWriter

class PMARNet(torch.nn.Module):
    def __init__(self, args, train_reg=True):
        super(PMARNet,self).__init__()
        self.args = args
        self.train_reg = train_reg
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu' )
        self.writer = SummaryWriter(Path(args.root_path) / args.train_tb)
        
        # 数据集加载
        abdomen_dataset = AbdomenDataset1(Path(args.train_dir),same_skip=True)
        self.train_loader = Data.DataLoader(abdomen_dataset,batch_size=args.batch_size,shuffle=False,num_workers=0, drop_last=True)

        test_dataset = AbdomenDataset2(Path(args.test_image_dir),Path(args.test_label_dir),same_skip=True)
        self.test_dataloader = Data.DataLoader(test_dataset,batch_size=args.test_batch_size,shuffle=False,num_workers=0,drop_last=True)

        #model加载
        id_map = torch.FloatTensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]).to(self.device)
        self.trans = Trans(inshape=args.img_shape,up_sample=True, freeze_e1=False, freeze_e2=False, freeze_d=False).to(self.device)
        self.affreg = VTNAffineStem(dim=len(args.img_shape), im_size=args.img_shape[0], id_map=id_map).to(self.device)
        self.defreg = Reg2(inshape=args.img_shape).to(self.device)
        # self.defreg = VxmDense().to(self.device)
        
        self.net_D = Discriminator().to(self.device)
        self.stn = SpatialTransformer(args.img_shape).to(self.device)

        self.affine = VTNAffineStem(dim=len(args.img_shape), im_size=args.img_shape[0],id_map=id_map).to(self.device)

        # initialize optimizer
        if args.adam:
            self.optimizerD = torch.optim.Adam(self.net_D.parameters(), lr=args.lrD, betas=(args.beta1, 0.999))
            self.optimizerT = torch.optim.Adam( filter(lambda p: p.requires_grad, self.trans.parameters()), lr=args.lrG, betas=(args.beta1, 0.999))
        else:
            self.optimizerD = torch.optim.RMSprop(self.net_D.parameters(), lr = args.lrD)
            self.optimizerT = torch.optim.RMSprop( filter(lambda p: p.requires_grad, self.trans.parameters()), lr = args.lrG)
        self.optimizerR = torch.optim.Adam(list(self.defreg.parameters()) + list(self.affreg.parameters()), lr=args.lr)

        if train_reg:
            self.optimize = self.optimize_reg
        else:
            self.optimize = self.optimize_trans

        self.train_data_len = len(self.train_loader)
        self.now_epoch = 0
        self.recoder = defaultdict(list)
        self.global_step = 0
        self.now_epoch = 0
        self.Diters = args.Diters

        self.ncc_loss = losses.NCC().loss
        self.flowshow = FlowShow(args.img_shape, 'grid_pic.jpg', self.device)
        self.one = torch.FloatTensor([1]).to(self.device)
        self.mone = self.one * -1

    def load_model(self, load_path):
        cpk = torch.load(load_path, map_location=torch.device(self.device))
        for key in cpk.keys():
            obj = getattr(self, key)
            obj.load_state_dict(cpk[key])

    def save_model(self, epoch):
        if self.train_reg:
            ckp = {}
            ckp['trans'] = self.trans.state_dict()
            ckp['defreg'] = self.defreg.state_dict()
            ckp['net_D'] = self.net_D.state_dict()
            ckp['optimizerD'] = self.optimizerD.state_dict()
            ckp['optimizerR'] = self.optimizerR.state_dict()
            ckp['optimizerT'] = self.optimizerT.state_dict()
            ckp['now_epoch'] = epoch
            torch.save(ckp, Path(args.root_path) / args.model_dir / ('parnet_%04d.pt' % epoch))
        else:
            ckp = {}
            ckp['trans'] = self.trans.state_dict()
            ckp['optimizerT'] = self.optimizerT.state_dict()
            ckp['now_epoch'] = epoch
            torch.save(ckp, Path(args.root_path) / args.model_dir / ('Trans_encoder_%04d.pt' % epoch))

    def write_tb(self, epoch):
        for k, v in self.recoder.items():
            loss_np = np.array(v)
            self.writer.add_scalar = (f'train/{k}', loss_np.mean(), epoch)

    def print_loss(self, epoch_info, is_write=False):
        count = 1
        loss_info = ''
        for k, v in self.recoder.items():
            loss_info = loss_info + '{}:{:.4e},'.format(k, np.array(v).mean())
            if count % 4 == 0:
                loss_info = loss_info + '\n'
            count += 1

        print(loss_info)
        if is_write:
            with open('log_train_loss.txt', 'a') as f:  # 设置文件对象
                print(epoch_info, flush=True, file = f)
                print(loss_info, flush=True, file = f)
            
    def clear_recoder(self):
        self.record = defaultdict(list)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def optimizeD(self,movings,fixeds,m_type, f_type):
        self.optimizerD.zero_grad()        
        self.set_requires_grad(self.net_D, True)
        for p in self.net_D.parameters():
            p.data.clamp_(args.clamp_lower, args.clamp_upper)

        errD_real = self.net_D(movings)
        self.recoder['errD_real'].append(errD_real.item())
        errD_real.backward(self.one)

        # train with fake
        self.forwardT(movings, fixeds, m_type, f_type)

        # noise = torch.rand(args.batchSiz,1,512,4,8,8).float().cuda()
        # noise.resize_(args.batchSize, 1, 512, 1).normal_(0, 1)
        # noisev = Variable(noise, volatile = True) # totally freeze netG
        # fake = Variable(netG(noisev).data)

        errD_fake = self.net_D(self.Tf)
        self.recoder['errD_fake'].append(errD_real.item())
        errD_fake.backward(self.mone)
        errD = errD_real - errD_fake
        self.recoder['errD'].append(errD.item())
        self.optimizerD.step()

    def forwardT(self, movings, fixeds, m_type, f_type, is_trans=True):
        # trans状态下只需要
        if is_trans:
            Tf, proto_loss, contra_loss = self.trans(movings, fixeds, m_type, f_type, is_trans)
            self.Tf = Tf
            return proto_loss, contra_loss
        else :
            Tf, proto_loss, contra_loss, style_loss, consis_loss = self.trans(movings, fixeds, m_type, f_type, is_trans)
            self.Tf = Tf
            return proto_loss, contra_loss, style_loss, consis_loss
    
    def forwardR(self, movings, fixeds):
        self.aff_flow, aff_loss = self.affreg(movings,fixeds)
        self.aff_warped = self.stn(self.Tf, self.aff_flow)
        self.deform_flow, _ = self.defreg(self.aff_warped, fixeds) # 计算仿射和柔性的flow
        self.deform_warped = self.stn(self.aff_warped, self.deform_flow) # 计算包装后的图像
        return aff_loss
    
    def optimizeT(self, movings, fixeds, m_type, f_type):
        self.optimizerT.zero_grad()
        self.set_requires_grad(self.net_D, False)
        # for p in self.net_D.parameters():
            # p.requires_grad = False # to avoid computation
        
        proto_loss, contra_loss = self.forwardT(movings, fixeds, m_type, f_type)
        sim_loss = torch.nn.functional.l1_loss(self.Tf, fixeds)

        # total_loss = errG + 1e-2*sim_loss + reg_loss + 1e2*proto_loss + style_loss
        total_loss =  sim_loss + 1e1*proto_loss + 1e2*contra_loss
        
        self.recoder['sim_loss'].append(sim_loss.cpu().item())
        self.recoder['proto_loss'].append(proto_loss.cpu().item())
        self.recoder['contra_loss'].append(contra_loss.cpu().item())
        self.recoder['total_loss'].append(total_loss.cpu().item())
        
        total_loss.backward()
        self.optimizerT.step()
        self.optimizerT.zero_grad()                       


    def optimizeR(self, movings, fixeds, m_type, f_type):
        self.optimizerT.zero_grad()
        # for p in self.net_D.parameters():
        #     p.requires_grad = False # to avoid computation
        self.set_requires_grad(self.net_D, False)
        proto_loss, contra_loss, style_loss, consis_loss = self.forwardT(movings, fixeds, m_type, f_type, is_trans=False)

        aff_loss = self.forwardR(movings, fixeds)

        sim_loss = torch.nn.functional.l1_loss(fixeds, self.deform_warped) + torch.nn.functional.l1_loss(fixeds, self.aff_warped)

        # ncc_loss = ncc_loss_func(fixeds, stn(movings, deform_flow))
        ncc_loss = self.ncc_loss(fixeds, self.deform_warped)
        # co_sim_loss = mse_loss_func(mf_warped_sec, mf_warped_first)

        reg_loss = losses.regularize_loss_3d(self.deform_flow)

        errG = self.net_D(self.Tf)
        total_loss = 1e2*errG + sim_loss + reg_loss + aff_loss + 1e1*style_loss + 1e1*consis_loss + 1e2*proto_loss + 1e2*contra_loss
        # total_loss =  errG + sim_loss + reg_loss + proto_loss + contra_loss + style_loss + consis_loss
        
        self.recoder['sim_loss'].append(sim_loss.cpu().item())
        self.recoder['co_sim_loss'].append(ncc_loss.cpu().item()) # co_sim_loss_list.append(co_sim_loss.cpu().item())
        self.recoder['errG'].append(errG.item())
        self.recoder['reg_loss'].append(reg_loss.item())
        self.recoder['proto_loss'].append(proto_loss.item())
        self.recoder['contra_loss'].append(contra_loss.item())
        self.recoder['style_loss'].append(style_loss.item())
        self.recoder['consis_loss'].append(consis_loss.item())
        self.recoder['total_loss'].append(total_loss.cpu().item())
                
        # errG.backward(self.one, retain_graph=True)
        total_loss.backward()
        self.optimizerT.step()
        self.optimizerR.step()


    def optimize_trans(self, epoch, movings, fixeds, m_type, f_type):
        self.global_step += 1
            
        movings = movings.float().to(self.device)
        fixeds = fixeds.float().to(self.device)
                
        # train G
        if self.global_step % args.Diters == 0:
            self.optimizeT(movings, fixeds, m_type, f_type)

        if self.global_step % args.tb_save_freq == 0:
            self.tb_save_step += 1
            ran_id = int((torch.rand(1)*0.8+0.1) * movings.size(2))
            self.writer.add_image('train/trans_m_img', (movings[0,0,ran_id,:,:]).unsqueeze(0), self.tb_save_step)
            self.writer.add_image('train/trans_f_img', (fixeds[0,0,ran_id,:,:]).unsqueeze(0), self.tb_save_step)
            self.writer.add_image('train/trans_mf', (self.Tf[0,0,ran_id,:,:]).unsqueeze(0), self.tb_save_step)

        if self.global_step % (5*500) == 0:
            id = 0
            jpg_path = Path(args.root_path) / args.trans_jpg
            idx = torch.randperm(movings.shape[2])
            for b in range(movings.shape[0]):
                out_dir = jpg_path / f'{m_type[b]}-{f_type[b]}'
                out_dir.mkdir(exist_ok=True)
                for z in range(10):
                    vutils.save_image(movings[0,0,idx[z]].float(), str(out_dir / f'{id}_{m_type[b]}.jpg'), normalize=True)
                    vutils.save_image(fixeds[0,0,idx[z]].float(), str(out_dir / f'{id}_{f_type[b]}.jpg'), normalize=True)
                    vutils.save_image(self.Tf[0,0,idx[z]].float(), str(out_dir / f'{id}_T.jpg'), normalize=True)
                    id += 1

    def optimize_reg(self, epoch, movings, fixeds, m_type, f_type):
        self.global_step += 1
        movings = movings.float().to(self.device)
        fixeds = fixeds.float().to(self.device)
        
        # train D
        self.optimizeD(movings, fixeds, m_type, f_type)
        
        # train G        
        if epoch > 50 or self.global_step % args.Diters == 0:
            self.optimizeR(movings, fixeds, m_type, f_type)

            if self.global_step % args.tb_save_freq == 0:
                self.tb_save_step += 1
                ran_id = int((torch.rand(1)*0.8+0.1) * movings.size(2))
                self.writer.add_image('train/reg_img', (movings[0,0,ran_id,:,:]).unsqueeze(0), self.tb_save_step)
                self.writer.add_image('train/reg_fixed', (fixeds[0,0,ran_id,:,:]).unsqueeze(0), self.tb_save_step)
                self.writer.add_image('train/reg_trans', (self.Tf[0,0,ran_id,:,:]).unsqueeze(0), self.tb_save_step)
                self.writer.add_image('train/reg_aff', (self.aff_warped[0,0,ran_id,:,:]).unsqueeze(0), self.tb_save_step)   
                self.writer.add_image('train/reg_def', (self.deform_warped[0,0,ran_id,:,:]).unsqueeze(0), self.tb_save_step)   

    def train_model(self):
        self.global_step = (self.now_epoch+1) * len(self.train_loader)
        self.tb_save_step = self.global_step // args.tb_save_freq
        for epoch in range(self.now_epoch+1,args.epoch+1):
            epoch_info = "Epoch:{}/{} - ".format(epoch, args.epoch) + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print(epoch_info)
            for movings, fixeds, m_type, f_type in self.train_loader:
                self.optimize(epoch, movings, fixeds, m_type, f_type)

            # 打印信息
            self.write_tb(epoch) 
            self.print_loss(epoch_info, is_write=True)
            self.clear_recoder()

            if self.train_reg and epoch % 25 == 0:
                self.test(epoch)

            if epoch % args.save_per_epoch == 0:
                self.save_model(epoch)

    def test(self, epoch):
        
        self.trans.eval()
        self.defreg.eval()

        dice_dict = defaultdict(list)
        thresholds = np.linspace(0,1,101)
        
        step = 0
        gap = len(self.test_dataloader)

        jpg_path = Path(args.root_path) / args.test_jpg_outpath if args.test_jpg_outpath is not None else None

        with torch.no_grad():
            for movings, fixeds, moving_lab, fixed_lab, p_type_m, p_type_f in self.test_dataloader:
                movings = movings.to(self.device).float()
                fixeds = fixeds.to(self.device).float()
                moving_lab = moving_lab.to(self.device).float()
                fixed_lab = fixed_lab.to(self.device).float()

                self.forwardT(movings, fixeds, p_type_m, p_type_f)
                self.forwardR(self.Tf, fixeds)

                warped_lab = self.stn(moving_lab, self.deform_flow)
                warped_img = self.stn(movings, self.deform_flow)

                ran_id = torch.argmax(torch.sum(fixed_lab,(3,4)),dim=-1)[0].squeeze()
                # ran_id = int((torch.rand(1)*0.8+0.1) * input_img.size(2))
                self.writer.add_image('test_pred/trans_img', (movings[0,0,ran_id,:,:]).unsqueeze(0), epoch*gap+step)
                self.writer.add_image('test_pred/warped_img', (warped_img[0,0,ran_id,:,:]).unsqueeze(0), epoch*gap+step)          
                self.writer.add_image('test_pred/warped_lab', (warped_lab[0,0,ran_id,:,:]).unsqueeze(0), epoch*gap+step)
                warped_fixed = (warped_lab>0)*0.5 + (fixed_lab)*0.5
                self.writer.add_image('test_pred/overlap_lab', (warped_fixed[0,0,ran_id,:,:]).unsqueeze(0), epoch*gap+step)

                flow_vis = self.flowshow.show(self.stn, self.deform_flow)
                self.writer.add_image('test_pred/flow_def', (flow_vis[0,0,ran_id,:,:]).unsqueeze(0), epoch*gap+step)

                self.writer.add_image('test/input_img',(movings[0,0,ran_id,:,:]).unsqueeze(0), epoch*gap+step)
                self.writer.add_image('test/fixed_img',(fixeds[0,0,ran_id,:,:]).unsqueeze(0), epoch*gap+step)
                self.writer.add_image('test/input_lab',(moving_lab[0,0,ran_id,:,:]).unsqueeze(0), epoch*gap+step)
                self.writer.add_image('test/fixed_lab',(fixed_lab[0,0,ran_id,:,:]).unsqueeze(0), epoch*gap+step)
                warped_fixed = (moving_lab>0)*0.5 + (fixed_lab)*0.5
                self.writer.add_image('test/overlap_lab',(warped_fixed[0,0,ran_id,:,:]).unsqueeze(0), epoch*gap+step)
                
                step = step + 1

                for i in range(len(p_type_m)):
                    key = p_type_m[i] + '_' + p_type_f[i]
                    dice_ = []
                    for threshold in thresholds:
                        dice_.append(vali.dice2(warped_lab[i], fixed_lab[i], threshold))
                    dice_dict[key].append(dice_)

            dice_detail = f'\nepoch:{epoch}\n'
            dice_info = f'\nepoch:{epoch}\n'
            for k in dice_dict.keys():
                dice_array =  np.array(dice_dict[k]).squeeze(-1)
                index = dice_array.mean(0).argmax()
                max_v = dice_array.mean(0)[index]
                std_v = dice_array.std(0)[index]
                dice_detail = dice_detail + str(dice_array.mean(0))[1:-1].replace(' ',',').replace('\n',' ') + '\n'
                dice_info = dice_info + f'{k}:{max_v},{std_v},{index}\n' 

            with open('expriment_test.txt', 'a') as f:  # 设置文件对象
                print(dice_detail, flush=True, file = f)
            
            print(dice_info)
            with open('log_test.txt', 'a') as f:  # 设置文件对象
                print(dice_info, flush=True, file = f)

    def test_trans(self, movings, fixeds,  moving_lab, fixed_lab, m_type, f_type):
        with torch.no_grad():
            _, _ = self.forwardT(movings, fixeds, m_type, f_type)
        return self.Tf

