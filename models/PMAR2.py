from models.UNet7 import Trans, Regs, Reg2,  SpatialTransformer
from models.transform import Encoder, Decoder
from models.base_networks import VTNAffineStem
from models.discriminator import Discriminator
from datagenerators import AbdomenDataset1, AbdomenDataset2
from BraTSgenerator import BraTSRegDataset1, BraTSRegDataset2
import models.losses as losses
import vali

import torch
import torch.utils.data as Data
import numpy as np
import time

from config import args
from collections import defaultdict
from pathlib import Path
from torchvision import utils as vutils
from models.utils import FlowShow
from models.voxelmorph.networks import VxmDense
from models.dis import Dis

from torch.utils.tensorboard import SummaryWriter

class PMARNet2(torch.nn.Module):
    def __init__(self, args):
        super(PMARNet2, self).__init__()
        self.args = args
        self.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu' )
        self.writer = SummaryWriter(Path(args.root_path) / args.train_tb)
        
        # 数据集加载
        # abdomen_dataset = AbdomenDataset1(Path(args.train_dir),same_skip=True)
        # self.train_loader = Data.DataLoader(abdomen_dataset,batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

        # test_dataset = AbdomenDataset2(Path(args.test_image_dir), Path(args.test_label_dir), same_skip=True)
        # self.test_dataloader = Data.DataLoader(test_dataset,batch_size=args.test_batch_size, shuffle=False, num_workers=0, drop_last=True)

        #model加载
        self.encoder = Encoder(inshape=args.img_shape, types=args.types, freeze_e1=False, freeze_e2=False).to(self.device)
        self.gen = Decoder().to(self.device)
        self.reg = Regs(VxmDense, 3, args.img_shape).to(self.device)
        # self.reg = Regs(Reg2, 2, args.img_shape, args={'inshape':args.img_shape}).to(self.device)
        self.net_D = Dis().to(self.device)
        self.stn = SpatialTransformer(args.img_shape).to(self.device)

        # initialize optimizer
        # if args.adam:
        if True:
            self.optimizerD = torch.optim.Adam(self.net_D.parameters(), lr=3e-4, betas=(args.beta1, 0.999))
            self.optimizerT = torch.optim.Adam(filter(lambda p: p.requires_grad, self.encoder.parameters()), lr=3e-4, betas=(args.beta1, 0.999))
            self.optimizerG = torch.optim.Adam(self.gen.parameters(), lr=3e-4, betas=(args.beta1, 0.999))
            self.optimizerR = torch.optim.Adam(self.reg.parameters(), lr=args.lr)
        else:
            self.optimizerD = torch.optim.RMSprop(self.net_D.parameters(), lr = args.lrD)
            self.optimizerT = torch.optim.RMSprop( filter(lambda p: p.requires_grad, self.encoder.parameters()), lr = args.lrG)
            self.optimizerG = torch.optim.RMSprop( self.gen.parameters(), lr = args.lrG)
            self.optimizerR = torch.optim.RMSprop(self.reg.parameters(), lr=args.lr)

        self.now_epoch = 0
        self.recoder = defaultdict(list)
        self.global_step = 0
        self.Diters = args.Diters

        self.flowshow = FlowShow(args.img_shape, 'grid_pic.jpg', self.device)

        self.ncc_loss = losses.NCC([9]*3).loss

    def load_model(self, load_path, is_best=True):
        print('load saved model')

        ckp = torch.load(load_path, map_location=torch.device(self.device))
        
        self.encoder.load_state_dict(ckp['encoder'])
        self.gen.load_state_dict(ckp['gen'])
        self.reg.load_state_dict(ckp['reg'])
        self.net_D.load_state_dict(ckp['net_D'])

        if is_best:
            return
        
        self.optimizerD.load_state_dict(ckp['optimizerD'])
        self.optimizerR.load_state_dict(ckp['optimizerR'])
        self.optimizerT.load_state_dict(ckp['optimizerT'])
        self.optimizerG.load_state_dict(ckp['optimizerG'])
        self.now_epoch = ckp['now_epoch']

        # for key in cpk.keys():
        #     obj = getattr(self, key)
        #     if isinstance(obj, torch.nn.Module) or isinstance(obj, torch.optim.Optimizer):
        #         obj.load_state_dict(cpk[key])
        #     else:
        #         setattr(self, key, cpk[key])

    def save_model(self, epoch=0, is_best=False):
        if is_best:
            ckp = {}
            ckp['encoder'] = self.encoder.state_dict()
            ckp['gen'] = self.gen.state_dict()
            ckp['reg'] = self.reg.state_dict()
            ckp['net_D'] = self.net_D.state_dict()
            torch.save(ckp, Path(args.root_path) / args.model_dir / 'pmarnet_best.pt')
            return 
        
        ckp = {}
        ckp['encoder'] = self.encoder.state_dict()
        ckp['gen'] = self.gen.state_dict()
        ckp['reg'] = self.reg.state_dict()
        ckp['net_D'] = self.net_D.state_dict()
        ckp['optimizerD'] = self.optimizerD.state_dict()
        ckp['optimizerR'] = self.optimizerR.state_dict()
        ckp['optimizerT'] = self.optimizerT.state_dict()
        ckp['optimizerG'] = self.optimizerG.state_dict()
        ckp['now_epoch'] = epoch
        torch.save(ckp, Path(args.root_path) / args.model_dir / ('pmarnet_%04d.pt' % epoch))

    def write_tb(self, epoch):
        for k, v in self.recoder.items():
            loss_np = np.array(v)
            self.writer.add_scalar(f'train/{k}', loss_np.mean(), epoch)

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
        self.recoder = defaultdict(list)
    
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

    def forward(self, movings, fixeds, m_type, f_type):
        self.encoder.set_input(movings, fixeds, m_type, f_type)
        s_m_proto, s_f_proto, s_m_content, s_f_content = self.encoder.forward1()
        self.gen_mf = self.gen(s_m_proto, s_f_proto, s_m_content, s_f_content)
        self.flows, self.warpeds = self.reg(movings, fixeds, self.gen_mf)
    
    def infer(self, movings, fixeds):
        flows, warpeds = self.reg.infer(movings, fixeds)
        return flows, warpeds
    
    def backward_dis(self, fixeds):
        dis_loss = self.args.w_dis * self.net_D.dis_loss(self.gen_mf.detach(), fixeds)
        dis_loss.backward()
        self.recoder['dis_loss'].append(self.args.w_dis * dis_loss.item())

    def backward_style_encoder(self):
        total_loss = self.args.w_proto*self.encoder.proto_loss + self.args.w_contra*self.encoder.contra_loss
        total_loss.backward(retain_graph=True)
        self.recoder['proto_loss'].append(self.args.w_proto*self.encoder.proto_loss.item())
        self.recoder['contra_loss'].append(self.args.w_contra*self.encoder.contra_loss.item())

    def backward_encoder(self, fixeds):
        # morp_loss = self.args.w_morp * torch.nn.functional.mse_loss(self.warpeds[0], fixeds)
        morp_loss = self.args.w_morp * self.ncc_loss(fixeds, self.gen_mf)
        
        total_loss = self.args.w_proto*self.encoder.proto_loss + self.args.w_contra*self.encoder.contra_loss + self.args.w_morp * morp_loss
        
        total_loss.backward(retain_graph=True)
        self.recoder['proto_loss'].append(self.args.w_proto * self.encoder.proto_loss.item())
        self.recoder['contra_loss'].append(self.args.w_contra * self.encoder.contra_loss.item())
        self.recoder['morp_loss'].append(self.args.w_morp * morp_loss.item())

    def backward_gen(self):
        # style and content loss
        style_loss, content_loss = self.encoder.forward2(self.gen_mf)
        # content_loss = self.ncc_loss(self.encoder.moving, self.gen_mf)
        total_loss = self.args.w_style * style_loss + self.args.w_content * content_loss
        # total_loss = self.args.w_content * content_loss
        # total_loss = self.args.w_style * style_loss
        total_loss.backward(retain_graph=True)
        self.recoder['style_loss'].append( self.args.w_style * style_loss.item())
        self.recoder['content_loss'].append(self.args.w_content * content_loss.item())

    def backward_reg(self, fixeds):
        # sim and smooth loss
        sim_loss = 0
        for i in range(len(self.warpeds)):
            sim_loss = sim_loss + torch.nn.functional.mse_loss(self.warpeds[i], fixeds)
        # sim_loss = torch.nn.functional.mse_loss(self.warpeds[-1], fixeds)
        smooth_loss = losses.reg_loss(fixeds, self.flows)
        
        # gan loss
        gan_loss = self.net_D.gen_loss(self.gen_mf)

        total_loss = self.args.w_sim * sim_loss + self.args.w_smooth * smooth_loss + self.args.w_gan * gan_loss 
                    # + self.args.w_proto * self.encoder.proto_loss + self.args.w_contra * self.encoder.contra_loss
        total_loss.backward()
        self.recoder['sim_loss'].append(self.args.w_sim * sim_loss.item())
        self.recoder['smooth_loss'].append(self.args.w_smooth * smooth_loss.item())
        self.recoder['gan_loss'].append(self.args.w_gan * gan_loss.item())
        # self.recoder['proto_loss'].append(self.encoder.proto_loss.item())
        # self.recoder['contra_loss'].append(self.encoder.contra_loss.item())
        self.recoder['total_loss'].append(total_loss.item())

    def optimize(self, epoch, movings, fixeds, m_type, f_type):
        self.global_step += 1
        self.forward(movings, fixeds, m_type, f_type)
        # backward dis

        if epoch % 1 == 0:
            self.set_requires_grad([self.encoder, self.gen, self.reg], False)
            self.set_requires_grad([self.net_D], True)
            self.optimizerD.zero_grad()
            self.backward_dis(fixeds)
            # torch.nn.utils.clip_grad_norm_(list(self.net_D.parameters()), max_norm=20, norm_type=2)
            self.optimizerD.step()
        
        # backward encoder
        self.set_requires_grad([self.net_D], False) # 固定dis参数
        self.set_requires_grad([self.encoder], True) # 更新解码器和编码器
        self.optimizerT.zero_grad()
        self.backward_style_encoder()

        # backward decoder
        self.set_requires_grad([self.encoder], False) # 固定dis参数
        self.set_requires_grad([self.gen], True) # 更新解码器和编码器
        self.optimizerG.zero_grad()
        self.backward_gen()

        # backward gen and reg
        self.set_requires_grad([self.reg], True) # 优化reg, gen
        self.optimizerR.zero_grad()
        self.backward_reg(fixeds)
        
        torch.nn.utils.clip_grad_norm_(list(self.gen.parameters()), max_norm=10, norm_type=2)
        torch.nn.utils.clip_grad_norm_(list(self.reg.parameters()), max_norm=10, norm_type=2)
        
        self.optimizerT.step()
        self.optimizerR.step()
        self.optimizerG.step()

        self.set_requires_grad([self.encoder], True)

        if self.global_step % self.args.tb_save_freq == 0:
            self.tb_save_step += 1
            ran_id = int((torch.rand(1)*0.8+0.1) * movings.size(2))
            self.writer.add_image('train/reg_img', (movings[0,0,ran_id:ran_id+1,:,:]), self.tb_save_step)
            self.writer.add_image('train/reg_fixed', (fixeds[0,0,ran_id:ran_id+1,:,:]), self.tb_save_step)
            self.writer.add_image('train/reg_trans', (self.gen_mf[0,0,ran_id:ran_id+1,:,:]), self.tb_save_step)
            for i in range(len(self.warpeds)):
                self.writer.add_image(f'train/warped_{i}', (self.warpeds[i][0,0,ran_id:ran_id+1,:,:]), self.tb_save_step)   


    def train_encoder(self):
        abdomen_dataset = AbdomenDataset1(Path(args.train_dir),same_skip=True)
        self.train_loader = Data.DataLoader(abdomen_dataset,batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=True)

        self.global_step = (self.now_epoch+1) * len(self.train_loader)
        self.tb_save_step = self.global_step // self.args.tb_save_freq

        for epoch in range(self.now_epoch+1, self.args.epoch+1):
            epoch_info = "Epoch:{}/{} - ".format(epoch, self.args.epoch) + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print(epoch_info)
            for movings, fixeds, m_type, f_type in self.train_loader:
                movings, fixeds = movings.float().to(self.device), fixeds.float().to(self.device)
                
                self.encoder.set_input(movings, fixeds, m_type, f_type)
                self.encoder.forward1()
                
                total_loss = self.args.w_proto*self.encoder.proto_loss + self.args.w_contra*self.encoder.contra_loss
                total_loss.backward()
                self.optimizerT.step()
                self.optimizerT.zero_grad()
                self.recoder['proto_loss'].append(self.args.w_proto*self.encoder.proto_loss.item())
                self.recoder['contra_loss'].append(self.args.w_contra*self.encoder.contra_loss.item())

            # 打印信息
            self.print_loss(epoch_info, is_write=True)
            self.clear_recoder()

            if epoch % args.save_per_epoch == 0:
                self.save_model(epoch=epoch)


    def train_model(self):
        # abdomen_dataset = AbdomenDataset1(Path(args.train_dir),same_skip=True)
        brats_train = BraTSRegDataset1(Path('E:\\datasets\\BRAST2018\\MICCAI_BraTS_2018_Data_Training\\HGG'), 
                                    fixed_seqs=['t1','t21','t1ce1'], moving_seqs=['t1','t21','t1ce1'], 
                                    resize=(64, 64, 64), label_flag=4, flag=0)
        self.train_loader = Data.DataLoader(brats_train, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

        # test_dataset = AbdomenDataset2(Path(args.test_image_dir), Path(args.test_label_dir), same_skip=True)

        brats_val = BraTSRegDataset1(Path('E:\\datasets\\BRAST2018\\MICCAI_BraTS_2018_Data_HGG_val'), fixed_seqs=['t1','t21','t1ce1'], moving_seqs=['t1','t21','t1ce1'], 
                                     resize=(64, 64, 64), label_flag=4, flag=1)
        self.test_dataloader = Data.DataLoader(brats_val, batch_size=args.test_batch_size, shuffle=False, num_workers=0, drop_last=True)

        self.global_step = (self.now_epoch+1) * len(self.train_loader)
        self.tb_save_step = self.global_step // self.args.tb_save_freq
        best_dice = 0
        for epoch in range(self.now_epoch+1, self.args.epoch+1):
            epoch_info = "Epoch:{}/{} - ".format(epoch, self.args.epoch) + time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print(epoch_info)
            for movings, fixeds, m_type, f_type in self.train_loader:
                movings, fixeds = movings.float().to(self.device), fixeds.float().to(self.device)
                self.optimize(epoch, movings, fixeds, m_type, f_type)

            # 打印信息
            self.write_tb(epoch) 
            self.print_loss(epoch_info, is_write=True)
            self.clear_recoder()

            if epoch % 10 == 0:
                mean_dice = self.test_model(epoch)
                if mean_dice > best_dice:
                    best_dice = mean_dice
                    self.save_model(is_best=True)

            if epoch % args.save_per_epoch == 0:
                self.save_model(epoch=epoch)

    def test_model(self, epoch):
        self.encoder.eval()
        self.gen.eval()
        self.reg.eval()

        dice_dict = defaultdict(list)
        thresholds = np.linspace(0,1,101)
        step = 0
        gap = len(self.test_dataloader)

        jpg_path = Path(self.args.root_path) / self.args.test_jpg_outpath

        with torch.no_grad():
            dices = []
            for movings, fixeds, moving_lab, fixed_lab, m_type, f_type in self.test_dataloader:
                moving_img = movings.to(self.device).float()
                fixed_img = fixeds.to(self.device).float()
                moving_lab = moving_lab.to(self.device).float()
                fixed_lab = fixed_lab.to(self.device).float()

                self.forward(moving_img, fixed_img, m_type, f_type)
                # flows, _ = self.reg.infer(moving_img, fixed_img)

                warped_lab = moving_lab
                for i in range(len(self.flows)):
                    warped_lab = self.stn(warped_lab, self.flows[i])

                if jpg_path is not None:
                    for b in range(len(m_type)):
                        out_dir = Path(jpg_path) / (m_type[b] + '-' + f_type[b])
                        out_dir.mkdir(exist_ok=True)
                        out_dir_name = out_dir / m_type[b]
                        out_dir_name.mkdir(exist_ok=True)
                        for z in range(movings.shape[2]):
                            vutils.save_image(moving_img[b,0,z], str(out_dir_name / f'{z}_m.jpg'), normalize=True)
                            vutils.save_image(fixed_img[b,0,z], str(out_dir_name / f'{z}_f.jpg'), normalize=True)
                            vutils.save_image(self.gen_mf[b,0,z], str(out_dir_name / f'{z}_t.jpg'), normalize=True)

                ran_id = torch.argmax(torch.sum(fixed_lab,(3,4)),dim=-1)[0].squeeze()
                # ran_id = int((torch.rand(1)*0.8+0.1) * input_img.size(2))
                self.writer.add_image('test_label/moving_lab', moving_lab[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                self.writer.add_image('test_label/fixed_lab', fixed_lab[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)          
                self.writer.add_image('test_label/warped_lab', warped_lab[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                warped_fixed = (warped_lab>0)*0.5 + (fixed_lab)*0.5
                self.writer.add_image('test_label/overlap', warped_fixed[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                warped_fixed = (moving_lab>0)*0.5 + (fixed_lab)*0.5
                self.writer.add_image('test_label/overlap_real', warped_fixed[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)

                self.writer.add_image('test_image/moving_img', moving_img[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                self.writer.add_image('test_image/fixed_img', fixed_img[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                self.writer.add_image('test_image/trans_img', self.gen_mf[0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                for i in range(len(self.warpeds)):
                    self.writer.add_image(f'test_image/wapred_{i}', self.warpeds[i][0,0,ran_id:ran_id+1,:,:], epoch*gap+step)
                    flow_vis = self.flowshow.show(self.stn, self.flows[i])
                    self.writer.add_image(f'test_image/flow_{i}', flow_vis[0,0,ran_id:ran_id+1,:,:]/255, epoch*gap+step)
                
                step = step + 1

                for i in range(len(m_type)):
                    key = m_type[i] + '_' + f_type[i]
                    dice_ = []
                    for threshold in thresholds:
                        dice_.append(vali.dice2(warped_lab[i], fixed_lab[i], threshold))
                    dice_dict[key].append(dice_)

            dice_detail = f'\nepoch:{epoch}\n'
            dice_info = f'\nepoch:{epoch}\n'
            key_max = 0
            for k in dice_dict.keys():
                dice_array =  np.array(dice_dict[k]).squeeze(-1)
                index = dice_array.mean(0).argmax()
                max_v = dice_array.mean(0)[index]
                std_v = dice_array.std(0)[index]
                dice_detail = dice_detail + str(dice_array.mean(0))[1:-1].replace(' ',',').replace('\n',' ') + '\n'
                dice_info = dice_info + f'{k}:{max_v},{std_v},{index}\n' 
                key_max += max_v

            with open(str(Path(self.args.root_path) / 'expriment_test.txt'), 'a') as f:  # 设置文件对象
                print(dice_detail, flush=True, file = f)
            
            print(dice_info)
            with open(str(Path(self.args.root_path) / 'log_test.txt'), 'a') as f:  # 设置文件对象
                print(dice_info, flush=True, file = f)

        self.encoder.train()
        self.gen.train()
        self.reg.train()
        return key_max / len(dice_dict.keys())

    def test2(self, moving, fixed, m_type, f_type):
        self.forward(moving, fixed, m_type, f_type)
        # flows, warpeds = self.infer(moving, fixed)
        return self.gen_mf, self.flows, self.warpeds


class PMARNet21(PMARNet2):
    def __init__(self, args):
        super(PMARNet21, self).__init__(args)

    def backward_gen(self, fixeds):
        # style and content loss
        style_loss, content_loss = self.encoder.forward2(self.gen_mf)
        
        # gan loss
        gan_loss = self.net_D.gen_loss(self.gen_mf)

        sim_loss0 = torch.nn.functional.mse_loss(self.warpeds[0], fixeds)
        smooth_loss0 = losses.reg_loss(fixeds, self.flows[0:1])
        # content_loss = self.ncc_loss(self.encoder.moving, self.gen_mf)
        total_loss = self.args.w_style * style_loss + self.args.w_content * content_loss + self.args.w_gan*gan_loss  + self.args.w_sim*sim_loss0 + self.args.w_smooth*smooth_loss0
        # total_loss = self.args.w_content * content_loss
        # total_loss = self.args.w_style * style_loss
        total_loss.backward(retain_graph=True)

        self.recoder['style_loss'].append( self.args.w_style * style_loss.item())
        self.recoder['content_loss'].append(self.args.w_content * content_loss.item())
        self.recoder['gan_loss'].append(self.args.w_gan * gan_loss.item())
        self.recoder['sim_loss0'].append( self.args.w_sim * sim_loss0.item())
        self.recoder['smooth_loss0'].append( self.args.w_smooth * smooth_loss0.item())

    def backward_reg(self, fixeds):
        # sim and smooth loss
        sim_loss = 0
        sim_loss = torch.nn.functional.mse_loss(self.warpeds[-1], fixeds)
        # sim_loss = torch.nn.functional.mse_loss(self.warpeds[-1], fixeds)
        smooth_loss = losses.reg_loss(fixeds, self.flows[1:])

        total_loss = self.args.w_sim * sim_loss + self.args.w_smooth * smooth_loss 

        total_loss.backward()
        self.recoder['sim_loss'].append(self.args.w_sim * sim_loss.item())
        self.recoder['smooth_loss'].append(self.args.w_smooth * smooth_loss.item())
        self.recoder['total_loss'].append(total_loss.item())

    def optimize(self, epoch, movings, fixeds, m_type, f_type):
        self.global_step += 1
        self.forward(movings, fixeds, m_type, f_type)
        # backward dis

        if epoch % 1 == 0:
            self.set_requires_grad([self.encoder, self.gen, self.reg], False)
            self.set_requires_grad([self.net_D], True)
            self.optimizerD.zero_grad()
            self.backward_dis(fixeds)
            # torch.nn.utils.clip_grad_norm_(list(self.net_D.parameters()), max_norm=20, norm_type=2)
            self.optimizerD.step()
        
        # backward encoder
        self.set_requires_grad([self.net_D], False) # 固定dis参数
        self.set_requires_grad([self.encoder], True) # 更新解码器和编码器
        self.optimizerT.zero_grad()
        self.backward_encoder(fixeds)

        # backward decoder
        self.set_requires_grad([self.encoder], False) # 固定编码器
        self.set_requires_grad([self.gen, self.reg], True)
        self.optimizerG.zero_grad()
        self.backward_gen(fixeds)

        # backward gen and reg
        self.set_requires_grad([self.gen], False) # 固定dis参数
        self.optimizerR.zero_grad()
        self.backward_reg(fixeds)
        
        # torch.nn.utils.clip_grad_norm_(list(self.gen.parameters()), max_norm=10, norm_type=2)
        # torch.nn.utils.clip_grad_norm_(list(self.reg.parameters()), max_norm=10, norm_type=2)
        
        self.optimizerT.step()
        self.optimizerR.step()
        self.optimizerG.step()

        self.set_requires_grad([self.encoder, self.gen], True)

        if self.global_step % self.args.tb_save_freq == 0:
            self.tb_save_step += 1
            ran_id = int((torch.rand(1)*0.8+0.1) * movings.size(2))
            self.writer.add_image('train/reg_img', (movings[0,0,ran_id:ran_id+1,:,:]), self.tb_save_step)
            self.writer.add_image('train/reg_fixed', (fixeds[0,0,ran_id:ran_id+1,:,:]), self.tb_save_step)
            self.writer.add_image('train/reg_trans', (self.gen_mf[0,0,ran_id:ran_id+1,:,:]), self.tb_save_step)
            for i in range(len(self.warpeds)):
                self.writer.add_image(f'train/warped_{i}', (self.warpeds[i][0,0,ran_id:ran_id+1,:,:]), self.tb_save_step)