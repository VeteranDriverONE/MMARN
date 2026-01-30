from audioop import mul
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from models.losses import ContrastiveLoss
from torch.distributions.normal import Normal
from torchvision import transforms

from pathlib import Path
# if __name__ == "__main__":
#     import losses as losses
# else:
#     import models.losses as losses

class DoubleConv(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, 
        conv_op=nn.Conv3d,conv_args={'kernel_size':3,'stride':1,'padding':1,'dilation': 1, 'bias': True}, conv_args2=None,
        norm_op=nn.BatchNorm3d, norm_op_args={'eps': 1e-5, 'affine': True, 'momentum': 0.1},
        non_line_op=nn.LeakyReLU, non_line_op_args={'negative_slope': 1e-2, 'inplace': True},
        first_op=None, first_op_args=None,
        last_op=None, last_op_args=None):

        super(DoubleConv, self).__init__()

        self.first_op = first_op
        self.last_op = last_op

        if conv_args2 is None:
            conv_args2 = conv_args

        self.conv1 = conv_op(in_channel, mid_channel, **conv_args)
        self.norm1 = norm_op(mid_channel, **norm_op_args)
        self.non_line1 = non_line_op(**non_line_op_args)

        self.conv2 = conv_op(mid_channel, out_channel, **conv_args2)
        self.norm2 = norm_op(out_channel, **norm_op_args)
        self.non_line2 = non_line_op(**non_line_op_args)
        
        if first_op is not None:
            self.first_op = first_op(**first_op_args)

        if last_op is not None:
            self.last_op = last_op(**last_op_args)

    def forward(self,x):
        if self.first_op is not None:
            x = self.first_op(x)

        x1 = self.non_line1(self.norm1(self.conv1(x)))
        x2 = self.non_line2(self.norm2(self.conv2(x1)))

        if self.last_op is not None:
            x2 = self.last_op(x2)

        return x2

class Trans(nn.Module):
    # 采用风格
    def __init__(self,inshape, in_ch=1, out_ch=1, stage_num=5, types=('CMP','NP','UP'),
        conv_pool=False, pool_args=None, up_sample=False, up_sample_args=None, freeze_e1=False, freeze_e2=False, freeze_d=False):
        super(Trans,self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        freeze_prototypes = True

        # 网络参数
        base_ch = 32
        self.stage_num = stage_num
        self.conv_pool = conv_pool
        self.up_sample = up_sample

        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        if up_sample and up_sample_args is None:
            up_sample_args = {'scale_factor':2, 'mode':'nearest', 'align_corners':None}
        
        if conv_pool:
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, conv_args2=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, conv_args2=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, conv_args2=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, conv_args2=pool_args),
                DoubleConv(8*base_ch, 16*base_ch,16*base_ch),
            ]
            encode_blocks2 = [
                DoubleConv(in_ch, base_ch, base_ch, conv_args2=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, conv_args2=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, conv_args2=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, conv_args2=pool_args),
                DoubleConv(8*base_ch, 16*base_ch,16*base_ch),
            ]
        else:
            encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(8*base_ch, 16*base_ch, 16*base_ch),
            ]
            encode_blocks2 = [
                DoubleConv(in_ch, base_ch, base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(8*base_ch, 16*base_ch, 16*base_ch),
            ]

        decode_blocks = [
            DoubleConv(2*16*base_ch, 16*base_ch, 8*base_ch),
            DoubleConv(2*8*base_ch, 8*base_ch, 4*base_ch),
            DoubleConv(2*4*base_ch, 4*base_ch, 2*base_ch),
            DoubleConv(2*2*base_ch, 2*base_ch, base_ch),
            DoubleConv(2*base_ch, base_ch, out_ch)
        ]

        if up_sample:
            self.up_sample = [nn.Upsample(**up_sample_args)] * 4
        else:
            self.up_sample = [
                nn.ConvTranspose3d(3*8*base_ch, 3*8*base_ch, 2,2),
                nn.ConvTranspose3d(3*4*base_ch, 3*4*base_ch, 2,2),
                nn.ConvTranspose3d(3*2*base_ch, 3*2*base_ch, 2,2),
                nn.ConvTranspose3d(3*base_ch, 3*base_ch, 2,2),
            ]

        self.encode_blocks = nn.ModuleList(encode_blocks)
        self.encode_blocks2 = nn.ModuleList(encode_blocks2)
        self.decode_blocks = nn.ModuleList(decode_blocks)
        self.up_sample = nn.ModuleList(self.up_sample)

        self.spatialtransformer = SpatialTransformer(inshape)
        conv_fn = getattr(nn, 'Conv%dd' % 3)
        self.flow = conv_fn(16, 3, kernel_size=3, padding=1)
        self.trans = nn.MaxPool3d(2,2,0)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        # self.batch_norm = getattr(nn, "BatchNorm{0}d".format(3))(3)

        self.contra_loss = ContrastiveLoss(1)

        input = torch.rand(inshape).unsqueeze(0).unsqueeze(0)
        prototypes = []
        self.masks = {}
        self.numel = []
        k_size = [5,5,3,3,3]
        for d in range(self.stage_num):
            input = self.encode_blocks[d](input)
            gap = k_size[d] // 2
            if d > 2:
                mask = []
                for z in range(input.shape[2]):
                    for h in range(input.shape[3]):
                        for w  in range(input.shape[4]):
                            tmp_mask = torch.zeros_like(input[0,0],dtype=torch.bool)
                            tmp_mask[max(z-gap,0):z+gap+1,max(h-gap,0):h+gap+1,max(w-gap,0):w+gap+1] = True
                            mask.append(tmp_mask.flatten())

                mask = (torch.stack(mask).unsqueeze(0) * ~torch.eye(input[0,0].numel(),dtype=bool)).cuda()
                mask_idx = torch.nonzero(mask).T  # 这里需要转置一下
                mask_data = mask[mask_idx[0], mask_idx[1], mask_idx[2]]
                coo_mask = torch.sparse_coo_tensor(mask_idx, mask_data, mask.shape)

                self.masks[str(d)] = coo_mask
                # self.numel.append(self.masks[str(d)].sum(-1).byte())
            prototype = {}
            for k in types:
                # prototype[k] = nn.Parameter(torch.rand_like(input.squeeze(0)))
                prototype[k] = nn.Parameter(torch.rand((input.shape[1], input.shape[1])))
                # prototype[k] = torch.rand((input.shape[1], input.shape[1]))
            prototypes.append(nn.ParameterDict(prototype))
            # prototypes.append(prototype[k])
        self.prototypes = nn.ParameterList(prototypes)
        # self.register_buffer('prototypes', prototypes)

        for d in range(self.stage_num):
            if freeze_e1:
                for _, p in self.encode_blocks[d].named_parameters():
                    p.requires_grad = False

            if freeze_e2:
                for _, p in self.encode_blocks2[d].named_parameters():
                    p.requires_grad = False
            
            if freeze_prototypes:
                for _, p in self.prototypes[d].items():
                    p.requires_grad = False
        
        for u in range(self.stage_num):
            if freeze_d:
                for _, p in self.decode_blocks[u].named_parameters():
                    p.requires_grad = False

    def forward(self, moving, fixed, moving_filename, fixed_filename, trans=False):
        # device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu' )
        x = moving
        x2 = moving
        y = fixed
        y2 = fixed
        bs = moving.shape[0]
        x_filename = moving_filename
        y_filename = fixed_filename

        prototype_loss = torch.tensor(0)
        content_loss = torch.tensor(0)
        style_loss = torch.tensor(0)
        contra_loss = torch.tensor(0)

        skip_content_moving = []
        skip_content_fixed = []
        skip_proto_moving =[]
        skip_proto_fixed =[]

        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            y = self.encode_blocks[d](y)
            x2 = self.encode_blocks2[d](x2)
            y2 = self.encode_blocks2[d](y2)
            
            skip_proto_moving.append(x)
            skip_proto_fixed.append(y)
            skip_content_moving.append(x2)
            skip_content_fixed.append(y2)
            
            x_gram = self.gram2(x)
            y_gram = self.gram2(y)

            if d>1:
                prototype_loss = prototype_loss + (x_gram - self.getPrototypes(x_filename, d)).square().mean() \
                            + (y_gram - self.getPrototypes(y_filename, d)).square().mean()
                contra_loss = contra_loss + self.contra_loss(x_gram, y_gram, 1)
            
            for i  in range(bs):
                self.setPrototype(x_gram[i].detach(), str(x_filename[i]), d)
                self.setPrototype(y_gram[i].detach(), str(y_filename[i]), d)

        # X -> Y        
        d_x = torch.concat([x2, y],dim=1)
        d_x = self.decode_blocks[0](d_x)
        for u in range(1,self.stage_num):
            # x = torch.concat([d_x, skip_content_moving[-(u+1)], self.getPrototypes(y_filename,-(u+1))], dim=1)
            # d_x = torch.concat([d_x, skip_content_moving[-(u+1)], skip_proto_fixed[-(u+1)]], dim=1)
            d_x = torch.concat([d_x, skip_content_moving[-(u+1)]], dim=1)
            d_x = self.decode_blocks[u](self.up_sample[u-1](d_x))

        if trans:
            return d_x, prototype_loss, contra_loss

        # encoder again
        warped_x = d_x
        warped_x2 = d_x
        
        for d in range(self.stage_num):
            warped_x = self.encode_blocks[d](warped_x)
            warped_x2 = self.encode_blocks2[d](warped_x2)
            
            # style_loss = style_loss + (self.gram2(warped_x) - self.getPrototypes(y_filename, d)).square().mean()
            # style_loss = style_loss + (self.gram(warped_x) - self.gram(skip_proto_fixed[d])).square().mean()
            if d > 1: 
                style_loss = style_loss + (self.gram2(warped_x) - self.getPrototypes(y_filename, d)).square().mean()

            if d > 2:
                geo_mx = self.geo_loss2(warped_x2,d) # m->f, B,C,ZHW,ZHW
                geo_fx = self.geo_loss2(skip_content_moving[d],d)

                tmp_we1 = geo_mx.square().sum(-1).sqrt()
                tmp_we2 = geo_fx.square().sum(-1).sqrt()
                we1_s = (torch.softmax(tmp_we1/tmp_we1.max(), dim=1)).unsqueeze(-1)
                we2_s = (torch.softmax(tmp_we2/tmp_we2.max(), dim=1)).unsqueeze(-1)

                # content_loss = content_loss + torch.mean((geo_mx*we1 - geo_fx*we2).square().sum(-1) / self.numel[d-2])
                # content_loss = content_loss + torch.mean((((geo_mx*we1 - geo_fx*we2).square()+ 1e-6).log()*self.masks[str(d)]).sum(-1) / self.numel[d-2])
                # content_loss = content_loss + torch.mean(((geo_mx-geo_fx).square()).sum(-1) / self.numel[d-2])
                # content_loss = content_loss - torch.mean( (torch.exp(-(geo_mx*we1-geo_fx*we2).square())*self.masks[str(d)]).sum(-1)/self.numel[d-2])
                
                # geo_mx = geo_mx * we1_s
                # geo_fx = geo_fx * we2_s
                cos_sim = F.cosine_similarity(geo_mx, geo_fx, dim=-1)

                # cos_sim = torch.sum(geo_mx * geo_fx, dim=-1) / (geo_mx.square().sum(-1).sqrt() * geo_fx.square().sum(-1).sqrt())
                # content_loss = content_loss + torch.mean(torch.exp(1-cos_sim))-1
                # content_loss = content_loss + torch.mean(torch.log(1 - cos_sim + 1e-6))
                content_loss = content_loss + torch.exp(1-cos_sim).mean()

                # sim = (geo_mx-geo_fx)/(geo_mx+geo_fx+1e-10)
                # content_loss = content_loss + torch.mean(torch.exp(cos_sim.sum(-1)/self.numel[d-2]))

                # diff = (geo_mx-geo_fx).square()
                # content_loss = content_loss + torch.mean( (diff / diff.sum(-1)) / self.numel[d-2] )
                # content_loss = content_loss - torch.mean( (-(diff / diff.sum(-1))).exp() * self.masks[str(d)]  ) 

        return d_x, prototype_loss, contra_loss, style_loss, content_loss

    def setPrototype(self,x,x_filename,stage):
        p = self.prototypes[stage][x_filename]
        prototype = (1-self.moment) * x + self.moment * p
        self.prototypes[stage][x_filename] = prototype
    
    def getPrototypes(self,x_filename,stage):
        prototypes = []
        for filename in x_filename:
            prototypes.append(self.prototypes[stage][str(filename)])
        return torch.stack(prototypes)

    def cosdist(self,fts,prototype,scaler=20):
        dist = F.cosine_similarity(fts, prototype, dim=1) * scaler
        return dist.detach().cpu().numpy()

    def gram(self, X):
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        return X.bmm(X.transpose(1,2)) / (c * n)
    
    def gram2(self, X):
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        gram = X.bmm(X.transpose(1,2))
        factor = X.square().sum(-1,keepdims=True).sqrt()  # b, c
        norm = factor.bmm(factor.transpose(1,2))
        return gram / norm

    def style_loss(self, moved_prototype, prototype, y_type):
        center_loss = torch.square(self.gram(moved_prototype) - self.gram(prototype[str(y_type)])).mean()
        return center_loss

    def mse_loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)

    def modlity_loss(self, source_ft, source_file, source_pros):
        source_pro = source_pros[source_file].unsqueeze(0)
        # loss = torch.pow((source_ft-source_pro),2).mean()
        # loss = F.cosine_similarity(source_ft,source_pro,1).mean()
        loss = (self.gram(source_ft) - self.gram(source_pro)).square().mean()
        return loss
    
    def geo_loss(self, X, d):
        # cosine dist
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n))
        gram = torch.bmm(X.transpose(1,2), X) / c
        # return self.masks[str(d)] * gram
        return torch.spares.mm(gram,self.masks[str(d)])
    
    def geo_loss2(self, X, d):
        # Euc dist
        b, c = X.shape[0], X.shape[1]
        n = X.numel() // (b*c)
        X = X.reshape((b, c, n)).unsqueeze(-1)
        gram = (X - X.transpose(-2,-1))
        # return self.masks[str(d)] * gram
        return gram * self.masks[str(d)].to_dense()
    
    def geo_loss(self, X, d):
        pass


class Reg(nn.Module):
    def __init__(self,inshape, stage_num=5, freeze=True, deep_supervision=False,
        conv_pool=False, pool_args=None, up_sample=False, up_sample_args=None):
        super(Reg,self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        
        # 网络参数
        self.stage_num = stage_num
        self.deep_supvision = deep_supervision
        self.up_sample = up_sample
        self.encode_blocks = []
        self.decode_blocks = []
        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        if up_sample and up_sample_args is None:
            up_sample_args = {'size':2, 'scale_factor':2, 'mode':'nearest', 'align_corners':None}

        if conv_pool:
            self.encode_blocks = [
                DoubleConv(1,32,32,conv_args2=pool_args),
                DoubleConv(32,64,64,conv_args2=pool_args),
                DoubleConv(64,128,128,conv_args2=pool_args),
                DoubleConv(128,256,256,conv_args2=pool_args),
                DoubleConv(256,512,512),
            ]
        else:
            self.encode_blocks = [
                DoubleConv(1,32,32,last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(32,64,64,last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(64,128,128,last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(128,256,256,last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(256,512,512),
            ]

        self.decode_blocks = [
            DoubleConv(512+256,512,128),
            DoubleConv(128*2,256,64),
            DoubleConv(64*2,128,32),
            DoubleConv(32*2,32,16)
        ]

        if up_sample:
            self.up_sample = [nn.Upsample(**up_sample_args)] * 4
        else:
            self.up_sample = [
                nn.ConvTranspose3d(512+256,512+256,2,2),
                nn.ConvTranspose3d(256,256,2,2),
                nn.ConvTranspose3d(128,128,2,2),
                nn.ConvTranspose3d(64,64,2,2),
            ]

        self.encode_blocks = nn.ModuleList(self.encode_blocks)
        self.decode_blocks = nn.ModuleList(self.decode_blocks)
        self.up_sample = nn.ModuleList(self.up_sample)

        self.spatialtransformer = SpatialTransformer(inshape)
        conv_fn = getattr(nn, 'Conv%dd' % 3)
        self.flow = conv_fn(16, 3, kernel_size=3, padding=1)
        self.trans = nn.MaxPool3d(2,2,0)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow.weight = nn.Parameter(nd.sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))
        # self.batch_norm = getattr(nn, "BatchNorm{0}d".format(3))(3)

        self.final_conv = conv_fn(16, 1, kernel_size=3, stride=2, padding=1) 

    def forward(self, moving, fixed):
        # device = torch.device('cuda:0'if torch.cuda.is_available() else 'cpu' )
        x = moving
        y = fixed

        skip_moving = []
        skip_fixed = []
        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            y = self.encode_blocks[d](y)
            if d!=self.stage_num-1:
                skip_moving.append(x)
                skip_fixed.append(y)

        # X -> Y
        for u in range(self.stage_num-1):
            x = torch.concat([x, skip_fixed[-(u+1)]], dim=1)
            # x = torch.concat([x,self.getPrototypes(y_filename,-(u+1)),skip_moving[-u]], dim=1)
            x = self.decode_blocks[u](self.up_sample[u](x))

        # Y -> X
        for u in range(self.stage_num-1):
            y = torch.concat([y, skip_moving[-(u+1)]], dim=1)
            # y = torch.concat([y,self.getPrototypes(x_filename,-(u+1)),skip_fixed[-u]], dim=1)
            y = self.decode_blocks[u](self.up_sample[u](y))

        x = self.flow(x)
        y = self.flow(y)

        return x, y

class Reg2(nn.Module):
    def __init__(self, inshape, in_ch=2, out_ch=3, stage_num=5, freeze=True, deep_supervision=False,
        conv_pool=False, pool_args=None, up_sample=False, up_sample_args=None):
        super(Reg2,self).__init__()

        self.pro_dict = [{} for i in range(stage_num)]
        self.prototypes = [{} for i in range(stage_num)]
        
        # 网络参数
        base_ch = 16
        self.stage_num = stage_num
        self.deep_supvision = deep_supervision
        self.up_sample = up_sample
        self.encode_blocks = []
        self.decode_blocks = []
        self.down_pool = []
        self.up_sample = []
        self.moment = 0.96

        if conv_pool is False and pool_args is None:
            pool_args = {'kernel_size':2, 'stride':2, 'padding':0, 'dilation':1,'return_indices':False, 'ceil_mode':False}
        if up_sample and up_sample_args is None:
            up_sample_args = {'size':2, 'scale_factor':2, 'mode':'nearest', 'align_corners':None}

        if conv_pool:
            self.encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, conv_args2=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, conv_args2=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, conv_args2=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, conv_args2=pool_args),
                DoubleConv(8*base_ch, 16*base_ch,16*base_ch),
            ]
        else:
            self.encode_blocks = [
                DoubleConv(in_ch, base_ch, base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(base_ch, 2*base_ch, 2*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(2*base_ch, 4*base_ch, 4*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(4*base_ch, 8*base_ch, 8*base_ch, last_op=nn.MaxPool3d, last_op_args=pool_args),
                DoubleConv(8*base_ch, 16*base_ch, 16*base_ch),
            ]

        self.decode_blocks = [
            DoubleConv((16+8)*base_ch, 8*base_ch, 8*base_ch),
            DoubleConv((8+4)*base_ch, 4*base_ch, 4*base_ch),
            DoubleConv((4+2)*base_ch, 2*base_ch, 2*base_ch),
            DoubleConv((2+1)*base_ch, base_ch, base_ch)
        ]

        if up_sample:
            self.up_sample = [nn.Upsample(**up_sample_args)] * 4
        else:
            self.up_sample = [
                nn.ConvTranspose3d((16+8)*base_ch, (16+8)*base_ch, 2,2),
                nn.ConvTranspose3d((8+4)*base_ch, (8+4)*base_ch, 2,2),
                nn.ConvTranspose3d((4+2)*base_ch, (4+2)*base_ch, 2,2),
                nn.ConvTranspose3d((2+1)*base_ch, (2+1)*base_ch, 2,2)
            ]

        self.encode_blocks = nn.ModuleList(self.encode_blocks)
        self.decode_blocks = nn.ModuleList(self.decode_blocks)
        self.up_sample = nn.ModuleList(self.up_sample)

        conv_fn = getattr(nn, 'Conv%dd' % 3)
        self.flow1 = conv_fn(base_ch, out_ch, kernel_size=3, padding=1)
        self.flow2 = conv_fn(base_ch, out_ch, kernel_size=3, padding=1)
        # Make flow weights + bias small. Not sure this is necessary.
        nd = Normal(0, 1e-5)
        self.flow1.weight = nn.Parameter(nd.sample(self.flow1.weight.shape))
        self.flow1.bias = nn.Parameter(torch.zeros(self.flow1.bias.shape))
        self.flow2.weight = nn.Parameter(nd.sample(self.flow2.weight.shape))
        self.flow2.bias = nn.Parameter(torch.zeros(self.flow2.bias.shape))
        # self.batch_norm = getattr(nn, "BatchNorm{0}d".format(3))(3)

    def forward(self, moving, fixed):
        # x = moving
        # y = fixed

        # skip_moving = []
        # skip_fixed = []
        # for d in range(self.stage_num):
        #     x = self.encode_blocks[d](x)
        #     y = self.encode_blocks[d](y)
        #     if d!=self.stage_num-1:
        #         skip_moving.append(x)
        #         skip_fixed.append(y)

        # # X -> Y
        # for u in range(self.stage_num-1):
        #     x = torch.concat([x, skip_fixed[-(u+1)]], dim=1)
        #     # x = torch.concat([x,self.getPrototypes(y_filename,-(u+1)),skip_moving[-u]], dim=1)
        #     x = self.decode_blocks[u](self.up_sample[u](x))

        # # Y -> X
        # for u in range(self.stage_num-1):
        #     y = torch.concat([y, skip_moving[-(u+1)]], dim=1)
        #     # y = torch.concat([y,self.getPrototypes(x_filename,-(u+1)),skip_fixed[-u]], dim=1)
        #     y = self.decode_blocks[u](self.up_sample[u](y))

        # x = self.flow(x)
        # y = self.flow(y)

        x = torch.concat([moving, fixed],dim=1)
        skips = []
        for d in range(self.stage_num):
            x = self.encode_blocks[d](x)
            if d!=self.stage_num-1:
                skips.append(x)

        for u in range(self.stage_num-1):
            x = torch.concat([x, skips[-(u+1)]], dim=1)
            x = self.decode_blocks[u](self.up_sample[u](x))

        # x = self.flow(x)

        return self.flow1(x), self.flow2(x)

    def attention(self, x, y):
        
        return


class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super(SpatialTransformer, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode

    def forward(self, src, flow):
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode, align_corners=True , padding_mode='border')
        # return F.grid_sample(src, new_locs, mode=self.mode, align_corners=False)

class Regs(nn.Module):
    def __init__(self, reg_module, n_cascades=1, inshape=None, args={}):
        super(Regs,self).__init__()
        
        stems = []
        for i in range(n_cascades):
            stems.append(reg_module(**args))
        self.stn = SpatialTransformer(inshape)
        self.stems = nn.ModuleList(stems)
        
    # old
    # def forward(self,moving, fixed, tf):
    #     flows = []
    #     warpeds = []

    #     flow, _ =self.stems[0](tf, fixed)
    #     warped = self.stn(tf, flow)
    #     flows.append(flow)
    #     warpeds.append(warped)
    #     for i in range(1, len(self.stems)):
    #         flow, _ = self.stems[i](warpeds[-1], fixed)
    #         flows.append(flow)
    #         warpeds.append(self.stn(warpeds[-1], flows[-1]))

    #     return flows, warpeds
    def cal_errors(self, fixed, tf):
        # calculation error directly
        # err = (fixed-tf.detach()).abs()
        # 3sigma
        err = (fixed-tf.detach())
        mean = err.mean()
        var = err.var()
        err = torch.max(mean-3*var, err)
        err = torch.min(mean+3*var, err)
        err = err.pow(2)
        # norm
        err = (err-err.min()) / (err.max()-err.min())
        return err

    def cal_errors2(self, fixed, tf):
        err = (fixed-tf.detach()).pow(2)
        return err
    
    # new
    def forward(self,moving, fixed, tf):
        flows = []

        warped_moving_ls = []
        warped_tf_ls = []
        residual_ls = []

        residual = self.cal_errors(fixed, tf.detach())
        residual_ls.append(residual)

        flow, _ =self.stems[0](moving, fixed)
        flows.append(flow)
        
        warped_moving = self.stn(moving, flow)
        warped_moving_ls.append(warped_moving)

        warped_tf = self.stn(tf, flow)
        warped_tf_ls.append(warped_tf)

        for i in range(1, len(self.stems)):
            residual = residual + self.cal_errors(fixed, warped_tf.detach())
            residual = residual / 2
            residual_ls.append(residual)

            flow, _ = self.stems[i](warped_moving_ls[-1], fixed)
            flows.append(flow)

            warped_moving_ls.append(self.stn(warped_moving_ls[-1], flows[-1]))

            warped_tf_ls.append(self.stn(warped_tf_ls[-1], flows[-1]))

        return flows, warped_tf_ls, warped_moving_ls, residual_ls
    
    def infer(self, moving, fixed):
        flows = []
        warpeds = [moving]
        for i in range(0, len(self.stems)):
        # for i in range(0,1): # cas1
            flow, _ = self.stems[i](warpeds[-1], fixed)
            flows.append(flow)
            warpeds.append(self.stn(warpeds[-1], flows[-1]))

        return flows, warpeds[1:]

class Regs2(nn.Module):
    def __init__(self, reg_module, n_cascades=1, inshape=None, args={}):
        super(Regs2,self).__init__()
        
        self.n_cascades = n_cascades
        self.stn = SpatialTransformer(inshape)
        self.stem = reg_module(**args)

    
    # new
    def forward(self,moving, fixed, tf):
        flows = []

        warped_moving_ls = []
        warped_tf_ls = []

        flow, _ =self.stem(moving, fixed)
        flows.append(flow)

        warped_moving = self.stn(moving, flow)
        warped_moving_ls.append(warped_moving)

        warped_tf = self.stn(tf, flow)
        warped_tf_ls.append(warped_tf)

        for i in range(1, self.n_cascades):
            flow, _ = self.stem(warped_moving_ls[-1], fixed)
            flows.append(flow)

            warped_moving_ls.append(self.stn(warped_moving_ls[-1], flows[-1]))

            warped_tf_ls.append(self.stn(warped_tf_ls[-1], flows[-1]))

        return flows, warped_tf_ls
    
    def infer(self, moving, fixed):
        flows = []
        warpeds = [moving]
        
        for i in range(0, self.n_cascades):
        # for i in range(0,2): # cas1
            flow, _ = self.stem(warpeds[-1], fixed)
            flows.append(flow)
            warpeds.append(self.stn(warpeds[-1], flows[-1]))

        return flows, warpeds[1:]

# if __name__ == "__main__":
#     model = UNet2((64,128,192))
#     moving = torch.rand(1,1,64,128,192)
#     fixed = torch.rand(1,1,64,128,192)
#     warped_init,feature,ini_reg_loss,Center_loss = model(moving,'CMP',fixed)
#     ini_reg_loss.backward()
#     print('运行结束')
